# -*- coding: utf-8 -*-
"""
sft_train.py —— 在 week04 的字符级 Transformer 语言模型上做 SFT（指令微调）

【预训练 vs SFT 的区别】
    预训练：语料是任意文本，每个位置都要预测下一个字，loss 覆盖全部 token。
    SFT  ：数据是「指令 → 回答」对，只对回答部分算 loss；
           指令（prompt）部分用 -100 屏蔽掉，不让模型去学"复述问题"。
    另外 SFT 的学习率要小、轮数要少，否则很容易把预训练学到的语言能力冲掉。

【数据格式】JSONL，每行一个样本，两种写法都支持：
    {"instruction": "介绍一下你自己", "input": "", "output": "我是一个字符级语言模型。"}
    {"prompt": "问：你好\\n答：", "response": "你好，很高兴认识你。\\n"}
    · instruction/input/output 是 Alpaca 风格，脚本会自动套上「问：… \\n答：」模板
    · prompt/response 是原始文本，脚本原样使用（想自定义模板时用这种）
    · 没有数据时可以直接 `--demo`，用内置的十几条样例先跑通流程

【模板与标签对齐】
    问：{instruction}
    [input 如果有的话]
    答：{response}\\n

    输入  x = 整段序列的前 T-1 个字符
    标签  y = 整段序列的后 T-1 个字符
    然后把 y 里属于 prompt 的位置全部置成 -100（CrossEntropy 的 ignore_index），
    只有 answer 的第一个字到结尾参与 loss。

【用法】
    python sft_train.py --demo                       # 内置样例，先跑通
    python sft_train.py --data sft_data.jsonl        # 自己的数据
    python sft_train.py --data sft_data.jsonl --base ..\\week04\\循环神经网络语言模型\\best_model_transformer_tiny.pt

    # 机器慢：减小 max_len / batch_size / epochs，模型本身很小，SFT 通常几分钟
    python sft_train.py --demo --epochs 10 --batch_size 4

【说明】
    · 字符表沿用预训练 checkpoint 的 char2idx；SFT 数据里没见过的字符会映射到
      新加的 <UNK>，脚本会打印 OOV 比例，比例高就说明数据域差得太远。
    · 训练结束会打印 SFT 前后的同一组回答，方便直观对比。
    · 保存的 checkpoint 结构与 transformer_lm.py 一致，额外多了 sft 元信息。
"""

import os
import sys
import json
import math
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK04_DIR = os.path.normpath(os.path.join(HERE, "..", "week04", "循环神经网络语言模型"))
sys.path.insert(0, WEEK04_DIR)
from transformer_lm import TransformerLM  # noqa: E402


# ─────────────────────────── 模型：支持 padding mask ───────────────────────────

class SFTModel(TransformerLM):
    """
    在 TransformerLM 的基础上多接受一个 key_padding_mask。

    SFT 的样本长短不一，一个 batch 里必须补齐到同一长度；补齐位不能参与注意力，
    否则短句子的表示会被一堆 padding 稀释。前向逻辑与父类一致，只多了这一个参数。
    """

    def forward(self, x, key_padding_mask=None):
        t = x.size(1)
        h = self.pos(self.embed(x) * self.embed_scale)          # (B, T, d)
        h = self.encoder(h, mask=self.causal_mask[:t, :t],
                         src_key_padding_mask=key_padding_mask)  # True = 忽略
        return self.fc(self.drop(h))                             # (B, T, V)


# ─────────────────────────── 模板 / 编码 ───────────────────────────

def build_prompt(instruction, input_text=""):
    """Alpaca 风格 → 字符级 LM 能读的「问 / 答」模板。"""
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        instruction = instruction + "\n" + input_text
    return f"问：{instruction}\n答："


def build_answer(output):
    """回答后面补一个换行当结束符，生成时遇到换行就停。"""
    output = output.strip()
    return output + "\n" if output else "\n"


def encode_text(text, char2idx, unk_id):
    return [char2idx.get(ch, unk_id) for ch in text]


# ─────────────────────────── 数据 ───────────────────────────

class SFTDataset(Dataset):
    """
    每个样本存 (完整 token 序列, prompt 长度)。

    prompt 长度用来算 loss mask：y 的前 prompt_len-1 个位置是"预测 prompt"，
    这些位置要设成 -100；从 prompt_len-1 开始才是"预测回答的第一个字"。
    """

    def __init__(self, rows, char2idx, unk_id, max_len):
        self.samples = []
        self.truncated = 0
        for prompt, answer in rows:
            p_ids = encode_text(prompt, char2idx, unk_id)
            a_ids = encode_text(answer, char2idx, unk_id)
            ids = (p_ids + a_ids)[:max_len + 1]     # 多留一位给错位
            if len(ids) <= len(p_ids):              # 回答被截没了 → 这条样本没有监督信号
                self.truncated += 1
                continue
            self.samples.append((ids, len(p_ids)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    """
    补齐到一个 batch 里的最大长度，返回：
        x     : (B, T-1)  输入字符
        y     : (B, T-1)  目标字符，prompt 和 padding 位置是 -100
        pad   : (B, T-1)  True = padding（给 key_padding_mask 用）
    """
    max_len = max(len(ids) for ids, _ in batch)
    x = torch.zeros(len(batch), max_len - 1, dtype=torch.long)
    y = torch.full((len(batch), max_len - 1), -100, dtype=torch.long)
    pad = torch.ones(len(batch), max_len - 1, dtype=torch.bool)

    for i, (ids, prompt_len) in enumerate(batch):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        y[i, :n] = torch.tensor(ids[1:], dtype=torch.long)
        y[i, :max(0, prompt_len - 1)] = -100    # prompt 部分不计 loss
        pad[i, :n] = False

    return x, y, pad


def load_rows(path):
    """读 JSONL，兼容 instruction/input/output 与 prompt/response 两种字段。

    返回 (rows, questions)：rows 是 (prompt, answer)，questions 只用于训练结束后打印示例。
    """
    rows, questions = [], []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path} 第 {line_no} 行不是合法 JSON：{e}")

            if "prompt" in obj and "response" in obj:
                prompt = str(obj["prompt"])
                rows.append((prompt, build_answer(str(obj["response"]))))
                questions.append(str(obj.get("question", prompt.strip()))[:40])
            elif "instruction" in obj or "output" in obj:
                instruction = str(obj.get("instruction", ""))
                prompt = build_prompt(instruction, str(obj.get("input", "")))
                rows.append((prompt, build_answer(str(obj.get("output", "")))))
                questions.append(instruction)
            else:
                raise SystemExit(
                    f"{path} 第 {line_no} 行缺少字段。\n"
                    '  需要 {"instruction": "...", "output": "..."} '
                    '或 {"prompt": "...", "response": "..."}'
                )
    if not rows:
        raise SystemExit(f"{path} 里没有读到任何样本")
    return rows, questions


# 没给数据时用的内置样例（--demo），只为了跑通流程，不代表真实任务
DEMO_RAW = [
    ("你好", "你好，很高兴认识你。"),
    ("你是谁", "我是一个字符级语言模型，刚刚学过 Transformer。"),
    ("介绍一下你自己", "我是用 PyTorch 训练的字符级语言模型，可以做简单的问答。"),
    ("今天天气怎么样", "今天天气不错，适合出去走走。"),
    ("什么是人工智能", "人工智能是让机器学习和模拟人类智能的技术。"),
    ("怎么学习编程", "先选一门语言，多写代码，多做项目，遇到问题就查文档。"),
    ("Python 是什么", "Python 是一门简单易学的编程语言，常用于数据分析和人工智能。"),
    ("一加一等于几", "一加一等于二。"),
    ("推荐一部电影", "可以看看肖申克的救赎，很经典。"),
    ("怎么保持健康", "规律作息，均衡饮食，坚持运动，保持好心情。"),
    ("北京是哪个国家的首都", "北京是中国的首都。"),
    ("再见", "再见，祝你今天过得愉快。"),
]


def demo_rows():
    return ([(build_prompt(q), build_answer(a)) for q, a in DEMO_RAW],
            [q for q, _ in DEMO_RAW])


# ─────────────────────────── 加载预训练模型 ───────────────────────────

def load_pretrained(path, device):
    """
    读 week04 的 checkpoint，并给字符表补一个 <UNK>。

    预训练字表是纯字符表，没有 <UNK>；SFT 数据可能带来没见过的字。
    这里在词表末尾加一行 <UNK>，同时把 embedding / 输出层的对应行扩展出来。
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"找不到预训练模型：{path}\n"
            f"先训练一个：python {os.path.join(WEEK04_DIR, 'transformer_lm.py')} --preset tiny"
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["args"]
    char2idx = {c: int(i) for c, i in ckpt["char2idx"].items()}
    idx2char = {int(i): c for i, c in ckpt["idx2char"].items()}

    kwargs = dict(
        vocab_size=len(char2idx),
        d_model=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        nhead=cfg["nhead"],
        ff_dim=cfg["ff_dim"] or 4 * cfg["embed_dim"],
        dropout=cfg["dropout"],
        max_len=max(cfg.get("max_len", 512), cfg.get("seq_len", 64)),
        tie_weights=False,          # 扩词表后不共享，避免和预训练权重纠缠
    )
    model = SFTModel(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])

    unk_id = None
    for token in ("<UNK>", "<unk>"):
        if token in char2idx:
            unk_id = char2idx[token]
            break

    if unk_id is None:
        unk_id = len(char2idx)
        char2idx["<UNK>"] = unk_id
        idx2char[unk_id] = "<UNK>"
        # 把 embedding 和输出层各扩一行
        sd = model.state_dict()
        d_model = kwargs["d_model"]
        sd["embed.weight"] = torch.cat(
            [sd["embed.weight"], torch.randn(1, d_model) * 0.02], dim=0)
        sd["fc.weight"] = torch.cat(
            [sd["fc.weight"], torch.randn(1, d_model) * 0.02], dim=0)
        sd["fc.bias"] = torch.cat([sd["fc.bias"], torch.zeros(1)], dim=0)
        model = SFTModel(**{**kwargs, "vocab_size": len(char2idx)}).to(device)
        model.load_state_dict(sd)

    return model, char2idx, idx2char, cfg, unk_id


# ─────────────────────────── 生成 ───────────────────────────

@torch.no_grad()
def answer_question(model, char2idx, idx2char, question, unk_id,
                    max_new=60, temperature=0.8, top_k=5, device="cpu", seed=42):
    """从「问：…\\n答：」开始续写，遇到换行或到长度上限就停。"""
    torch.manual_seed(seed)          # 前后对比时固定随机性
    model.eval()
    ids = encode_text(build_prompt(question), char2idx, unk_id)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []

    for _ in range(max_new):
        logits = model(x[:, -model.max_len:])[0, -1] / max(temperature, 1e-6)
        if top_k and top_k > 0:
            topv = torch.topk(logits, min(top_k, logits.size(-1))).values
            logits[logits < topv[-1]] = -float("inf")
        nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        ch = idx2char.get(nxt, "")
        if ch == "\n":               # 换行 = 回答结束
            break
        out.append(ch)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)

    return "".join(out) or "（空回答）"


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0

    for x, y, pad in loader:
        x, y, pad = x.to(device), y.to(device), pad.to(device)
        logits = model(x, key_padding_mask=pad)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-100)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = int((y != -100).sum())
        total_loss += loss.item() * n
        total_tokens += n

    avg = total_loss / max(1, total_tokens)
    return avg, math.exp(avg)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.join(
                            WEEK04_DIR, "best_model_transformer_tiny.pt"),
                        help="预训练 checkpoint（week04 的 transformer_lm.py 产物）")
    parser.add_argument("--data", default="",
                        help="SFT 数据 JSONL；不填则必须加 --demo")
    parser.add_argument("--demo", action="store_true", help="使用内置样例（没有数据时先跑通）")
    parser.add_argument("--save", default="sft_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4, help="SFT 学习率，比预训练小")
    parser.add_argument("--max_len", type=int, default=128, help="prompt+回答的最大字符数")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="从数据里留多少做验证")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.demo and args.data:
        raise SystemExit("--demo 和 --data 只能选一个")
    if not args.demo and not args.data:
        raise SystemExit(
            "需要 SFT 数据：--data xxx.jsonl，或者先用 --demo 跑内置样例。\n"
            '数据每行：{"instruction": "...", "output": "..."}'
        )
    if args.demo and args.save == "sft_model.pt":
        args.save = "sft_demo.pt"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 预训练模型
    model, char2idx, idx2char, cfg, unk_id = load_pretrained(args.base, device)
    print(f"device: {device}")
    print(f"预训练模型：{args.base}")
    print(f"  结构：d_model={cfg['embed_dim']}  layers={cfg['num_layers']}  "
          f"heads={cfg['nhead']}  词表={len(char2idx)}")

    # 2. SFT 数据
    all_rows, all_questions = demo_rows() if args.demo else load_rows(args.data)
    total_chars = sum(len(p) + len(a) for p, a in all_rows)
    oov_chars = sum((ch not in char2idx) for p, a in all_rows for ch in p + a)
    print(f"\nSFT 样本：{len(all_rows)} 条  "
          f"（{'内置 demo' if args.demo else args.data}）")
    print(f"  字符数 {total_chars}，OOV {oov_chars} 个 "
          f"（{oov_chars / max(1, total_chars):.2%}）")

    # 打乱后切训练/验证
    order = list(range(len(all_rows)))
    random.Random(args.seed).shuffle(order)
    n_val = max(1, int(len(order) * args.val_ratio)) if len(order) >= 5 else 0
    val_rows = [all_rows[i] for i in order[:n_val]]
    train_rows = [all_rows[i] for i in order[n_val:]]

    train_ds = SFTDataset(train_rows, char2idx, unk_id, args.max_len)
    val_ds = SFTDataset(val_rows, char2idx, unk_id, args.max_len) if val_rows else None
    if len(train_ds) == 0:
        raise SystemExit("所有样本都被 max_len 截断了，请调大 --max_len 或检查数据")
    if train_ds.truncated:
        print(f"  [提示] {train_ds.truncated} 条样本的回答被 max_len 截掉，未参与训练")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate) if val_ds else None
    # 验证集太小（比如 demo 只有 1 条）时，best 选择会被单条样本的噪声带偏，
    # 这时直接用最后一轮的权重，不按验证集挑模型。
    use_val = val_loader is not None and len(val_ds) >= 3
    print(f"  训练 {len(train_ds)} 条 / 验证 {len(val_ds) if val_ds else 0} 条，"
          f"每个 epoch {len(train_loader)} 个 batch"
          + ("" if use_val else "（验证集太小，按最后一轮保存）"))

    # 3. SFT 前的回答（留作对比）
    demo_questions = all_questions[:3]
    print("\n--- SFT 之前 ---")
    for q in demo_questions:
        print(f"[{q}] {answer_question(model, char2idx, idx2char, q, unk_id, args.max_new, args.temperature, args.top_k, device)}")

    # 4. 微调
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl = run_epoch(model, train_loader, optimizer, device, train=True)
        if val_loader:
            with torch.no_grad():
                va_loss, va_ppl = run_epoch(model, val_loader, optimizer, device, train=False)
        else:
            va_loss, va_ppl = float("nan"), float("nan")

        marker = "  *" if use_val and va_ppl < best_val else ""
        if use_val and va_ppl < best_val:
            best_val = va_ppl
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": {**cfg, **vars(args),
                                                        "sft": True, "base": args.base}},
                       args.save)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}  {time.time()-t0:>5.0f}s")

    if not use_val:
        torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                    "idx2char": idx2char, "args": {**cfg, **vars(args),
                                                    "sft": True, "base": args.base}},
                   args.save)

    # 5. 有可信验证集时加载最好的权重；否则就是最后一轮
    if use_val and os.path.exists(args.save):
        ckpt = torch.load(args.save, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    print("\n--- SFT 之后 ---")
    for q in demo_questions:
        print(f"[{q}] {answer_question(model, char2idx, idx2char, q, unk_id, args.max_new, args.temperature, args.top_k, device)}")

    print(f"\nSFT 完成，模型已保存：{args.save}")
    print("提示：换数据/换模板时，prompt 的写法必须和训练时保持一致，否则效果会明显变差。")


if __name__ == "__main__":
    main()
