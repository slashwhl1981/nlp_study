# -*- coding: utf-8 -*-
"""
train_langid_cls_v12.py —— 多语种「语种识别」分类器（课后作业 · 12 语种版）

任务：一句话 → 它是什么语言（12 分类）
数据：data/langid/v12（课堂 8 语种 + ko / th / vi / tr）
模型：与课堂 8 语种版完全相同（Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear）
损失：CrossEntropyLoss      优化器：Adam
运行：CPU 单线程约 30 秒（12 个 epoch）。

     python train_langid_cls_v12.py
     python train_langid_cls_v12.py --save langid_rnn_v12.pt

【本版相对课堂 8 语种版，只改了这几处】
  1. LABELS 从 8 个扩到 12 个 —— 模型输出维度 = len(LABELS) = 12
  2. 数据目录从 v8 换成 v12
  3. 字符表不再读课件发的 vocab_v8.txt，而是从 v12/train.csv 重新统计生成
     （DATA_CARD 里明确说了 12 类的字符表不随课件提供，要自己生成）。
     生成逻辑直接复用课堂的 build_char_vocab()：
     统计每个字符出现次数 → 按次数从高到低编号，第 0 号 <PAD>、第 1 号 <UNK>。
     第一次运行会把结果写成 data/langid/vocab/vocab_v12.txt，之后直接读文件。
  4. epoch 从课堂的 8 改成 12（数据从 8 类变 12 类，多跑 4 轮能再提一点，
     主要是 pt / es 这一对；CPU 上也就多花 9 秒）。

【为什么只需改这几处】
  模型和训练代码不关心"是哪 12 种语言"：字符级输入把汉字、假名、谚文、泰文、
  西里尔字母、拉丁字母一视同仁，类别数由数据决定。
  所以要换语种，永远是「改 LABELS + 换数据 + 重建字符表」这三步。
"""

import os
import sys
import csv
import time
import random
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 直接复用课堂脚本里的组件：同一份模型、同一套字符表/编码/评测工具
from train_langid_cls import (                      # noqa: E402
    LangIDClassifier, build_char_vocab, load_vocab_file, encode, normalize,
    TextDataset, macro_f1, confusion, print_confusion,
    MAX_LEN, EMBED_DIM, HIDDEN_DIM, DROPOUT, BATCH_SIZE, LR, SEED,
)

# ─── 12 语种配置 ───────────────────────────────────────────────────────
LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh",
          "ko", "th", "vi", "tr"]          # = 课堂 8 个 + 新增 4 个
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

LANG_NAMES = {
    "ar": "阿拉伯语", "en": "英语", "es": "西班牙语", "hi": "印地语",
    "ja": "日语", "pt": "葡萄牙语", "ru": "俄语", "zh": "中文",
    "ko": "韩语", "th": "泰语", "vi": "越南语", "tr": "土耳其语",
}


# ─── 字符表：没有就现场生成 ────────────────────────────────────────────
def build_vocab_file(path, train_rows):
    """
    从训练集统计字符表并写成文件（格式与 vocab_v8.txt 一致）。

    第 0 号 <PAD>、第 1 号 <UNK>，其余按出现次数从高到低编号。
    出现次数相同时按字符编码排序，保证每次生成的编号完全一样。
    """
    vocab = build_char_vocab([t for t, _ in train_rows])
    freq = Counter()
    for t, _ in train_rows:
        freq.update(normalize(t))     # 与 build_char_vocab 用的是同一种归一化

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")
    return vocab


def ensure_vocab(path, train_rows, rebuild=False):
    """字符表存在就直接读，不存在（或 --rebuild-vocab）就按 v12 训练集重新生成。"""
    if os.path.exists(path) and not rebuild:
        return load_vocab_file(path), False
    vocab = build_vocab_file(path, train_rows)
    return vocab, True


# ─── 读数据 ────────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="强制按训练集重新生成字符表（默认已有文件就直接用）")
    ap.add_argument("--save", default=os.path.join(HERE, "langid_rnn_v12.pt"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须都在 LABELS 里，且 12 个类都得有训练数据
    seen = {l for _, l in train_rows}
    extra = sorted(seen - set(LABELS))
    if extra:
        raise SystemExit(f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}")
    missing = [l for l in LABELS if l not in seen]
    if missing:
        raise SystemExit(f"这些标签在训练集里没有样本：{' '.join(missing)}")

    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    vocab, built = ensure_vocab(args.vocab, train_rows, rebuild=args.rebuild_vocab)
    unk = sum(1 for t, _ in train_rows for ch in normalize(t) if ch not in vocab)

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：" + " ".join(LABELS))
    print(f"  字符表 {len(vocab)} 个字符"
          + (f"（已从训练集生成 → {args.vocab}）" if built else f"（读取 {os.path.basename(args.vocab)}）"))
    print(f"  训练集 <UNK> 字符数：{unk}")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            loss = criterion(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    print(f"\n训练耗时：{time.time()-t0:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 测试集
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=5)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}({LANG_NAMES[lab]}): {hit}/{len(idx)} = {hit/max(1, len(idx)):.3f}")

    # 每个语种抽一条测试句做推理演示
    print("\n--- 推理示例（每个语种一条）---")
    model.eval()
    picks = {}
    for text, lab in test_rows:
        if lab not in picks:
            picks[lab] = text
        if len(picks) == num_classes:
            break
    with torch.no_grad():
        for lab in LABELS:
            text = picks[lab]
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            mark = "✓" if LABELS[pid] == lab else "✗"
            print(f"  {mark} 真实[{lab}] 预测[{LABELS[pid]}] p={prob[pid]:.2f}  {text[:46]}")

    torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                "labels": LABELS, "maxlen": MAX_LEN,
                "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT}, args.save)
    print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
