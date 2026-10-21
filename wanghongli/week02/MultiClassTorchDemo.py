# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写多分类模型训练
规律：x是一个5维向量，5个数字中最大的数字在第几维，就属于第几类
例如：x = [0.1, 0.9, 0.3, 0.2, 0.5]，最大的是第2个数，所以属于第2类(从0开始计数，标签为1)

与二分类任务的区别：
1. 输出维度由1变为类别数5，使用softmax把5个输出归一化成概率
2. 标签由0/1变为0~4的整数(LongTensor)，损失函数使用交叉熵CrossEntropyLoss
3. 预测时取概率最大的那一维作为预测类别(argmax)

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)  # 线性层：5维输入 -> 5个类别的得分
        self.activation = torch.softmax  # softmax把得分转换成概率
        self.loss = nn.functional.cross_entropy  # 多分类使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值(每个类别的概率)
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            # torch的cross_entropy内部自带softmax，所以直接传入未归一化的得分即可
            return self.loss(logits, y)  # 预测得分和真实标签计算损失
        else:
            return self.activation(logits, dim=1)  # 输出每个类别的概率


# 生成一个样本
# 随机生成一个5维向量，最大数字在第几维(从0开始数)，标签就是几
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 最大值的下标，范围0~4
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = np.zeros((total_sample_num, 5), dtype=np.float32)
    Y = np.zeros(total_sample_num, dtype=np.int64)
    for i in range(total_sample_num):
        x, y = build_sample()
        X[i] = x
        Y[i] = y
    return torch.from_numpy(X), torch.from_numpy(Y)  # 标签是整数，所以用LongTensor


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本，各类别数量：%s" % (test_sample_num, np.bincount(y.numpy(), minlength=5).tolist()))
    with torch.no_grad():
        y_pred = model(x)  # 得到每个样本在5个类别上的概率
        pred = torch.argmax(y_pred, dim=1)  # 概率最大的类别即预测结果
    correct = int((pred == y).sum())
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    class_num = 5  # 类别数：最大值可能出现在5个维度中的任意一个
    learning_rate = 0.05  # 学习率
    # 建立模型
    model = TorchModel(input_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_multiclass.bin")
    torch.save(model.state_dict(), model_path)
    print("模型已保存到：%s" % model_path)
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.xlabel("epoch")
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_curve.png"))
    plt.show()
    return model_path


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    class_num = 5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，得到5个类别的概率
    for vec, res in zip(input_vec, result):
        true_class = int(np.argmax(vec))  # 真实类别：最大值所在维度
        pred_class = int(torch.argmax(res))  # 预测类别：概率最大的维度
        print("输入：%s, 最大值在第%d维(真实类别%d), 预测类别：%d, 概率值：%f"
              % (["%.4f" % v for v in vec], true_class + 1, true_class + 1,
                 pred_class + 1, float(res[pred_class])))


if __name__ == "__main__":
    # 训练模型并得到模型保存路径
    saved_model_path = main()
    # 用训练好的模型做预测
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]
    predict(saved_model_path, test_vec)
