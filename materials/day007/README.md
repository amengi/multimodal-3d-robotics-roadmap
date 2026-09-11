# Day 007 参考实现

这个小型周验收包把第 1 周的 shape、函数/模块、异常、向量化、PyTorch 训练循环、验证指标和可复现记录放进同一个闭环。

## 运行

在项目根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r materials/day007/requirements.txt
python -m materials.day007.weekly_assessment
python -m unittest materials.day007.test_day007 -v
```

脚本默认把 `assessment_report.json` 和 `loss_curve.png` 写入 `experiments/Day007/reference_outputs/`。自动校验时可以改用临时目录：

```bash
python -m materials.day007.weekly_assessment --output /tmp/day007-check
```

## 数据契约

- `X.shape == (B, 2, 2)`：第二维是两个配对的合成传感器 A/B，第三维是每个传感器的两个无量纲特征。
- `Y.shape == (B, 1)`：二分类标签 `0/1`。
- 训练/验证索引互斥；均值和标准差只从训练样本拟合，shape 为 `(1,2,2)`。
- 普通基线为训练集多数类和仅使用稳定传感器 A 的线性分类器；融合模型使用 A+B。
- 打乱 B 只是一个配对故障对照，不是互信息估计，也不是因果证据。

## 成功标志

- 脚本报告训练 `(360,2,2)`、验证 `(120,2,2)` 且索引交集为 0。
- 四组方法都同时报告 accuracy、NLL（nats/example）和 Brier。
- 融合模型明显高于多数类基线，打乱 B 后融合准确率下降。
- 图和 JSON 均为非空文件，15 项单元测试显示 `OK`。

## 边界

这是合成数据的程序验收，不是传感器、机器人或医学性能结论。Accuracy 是判别指标；NLL 与 Brier 会惩罚坏的概率输出，但三个数值仍不足以证明校准良好。单个 split、seed 和合成生成规则不支持普遍优劣排名。
