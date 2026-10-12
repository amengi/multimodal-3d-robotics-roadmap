# Day 029 配套材料：导数、泰勒与表示审计

本目录用两个互相独立但接口清楚的最小实验完成 Day 029：

1. 对 `sin(theta)` 扫描有限差分步长，观察截断误差与浮点舍入误差；比较一阶、二阶泰勒局部模型。
2. 枚举离散随机变量 `X=(S,N)`、标签 `Y` 和表示 `Z=f(X)`，精确检查数据处理不等式、任务充分性、准确率、NLL、Brier、校准误差、表示成本和分布变化失败。

所有数据均为合成数据，只用于数学与研究教学，不作临床诊断建议。

## 环境与运行

从仓库根目录运行：

```bash
python3 -m venv /tmp/day029-venv
source /tmp/day029-venv/bin/activate
python3 -m pip install -r materials/day029/requirements.txt
MPLCONFIGDIR=/tmp/day029-mpl \
python3 materials/day029/derivative_representation_lab.py \
  --output-dir /tmp/day029-output
```

成功时最后一行是 `SUCCESS`，并生成：

- `/tmp/day029-output/report.json`
- `/tmp/day029-output/derivative_taylor.png`
- `/tmp/day029-output/representation_audit.png`

## 数学与数据契约

### 导数与泰勒

- `theta`、展开中心 `a`、步长 `h` 的单位都是 rad；弧度在 SI 中无量纲，但本课保留 `rad` 以避免角度单位混淆。
- `sin(theta)` 无量纲，导数单位记为 `per rad`。
- 前向差分 `(f(x+h)-f(x))/h` 的截断误差是一阶；中心差分 `(f(x+h)-f(x-h))/(2h)` 是二阶。
- 扫描 `h=10^-1,...,10^-14 rad`；极小 `h` 会放大相近浮点数相减造成的舍入误差。
- 泰勒实验以 `a=0.7 rad` 展开 `sin`，在 `1.2 rad` 核对一阶和二阶局部模型。

### 表示、任务与预测

- `S,N` 是相互独立的公平二元变量；`X=(S,N)` 有 4 个状态。
- `Y=S xor E`，其中 `E~Bernoulli(0.1)`；因此 `I(X;Y)=1-H_b(0.1)=0.531004 bit/example`。
- `full_x` 保留两个 bit；`signal_z=S` 只保留一个 bit，但对 `Y` 充分；`nuisance_z=N` 和常量表示不含任务信息。
- 所有 MI 都由已知 PMF 精确枚举，不是有限样本估计或下界。
- 固定长度表示成本定义为 `log2(|Z|)`；这只是课堂编码成本，不是训练后神经网络的真实压缩率。
- 预测指标同时报告 accuracy、NLL（nat/example）、Brier、精确分组校准误差。准确率相同不代表概率质量相同。
- 鲁棒性失败把测试标签翻转率从 `0.1` 改为 `0.3`，但保留训练条件概率；没有在测试集重新拟合校准参数。

## 预期关键结果

不同平台的最优有限差分步长可能相差一个数量级；这是浮点实现差异，不要求逐字相同。以下离散结果应一致：

```text
full_x:    I=0.531004 bit, accuracy=0.900, NLL=0.325083 nat, Brier=0.090, ECE=0
signal_z:  I=0.531004 bit, accuracy=0.900, NLL=0.325083 nat, Brier=0.090, ECE=0
nuisance_z:I=0 bit,      accuracy=0.500, NLL=0.693147 nat, Brier=0.250, ECE=0
overconfident signal: accuracy=0.900, but NLL/Brier/ECE worsen
test epsilon=0.3: accuracy=0.700 and ECE=0.200 when training probabilities are frozen
```

## 测试

```bash
MPLCONFIGDIR=/tmp/day029-mpl \
python3 -m unittest materials.day029.test_day029 -v
python3 -m compileall -q materials/day029
python3 -m json.tool /tmp/day029-output/report.json >/dev/null
```

21 项测试覆盖差分阶数、极小步长失败、泰勒局部误差、PMF、解析 MI、数据处理不等式、充分性、普通/概率基线、分布变化与输出产物。

## 边界

- 导数实验只验证光滑标量函数，不能外推到不可导点或高维病态优化。
- `signal_z` 的充分性来自已知合成生成机制；真实胸片与报告的表示是否充分，不能靠降维或高 MI 宣称。
- 高 MI 不证明因果、临床效用或隐私；任务性能、校准、鲁棒性、成本和失败病例必须分别报告。
