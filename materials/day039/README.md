# Day 039 配套材料：线性高斯系统与预测不确定性审计

本材料用 NumPy 完成两个可复现的合成实验：一维位置—速度线性高斯系统的先验传播，以及 12 个二分类反例上的预测熵/ensemble disagreement 审计。它不执行 Kalman 观测更新；更新公式留到 Day 041。

## 数据与数学契约

- 状态 `x_k=[position_m, velocity_m_s]`，shape `(2,)`；位置在一维 world-x 坐标轴，单位分别为 `m`、`m/s`。
- 控制 `u_k` 是已知加速度，shape `(1,)`，单位 `m/s²`；默认全零。
- 状态矩阵 `F` shape `(2,2)`；控制矩阵 `B` shape `(2,1)`；观测矩阵 `H=I` shape `(2,2)`。
- 过程噪声由未建模加速度 `a_noise ~ N(0,sigma_a²)` 产生，`Q=sigma_a² B B^T`；`Q` 的各元素单位随对应状态分量而变。
- 两个传感器观测位置和速度，`z_k=[position_sensor_m, velocity_sensor_m_s]`，`R=diag(sigma_p²,sigma_v²)`。
- 分类部分输入 shape `(12,5)`：12 个样本、5 个 ensemble 成员的 `P(Y=1|x,member)`；标签 shape `(12,)`。
- 输出：`report.json`、`linear_gaussian_trajectory.png`、`uncertainty_audit.png`。

## 环境与运行

从项目根目录执行：

```bash
python3 -m venv /tmp/day039-venv
source /tmp/day039-venv/bin/activate
python3 -m pip install -r materials/day039/requirements.txt
MPLCONFIGDIR=/tmp/day039-mpl python3 materials/day039/linear_gaussian_lab.py \
  --output-dir /tmp/day039-output --seed 39 --steps 30 --mc-samples 4000
python3 -m unittest materials.day039.test_day039 -v
python3 -m compileall -q materials/day039
```

已有满足依赖范围的隔离环境时可复用。成功标志：脚本最后打印 `SUCCESS`，22 项测试全部为 `OK`，三个输出文件均非空。

## 预期趋势

- `F=[[1,0.5],[0,1]]`，`B=[[0.125],[0.5]]`。
- 不做观测更新时，位置的预测标准差随时间增大。
- 4000 条 Monte Carlo 轨迹在最终位置的解析 95% 区间覆盖率应接近 0.95，但不会恰好等于 0.95。
- 构造数据同时出现“高熵但预测正确”和“低熵但预测错误”；预测熵或成员分歧不能单独保证错误检测。
- `report.json` 同时给出 accuracy、NLL、Brier、ECE、错误检测 AUC 与每个样本所需模型前向次数。

## 报错处理

- `ModuleNotFoundError`：确认激活环境并从项目根目录安装依赖。
- `No module named materials`：当前目录不是项目根目录；先运行 `pwd`。
- Matplotlib 缓存不可写：保留 `MPLCONFIGDIR=/tmp/day039-mpl`。
- 协方差出现负特征值：检查是否写成 `F @ P @ F.T + Q`，并检查转置。
- 覆盖率偏离很大：确认 `--mc-samples 4000`、`seed=39`，且区间使用标准差 `sqrt(P[0,0])` 而不是方差本身。

## 实验边界

线性、Gaussian、噪声独立且尺度已知都是声明的建模假设。合成分类反例只用于教学；预测熵、ensemble disagreement、低 NLL 或低 ECE 都不构成因果、临床安全或隐私保证。涉及医学迁移时仅作研究与教学，不作临床诊断建议。
