# Day 034 配套材料：鲁棒直线拟合与二比特瓶颈

## 数据与数学契约

- 几何输入：`x_m, y_m` shape 都是 `(N,)`，单位米，在同一二维直角坐标系。
- 直线模型：`y_hat = slope * x + intercept_m`；`slope` 无量纲，`intercept_m` 单位米。
- 真值：`slope=1.7`、`intercept=-0.4 m`；100 个点中精确 20% 被加入大幅 `y` 方向离群。
- 拟合比较：相同数据、初值和 SciPy 求解器，只换 `linear`、`huber`、`cauchy` 损失；鲁棒损失用 `f_scale=0.12 m`。
- 瓶颈输入：`X=(S,N)` 是两个独立均匀比特，标签 `Y=S`，表示 `Z=(Zs,Zn)` 通过两个独立二元对称信道得到。
- `capacity_bits = KL(q(Z|X)||r(Z))/ln 2`，且本例 `r(Z)` 为均匀分布。这是声明的离散玩具分布上的精确值，不是对连续确定性网络真实 MI 的估计。

## 环境与运行

从仓库根目录执行：

```bash
python3 -m venv /tmp/day034-venv
source /tmp/day034-venv/bin/activate
python3 -m pip install -r materials/day034/requirements.txt
MPLCONFIGDIR=/tmp/day034-mpl python3 materials/day034/robust_bottleneck_lab.py \
  --output-dir /tmp/day034-output --seed 34
python3 -m unittest materials.day034.test_day034 -v
python3 -m compileall -q materials/day034
```

已有满足依赖范围的隔离环境时可复用。

## 输出与成功标志

`/tmp/day034-output/` 会出现：

- `report.json`：配置、三种损失的参数/误差、VIB-like/β-VAE-like 扫描、dropout 基线。
- `robust_line_fit.png`：同一数据的拟合线和内点残差分布。
- `bottleneck_tradeoff.png`：KL 容量代理与任务/重建失真。

终端应以
`SUCCESS: robust fits and exact binary bottleneck audit completed` 结尾；22 项测试应全部通过。

## 报错处理

- `ModuleNotFoundError`：确认已激活环境，并在仓库根目录重跑 `pip install -r ...`。
- Matplotlib 缓存不可写：保留 `MPLCONFIGDIR=/tmp/day034-mpl`。
- 鲁棒结果与文档不同：检查 `--seed 34`、SciPy 版本与 `f_scale=0.12 m`；不要在测试集调这个尺度。
- 曲线重合：不是绘图错误；查看 JSON 的 `signal_error`、`nuisance_error` 和 `capacity_bits`。

## 实验边界

这是合成数据上的教学对照。鲁棒损失不会自动修正错误模型、高杠杆点或系统偏差；更小的 KL 代理、更好的重建或更高的准确率也不证明因果性、临床可靠性或隐私保证。
