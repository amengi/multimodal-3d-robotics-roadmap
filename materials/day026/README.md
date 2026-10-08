# Day 026 配套材料：PCA 主方向与率—失真任务

这是一套只依赖 NumPy 与 Matplotlib 的可复现实验。它先用二维点集的样本协方差和 `eigh` 找主方向，再在同一名义码率下比较“只保留高方差方向”和“给低方差任务方向保留 1 bit”两种量化策略。数据完全合成，仅用于研究教学，不是医学数据或临床结论。

## 数据契约

- 点 `X`：shape `(N,2)`，每行是同一世界坐标系中的 `(x,y)`，单位 m。
- 校准集/评估集：各 `(600,2)`；PCA、量化范围只在校准集拟合，指标只在评估集报告。
- 协方差 `C`：shape `(2,2)`，单位 m²，采用 `1/(N-1)` 样本协方差。
- 特征值 `lambda`：shape `(2,)`，单位 m²，降序；特征向量 `V`：shape `(2,2)`，列向量、无量纲。
- 标签 `Y`：生成时由真实低方差轴坐标是否非负给出；评估使用仅由校准集确定的 PCA 次轴线性分类基线。它只是教学任务，展示几何失真与任务失真可能冲突。
- 率 `R`：固定长度分配的名义 bit/point；失真 `D`：平均平方欧氏距离 m²/point，另报告 RMSE m；任务指标为 accuracy。

## 安装与运行

从仓库根目录执行：

```bash
python3 -m venv /tmp/day026-venv
source /tmp/day026-venv/bin/activate
python3 -m pip install -r materials/day026/requirements.txt
python3 materials/day026/pca_rate_distortion.py --output-dir /tmp/day026-output
```

复现默认结果时保持默认 `--seed 2608`；动手题使用独立输出目录：

```bash
python3 materials/day026/pca_rate_distortion.py \
  --output-dir /tmp/day026-seed42 --seed 42
```

若仓库现有环境已满足版本范围，可不新建环境。成功标志是最后一行 `SUCCESS`，并生成：

```text
/tmp/day026-output/report.json
/tmp/day026-output/pca_direction.png
/tmp/day026-output/rate_distortion.png
```

关键输出的精确末位会随 NumPy/BLAS 有极小差异；验收应使用测试中的容差。预期主方向约 30°、角度误差小于 2°、第一主成分解释方差超过 97%。原始坐标的固定线性任务基线应超过 0.99。在 8 bit/point 下，只编码高方差轴的任务 accuracy 接近随机基线，而给低方差任务轴 1 bit 的策略应超过 0.95。

## 测试

```bash
python3 -m unittest materials.day026.test_day026 -v
python3 -m compileall -q materials/day026
python3 -m json.tool /tmp/day026-output/report.json >/dev/null
test -s /tmp/day026-output/pca_direction.png
test -s /tmp/day026-output/rate_distortion.png
```

16 项测试覆盖 shape/非有限值、手算协方差、特征方程、正交性、投影近似回环、主方向、符号不唯一、量化预算、普通原始数据基线、两种策略差异、近各向同性失败、随机种子参数和产物。

## 如何读两张图

`pca_direction.png` 中箭头从样本均值出发，长度为 `2*sqrt(lambda)`。特征向量的正负号不唯一，因此应比较无向夹角 `acos(|v^T u|)`，不能因为箭头反向就判错。

`rate_distortion.png` 左图是几何 RMSE，右图是任务 accuracy。两种编码在相同名义码率下选择了不同失真目标。曲线只是一个有限样本、标量量化器的经验结果；它不是 Shannon 最优 `R(D)`，也不证明任何真实医学压缩器安全有效。

## 预期失败：主方向不可辨认

`near_isotropic_failure` 重复生成只有 40 个点的近各向同性云。当两个特征值接近时，`eigengap=(lambda1-lambda2)/lambda1` 小；很小的样本扰动即可旋转主特征向量。此时输出方向“数值上存在”，但科学解释不稳定。修复不是强行固定符号，而是增加样本、报告 eigengap/区间，并承认方向不可辨认。

## 常见报错

- `ModuleNotFoundError`：确认已激活环境并安装 requirements。
- `points must have shape (N, 2)`：不要把 `(2,N)` 误当成每行一个点。
- 图后端报错：脚本已强制使用无界面的 `Agg`；不要删掉 `matplotlib.use("Agg")`。
- 主方向相差 180°：这是同一条无向直线，使用绝对内积计算夹角。
