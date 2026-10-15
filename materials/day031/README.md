# Day 031 配套材料：投影雅可比与配对依赖审计

本目录完成两个接口相互独立、最后统一验收的最小实验：

1. 对针孔相机投影 `p_c=(X,Y,Z) -> q=(u,v)` 手写解析雅可比，并用中心有限差分逐列检查。
2. 对成对视图比较 Pearson、线性 CCA、RBF-HSIC 和仅在联合高斯假设下成立的解析 MI；再加入错配和共享标识泄漏失败案例。

所有数据都是合成数据，仅作数学与研究教学，不作临床诊断建议。

## 环境与运行

从仓库根目录执行：

```bash
python3 -m venv /tmp/day031-venv
source /tmp/day031-venv/bin/activate
python3 -m pip install -r materials/day031/requirements.txt

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/day031-mpl \
python3 materials/day031/jacobian_dependence_lab.py \
  --output-dir /tmp/day031-output \
  --seed 31 \
  --sample-count 400
```

成功标志是最后一行 `SUCCESS`，并生成：

- `/tmp/day031-output/report.json`
- `/tmp/day031-output/jacobian_sweep.png`
- `/tmp/day031-output/dependence_audit.png`

## 输入、输出、shape、单位与坐标系

### 针孔投影

- 输入点 `p_c=[X,Y,Z]`：shape `(3,)`，单位 m，**相机坐标系**，`Z>0`。
- 内参 `[fx,fy,cx,cy]`：shape `(4,)`，单位 pixel；本实验无 skew、无畸变。
- 输出像素 `q=[u,v]`：shape `(2,)`，单位 pixel，原点和轴方向按普通图像坐标约定。
- 雅可比 `J=d[u,v]/d[X,Y,Z]`：shape `(2,3)`，单位 pixel/m。
- 有限差分步长 `h`：单位 m；每次只扰动一个 3D 坐标。

默认数据：

```text
p_c = [0.4, -0.2, 2.0] m
[fx,fy,cx,cy] = [800,820,640,360] pixel
q = [800,278] pixel
J = [[400, 0, -80],
     [0, 410, 41]] pixel/m
```

解析式只覆盖无畸变针孔模型。OpenCV `projectPoints` 对旋转、平移、内参和畸变参数的完整雅可比列更多，不能把这里的 `(2,3)` 与其完整输出直接混为一谈。

### 配对依赖

- 每个视图用 `X`、`Y` 表示，样本矩阵 shape 是 `(N,Dx)`、`(N,Dy)`，同一行必须来自同一对象。
- Pearson 只比较两个第一坐标，输出无量纲相关系数。
- CCA 先白化协方差，再对 `Cxx^(-1/2) Cxy Cyy^(-1/2)` 做 SVD；输出最多 `min(Dx,Dy)` 个无量纲典型相关。
- RBF-HSIC 使用中位距离带宽和有偏有限样本统计量。数值依赖核和带宽，不是 bit，也不是 MI。
- 仅对标量联合高斯对，`I(X;Y)=-0.5 log2(1-rho^2)`，单位 bit/pair。
- 任务基线只在训练集拟合岭化最小二乘概率分数，测试集只报告 accuracy、Brier 和 NLL；未在测试集拟合阈值或校准参数。

## 预期关键结果

NumPy/平台差异会使最后几位略有不同；下面是当前固定种子结果：

```text
projected_pixel=[800.0, 278.0]
best_step_m=1.0e-05
best_jacobian_error_px_per_m=2.655725e-09
near_over_far_sensitivity=4.665730

linear_gaussian:      Pearson=0.765587, CCA1=0.765587, HSIC=0.031729
nonlinear:            Pearson=0.029071, CCA1=0.029071, HSIC=0.031506
nonlinear_shuffled:   Pearson=0.064981, CCA1=0.064981, HSIC=0.000858
shared_identifier:    Pearson=0.999592, CCA1=0.999592, HSIC=0.074068

stable_single_view:   accuracy=0.700000, Brier=0.178269, NLL=0.522847 nat/example
clean_fusion:         accuracy=0.800000, Brier=0.139923, NLL=0.425025 nat/example
shared_identifier:    accuracy=0.541667, Brier=0.249594, NLL=0.692346 nat/example
```

高斯总体参数是 `rho=0.8`，因此总体解析 MI 是 `0.736966 bit/pair`；把有限样本相关系数代回公式得到 `0.636364 bit/pair`，后者只是带高斯假设的 plug-in 值，不是真实一般 MI。

## 测试

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/day031-mpl \
python3 -m unittest materials.day031.test_day031 -v

python3 -m compileall -q materials/day031
python3 -m json.tool /tmp/day031-output/report.json >/dev/null
```

21 项测试覆盖：投影手算、无效深度/shape、解析与数值雅可比、步长舍入失败、近相机敏感性、CCA/Pearson 关系、高斯 MI 前提、非线性依赖、错配、共享标识、监督/概率基线和三项输出。

## 成本与限制

- 本实现的 RBF-HSIC 构造两个 `(N,N)` float64 核矩阵，`N=400` 时仅这两张矩阵约 `2.44 MiB`，时间和内存均为 `O(N^2)`；大数据不能原样照搬。
- 单次打乱只用于制造失败对照，不构成正式置换检验，也不给显著性结论。
- Pearson 和线性 CCA 会漏掉某些非线性依赖；HSIC 的数值又依赖核、带宽和样本量。
- 共享标识导致高依赖不等于对标签有用，更不证明因果关系、临床可靠性或隐私安全。
- 雅可比只描述当前点附近的一阶变化；大位移、`Z` 接近 0、畸变未建模或坐标系写错时，线性近似会失败。
- 图片看起来合理不能替代 JSON 指标、测试、单位核对和失败案例。
