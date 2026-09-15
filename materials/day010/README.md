# Day 010 参考实现

这个小包把常用线性代数 API 放进两个闭环：用 `solve`/`lstsq` 求方程和超定直线拟合，用 `svd`/`eigh` 实现中心化 PCA。代码不显式求逆，且只在同一数据上对比合理基线。

## 环境与运行

从项目根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r materials/day010/requirements.txt
python -m materials.day010.linear_algebra_lab
python -m unittest materials.day010.test_day010 -v
```

默认把 `day010_report.json` 和 `line_fit_residuals.png` 写入 `experiments/Day010/reference_outputs/`。自动校验时用临时目录：

```bash
python -m materials.day010.linear_algebra_lab --output /tmp/day010-check
```

## 数据契约

- 方形系统：`A:(n,n)`、`b:(n,)`，求 `A @ x = b`；`A` 必须非奇异。
- 直线拟合：`x,y:(N,)`，单位都是米；设计矩阵 `[x,1]:(N,2)`，斜率无量纲，截距与残差的单位为米。
- PCA：`X:(N,F)` 是无量纲合成特征；中心化后 `X_c = U @ diag(s) @ Vt`；前 `k` 个主方向为 `Vt[:k]:(k,F)`，scores 为 `(N,k)`。
- `eigh` 只用于对称协方差矩阵；特征向量符号可翻转，比较方向时使用内积绝对值。

## 成功标志与边界

- 方程解为 `[2,3]`，直线 `lstsq` 明显优于只预测 `mean(y)` 的基线。
- PCA 一维重建 RMSE 低于“保留第 0 个原始坐标”基线；SVD/eigh 的首方向在符号不定意义下一致。
- JSON 和 PNG 非空，15 项单元测试显示 `OK`。
- PCA 最大化样本方差，不直接使用标签；保留方差不等于保留任务信息、真实语义或因果信息。合成数据结果不是真实机器人或医学性能证据。
