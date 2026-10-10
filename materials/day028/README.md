# Day 028 配套材料：有限样本信息度量周验收

这套脚本用完全合成、无量纲数据比较固定直方图 plug-in 熵、KL/JS、MI/NMI，同时报告 Pearson 相关和离散查表预测的普通基线。它还实现公平 Bernoulli 源在 Hamming 失真下的率—失真小算例，以及一个明确声明编码规则的二段长度代理。

## 数据与估计器契约

- 场景：`independent`、`linear`、`nonlinear` 是成对样本 `X_i,Y_i`；`distribution_shift` 是源/目标两组未配对样本。
- shape：每次输入 `X:(n,)`、`Y:(n,)`，`n in {80,200,800}`；每个设置重复 40 次。
- 单位：原始合成数据无量纲；固定 8 个 `[-4,4]` bins；信息量用 bit；Pearson 和 accuracy 无量纲。
- 熵和边缘 KL/JS 对直方图计数加 `alpha=0.5`，以避免零支持导致无穷 KL；改变 alpha 就改变了估计器。
- MI 是联合直方图 plug-in 估计，NMI 固定为 `2I/(H(X)+H(Y))`。分布偏移场景未配对，因此 MI/NMI/Pearson/预测准确率标记为不适用，不伪造对齐。
- 区间是 40 次结果的 5%/95% 经验分位数，不是解析置信区间。
- 所有估计都是教学基准，不是真实熵/真实 MI 的证明。

## 安装与运行

从仓库根目录执行：

```bash
python3 -m venv /tmp/day028-venv
source /tmp/day028-venv/bin/activate
python3 -m pip install -r materials/day028/requirements.txt
MPLCONFIGDIR=/tmp/day028-mpl \
python3 materials/day028/weekly_information_audit.py --output-dir /tmp/day028-output
```

成功标志是末行 `SUCCESS`，并生成：

```text
/tmp/day028-output/report.json
/tmp/day028-output/finite_sample_metrics.png
/tmp/day028-output/rate_distortion_mdl.png
```

`finite_sample_metrics.png` 同时画均值和 5–95% 重复区间。独立场景的 plug-in MI 在小样本时常偏大；非线性 `Y=X^2+noise` 的 Pearson 可接近 0，但 MI 和预测基线仍能显示依赖。KL/JS 图只比较边缘分布，不证明配对依赖或因果。

`rate_distortion_mdl.png` 左图固定源 `Bernoulli(0.5)`、Hamming 失真和 `0<=D<=0.5`，绘制 `R(D)=1-h2(D)`。右图固定每个自由概率参数 4 bit，将模型代理与测试 NLL bit 相加。这是课堂二段代理，不是 exact MDL、NML 或 stochastic complexity。

## 测试

```bash
source /tmp/day028-venv/bin/activate
MPLCONFIGDIR=/tmp/day028-mpl python3 -m unittest materials.day028.test_day028 -v
python3 -m compileall -q materials/day028
python3 -m json.tool /tmp/day028-output/report.json >/dev/null
test -s /tmp/day028-output/finite_sample_metrics.png
test -s /tmp/day028-output/rate_distortion_mdl.png
```

22 项测试覆盖 shape/非有限值检查、PMF、熵、KL 的零支持、JS 对称性、已知 MI/NMI、配对边界、有限样本趋势、非线性失效案例、率—失真单调性、MDL 代理取舍和最终产物。

## 常见报错

- `values must be finite`：先处理 NaN/Inf，不要让它们默默落入某个 bin。
- `x and y must be paired`：只有同一样本的 `X_i,Y_i` 才能估 MI；两个未配对数据集只比较边缘分布。
- KL 为无穷：`P` 有正质量而 `Q` 为 0；平滑可避免数值无穷，但必须报告 alpha。
- 图形缓存不可写：保留 `MPLCONFIGDIR=/tmp/day028-mpl`。
- 末位小数不同：NumPy/平台浮点实现可带来极小差异；以测试容差和趋势为准，不手改 JSON。
