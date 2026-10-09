# Day 027 配套材料：SVD、条件数与 MDL 代理

这是一套只依赖 NumPy 和 Matplotlib 的可复现实验。它生成一张 `64×96` 合成灰度图，用截断 SVD 比较不同秩的重建误差，演示病态线性系统对微小观测误差的放大，并用声明清楚的二段编码代理比较“模型复杂度 + 残差”。它还用固定的合成预测概率比较稳定单模态基线与双模态融合模型的参数量、NLL 和描述长度代理。数据完全合成，仅用于研究教学。

## 数据契约

- 图像 `A`：shape `(64,96)`；每个元素是 `[0,1]` 内的无量纲灰度强度。
- 像素坐标：第 0 轴是行并向下增加，第 1 轴是列并向右增加；`imshow(origin='upper')` 与数组布局一致。
- 紧凑 SVD：`U:(64,64)`、`s:(64,)`、`Vh:(64,96)`，`A=(U*s)@Vh`。
- rank-`r` 参数量代理：`r*(m+n+1)`，即存 `U_r`、`s_r`、`V_r^T` 的标量个数；未考虑文件头、熵编码和正交约束，因此只是统一比较口径。
- 图像指标：MSE 为强度平方、RMSE 为强度、PSNR 为 dB；PSNR 取 `data_range=1`。
- 条件数例子：`A=diag(1,1e-4)`，无单位教学系统；`kappa_2=1e4`。

## 安装与运行

从仓库根目录执行：

```bash
python3 -m venv /tmp/day027-venv
source /tmp/day027-venv/bin/activate
python3 -m pip install -r materials/day027/requirements.txt
MPLCONFIGDIR=/tmp/day027-mpl \
python3 materials/day027/svd_condition_mdl.py --output-dir /tmp/day027-output
```

若已有满足版本范围的隔离环境，可直接使用。默认随机种子为 `2709`；动手题可用 `--seed 42` 和新的输出目录。成功标志是最后一行 `SUCCESS`，并生成：

```text
/tmp/day027-output/report.json
/tmp/day027-output/reconstructions.png
/tmp/day027-output/svd_mdl_curves.png
```

## 如何读输出

`reconstructions.png` 固定显示 rank 0、4、16 和 full。rank 0 是全零普通基线；rank 增大时，截断 SVD 的 Frobenius 误差不会增加。完整秩能在浮点容差内重建原图，但存储代理未必更小。

`svd_mdl_curves.png` 有三栏：奇异值谱、rank—RMSE、rank—描述长度代理。代理明确假定每个模型标量 16 bit，残差用固定 `sigma=0.03` 的零均值高斯密度并乘 `1/255` 量化 bin 近似概率。它用于演示拟合与复杂度的权衡，不是 exact MDL、NML、stochastic complexity，也不是跨项目可比的通用分数。

`conditioning` 同时报告直接解和三个 `pinv` cutoff。小 cutoff 保留 `1e-4` 奇异方向，拟合 noisy `b` 很好但放大噪声；`rcond=1e-3` 截断该方向，牺牲残差并产生偏差。条件数给出最坏情形敏感度上界，不保证每个扰动都达到上界。

`fusion_model_comparison` 使用 800 个固定合成标签/概率。两模型 accuracy 都为 1，因此必须再比较 NLL；融合模型参数更多但 NLL 更低。本例的代理按每个模型标量 8 bit 加上 `-log2 p(y|x)`。这个人为码长约定只用于课堂比较，不能当成 exact MDL、校准证明或真实多模态效果。

## 测试

```bash
source /tmp/day027-venv/bin/activate
MPLCONFIGDIR=/tmp/day027-mpl \
python3 -m unittest materials.day027.test_day027 -v
python3 -m compileall -q materials/day027
python3 -m json.tool /tmp/day027-output/report.json >/dev/null
test -s /tmp/day027-output/reconstructions.png
test -s /tmp/day027-output/svd_mdl_curves.png
```

19 项测试覆盖数据契约、固定种子、错误输入、SVD shape/排序/正交/回环、rank 边界、误差单调性、Eckart–Young 的 Frobenius 尾和、指标 shape、MDL 代理加和、条件数误差放大、伪逆 cutoff 权衡、融合模型 NLL/复杂度权衡和最终产物。

## 常见报错

- `matrix must be 2-D`：不要把展平后的 `(6144,)` 向量直接做本课的图像 SVD。
- `rank must be ...`：rank 只能在 `0..min(m,n)`；本图最大为 64。
- `LinAlgError: SVD did not converge`：先检查 NaN/Inf；脚本会在调用 SVD 前拒绝非有限值。
- 图缓存不可写：保留 `MPLCONFIGDIR=/tmp/day027-mpl`；脚本已使用无界面 `Agg` 后端。
- 输出数值末位不同：BLAS/LAPACK 可能带来极小浮点差；使用测试容差，不手改 JSON。
