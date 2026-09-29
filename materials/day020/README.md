# Day 020 配套材料：Eigen 最小二乘、SVD 与互信息基础

本材料使用 Eigen 3.4.0/C++17 完成 `(5,2)` 超定直线拟合，并用 QR、SVD 与 NumPy 三路核对；同一可执行程序从二元配对计数构造联合、边缘和条件分布，用三条等价公式计算互信息，并与多数类准确率、`Y=X` 规则准确率及破坏配对后的失败案例对照。

## 输入、shape、单位与随机变量

- 设计矩阵 `A:(5,2)`：第一列是自变量 `x`，无量纲；第二列全 1，表示截距。
- 观测 `b:(5,)`：距离，单位 m；模型 `b_hat = slope*x + intercept_m`。
- 参数 `beta:(2,)=[slope,intercept_m]`；残差和 RMSE 单位 m。
- 配对变量 `X,Y∈{0,1}`；行是 `X`，列是 `Y`，计数矩阵为 `[[3,1],[1,3]]`，总样本数 8。
- 概率无量纲；所有信息量用 `log2`，单位 bit。
- 这 8 个样本上的 MI 是经验 plug-in 值，不是已知的总体真实 MI，也不是因果效应。

## 获取 Eigen 3.4.0

Eigen 是 header-only；本机没有系统安装，因此验证时只把官方固定版本解压到 `/tmp`，不把第三方源码提交到仓库：

```bash
curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
  -o /tmp/eigen-3.4.0.tar.gz
shasum -a 256 /tmp/eigen-3.4.0.tar.gz
```

预期 SHA-256：

```text
8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
```

确认哈希后再解压：

```bash
rm -rf /tmp/day020-eigen
mkdir -p /tmp/day020-eigen
tar -xzf /tmp/eigen-3.4.0.tar.gz -C /tmp/day020-eigen --strip-components=1
test -f /tmp/day020-eigen/Eigen/Dense && echo EIGEN_OK
export DAY020_EIGEN_DIR=/tmp/day020-eigen
```

若系统已有 Eigen，令 `DAY020_EIGEN_DIR` 指向包含 `Eigen/Dense` 的目录即可。外网临时不可用时，正文仍可完成手算；代码实践延后到能取得官方头文件时补跑，不能声称已验证。

## 配置、构建与运行

本机普通 `PATH` 没有 CMake，但 CLion 自带 CMake 4.3.1：

```bash
export DAY020_CMAKE=/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake
```

其他机器若 `cmake --version` 成功，使用 `export DAY020_CMAKE=cmake`。然后从仓库根目录执行：

```bash
rm -rf /tmp/day020-build
"$DAY020_CMAKE" -S materials/day020 -B /tmp/day020-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN3_INCLUDE_DIR="$DAY020_EIGEN_DIR"
"$DAY020_CMAKE" --build /tmp/day020-build --parallel 2
/tmp/day020-build/eigen_mi_demo
/tmp/day020-build/eigen_mi_demo --self-test
```

预期关键输出：

```text
eigen_version=3.4.0
top_block_shape=(3,2) top_block_sum=6.000000
qr_slope=1.990000
qr_intercept_m=1.040000
svd_slope=1.990000
svd_intercept_m=1.040000
beta_max_abs_diff=0.000000
rmse_m=0.146287
condition_number=4.738720
h_joint_bits=1.811278
h_x_given_y_bits=0.811278
mi_conditional_bits=0.188722
mi_entropies_bits=0.188722
mi_kl_bits=0.188722
match_rule_accuracy=0.750000
majority_y_accuracy=0.500000
shifted_mi_bits=0.000000
SELF_TEST_OK: 19 checks
```

## NumPy、CTest、黑盒测试与证据文件

仓库已有的 Conda `pyt` 环境含 NumPy；若你有其他 NumPy 环境，可替换命令：

```bash
conda run -n pyt python materials/day020/verify_numpy.py \
  --json /tmp/day020-build/numpy_report.json --self-test
"$DAY020_CMAKE" --build /tmp/day020-build --target test
/tmp/day020-build/eigen_mi_demo --csv /tmp/day020-build/pairing.csv
DAY020_EIGEN_DIR=/tmp/day020-eigen \
  conda run -n pyt python -m unittest materials.day020.test_day020 -v
conda run -n pyt python -m compileall -q materials/day020
```

成功标志：NumPy 10 项内测、C++ 19 项内测、CTest 14 项内测和 Python 18 项黑盒测试全部通过；JSON 与两行 CSV 非空；Eigen/NumPy 的 slope、intercept、RMSE、奇异值、condition number 和 MI 一致。

## 两个常见误区

1. 不要显式求逆 `beta=(A^T A)^{-1}A^T b` 作为默认解法；正规方程会平方条件数。今天用列主元 QR 和 JacobiSVD，并报告奇异值与条件数。
2. 不要把 `MI=0.188722 bit` 写成“X 导致 Y”或“模型性能提高 18.9%”。MI 衡量这个经验联合分布相对边缘乘积的依赖；准确率是另一个任务指标，必须单独报告。
