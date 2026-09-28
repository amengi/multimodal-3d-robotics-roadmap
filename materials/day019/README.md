# Day 019 配套材料：CMake targets、KL 与 JS 散度

这是一个只依赖 C++17 标准库的最小 CMake 工程。CMake 生成三个 target：静态库 `divergence`、可执行程序 `divergence_demo`、测试程序 `divergence_tests`。程序计算二元分布的熵、交叉熵、双向 KL 与 JS，并把支持集错配和平滑扫描写成可检查证据。

## 输入、输出与契约

- `p,q`：shape `(K,)` 的离散概率分布，无量纲；元素有限、非负且和为 1；今天 `K=2`。
- 对数底：2，因此熵、交叉熵、KL、JS 的单位都是 bit。
- 主例：`p=(0.5,0.5)`，`q=(0.75,0.25)`；同时报告不依赖对数方向的概率基线 `TV=0.5*sum|p-q|=0.25`。
- 失败例：`p=(1,0)`，`q=(0,1)`。因为 `p_i>0` 处 `q_i=0`，`D_KL(p||q)=∞`；JS 仍为 1 bit。
- `epsilon` 平滑是人为建模选择，不是真相恢复；必须报告其数值，不能用它隐藏支持集错配。

## 环境与 CMake 路径

需要 CMake 3.20+、C++17 编译器和 Python 3。先执行：

```bash
cmake --version
c++ --version
python3 --version
```

若 macOS 的 `PATH` 找不到 CMake，但已安装 CLion，本机可用：

```bash
export DAY019_CMAKE=/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake
"$DAY019_CMAKE" --version
```

变量只在当前终端有效。若你的 CMake 已在 `PATH`，可执行 `export DAY019_CMAKE=cmake`。

## 配置、构建、运行

从仓库根目录执行 out-of-source build：

```bash
export DAY019_CMAKE=${DAY019_CMAKE:-cmake}
rm -rf /tmp/day019-build
"$DAY019_CMAKE" -S materials/day019 -B /tmp/day019-build \
  -DCMAKE_BUILD_TYPE=Release
"$DAY019_CMAKE" --build /tmp/day019-build --parallel 2
/tmp/day019-build/divergence_demo
/tmp/day019-build/divergence_demo --self-test
```

预期关键输出：

```text
entropy_p_bits=1.000000
cross_entropy_pq_bits=1.207519
kl_pq_bits=0.207519
kl_qp_bits=0.188722
identity_error_bits=0.000000
js_bits=0.048795
total_variation=0.250000
support_mismatch_kl_bits=inf
SELF_TEST_OK: 11 checks
```

`-S` 指定源码树，`-B` 指定构建树。`CMakeCache.txt`、对象、静态库和可执行文件都应留在 `/tmp/day019-build`，不要写入或提交源码目录。

## CTest、黑盒测试与扫描

```bash
"$DAY019_CMAKE" --build /tmp/day019-build --target test
/tmp/day019-build/divergence_demo --csv /tmp/day019-build/smoothing.csv
sed -n '1,8p' /tmp/day019-build/smoothing.csv
python3 -m unittest materials.day019.test_day019 -v
python3 -m compileall -q materials/day019
```

成功标志：CTest 显示 `100% tests passed`；CSV 有表头和 5 行数据；16 项 Python 黑盒测试全部 `OK`。扫描中 `epsilon=0` 时 KL 为 `inf`；正平滑增大时 KL 与 JS 都下降。下降只说明定义后的分布变近，不证明两个真实域、传感器或医院等价。

## 预期失败：禁止源码内构建

不要在真实材料目录执行。测试会复制工程到临时目录，再确认这条命令失败：

```bash
cmake -S /tmp/in-source-copy -B /tmp/in-source-copy
```

失败信息应包含 `In-source builds are disabled`。修复不是删除源码，而是选择独立构建目录。若不慎在自己的副本中源码内构建，只删除副本内 CMake 生成物；不要对仓库根目录执行递归删除。

## 文件职责

```text
CMakeLists.txt
  add_library(divergence) <--- src/divergence.cpp + include/day019/divergence.hpp
          ^ PUBLIC include path and C++17 requirement propagate
          |
  target_link_libraries
     /                         \
divergence_demo             divergence_tests --add_test--> CTest
app/main.cpp                tests/test_divergence.cpp
```

`PUBLIC` include 路径是因为库自己和链接它的消费者都要包含公共头文件；`PRIVATE` 链接表示可执行程序内部使用该库，不把依赖继续传播给它的消费者。
