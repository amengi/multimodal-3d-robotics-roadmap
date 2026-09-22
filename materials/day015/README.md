# Day 015 配套材料：C++ 二维点统计与自信息量

本材料只使用 C++17 标准库。程序用变量、函数、`std::vector`、范围 `for` 和 `const` 引用统计二维点，并计算同一组概率的 bit/nat 自信息量。

## 数据与数学契约

- 点集：`points_m:(N,2)`，每行 `[x,y]`，单位 m，坐标系 `world`，`N>0`，坐标必须有限。
- 输出：点数、质心、AABB 最小/最大坐标；坐标单位均为 m。
- 自信息：事件概率 `p` 无量纲且满足 `0<p<=1`；`-log2(p)` 的单位是 bit，`-ln(p)` 的单位是 nat。
- 边界：这个固定小样本只用于学习语法和手算，不代表传感器精度或概率模型质量。

## 编译、运行与验证

从仓库根目录执行：

```bash
mkdir -p /tmp/day015-build
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  materials/day015/point_stats.cpp -o /tmp/day015-build/point_stats
/tmp/day015-build/point_stats
/tmp/day015-build/point_stats --self-test
python3 -m unittest materials.day015.test_day015 -v
```

成功标志：普通运行末尾显示 `SUCCESS`，程序自检显示 `SELF_TEST_OK: 10 checks`，Python 黑盒测试显示 `Ran 15 tests` 和 `OK`。

若找不到 `c++`，先安装 Xcode Command Line Tools（macOS）或系统 C++ 编译器；不要为了完成本日任务引入 Eigen、CMake 或第三方包，它们会在后续学习日单独学习。
