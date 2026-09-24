# Day 016 配套材料：引用、指针、生命周期与 Bernoulli 熵

本材料只依赖 C++17 标准库和 Python 3 标准库。C++ 程序演示别名、按值/按引用传参、非拥有指针与边界检查，并把 Bernoulli 熵表写成 CSV；Python 脚本把 CSV 验证后绘成无需第三方库的 SVG。

## 数据与数学契约

- 距离：`ranges_m:(N,)`，`double`，单位 m，传感器坐标系，`N>0`，每项有限且非负。
- 引用：`T&` 是已存在对象的别名；`const T&` 允许只读访问。引用本身不延长普通具名对象的生命周期。
- 指针：示例中的 `const double*` 是可为空的非拥有观察者，不负责 `delete`；只有在被指对象仍存活且地址未因容器重分配失效时才可解引用。
- Bernoulli 随机变量：`X∈{0,1}`，`P(X=1)=p`，`P(X=0)=1-p`，`0≤p≤1`，概率无量纲。
- 二进制熵：`H₂(X)=-p log₂p-(1-p)log₂(1-p)`，单位 bit；约定 `0 log₂0=0`。

## 编译、运行与验证

从仓库根目录执行：

```bash
mkdir -p /tmp/day016-build
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  materials/day016/lifetime_entropy.cpp -o /tmp/day016-build/lifetime_entropy
/tmp/day016-build/lifetime_entropy
/tmp/day016-build/lifetime_entropy --self-test
/tmp/day016-build/lifetime_entropy --write-csv /tmp/day016-build/entropy.csv
python3 materials/day016/plot_entropy.py \
  /tmp/day016-build/entropy.csv /tmp/day016-build/entropy.svg
python3 -m unittest materials.day016.test_day016 -v
```

成功标志：普通运行末尾有 `SUCCESS`；程序自检显示 `SELF_TEST_OK: 14 checks`；CSV 有 11 个数据点；绘图显示 `PLOT_OK`；黑盒测试显示 `Ran 18 tests` 和 `OK`。

不要编写或运行会解引用悬空指针的代码来“观察结果”：那属于未定义行为，任何表象都不能作为正确结论。今天用生命周期图、容器失效规则、受控空指针/越界拒绝来学习失败边界。
