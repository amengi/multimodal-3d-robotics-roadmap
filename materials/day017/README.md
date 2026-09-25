# Day 017 配套材料：Camera 类、RAII 与条件熵

本材料只依赖 C++17 标准库和 Python 3 标准库。C++ 程序用私有成员与构造函数保护相机参数不变式，用 `std::unique_ptr<Camera>` 表达独占所有权，用 `ProjectionReport` 的构造/析构把 CSV 文件寿命绑定到作用域，并计算固定二元联合分布的边缘熵、联合熵和条件熵。

## 数据、坐标和数学契约

- 世界点：`points_world_m:(N,3)`，顺序 `(X_w,Y_w,Z_w)`，单位 m。
- 外参：`P_c=R_cw P_w+t_cw`；`R_cw:(3,3)` 无量纲，`det=+1`；`t_cw_m:(3,)` 单位 m。
- 内参：`fx,fy,cx,cy` 均以 pixel 表示；简化针孔模型不含镜头畸变。
- 投影：`u=fx*X_c/Z_c+cx`、`v=fy*Y_c/Z_c+cy`；要求 `Z_c>0`。
- 随机变量：`X=0/1` 表示近/远深度，`Y=0/1` 表示主点左/右。固定样本联合计数为 `(3,1,1,3)`。
- 熵：`H(X,Y)=-Σ p(x,y)log₂p(x,y)`，`H(Y|X)=H(X,Y)-H(X)`，单位 bit。

## 编译、运行与验证

从仓库根目录执行：

```bash
mkdir -p /tmp/day017-build
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  materials/day017/camera_raii_entropy.cpp \
  -o /tmp/day017-build/camera_raii_entropy
/tmp/day017-build/camera_raii_entropy
/tmp/day017-build/camera_raii_entropy --self-test
/tmp/day017-build/camera_raii_entropy \
  --write-report /tmp/day017-build/projection.csv
python3 -m unittest materials.day017.test_day017 -v
python3 -m compileall -q materials/day017
```

成功标志：普通运行末尾有 `SUCCESS`；程序内测显示 `SELF_TEST_OK: 16 checks`；报表运行显示 `active_inside=1`、`active_after=0` 和 `REPORT_OK`；Python 黑盒测试显示 `Ran 19 tests` 和 `OK`。

注意：`unique_ptr` 只在需要动态寿命/所有权转移时使用；普通局部 `Camera camera{...}` 更简单时不应为了“智能指针”而堆分配。本例的熵只描述离散经验分布，不证明投影准确、因果关系或模型可靠。
