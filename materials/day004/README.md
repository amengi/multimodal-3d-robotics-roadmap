# Day 004 配套材料

本目录把 Day 003 的二维点统计重构为可导入模块，并提供两个可验证的 PyTorch 自动微分例子。数据是合成或手写教学数据，不包含医学数据；本日不估计互信息。

## 环境与运行

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r materials/day004/requirements.txt
python -m materials.day004.geometry
python -m materials.day004.gradient_lab
python -m unittest materials/day004/test_day004.py -v
```

成功时几何程序最后显示 `Geometry checks passed.`；自动微分程序应显示 `y=10.000000, autograd=7.000000` 并以 `All autograd checks passed.` 结束；测试共 10 项并显示 `OK`。

## 文件和接口

- `geometry.py`：输入同一 `world` 笛卡尔坐标系中的点 `(N, 2)`，坐标单位米；输出质心、轴对齐包围盒、最近点和距离平方。它不负责坐标系变换。
- `sample_points.csv`：带 `x,y` 表头的四点样例。
- `gradient_lab.py`：先验证无量纲标量函数 `y=x²+3x` 的导数，再对形状 `(B=3, M=2)` 的双模态标准化特征做线性融合和均方误差反向传播。
- `test_day004.py`：覆盖正常路径、空输入、非有限数、布尔值、CSV 行号、数值梯度、张量 shape 和融合梯度。

先独立完成课件中的练习，再用这里的实现和测试核对。异常信息是接口的一部分；不要用裸 `except:` 隐藏未知错误。
