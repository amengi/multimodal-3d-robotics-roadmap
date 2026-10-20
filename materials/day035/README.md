# Day 035 配套材料：Gauss–Newton 与对比表示审计

这是一个不依赖外部数据的可复现周项目。第一部分用 NumPy 实现二维点到直线的 Gauss–Newton（GN）拟合，用中心差分检查雅可比，并与全最小二乘（TLS/PCA）几何基线比较。第二部分对合成的图像—文本表示计算对称 InfoNCE，分开报告严格配对检索、语义检索、下游分类、NLL、Brier 与 ECE。

## 输入、输出与边界

- 几何输入：`points_m` shape `(N, 2)`，单位 m，合成 world-xy 坐标系。
- 几何参数：`[theta_rad, rho_m]` shape `(2,)`，法线式 `x cos(theta)+y sin(theta)-rho=0`。
- 对比输入：`image`/`text` shape `(64, 8)`，每行是一个 L2 归一化表示；行号定义严格配对。
- 主输出：`report.json`、`gn_convergence.png`、`contrastive_diagnostics.png`。
- 边界：合成表示只用于教学。`log(batch_size)-InfoNCE` 是依赖 batch 构造的下界诊断，不是真实 MI 的精确测量，也不是因果证据。

## 从项目根目录运行

```bash
python -m pip install -r materials/day035/requirements.txt
python materials/day035/day035_project.py --output-dir /tmp/day035_outputs
python -m unittest materials.day035.test_day035 -v
```

成功标志：程序最后输出 `SUCCESS`，测试全部为 `OK`，且三个输出文件均非空。典型输出的最后几行形如：

```text
Jacobian max abs error: < 1e-7
GN/TLS orthogonal RMSE: 两者几乎相同 m
Downstream test accuracy/NLL/Brier/ECE: 四个有限数
Wrote /tmp/day035_outputs/report.json
SUCCESS
```

## 快速故障处理

- `ModuleNotFoundError: numpy/matplotlib`：先运行上面的 `pip install` 命令，再记录 `python --version`。
- `No module named materials`：你没有在项目根目录运行测试；先 `pwd`。
- 数值雅可比误差过大：检查中心差分是否同时计算 `f(p+h)` 和 `f(p-h)`，以及 `theta` 是否用 rad。
- 换机器后小数末位不同：看阈值和趋势，不要要求浮点字符串逐位相同。
- 图像无法弹窗：本项目使用 `Agg` 后端，直接打开 PNG 文件，不依赖 GUI。
