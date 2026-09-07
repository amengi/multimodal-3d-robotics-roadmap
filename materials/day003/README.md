# Day 003 配套材料

本目录提供两个可复现参考实现：二维点统计，以及成对变量在保持/打乱配对关系时的三种视图。它们使用合成数据，不包含医学数据，也不计算互信息。

## 环境与运行

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r materials/day003/requirements.txt
python materials/day003/point_stats.py
python materials/day003/paired_views.py --out-dir artifacts/day003
python -m unittest materials/day003/test_day003.py -v
```

`point_stats.py` 的成功标志是最后显示 `All point-stat checks passed.`。可视化脚本应生成 `paired_original.png` 和 `paired_shuffled.png`，并显示排序后 `Y` 的最大差为 `0.0`。测试共 9 项，最后应显示 `OK`。

## 输入、输出与边界

- `point_stats.py`：输入是同一 `world` 坐标系中的有限二维点，坐标单位为米；输出质心、轴对齐包围盒、最近点和距离平方
- `paired_views.py`：输入是 `seed`、样本数 `n` 和目标相关参数 `rho`；输出两张 PNG 与两次样本相关系数
- `rho` 必须在 `[-1,1]`，`n` 至少为 2；有限样本相关不保证恰好等于目标值
- 打乱 `Y` 只改变它与 `X` 的对应关系，不改变 `Y` 自身的多重集合；这不是独立性证明，也不是互信息估计

先自行完成课件中的核心练习，再用这里的实现和测试核对。不要直接把参考实现当作个人实验提交物。
