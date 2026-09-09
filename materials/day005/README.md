# Day 005 参考实现

今天用 NumPy 掌握 `shape`、`dtype`、索引、`axis` 与广播，并把这些概念用于批量三维点坐标变换；随后用 PyTorch 组装一个两层 MLP，只做一次可检查的优化步骤，为 Day 006 的完整训练循环做准备。

## 环境与运行

在项目根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r materials/day005/requirements.txt
python -m materials.day005.vectorized_points
python -m materials.day005.mlp_shapes
python -m unittest materials.day005.test_day005 -v
```

参考程序只生成固定随机种子的合成数据，不下载数据集。测试共 12 项；全部通过时显示 `OK`。

## 文件说明

- `vectorized_points.py`：验证 `(N, 3)` 点云、旋转矩阵和位移向量，比较 Python 循环与 NumPy 向量化坐标变换。约定使用列向量公式 `p_target = R @ p_source + t`；数组实现等价写成 `points @ R.T + t`。
- `mlp_shapes.py`：构造 `2 → 8 → 1` 的二分类 MLP，输出各层 shape、logit、概率、损失与梯度范数，并执行一次 SGD 更新。
- `test_day005.py`：检查数值结果、广播、非法 shape、非有限值、非旋转矩阵、网络 shape 和单步更新。

计时结果依赖机器负载、NumPy 构建和数组规模，不把某个固定加速倍数当作测试条件。模型输入是同一坐标系中的二维点，不把 `x/y` 两列称作两个“模态”。
