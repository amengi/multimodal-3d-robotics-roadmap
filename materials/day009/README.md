# Day 009 参考实现

这个小包把“广播与向量化”放进两个同构问题：三维点的刚体变换，以及两种配对模态到共享表示空间的投影。代码先用逐点循环建立几何正确性基线，再用矩阵乘法和广播批处理；性能数据只在正确性通过后解释。

## 环境与运行

在项目根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r materials/day009/requirements.txt
python -m materials.day009.rigid_and_projection
python -m unittest materials.day009.test_day009 -v
```

脚本默认把 `day009_report.json` 写入 `experiments/Day009/reference_outputs/`。自动校验时可避免污染工作区：

```bash
python -m materials.day009.rigid_and_projection --output /tmp/day009-check
```

## 数据与坐标契约

- 点云 `P_source.shape == (N,3)`；每行是源坐标系中的 `[x,y,z]`，单位为米。
- `R.shape == (3,3)` 是主动、右手、源到目标坐标系的旋转；`t.shape == (3,)` 是目标坐标系中的平移，单位为米。
- 单点列向量公式是 `p_target = R @ p_source + t`；把点存成行时，批处理公式是 `P_target = P_source @ R.T + t`。
- 配对模态分别是 `A:(B,d_a)`、`B:(B,d_b)`；投影矩阵是 `W_a:(d_a,d)`、`W_b:(d_b,d)`；结果是 `Z_a,Z_b:(B,d)`，均无量纲。
- 同一行号是配对键。交换第二模态的行只用于演示配对失败，不是互信息估计或因果实验。

## 成功标志

- 三个手算点变换为 `[[10,2,-1],[8,1,-1],[10,0,0]]` 米。
- 逐点循环与向量化的最大绝对误差不超过 `1e-12 m`。
- 正确配对相似度矩阵为单位阵；交换第二模态后高相似度离开主对角线。
- 15 项测试显示 `OK`，JSON 非空。
- 加速比因机器、BLAS、温度与负载而变；不要把某个固定倍数写进正确性测试。

## 边界

向量化常常减少 Python 循环开销，但也可能创建大中间数组并增加内存压力。余弦相似度只是共享空间中的几何比较；它不证明语义等价、互信息大小或因果关系。真实机器人数据还要明确外参标定、时间同步、坐标系方向与单位。
