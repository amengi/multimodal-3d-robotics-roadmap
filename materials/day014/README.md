# Day 014 配套材料：点云几何与非线性依赖周项目

本项目用固定随机种子完成两条互相校验的主线：对立方体表面点云施加已知刚体变换和高斯噪声，报告质心、轴对齐包围盒（AABB）与配对 RMSE；生成 `X~N(0,1), Y=X²+ε`，比较 Pearson 相关、事件独立性反例和常数/线性/二次回归基线。

## 环境与运行

从仓库根目录执行：

```bash
python -m pip install -r materials/day014/requirements.txt
MPLCONFIGDIR=/tmp/day014-mpl-cache python -m materials.day014.weekly_project --output /tmp/day014-output
MPLCONFIGDIR=/tmp/day014-mpl-cache python -m unittest materials.day014.test_day014 -v
python -m json.tool /tmp/day014-output/day014_report.json >/dev/null
```

成功标志：程序末尾显示 `SUCCESS`；测试显示 `Ran 15 tests` 与 `OK`；输出目录中有非空 JSON 和两张 PNG。

## 数据契约

- 点云：`points_m:(N,3)`、`float64`、单位 m；源坐标系 `cube_local`，目标坐标系 `world`，以 `point_index` 配对。
- 行向量变换：`p_target = p_source @ R.T + t + noise`；`R:(3,3)` 必须正交且 `det(R)=+1`，`t:(3,)` 单位 m。
- AABB：各轴 `max-min`，单位 m；它依赖坐标轴，旋转后即使物体没有变形，AABB 也会变化。
- 随机变量：`X,Y:(1200,)`、无量纲；`Y=X²+ε`。接近零的 Pearson 相关只表示线性关系弱，不证明独立。

## 输出与边界

- `cube_transform.png`：源/目标点云、质心、相同数值跨度三轴。
- `nonlinear_dependence.png`：散点、二次拟合与经验联合计数。
- `day014_report.json`：几何指标、事件概率、回归基线、数据契约与限制。
- 所有结论仅针对固定合成数据；不证明因果、真实传感器性能或一般独立性。
