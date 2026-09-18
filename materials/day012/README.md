# Day 012 配套材料：可审计的图与条件概率

本项目用固定随机种子生成世界坐标系中的相机真值/估计轨迹，输出曲线、散点、直方图和三维轨迹；同时从 `2×2` 联合概率表计算边缘概率、两个方向的条件概率和 Bayes 校验。

## 环境与运行

从仓库根目录执行：

```bash
python -m pip install -r materials/day012/requirements.txt
MPLCONFIGDIR=/tmp/day012-mpl-cache python -m materials.day012.plotting_probability_lab --output /tmp/day012-output
MPLCONFIGDIR=/tmp/day012-mpl-cache python -m unittest materials.day012.test_day012 -v
python -m json.tool /tmp/day012-output/day012_report.json >/dev/null
```

成功标志：程序末尾显示 `SUCCESS`；测试显示 `Ran 15 tests` 和 `OK`；输出目录内有非空 JSON 和两张 PNG。

## 数据契约

- 轨迹：`truth_m, estimate_m:(T,3)`，单位 m，世界坐标系，以 `frame_index` 一一配对。
- 误差：每帧欧氏位置误差 `||estimate-truth||₂`，单位 m；RMSE 也用 m。
- 联合 PMF：`joint[x,y]:(2,2)`，元素有限且非负、总和为 1；被条件化事件的边缘概率必须大于 0。
- 图：每个轴写单位；2D 顶视图使用 equal aspect；3D 三轴使用相同数值跨度，避免形状被拉伸。

## 输出与边界

- `trajectory_error_report.png`：三维轨迹、逐帧误差曲线、等比例顶视散点和误差直方图。
- `conditional_probability_report.png`：联合 PMF 热图和边缘/条件概率柱图。
- `day012_report.json`：数值指标、数据契约、失败对照与限制。
- 错帧 RMSE 应明显高于正确配对，但这只是同步失败的合成示例；条件概率不等于因果关系，图形美观也不等于结论可靠。
