# Day 011 配套材料：可验证的数据读写

这个小项目生成并重新读取四种数据：相机配置 `config.json`、位姿表 `poses.csv`、世界坐标系点云 `points.npy` 和标定图 `calibration.png`。它还比较 CSV 小数位数对几何误差和文件大小的影响，并验证链式法则与稳定 `log-sum-exp`。

## 环境

- Python 3.11 或更新
- NumPy 2.x
- Matplotlib 3.8 或更新

如需安装：

```bash
python -m pip install -r materials/day011/requirements.txt
```

## 运行

在仓库根目录执行：

```bash
python -m materials.day011.data_io_lab --output /tmp/day011-output
python -m unittest materials.day011.test_day011 -v
python -m json.tool /tmp/day011-output/report.json >/dev/null
```

成功标志：第一条命令末尾打印 `SUCCESS: all files reloaded and validated`；测试显示 `Ran 15 tests` 和 `OK`。

## 输入、输出和检查点

- JSON 记录 schema、图像尺寸、内参矩阵和坐标系；写入时禁止 `NaN/Infinity`。
- CSV 表头与单位写在列名；时间戳必须严格递增。
- NPY 保存 `(N,3)` 的 `float64` 米制点云，读取时使用 `allow_pickle=False`。
- PNG 解码后取 RGB，shape 是 `(H,W,3)`，值域为 `[0,1]`。
- `report.json` 汇总 shape、NPY 往返误差、CSV 精度扫描、梯度差和稳定损失。

请不要提交 `/tmp/day011-output`；它是可重建的运行产物。
