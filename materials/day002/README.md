# Day 002 配套材料

本目录只支持今天的两个核心任务：检查终端创建的目录树，以及练习 NumPy 的 `(B, M, F)` 轴语义。没有隐藏数据或预训练模型。

## 依赖

- Python 3.10+
- NumPy 1.24–2.x；版本范围见 `requirements.txt`

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r materials/day002/requirements.txt
python materials/day002/array_shape_lab.py
python -m unittest materials/day002/test_day002.py -v
```

预期脚本最后三行包含：

```text
weights shape: (4, 2)
projected shape: (2, 3, 2)
All shape checks passed.
```

测试成功标志：共 7 个测试，最后显示 `OK`。

## 终端目录检查

完成课件中的终端练习后运行：

```bash
python materials/day002/check_terminal_tree.py experiments/Day002/terminal_lab
```

成功时输出：

```text
Tree check passed: 4 directories and 4 required files are in place.
```

检查器只读取目标目录，不修改文件。若失败，它会逐条列出缺少或移动错误的路径。
