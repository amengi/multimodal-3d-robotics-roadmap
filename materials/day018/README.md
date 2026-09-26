# Day 018 配套材料：多文件编译、链接与三分类交叉熵

本材料把 Day 017 的 `Camera` 缩成一个最小多文件工程：`camera.hpp` 放接口，`camera.cpp` 放定义，`main.cpp` 放调用者。它只用 C++17 标准库。`verify_cross_entropy.py` 依赖 Python 3 与 PyTorch，用相同三分类样本核对手算、one-hot 交叉熵和 `torch.nn.functional.cross_entropy`。

## 输入、输出与契约

- 相机点 `P_c=(X_c,Y_c,Z_c)`：shape `(3,)`，camera frame，单位 m，且 `Z_c>0`。
- 相机参数 `(f_x,f_y,c_x,c_y)`：shape `(4,)`，单位 px，且焦距为正。
- 像素 `(u,v)`：shape `(2,)`，单位 px；`u=f_x X_c/Z_c+c_x`，`v=f_y Y_c/Z_c+c_y`。
- 分类 logits `z:(1,3)`：无量纲、未归一化分数；标签 `y:(1,)` 是类索引 0。
- softmax 概率 `q:(1,3)`：无量纲、非负、和为 1。
- 单样本 NLL/交叉熵 `L=-ln q_y`：单位 nat/example。准确率和 multiclass Brier 同时报告，避免只看一种指标。

## 编译、链接、运行

从仓库根目录执行：

```bash
mkdir -p /tmp/day018-build
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -c materials/day018/camera.cpp -o /tmp/day018-build/camera.o
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -c materials/day018/main.cpp -o /tmp/day018-build/main.o
c++ /tmp/day018-build/main.o /tmp/day018-build/camera.o \
  -o /tmp/day018-build/camera_app
/tmp/day018-build/camera_app
/tmp/day018-build/camera_app --self-test
```

成功标志：出现 `projection_px=(270.000000,190.000000)`、`geometry_error_px=0.000000`、`SUCCESS` 和 `SELF_TEST_OK: 6 checks`。

有意制造链接失败（命令应失败，随后继续）：

```bash
c++ /tmp/day018-build/main.o -o /tmp/day018-build/broken_app \
  2> /tmp/day018-build/link-error.txt || true
sed -n '1,12p' /tmp/day018-build/link-error.txt
```

看到 `undefined symbol` 或 `undefined reference` 即成功制造故障：`main.o` 有 `Camera` 的调用和声明，但缺少 `camera.o` 中的定义。不要用 `#include "camera.cpp"` 掩盖问题；应把缺失的对象文件加入链接命令。

## 交叉熵验证与全套测试

本机系统 `python3` 未安装 PyTorch，仓库已有 `pyt` Conda 环境，因此验证命令显式使用它。若你已经激活含 PyTorch 的环境，可把 `conda run -n pyt python` 换成 `python`。

```bash
conda run -n pyt python materials/day018/verify_cross_entropy.py \
  --csv /tmp/day018-build/loss_scan.csv
conda run -n pyt python materials/day018/verify_cross_entropy.py --self-test
conda run -n pyt python -m unittest materials.day018.test_day018 -v
conda run -n pyt python -m compileall -q materials/day018
```

成功标志：手算和 PyTorch 差值约为 0；预测正确且同时输出 `accuracy`、`multiclass_brier`；扫描中真实类 logit 从 `-2` 增至 `4` 时 `p_true` 单调上升、NLL 单调下降；Python 内测为 7 项，黑盒测试为 16 项且全部 `OK`。

注意：`CrossEntropyLoss` 接受 logits，不要先做 softmax 再把概率当 logits 传入。单个正确样本不能证明模型校准良好；交叉熵低也不证明因果、几何正确或现实可靠。
