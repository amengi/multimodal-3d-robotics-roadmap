# Day 006 参考实现

今天把两个循环变成可审计证据：Git 的“修改—暂存—提交—分支—合并”循环，以及 PyTorch 的“前向—损失—清梯度—反传—更新—验证”循环。

## 环境与运行

在项目根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r materials/day006/requirements.txt
python -m materials.day006.git_sandbox
python -m materials.day006.train_moons
python -m materials.day006.compare_methods
python -m unittest materials.day006.test_day006 -v
```

`git_sandbox` 会在系统临时目录新建一个独立仓库，并故意停在一次内容冲突上；它拒绝覆盖非空目录，也拒绝在任何现有 Git 仓库内部创建嵌套仓库。请进入脚本打印的绝对路径，亲手删除冲突标记、保留你能解释的最终值，再 `git add` 和 `git commit`。把 `status/diff/log` 复制到 `experiments/Day006/git-evidence.md`；不要在当前课程仓库里制造冲突。

训练程序只生成固定随机种子的合成数据，不下载数据集。默认把 `loss_curve.png` 和 `decision_boundary.png` 保存到 `experiments/Day006/reference_outputs/`。测试共 16 项；全部通过时显示 `OK`。

## 文件说明

- `git_sandbox.py`：在新目录内创建三次语义提交和一次可复现冲突，留下 `status`、冲突标记与分支图供检查。
- `train_moons.py`：分层切分 `make_moons`，只用训练集统计量标准化，训练小型 MLP，并在留出的验证集上报告 BCE 与 accuracy。
- `compare_methods.py`：在同一验证集、同一 accuracy 指标下比较最近质心几何基线和 MLP；单次合成实验不能证明普遍优劣。
- `test_day006.py`：检查沙盒隔离、三次提交、冲突状态、数据切分、shape、复现性、训练、验证、图像和基线。

## 关键边界

- `git add` 暂存的是运行当时的文件内容；之后继续编辑会同时出现 staged 和 unstaged 差异。
- commit 是本地快照，不等于已经推送、备份或复现实验；仍需保存环境、数据与配置。
- 验证集不参与参数更新，也不能在反复调参后继续冒充一次性的最终测试集。
- 决策边界图中的 sigmoid 输出不是经过校准的真实概率，也不能迁移为医学或机器人安全结论。
