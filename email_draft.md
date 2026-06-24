老师您好，

关于 Submission 1 的复现，我之前在 README 里写的是 "the exact code state was not version-controlled"，以为无法完全复现。但昨晚重新排查实验日志后发现，实际配置是可以确定的，而且刚刚验证成功了——生成的 portfolio 与原版 50/50 匹配，权重相关性 1.000。

Sub1 和 Sub2 的关键差异如下：

| | Submission 1 | Submission 2 |
|---|---|---|
| Target | 3-day forward return | 5-day forward return |
| 特征 | 15 个（短动量，无 ret_20d/ret_60d） | 17 个（+中期动量） |
| 模型 | LightGBM only | LightGBM + XGBoost ensemble |
| as-of | 20260430 | 20260508 |

复现方式：我单独写了一个 `features_sub1.py`（覆盖 `features.py` 即可切到 Sub1 模式，跑完 git checkout 恢复）。具体步骤在更新后的 `README_MODEL.md` 里有详细说明。

这次更新主要涉及三个文件，随报告一并提交：

- `features_sub1.py` — Sub1 特征模块（drop-in replacement）
- `README_MODEL.md` — 更新了 Sub1 复现步骤和 Sub1/Sub2 对比表
- `self_test.py` — self-test 脚本（train/test split + baseline vs final 对比）

抱歉之前沟通不准确，现在所有提交都可以完整复现了。如有问题请随时指出。

谢谢！
