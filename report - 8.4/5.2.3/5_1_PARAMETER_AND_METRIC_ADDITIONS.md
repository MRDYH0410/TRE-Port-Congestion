# 5.2.3 对 5.1 参数与指标表的补充报告

## 结论

- 数据部分：**NO CHANGE**。本实验不引入新数据，唯一上游是已验收的 5.2.2 配置、88 条测试路径、轨迹和 checkpoint。
- 经济与行为参数：**NO ADDITION**。网络、损失、预算、SUE、能力交付和数值容差均继承 5.2.2，不根据 5.2.3 结果重新校准。
- 需要在 5.1 中保持可追踪的仅是实验协议和派生指标；这些不是新的经济参数。

## 实验协议

1. 正式推断单位为 physical path，共 88 条；学习种子必须先在路径内平均。
2. 机制政策集包含 Passive、5.2.2 的唯一置信集领先者 `Reactive`、以及 Model-guided constrained SAC。
3. 代表路径 `test_017_ebd333a3cdba` 由 total blocked mass、peak blocked mass、mean/minimum serviceability 和 recovery rate 标准化后选择 physical-path medoid；政策损失不进入选择。
4. Full action 的 5.2.2 复现绝对容差为 $10^{-6}$。
5. No release pacing 使用非控制基准 $ho_t^{base}=1$，即所有当前可释放 waiting cargo 通过同一 oldest-first operator 立即进入 SUE。
6. 所有限制在冻结 Reactive 原始动作之后、共同凸投影之前施加；策略不重新训练或重新优化。

## 派生指标

- 受限动作路径差：$D_{r,m}^Y=Y_m^{(r)}-Y_m^{(full)}$。
- 每个 outcome 内对四项限制使用 Holm 调整，并报告 Bonferroni-$t$ simultaneous 95% interval。
- waiting exposure 以 model-unit weeks 计量；clearance probability、restricted mean clearance time 和 final outstanding 单独报告右删失边界。
- route-stage exposure 是 medoid 路径上的累计 preservice workload，仅作过程解释，不作为统计推断样本。

不得把这些诊断写成因果边际价值、重新优化后的 action-right value，或管理者删除某项权利后的最优结果。
