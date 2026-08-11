# 5.2.3 Action and Congestion Mechanisms：结果与图形分析

## 结果摘要

5.2.3 使用全部 88 条已验收 5.2.2 物理路径。Passive、`Reactive` 和 MG constrained SAC 的 5.2.2 平均总损失分别为 286,516.53、100,056.01 和 115,978.94。这些数值只用于确定需要解释的机制政策，不构成新的政策排名。

Full action 对三类机制政策和全部路径进行了生产引擎重放，并在 $10^{-6}$ 内逐项复现 5.2.2。受限动作部分冻结 `Reactive` 的策略映射，只限制声明的动作块并重新投影；其结果属于 fixed-policy restricted-action diagnostic。

## Figure 5.2.3a：动作与拥堵轨迹

折线与路径四分位带分别显示 readiness order、direct order、readiness exercise、waiting release、disclosure，以及 external waiting 加四阶段队列负担。学习种子先在同一路径内平均，周数未被当作独立样本。该图回答动作何时激活以及负担积累在哪里。

## Figure 5.2.3b：route-stage 与 provenance 热力图

热力图使用外生 medoid `test_017_ebd333a3cdba`，按 policy、committed/adaptive provenance、route 和 maritime/berth/yard/gate/landbridge stage 展示累计 preservice exposure。代表路径不根据政策表现选择；该图仅解释 route-tagged 状态转移。

## Figure 5.2.3c：固定策略受限动作配对效应

森林图以 88 条 physical paths 为配对单位，报告相对 full action 的均值差和 outcome 内 simultaneous 95% interval。

- `no_direct_capacity`：相对 full action 增加总损失 7,166.45；同时 95% 区间 [6,977.82, 7,355.08]；Holm 调整 p=6.192e-90。
- `no_disclosure`：相对 full action 增加总损失 3,973.68；同时 95% 区间 [3,839.43, 4,107.93]；Holm 调整 p=4.322e-81。
- `no_readiness`：相对 full action 增加总损失 1,978.82；同时 95% 区间 [1,920.32, 2,037.32]；Holm 调整 p=9.116e-86。
- `no_release_pacing_authority`：相对 full action 降低总损失 4,967.21；同时 95% 区间 [-5,058.44, -4,875.98]；Holm 调整 p=2.566e-103。

负差值必须保留为冻结策略不协调或训练不足的诊断，不能解释为删除动作权利具有因果收益。

## Proposed MG 模块激活审计

- BC-SAC selector / BC_proposal_selected_count: 4514 (denominator=5544)。
- BC-SAC selector / SAC_proposal_selected_count: 1030 (denominator=5544)。
- BC-SAC selector / fallback_count: 0 (denominator=5544)。
- readiness order / activation_count: 5544 (denominator=5544)。
- readiness order / range_across_test_decisions: 48.2319 (denominator=5544)。
- direct order / activation_count: 5544 (denominator=5544)。
- direct order / range_across_test_decisions: 49.5728 (denominator=5544)。
- readiness exercise / activation_count: 3432 (denominator=5544)。
- readiness exercise / range_across_test_decisions: 44.3799 (denominator=5544)。
- release / activation_count: 5544 (denominator=5544)。
- release / range_across_test_decisions: 0.620682 (denominator=5544)。
- disclosure / activation_count: 5544 (denominator=5544)。
- disclosure / range_across_test_decisions: 0.360943 (denominator=5544)。

## 负面、较弱或不确定结果

- Fixed Reactive restriction no_release_pacing_authority reduced mean loss by 4967.209071; retain as a negative coordination diagnostic, not causal value.

区间跨零、动作块未激活、限制动作反而改善损失或 clearance 发生转移，均不是代码失败；它们限定了 5.2.3 可支持的机制结论。

## 证据边界

- 不重新训练或重新优化任何受限动作控制器。
- 不把 medoid 路径当作推断样本。
- 不将 queue 降低单独解释为系统改善，必须同时检查 waiting、两类 exit、overload、terminal mass 和 clearance。
- 本实验不修改 5.2.2 的外生路径、随机数、released information、checkpoint、共同 projector、RC-MSA、tagged transition 或 loss。
