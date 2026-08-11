# 5.2.2 Common Authority Policy Benchmark：结果与图形分析

## 1. 正式运行与证据边界

本次运行在最新版 `src/tre84` 上从头训练，状态为 `complete`。唯一历史输入为 5.2.1 的 `historical_information_event_path.csv`，其 SHA256 为 `407b06106c86f7173d399d8a66283f48ac6746b146c869c14cf77bad3ba3a976`。实验没有读取旧 checkpoint、旧正式输出或旧政策结论，也没有修改 TeX。

正式验收的 41 项阻断检查全部通过，包括输入哈希锁定、training/validation/test 分离、共同外生路径、开始周信息防火墙、共同 ModelKernel、正式 ActionProjector、RC-MSA、route-tagged transition、不可用路线保持、能力 pipeline、完整损失恒等式、路径内 seed 聚合、Holm 多重校正、精度规则和右删失处理。36/36 个跨政策非预见性探针通过，最大 raw-action 差为 0；44/44 个 SAC actor 独立有限差分梯度坐标通过，最大相对误差为 `1.75e-11`。

最终使用 88 条 matched physical paths。学习政策每条路径使用三个独立训练 seed，并先在路径内平均，再以 physical path 为统计推断单位。结论只适用于声明的三网关、共享 corridor、`chi_ref=0.5`、成本代理、共同权限和冻结信息设置。

## 2. Figure 5.2.2a：政策表现与配对不确定性

| 政策 | 平均总损失 | 路径 95% 区间 | 相对 Passive 的配对差 |
|---|---:|---:|---:|
| Reactive | 100,056.0 | [98,924.9, 101,187.1] | -186,460.5 |
| Projected stochastic MPC | 109,237.9 | [107,943.8, 110,532.0] | -177,278.6 |
| Behaviour cloning | 115,468.5 | [114,224.3, 116,712.6] | -171,048.1 |
| Model-guided constrained SAC | 115,978.9 | [114,727.9, 117,229.9] | -170,537.6 |
| PPO | 174,868.0 | [173,318.7, 176,417.3] | -111,648.6 |
| Vanilla SAC | 286,464.1 | [283,712.6, 289,215.6] | -52.4 |
| Constrained SAC | 286,485.5 | [283,734.1, 289,237.0] | -31.0 |
| Passive | 286,516.5 | [283,767.1, 289,266.0] | 0 |

Reactive 是本次参考设计下的 benchmark leader。全 28 个两两比较的 simultaneous confidence set 只保留 Reactive，其相对第二名 MPC 的平均差为 9,181.9，simultaneous 95% 区间为 `[8,801.4, 9,562.4]`。这支持“在本次冻结参考设计中 Reactive 的平均损失最低”，但不支持其对其他网络、成本、事件或 committed share 普遍最优。

所有七个主动政策相对 Passive 的 Holm-adjusted paired test 均排除零。不过效应量差异极大：Reactive、MPC、BC、MG 和 PPO 的下降具有运行意义；Vanilla SAC 和 Constrained SAC 分别仅降低 52.4 和 31.0，虽然在 88 条高度配对路径下区间排除零，却几乎复制了 Passive 的运行结果。因此，统计可分辨不等于经济或运行上重要。

学习政策没有显示超过透明政策的增量价值。BC 比 MPC 高 6,230.6，比 Reactive 高 15,412.5；MG 比 BC 高 510.5，并未把两提案选择转化为更低的路径平均损失。SAC 已按第四章合同实现并通过数值梯度验收，但 Vanilla 和 Constrained SAC 的测试表现仍接近 Passive，这是应保留的负面算法结果，而不是代码失败。

## 3. Figure 5.2.2b：完整损失分解

每个政策的总损失均由 queue、waiting、exit、overload、route resource and transport、action 和 terminal 七项精确重构，最大闭合误差为 `1.16e-10`。运输/route-resource 分项在所有政策中均非零，没有因相同阶段拓扑而被遗漏。

主要差异来自 waiting：Passive 的 waiting loss 为 265,139.3，占总损失 92.5%；Reactive 将其降至 83,989.7，MPC 降至 90,843.2，BC 和 MG 分别为 97,189.3 和 97,847.7。Reactive 的 queue、overload 和 action 分项高于 Passive，说明最低总损失并非所有拥堵分项同时降低，而是 waiting 大幅下降超过了物理拥堵和行动成本的增加。

MPC 的 action loss 为零，但这不表示零协调：其主要动作是无现金成本的 waiting release。BC、PPO、Reactive 和 MG 的 action loss 分别为 1,172.7、1,041.9、599.9 和 940.6。Vanilla/Constrained SAC 的 action loss仅约 3.5/3.4，结合其等待分项几乎不变，说明其训练后动作虽然非零，却没有形成实质性协调响应。

## 4. Figure 5.2.2c：清空和右删失

所有政策的 88 条 physical paths 均在 104 周 cap 前清空，clearance probability 均为 1，右删失路径数为 0，terminal loss 为 0。Passive、Vanilla SAC 和 Constrained SAC 的 restricted mean clearance time 为 78 周；Reactive、MPC、BC 和 MG 为 79 周；PPO 为 78.996 周。最终 outstanding mass 均低于注册的 `1e-6` 清空容差。

因此，本次重跑中 clearance 和右删失没有改变平均损失排序。图仍保留完整经验清空曲线和删失计数，以防止将 cap 机械写成实际清空时间。

## 5. SAC、selector 与计算结果

Vanilla SAC 和 Constrained SAC 均执行每个训练期 21 次 actor、双奖励 critic 和 entropy-temperature 更新；Constrained SAC 同时执行 21 次 constraint-critic 和 dual 更新。训练期间 `log_standard_deviation` 和 entropy temperature 均发生变化，后者由初始 0.05 更新到约 0.007--0.030 区间。Constrained SAC 的 dual 进入约 41.8--242.8 区间。checkpoint 仅根据两条独立 validation paths 选择，且在 test replay 前冻结。

MG selector 的 test 决策共有 5,544 个：选择 BC proposal 4,514 次，选择 Constrained-SAC proposal 1,030 次；11,088 个候选评价全部 solver-valid。MG 的平均损失仍比 BC 高 510.5，说明正式嵌套 selector 在本样本中没有带来增量损失改善。

平均单次决策时间分别为：Reactive `0.00020 s`、学习 actor约 `0.00154 s`、MG selector `3.150 s`、Projected MPC `10.490 s`。这些仅是当前机器上的 computational profile，不能解释为满足外部实时部署时限。

## 6. 路径精度与可写结论

四条 pilot paths 给出的最大所需路径数为 88；计算上限为 196，因此正式运行执行了全部 88 条路径。目标 half-width 为 2,255.64，最大实现 half-width 为 2,237.54，精度目标达成。相比旧的六路径运行，这次不是仅报告未达标，而是实际执行了精度公式要求的路径数。

论文可安全表述为：在声明的参考网络、共同权限和 matched-path 设计下，Reactive 是条件 benchmark leader；MPC、BC、MG 和 PPO 也显著降低了相对 Passive 的完整运行损失，但学习政策没有超过 Reactive 或 MPC。正确闭合的 Vanilla/Constrained SAC 仅产生很小的相对 Passive 改善，MG selector 也未超过 BC。总损失差异主要由 waiting burden 驱动，所有政策均在清空上限前完成清空。

不得据此声称：Reactive 在所有霍尔木兹情景中普遍最优；`chi_ref=0.5` 是历史估计；route-resource 指数是实际运价；学习方法理论上无效；或最低样本均值等于全局最优。
