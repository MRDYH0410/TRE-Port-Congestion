# 5.2.2 对 Section 5.1 的参数与指标补充报告

## 结论

本报告只指出最新版正式运行与当前 5.1 表之间需要同步的内容，不修改 TeX。数据、网络、成本和行为参数的主体已经在 5.1 中登记；主要新增或修订项来自本次实验层适配、完整 SAC 合同和按精度规则扩展到 88 条测试路径。

## 已由 5.1 覆盖且本次不变

- `U_Q=1000`、21 个 Monday-based event weeks、三网关及四阶段共享 corridor 网络。
- `chi_ref=0.5` 及 committed itinerary shares `(0.5952, 0.1071, 0.2976)`。
- readiness lead 8 周、direct procurement lead 0 周、decision horizon 21 周、clearance cap 104 周和清空容差 `1e-6`。
- RC-MSA 容差 `1e-6`、最多 500 次迭代、去重容差 `1e-7`；投影容差 `1e-8`、最多 300 次迭代。
- 未识别晚期退出成本 `c^{E,late}_{kj}=0`；route-wise waiting forecast RMSE、`gamma_I=1` 和公共等待信号构造。
- queue/waiting/exit/overload/route-resource/action/terminal 的完整损失定义，以及 route-resource 增量 `(0,1,2)` 的设计性物理代理边界。
- MPC 的 8 周 horizon、slow/central/fast 三情景、七个候选方向和正式 terminal value。
- 三条 training paths、两条 validation paths、三个学习 seed、9--12 episodes、每 3 episodes 验证、patience 2。

## 需要补充或修订的参数

| 参数或合同 | 本次正式值 | 依据与应写边界 |
|---|---:|---|
| 唯一 5.2.1 接口哈希 | `407b06106c86f7173d399d8a66283f48ac6746b146c869c14cf77bad3ba3a976` | 新验收的唯一历史输入；哈希不一致时 fail closed |
| test path minimum / maximum | 6 / 196 | 最小值只保证 pilot 后继续执行；196 是 216 个事件前残差中可形成的不同 21 周连续块上限 |
| 正式 executed paths | 88 | 由四条 pilot paths 和预注册 paired half-width 公式产生，不是结果后选取 |
| target half-width | 2,255.637825 | `0.05 J^{fail,ref}`，沿用已声明的统计精度规则 |
| achieved maximum half-width | 2,237.542408 | 88 条 matched paths 下的验收结果，不是新参数 |
| SAC temperature 0.05 的角色 | 初始值 | 不再表述为固定 temperature；每个训练决策期更新 |
| entropy-temperature learning rate | 0.02 | 与冻结 actor optimizer step 相同，避免新增测试调参尺度 |
| target entropy | `-dim(a)` | 连续动作 SAC 的维度规则；动作维度为 34 |
| preprojection policy | diagonal Gaussian latent + logistic map | 对应第四章重参数化随机策略，之后进入共同正式投影 |
| projection gradient | active-set piecewise Jacobian/subgradient | 正式凸投影在固定 active set 内的局部导数；active-set tie 处按注册分段次梯度处理 |
| SAC gradient-check step | `6.055454452393343e-6` | IEEE-754 double machine epsilon 的立方根，用于中心有限差分 |
| SAC gradient relative tolerance | `1e-4` | 仅用于独立数值梯度验收，不改变训练目标或动作可行性 |
| critic projection dimension | 32 | 已有冻结特征投影维度；双 reward critic 分别更新 |
| critic interaction head | 4 | 已有冻结交互头；Constrained SAC 另更新 constraint critic |
| per-period SAC update contract | Q1、Q2、actor、temperature 每期各一次；约束版本另更新 Qg 和 dual | 直接对应第四章 actor/critic/entropy/dual 合同；不可再用 episode-level字段变化代替 |
| beginning-of-week preparer | released information + event-aligned scenario bundle before action | 同一 preparer 同时用于 SAC training、teacher、validation 和所有正式执行；未来 payload 仅由环境随后实现 |
| nonanticipativity tolerance | `1e-8` action scale | 与正式 projection numerical scale 一致；future-payload 和 same-week unrealised-outcome 探针均要求动作不变 |

当前 TeX 的 5.1 若仍写“minimum 与 maximum final path count 均为 6”“正式 test paths 为 6”或“SAC temperature 固定为 0.05”，必须在后续写作阶段修订为上述正式合同。这里的 88 是本次精度计算结果，196 是计算支持上限，两者不能混写。

## 需要报告的指标

1. 路径内 seed 聚合后的 total operational loss，以及 queue、waiting、两类 exit、overload、route-resource、action 和一次 terminal correction 的闭合分解。
2. 相对 Passive 的 matched-path mean difference、standard error、普通 95% 区间、simultaneous 95% 区间、unadjusted p-value 和 Holm-adjusted p-value。
3. sample-best confidence set，但只能称为当前参考设计的 benchmark leader/confidence-set result。
4. pilot required paths、executed paths、target half-width、achieved half-width 和 precision-target status。
5. clearance probability、restricted mean clearance time、右删失路径数、final outstanding 和 terminal loss；不能把 cap 写成清空周。
6. SAC 的 Q1/Q2/Qg、actor、entropy-temperature、log-standard-deviation 和 dual 更新计数；actor 独立有限差分误差。
7. future-payload 与 same-week unrevealed outcome 的非预见性动作差，以及 information/observation/controller hashes。
8. requested/implemented action、projection constraints、solver status、MG proposal source及两个正式嵌套目标、checkpoint hashes 和决策时间。

## 证据边界

这些参数与指标支撑的是声明参考网络、成本指数、`chi_ref=0.5` 和共同权限下的条件政策比较。它们不把 AIS proxy 解释为实际 handled cargo，不把 route-resource proxy 解释为实际运输价格，不把 designed training paths 解释为历史事实，也不把 SAC 负面表现解释为算法在所有问题上的一般结论。
