# Adroit Pen 视觉灵巧操作：面试复述与问答稿

> 这份材料用于机器人学习、强化学习、具身智能和视觉控制实习面试。口头表达时先说结论，再补证据；不要把开发集峰值、不同 bank 或训练与评测数据混在一起。

## 1. 四种复述长度

### 一句话项目概述

我在纯仿真的 Adroit Pen 上从 25 条人类示范训练 offline AWAC，并把多条不稳定的 online/action-residual 路线转化为 8-step Temporal Visual-State Distillation；固定 E2 在独立 300-episode paired confirmation 上将 Benchmark 从 52.3% 提高到 68.3%，部署时只使用 RGB、proprioception 和历史动作。

### 30 秒版本

我做的是 Adroit Pen 的视觉灵巧操作。先用 25 条 human demos、4,975 个合法 transition 建立 Visual BC 和 offline AWAC，并用黑屏、跨 episode 打乱和目标遮挡证明策略确实依赖 RGB。PPO、Residual SAC 和反事实局部修正都不稳定，核心原因是接触任务里很小的动作也会改变接触模式，而且长时 gain/harm 很难从视觉历史可靠预测。最后我把问题改成时序状态蒸馏：用 544 条仿真轨迹、108,800 个稠密样本训练 8-step CNN+GRU 状态估计器，再驱动 Frozen Oracle AWAC。独立 300 episodes 上 Benchmark 52.3%→68.3%，Strict 27.0%→37.0%；项目纯仿真，没有 sim-to-real。

### 2 分钟版本

任务是用单相机 RGB 控制 24 维 Adroit Hand，把笔重定向到目标姿态并稳定保持。第一步不是先堆算法，而是把 Minari `D4RL/pen/human-v2` 恢复正确：25 个 episode、5,000 source states，严格切边界后是 4,975 transitions；随机状态 observation 恢复误差约 `5.6e-16`，一步 qpos/qvel/reward 也到机器精度，跨 episode transition 是 0。

基线上，Visual BC 是 37.3% Benchmark，offline AWAC 提到 51.0%。视觉干预里正常输入 52.7%，全黑降到 22.7%，跨 episode 打乱降到 27.0%，说明 RGB 有因果作用；但 temporal shuffle 还是 53.5%，说明旧 encoder 没真正利用顺序和动态。

之后我系统试了 PPO、Residual SAC、Safe Residual、同状态 counterfactual branch 和 candidate ranker。它们不是简单“没调好”：Safe Residual 已把 correction RMS 压到 0.0051、没有数值 clip，Strict 还是从 34% 降到 23%，同时 Q–MC bias 增到 +40。真实 branch 又证明 52/240 roots 确实有局部改进，但视觉 gate precision 只有 30.8%；更进一步，privileged ranker 也不安全。我的判断是，预测一个微小动作几十步后的 gain/harm，本质上是困难的长时 action-value，而不是单纯的 residual 幅度问题。

最终我改成 Temporal Visual-State Distillation。输入最近 8 步 RGB、手部 proprio 和 executed actions，CNN+GRU 预测笔的位置、速度、角速度、object/target 6D rotation，再确定性重建 Oracle observation，经过原 normalizer 驱动 Frozen Oracle AWAC。privileged state 和 Oracle action 只做训练监督，部署不读取仿真真值。E2 只解冻 estimator 的最后一个 CNN block。独立 300-episode paired confirmation 上，Benchmark 52.3%→68.3%，Strict 27.0%→37.0%；one-shot frozen test 的 Benchmark 54.5%→75.0%，Strict 26.0%→33.5%，但后者 CI 下界为 0，所以我只说方向性提升。

### 5 分钟完整版本

我把这个项目分成“数据可信、离线控制、失败定位、方法转向、独立验证”五步来讲。

第一步是数据。Adroit 的 offline 数据如果边界或 state restore 错一点，Q-learning 会把错误反复 bootstrap。我固定了 Gymnasium-Robotics 1.2.3、Minari 0.5.3、MuJoCo 2.3.7、dense reward 和相机参数。从 25 条 human demonstrations 里得到 5,000 source states、5,025 container observations，合法 transition 是每个 episode 少一个 next state，所以一共 4,975，不是跨 episode 拼 4,999。随机 100 states 的 observation 恢复最大误差 `5.55e-16`；一步 replay 的 qpos `1.17e-15`、qvel `3.12e-13`、reward `7.11e-15`，跨 episode 数为 0。

第二步是基础策略。Vision actor 只看 4-frame 84×84 RGB、hand qpos/qvel 和 previous action，critic 在训练时可以 asymmetric 地看完整仿真状态。Visual BC 的 Benchmark/Strict 是 37.3%/14.7%，critic warm-up 加 offline AWAC 后是 51.0%/20.3%。AWAC 的好处是仍然在数据动作上做加权模仿，只让 critic 估计的正优势决定权重，不会像纯 online policy gradient 一开始就离开示范分布。

我还专门验证视觉因果性：Full policy 正常 RGB 是 52.7%/19.9%，全黑变 22.7%/9.5%，跨 episode 图像变 27.0%/5.8%，遮挡绿色目标是 40.8%/12.5%。所以 RGB 和目标外观确实重要。不过 temporal shuffle 是 53.5%/19.4%，几乎没降，这说明原 CNN 更依赖静态外观，没有可靠编码角速度和帧顺序。

第三步是 offline-to-online 交接。Direct PPO 会改完整 actor，很多 seed 最优点就是 0 online step。Residual SAC 想把改动限制在 AWAC 周围，但 scale 0.2 经过 100k 从 64%/40% 退到 28%/26%，还有 hard clip、约 28% saturation 和 Q drift。之后的 Safe Residual 用 action headroom 天然保证动作有界，固定低熵并加 zero-correction anchor；30k 时 correction RMS 只有 0.0051，clamp 为 0，但还是从 62%/34% 退到 59%/23%，Q 对 Monte Carlo return 的 bias 到 +40.08。也就是说，小动作在接触动力学里不等于小轨迹变化，critic 又不能可靠排序这种微小 correction。

然后我用 simulator snapshot 做同状态真实 branch，不再相信 Q。240 roots 里 52 个确实能被局部候选改善，pending-exit roots 有 48.3% 可改善，说明 action headroom 是存在的。但 gate 在 branch-dev 的 precision 只有 30.8%，达不到 80% precision、5% harm 的部署标准。V5 把数据扩到 1,200 roots、20,400 candidate records，甚至让 privileged ranker 直接看真值，selector 仍只有约 28.9% precision、11.1% harm。Oracle 5-step takeover 也同时有 205 gain 和 223 harm，因为短时 Oracle controller 再切回 Vision controller 会产生 stitching 问题。

第四步是核心转向。我意识到之前一直在问“这个动作会不会在几十步后有益”，这是一个稀疏、高方差、部分可观测的 counterfactual value 问题。仿真却能为每一步提供稠密物理状态，所以改问“当前隐藏状态是什么”。我收集 544 条 simulation trajectories、108,800 samples。E2 输入 8-step causal RGB、proprio 和 executed-action history，用 CNN+GRU-128 预测 object position、linear/angular velocity 和 object/target 6D rotation；可部署的手部状态直接复制，derived 45D Oracle observation 由 primitive state 确定性重建。position/velocity 用 Huber，rotation 用 6D 加 SO(3) geodesic loss，还把预测状态送进 Frozen Oracle AWAC，利用 action-consistency 把梯度穿过冻结 actor 回到 estimator。Oracle 参数不更新，部署也不读取 simulator state。

消融很关键：E1 完全冻结 CNN，是 65%/22%；E2 只解冻 estimator 私有 CNN 最后一个 block，是 78%/38%。说明 GRU 不能恢复 encoder 已丢失的信息，有限 representation adaptation 才是关键。普通 D1 DAgger 又降到 66%/31%，而较低 exit ratio 主要伴随 entry 减少，不能说 hold 恢复。

第五步是验证。V6 旧 100-episode gate 因 exit 超阈值 0.143pp，历史状态仍保持 STOP；V6.1 没回写这个结论，而是冻结 E2 后开新 bank。300-episode paired confirmation：E0 52.3%/27.0%，E2 68.3%/37.0%；Benchmark 有 66 wins、18 losses，McNemar `p=1.33e-7`，Strict 60/30，`p=0.00206`。三个训练 seed 平均是 61%±4.0pp Benchmark、35%±6.24pp Strict，说明方向可复现但 Strict 方差明显。一次性 frozen test 上 Benchmark 54.5%→75.0%，CI `[+13,+28]pp`；Strict 26.0%→33.5%，CI `[0,+15]pp`，所以 Strict 只说方向性提升。

最后我会主动讲限制：纯仿真、单相机单任务，最终不止使用 25 demos，还用了 108,800 个仿真监督样本；fixed-entry 50-step hold 的 E2 是 47%，还低于 E0 的 50% 和 Oracle 的 53%。项目最有价值的地方不是某个峰值，而是通过可复现负结果把问题从不稳定的 action correction 改写成稠密时序状态估计，并用独立 paired 数据验证了这个判断。

## 2. 必须记住的数字

| # | 数字 | 一句话含义 |
| ---: | --- | --- |
| 1 | 25 human demos | 离线 AWAC 的人类示范来源 |
| 2 | 4,975 offline transitions | 5,000 states 按 25 个 episode 合法切边界 |
| 3 | 544 simulation episodes | E1/E2 state-distillation source |
| 4 | 108,800 transitions | 稠密状态、rotation、risk 和 Oracle action 监督 |
| 5 | 300 confirmation episodes | 固定 E2 的独立 paired confirmation |
| 6 | 52.3%→68.3% | confirmation Benchmark，+16pp |
| 7 | 27.0%→37.0% | confirmation Strict，+10pp |
| 8 | 61%±4.0pp | 三训练 seed 的 Benchmark 均值±sample std |
| 9 | 35%±6.24pp | 三训练 seed 的 Strict 均值±sample std |
| 10 | 54.5%→75.0% | one-shot paired frozen Benchmark |
| 11 | 26.0%→33.5% | frozen Strict，方向性提升，CI 下界 0 |
| 12 | `cc35a4130cb3…` | 最终单一 E2 checkpoint SHA-256 缩写 |

## 3. 高频面试问题与口语化回答

### 1. 为什么选择 Adroit Pen？

**结论：它同时检验高维接触控制、视觉部分可观测和稳定保持，能暴露普通 locomotion 看不到的策略交接问题。**

Adroit Hand 有 24 维动作，笔的姿态会受多指接触和摩擦切换影响。Benchmark 只要求到过目标，Strict 还要求最后 20 步连续保持，所以可以把 acquisition 和 hold 分开分析。它比单纯到达任务更适合研究视觉动态和闭环稳定性。

### 2. 为什么使用 AWAC？

**结论：AWAC 能利用示范启动策略，同时把更新限制在离线数据支持的动作附近。**

它本质上是按优势加权的行为克隆：仍然最大化数据动作似然，只让 critic 认为更好的样本权重更高。25 条 demo 很少，直接 online exploration 风险高；AWAC 把 Visual BC 的 Benchmark 从 37.3% 提到 51.0%，证明它在这个数据规模下是有效起点。

### 3. 25 条示范是不是太少？

**结论：对启动 AWAC 足够，对独立学习稳定时序视觉状态不够。**

25 条轨迹给了抓持和操纵动作先验，但只有 4,975 transitions，覆盖不了遮挡、角速度和各种接触相位。最终 E2 还用了 544 条仿真轨迹、108,800 个稠密监督样本；我不会说最终策略只靠 25 条示范训练。

### 4. 最终模型到底用了多少数据？

**结论：基础 controller 用 25 demos；最终 state estimator 额外用 544 simulation episodes、108,800 transitions。**

其中 244 episodes 来自已有 training rollout，300 episodes 来自 Frozen Oracle training rollout。D1 的 300 episodes、60,000 transitions 只训练 D1，不属于最终 E2；confirmation、fixed-entry 和 frozen test 都只评测不训练。

### 5. privileged supervision 算不算作弊？

**结论：它是仿真中的 teacher supervision，不是部署输入；是否可接受取决于问题定义。**

E2 训练时用 simulator state 和 Frozen Oracle action 标注，但部署只输入 8-step RGB、hand proprio 和 executed-action history，代码路径不读真值。若目标是“仅用真实可观测数据训练”，这不满足；若目标是“仿真训练、可部署视觉推理”，这是常见 asymmetric/privileged learning 设定。报告明确披露了 108,800 个仿真标签。

### 6. 最终部署输入是什么？

**结论：最近 8 步单相机 RGB、可部署手部 proprioception 和实际执行动作历史。**

不输入 pen position、velocity、orientation、target orientation 或 simulator goal error。模型先预测缺失 primitive state，再重建 45D Oracle observation；这个 45D 是模型输出和可部署 proprio 的组合，不是环境真值直读。

### 7. 如何证明 RGB 真正起作用？

**结论：在 proprio 和动作历史不变时破坏 RGB，闭环成功率显著下降。**

V1 Normal 是 52.7%/19.9%；全黑降到 22.7%/9.5%，跨 episode 图像降到 27.0%/5.8%，绿色目标遮挡降到 40.8%/12.5%。这说明策略依赖图像内容和目标外观，不是只走 proprio 捷径。

### 8. temporal shuffle 为什么没有退化？

**结论：基础 4-frame CNN 用到了视觉内容，却没有可靠利用时间顺序。**

temporal shuffle 后是 53.5%/19.4%，与 Normal 52.7%/19.9% 基本相同。最合理的判断是旧 encoder 更像提取静态姿态/外观，而不是角速度和有向动态；这直接支持后来使用 8-step causal GRU 和稠密速度监督。

### 9. 为什么 PPO 失败？

**结论：小规模 on-policy advantage 不足以保护一个已经工作的接触闭环，完整 actor 更新造成分布漂移。**

实现里区分了 rollout 的 `pi_old` 和 Frozen AWAC `pi_ref`，也有 KL/BC anchor，但很多 seed 最优 checkpoint 仍是 0 step。接触任务里 action likelihood 的小改动会改变接触轨迹，anchor 没有稳定优于 no-anchor，所以问题不只是实现少了正则。

### 10. 为什么 Residual SAC 失败？

**结论：早期既有 hard clip/saturation，也有更根本的 critic 排序错误。**

scale 0.2 在同 50-episode bank 从 64%/40% 降到 28%/26%，saturation 约 28%，Q mean 漂到约 610。缩小 scale 只延缓退化；后续 Safe Residual 消除 clipping 后仍失败，说明 Q 对微小 correction 的排序才是核心瓶颈。

### 11. 为什么极小 residual 也会破坏 hold？

**结论：动作空间的小距离不代表接触轨迹的小距离。**

Safe Residual 的 per-dimension correction RMS 只有 0.0051、数值 clamp 为 0，Strict 仍从 34% 降到 23%。一个小动作可以改变是否滑移、某根手指是否接触，之后 Frozen AWAC 面对的状态分布完全不同；同时 Q–MC bias 到 +40.08，actor 会持续选择错误方向。

### 12. counterfactual candidate 明明有 headroom，为什么不能部署？

**结论：存在好候选不等于能从视觉历史高精度识别它。**

V4 有 52/240 positive roots，pending-exit 是 48.3%，说明局部修正客观存在。但 gate precision 只有 30.8%，且没有 threshold 同时满足 80% precision、5% FPR、5% coverage；V5 连 privileged selector 也没过安全门槛。瓶颈是长时 gain/harm 的可预测性，不是候选完全无效。

### 13. 为什么 Oracle 短时 takeover 也会失败？

**结论：完整 Oracle 强，不代表把它的 5-step chunk 插入另一个 controller 就安全。**

Oracle 动作基于自己的真值 closed loop；切回 Vision AWAC 后，视觉 controller 会面对 Oracle 制造的新接触状态和历史，形成 controller stitching。1,200 roots 中 5-step Oracle takeover 有 205 gains，也有 223 harms，所以不能无条件做 teacher chunk imitation。

### 14. 最终为什么改成状态蒸馏？

**结论：当前隐藏物理状态有稠密、平滑、可诊断的监督，比局部动作的长时 gain/harm 更可学。**

candidate selector 要同时推断隐藏状态、接触切换和几十步未来，相当于困难的 counterfactual Q。仿真则能给每个时刻的 position、velocity、rotation 和 teacher action；108,800 个稠密样本让问题从稀疏事件排序变成时序状态辨识，再复用已有 Frozen Oracle controller。

### 15. 6D rotation 为什么优于 Euler 或 quaternion MSE？

**结论：6D 表示在神经网络输出空间连续，SO(3) geodesic 又直接度量姿态最短转角。**

Euler 有周期边界和万向节问题；quaternion 的 `q` 与 `-q` 表示同一旋转，普通 MSE 会产生二义性。6D 输出经 Gram–Schmidt 还原旋转矩阵，再用 trace 公式算 geodesic，控制意义更直接。

### 16. action-consistency loss 如何反向传播？

**结论：Oracle 参数冻结，但预测状态到 Oracle action 的计算图保留，所以梯度回到 estimator。**

路径是 predicted state → deterministic adapter → original Oracle normalizer → Frozen Oracle AWAC → action MSE。`requires_grad=False` 只阻止更新 Oracle 权重，不会阻止对输入求梯度；teacher truth branch 才在 `no_grad` 下。这样 estimator 会优先修正真正影响 controller action 的状态误差。

### 17. 为什么只解冻 CNN 最后一个 block？

**结论：E1/E2 对照说明旧高层特征丢了动态信息，而有限适配比全量改 backbone 风险更小。**

E1 冻结 CNN 是 65%/22%，E2 只解冻最后 block 是 78%/38%。低层边缘颜色仍可复用，高层需要为 rotation/velocity 重新组织；CNN 学习率又比 GRU/head 小十倍，减少遗忘和 seed 方差。

### 18. D1 DAgger 为什么退化？

**结论：普通 student-visited 数据聚合改变了 phase 分布，并破坏 E2 的 acquisition 表征。**

D1 从 E2 的 78%/38% 降到 66%/31%。Exit/Entry 虽从 62.8% 降到 57.6%，但 entry 本身也减少，所以不能说 hold 改善；paired benchmark wins/losses 是 27/29。数据更多不等于监督更合适。

### 19. 为什么 E2 有时名义高于 Oracle？

**结论：这是有限 bank 的名义排序，可能来自样本波动或 estimator 的平滑偏差，不代表普遍超过真值策略。**

300 bank 上 E2 是 68.3%/37.0%，Oracle 64.7%/36.3%；旧 100 bank 上 Oracle 又是 75%/42%。主要 paired 假设检验是 E2 对 E0，而不是 E2 对 Oracle，所以我只说“该 bank 上接近 Oracle 水平”。

### 20. 如何解释 fixed-entry 50-step 差距？

**结论：E2 改善 acquisition 和短 hold，但长期保持仍有方向性弱点。**

共同 200 first-entry roots 上，20-step E0/E2 都是 61.5%；50-step E2 47%，低于 E0 50% 和 Oracle 53%。没有 discordance CI，所以不能说差距显著，也不能直接断言一定由速度误差或 GRU 延迟造成。

### 21. 3-seed 方差说明什么？

**结论：改进方向能复现，但 Strict 对训练随机性明显敏感。**

三个 seed Benchmark 是 65/61/57%，均值 61%±4.0pp；Strict 是 40/37/28%，均值 35%±6.24pp。它们评的是同一组 100 initial states，不能 pooled 成 300 个独立 episode；也不能说每个 seed 都达到 75%。

### 22. paired McNemar 检验在这里有什么意义？

**结论：它只使用同一 episode 上 E0/E2 不一致的结果，直接检验谁更常赢。**

confirmation Benchmark 有 66 个 E2-only success、18 个 E0-only success，ties 216；McNemar 检验比较 66 和 18，而不是假装两组互相独立，得到 `p=1.33e-7`。Strict 是 60 对 30，`p=0.00206`。

### 23. 如果进行真机迁移，需要增加什么？

**结论：先解决 observation/action domain gap 和安全，而不是直接把 checkpoint 上机械手。**

至少需要真实相机标定、图像 domain randomization 或真实视觉微调、关节/动作延迟建模、摩擦与几何辨识、动作限速和独立安全监控。还要重新验证 proprio 定义、目标姿态视觉表示和掉落检测；本项目没有执行任何真实硬件命令。

### 24. 如果再做一个实验，会做什么？

**结论：我会在共同 first-entry roots 上做 hold-focused state-component intervention。**

保持 E2 其他预测不变，分别把 predicted angular velocity、object rotation 或 target rotation 替换成真值，比较 20/50-step survival。这样可以区分速度误差、姿态误差和时序延迟，而不再靠自然 rollout 的条件 Exit/Entry 猜根因。

## 4. 白板题

### 白板题 1：AWAC 的优势加权行为克隆

从数据集 \(D\) 中采样 \((s,a)\)，用 double-Q 得到：

\[
A(s,a)=\min(Q_1,Q_2)(s,a)-V^\pi(s),\qquad
w=\min\left(\exp(A/\lambda),w_{\max}\right).
\]

actor 最小化：

\[
\mathcal L_{\text{actor}}=-\mathbb E_D[w\log\pi_\theta(a\mid s)].
\]

**口头解释：** 它没有让 actor 随便最大化 Q，而是在数据动作里“挑重点模仿”。优势越高，动作权重越大；指数截断防止错误 Q 把少量样本放大。这个性质适合 25-demo 的保守离线起点，但仍依赖 critic 排序质量。

### 白板题 2：6D rotation 与 SO(3) geodesic loss

网络输出 \(a_1,a_2\in\mathbb R^3\)。先正交化：

\[
b_1=\frac{a_1}{\|a_1\|},\quad
b_2=\frac{a_2-(b_1^Ta_2)b_1}{\|a_2-(b_1^Ta_2)b_1\|},\quad
R=[b_1,b_2,b_1\times b_2].
\]

两个旋转的最短角距离：

\[
d(\hat R,R)=\arccos\left(\frac{\operatorname{tr}(\hat R^TR)-1}{2}\right).
\]

**口头解释：** Euler 的 179° 和 −181° 实际接近但数值跳变，quaternion 又有正负二义性。6D 输出连续，最后投影到合法旋转矩阵；geodesic 直接回答“还差多少度旋转”。

### 白板题 3：GRU 因果历史

对每步特征 \(x_t=[f_{\text{cnn}}(I_t),p_t,a_{t-1},m_t]\)，更新：

\[
h_t=\operatorname{GRU}(x_t,h_{t-1}),\qquad
\hat s_t=g(h_t).
\]

只允许使用 \(I_{t-7:t}\) 和已经执行的 \(a_{t-7:t-1}\)，不能看未来帧。episode 开头用 mask 区分 padding 与真实历史。

**口头解释：** 单帧看不出角速度，连续帧和自己刚执行的动作能帮助判断物体为什么运动。用 executed action 而不是 planned action，确保历史对应真实闭环。

### 白板题 4：Frozen Oracle action-consistency 的梯度路径

\[
\hat s_\theta\rightarrow g(\hat s_\theta,p)
\rightarrow N_{\text{oracle}}
\rightarrow \pi_{\text{oracle}}
\rightarrow \hat a,
\quad
\mathcal L=\|\hat a-a^*\|^2.
\]

Oracle 权重 \(\phi\) 冻结，即不计算或不应用 \(\nabla_\phi\mathcal L\)；但需要 \(\partial \hat a/\partial \hat s\)，所以：

\[
\nabla_\theta\mathcal L
=\frac{\partial\mathcal L}{\partial\hat a}
\frac{\partial\hat a}{\partial\hat s}
\frac{\partial\hat s_\theta}{\partial\theta}.
\]

**口头解释：** 冻结模块像一个固定可微函数。它自己不学，但可以告诉前面的 estimator：哪一个状态误差最会改变动作。

### 白板题 5：paired wins/losses/ties 与 McNemar

对同一批 episode，把 E2 与 E0 二元结果分成：both success、both fail、E2-only success \(b\)、E0-only success \(c\)。ties 不提供方向信息，McNemar exact test 在零假设下认为 \(b\) 与 \(c\) 等可能，相当于对 \(b\) 次“赢”在 \(b+c\) 个 discordant episode 中做双侧 binomial test。

confirmation Benchmark：\(b=66,c=18\)，ties 216；Strict：\(b=60,c=30\)，ties 210。

**口头解释：** 同一个初始状态本来就有难易差异，paired 设计把这种差异控制住。真正有信息的是两策略结果不一致的 episode，而不是把 300+300 条当成独立样本。

## 5. 可以说与不能说

| 主题 | 可以说 | 不能说 | 原因 |
| --- | --- | --- | --- |
| 总体水平 | 在固定 Adroit Pen 仿真协议上得到可复现提升 | 达到 SOTA | 没有统一协议下重跑公开方法 |
| Oracle 对比 | E2 在 300 bank 上达到接近 Oracle 的闭环水平 | E2 普遍超过 Oracle | 不同 bank 名义排序变化，未做普遍性检验 |
| 数据规模 | 25 demos 训练 AWAC；108,800 sim samples 训练 E2 estimator | 仅用 25 条示范训练最终视觉策略 | 隐瞒了 544 条仿真轨迹的 privileged supervision |
| frozen Benchmark | paired 54.5%→75.0%，CI `[+13,+28]pp` | 每个 seed 都有 75% | 75% 是单一预选 E2 在一个 frozen bank 的点估计 |
| frozen Strict | 26.0%→33.5%，方向性提升 | frozen-test Strict 显著提升 | CI `[0,+15]pp`，下界触及 0，`p=0.0722` |
| 硬件 | 纯仿真、单 RTX 5090 配置 | 完成 sim-to-real | 没有跨域实验 |
| 部署 | 推理路径只读 RGB/proprio/action history | 完成真实硬件部署 | 没有真机或 HIL |
| 方法类型 | 监督式时序视觉状态蒸馏 | 最终方法是 online RL | E2 的核心优化是监督状态/action consistency loss |
| hold | acquisition 与 20-step window 改善，50-step 仍弱 | 已解决长期 hold | fixed-entry E2 50-step 低于 E0/Oracle |
| 3-seed | 61%±4.0pp / 35%±6.24pp，方向复现但 Strict 有方差 | 3-seed 每次都达到 75% | 实际各 seed 为 65/61/57% Benchmark |
| V6 历史 | V6 STOP 保持，V6.1 是固定 E2 的新 bank 证据 | V6 当时其实已经通过 | 旧 exit gate 超 0.143pp |
| 原创性 | 本项目实现并验证了这条组合路线 | 首次提出、彻底解决 | 没有足够文献和统一比较支持 |

### 面试时容易踩的三个口径坑

1. **不要把不同 bank 连成学习曲线。** 37.3%、51.0%、62%、68.3% 和 75% 分别来自不同训练/评测协议。可以按阶段描述，但不能说策略沿同一个 test set 单调增长。
2. **不要把条件 Exit/Entry 当作独立成功率。** D1 的 exit 看似更低，却伴随 entry 减少；真正比较 hold 要看共同 first-entry roots。
3. **不要把三个训练模型当 300 个独立 episode。** 3-seed 复现共享同一组 100 evaluation states，报告训练 seed 的均值与标准差，不做普通 pooled binomial CI。

## 6. 简历表述

### 项目标题

**Adroit Pen 视觉灵巧操作：离线强化学习与时序视觉状态蒸馏**

### 中文简历项目描述

在 MuJoCo Adroit Pen 纯仿真环境中，基于 25 条人类示范（4,975 个合法离线 transition）实现 Visual BC、double-Q critic warm-up 与 offline AWAC，并通过 RGB 黑屏/打乱/目标遮挡验证视觉因果性；针对 PPO、Residual SAC 与反事实动作修正的闭环退化，设计 8-step CNN+GRU Temporal Visual-State Distillation，利用 544 条仿真轨迹、108,800 个稠密 privileged-state/Oracle-action 监督样本，在部署仅使用 RGB、proprioception 与动作历史。固定 E2 在 300-episode paired confirmation 上将 Benchmark/Strict 从 52.3%/27.0% 提升至 68.3%/37.0%，one-shot 200-episode frozen test Benchmark 达到 75.0%。

### 英文简历 bullet

Implemented a reproducible visual control pipeline for simulated Adroit Pen reorientation, bootstrapping offline AWAC from 25 human demonstrations (4,975 valid transitions) and replacing unstable PPO/residual handoffs with an 8-step temporal visual-state distillation model trained on 108,800 simulator-supervised samples; improved paired 300-episode Benchmark/Strict success from 52.3%/27.0% to 68.3%/37.0%, with RGB-only-at-deployment inference and a 75.0% Benchmark rate on a one-shot 200-episode frozen test.

### 三个面试展开关键词

1. **Offline-to-online failure analysis**：PPO drift、Q–MC calibration、接触模式放大和 counterfactual gain/harm。
2. **Privileged temporal state distillation**：8-step causal GRU、6D rotation、Frozen Oracle action-consistency。
3. **Paired closed-loop evaluation**：独立 bank、wins/losses/ties、McNemar、fixed-entry hold 和证据边界。

## 7. 面试前 60 秒自检

- 我有没有先说“纯仿真、无 sim-to-real”？
- 我有没有把 25 demos 和 108,800 simulation transitions 分开？
- 我有没有说明 E2 部署不读 simulator state？
- 我有没有把 final method 说成监督式 state distillation，而不是 online RL？
- 我有没有用 300 confirmation 的 52.3%→68.3%、27.0%→37.0% 作为主证据？
- 我有没有说明 frozen Strict 的 CI 下界为 0？
- 我有没有说明 3-seed 共用同一 100 evaluation states？
- 我有没有主动承认 fixed-entry 50-step hold 仍弱？
- 我能否在白板上画出 action-consistency 的梯度路径？
- 如果被问“为什么不继续调 SAC”，我能否回答 Q–MC bias +40.08 和极小 residual 仍退化？

更完整的方法、结果文件和复现命令见[中文技术报告](adroit_pen_technical_report_zh.md)。
