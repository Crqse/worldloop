# 为什么 LLM Agent 应该「提议」，而不是「裁决」

> 一篇短小、偏执的说明：为什么我们把「世界」重新放到权威位置，以及这对
> 多智能体仿真与研究意味着什么。写给正在构建 Agent 框架、MARL/模拟器工具链、
> 以及合成数据管线的人。

## 一句话主张

> **LLM / 策略只负责 *提议*；一个确定性的世界负责 *裁决* 每一个后果 ——
> 每一步变化都是一条可验证、带哈希链的状态差，可重放、可分支、可反事实比对。**

## 「LLM 就是世界」的问题

大多数 Agent 框架把状态放在聊天历史里。模型既是**提议者**，也是**裁判**，
还是**记录员**。这让三个问题根本无法诚实回答：

1. **世界到底发生了什么？** —— 谁提议了什么、实际执行了什么、结算结果如何？
2. **能否重放？** —— 两次运行能否逐 tick 复现？
3. **某个数字能否追溯到某次状态转移？** —— 还是每个指标都只是上下文窗口里的「感觉」？

当 LLM 既提议又裁决时，你无法把「模型想要的」与「系统实际做的」分开；
你也没法确定性地给轨迹分叉 —— 因为「世界」是一串非确定性的 token。

## WorldLoop 的答案

WorldLoop 把分层**倒过来**：

```
observe S_t → propose → world validates & settles → write S_{t+1}
→ verify hash & invariants → record / replay / branch / export
```

- 一个 LLM 或策略只能**提交一个候选动作**。它不能直接改能量、位置、资源数量，
  更不能决定生死。
- 世界负责**合法性检查、冲突处理、成本/资源结算、状态回写** —— 全部由确定性规则完成。
- 每一步转移都记录为**状态差 + 哈希链**。

因为世界确定且权威，轨迹就变成**一等公民、可审查的工件**：你能精确重放它，
能在任意 tick 用一个不同动作给它分叉，并问出「如果 agent_0 在第 4 tick 往右
而不是往上，结果会怎样？」

## 一个 10 行的尝鲜

完整 gist 在这里：
https://gist.github.com/Crqse/0ad906b3aa956133bd1391b5c7404796

```python
def run(seed, policy):
    env = make_simple_spread_env(n_agents=2, n_landmarks=2, max_cycles=25)
    adapter = PettingZooParallelAdapter(env=env, env_id="simple_spread_v3")
    adapter.reset(seed=seed)
    chain = []
    for tick in range(N_TICKS):
        proposal = ActionProposal(
            agent_id=AGENT, action_type="move",
            params={"discrete_action": policy(tick)},
            proposed_at_tick=tick, proposer="demo")
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)   # 其它 agent 默认 STAY
        chain.append(record.state_after_hash)
    return chain

# 确定性重放
assert run(42, policy_a) == run(42, policy_a)           # 逐 tick 一致
# 反事实分叉
assert run(42, policy_a)[4] != run(42, policy_b)[4]      # 第 4 tick 分叉
```

输出：`deterministic replay same? True`，`branches diverge by tick: [4,5,6,7]`。

## 为什么不是「又一个 Agent 框架」？

WorldLoop **不是**通用 Agent 框架。它是**环境权威、可验证的状态转移基座**。
这带来三个具体场景：

1. **确定性重放与反事实研究** —— 让一项「改动」的声明变成可复现的 diff，
   而不是一种感觉。
2. **场景即数据** —— 场景是 YAML、schema 校验、运行**前**编译（非法场景在编译期
   就拒绝，而不是运行到一半才崩）。
3. **无损导出、可做泄漏检查** —— 转移导出为结构化数据集并附带泄漏报告，
   让你能想清楚到底把什么交给了下游训练器。

## 诚实的边界（我们主动报告负结果）

- 这是**研究级** alpha，不是久经生产考验的系统。
- M8 反事实数据价值是**负结果**：在这个规模下，match 过的反事实并没有可测量地
  提升下游轨迹数据集价值。我们公开报告它，而不是粉饰。
- 对*任意*环境的精确状态恢复，只在机械验证过的地方声明（当前是 MPE2
  Simple Spread / Simple Tag 一族）。我们从不声称自己没证明过的能力。
- 全文**不**做任何「训练增益」主张。我们发布的是**机制**（重放、分叉、序列化、
  泄漏报告），不是某个吸睛的涨点数字。

## 盒子里面有什么

四个兄弟包，一个扁平仓库，Ubuntu + Windows × Python 3.10 / 3.12 下经由
GitHub Actions `ci.yml` 全绿：

| 包 | 职责 |
|---|---|
| `worldloop-kernel` | 协议、状态、动作、转移、记录器、哈希链、重放、分叉、检查点 |
| `worldloop-scenarios` | ScenarioSpec v0 YAML schema → 校验 → 编译 → 参数化世界 |
| `worldloop-adapters` | PettingZoo Parallel / Gymnasium / OpenEnv → kernel `WorldProtocol` |
| `worldloop-data` | 覆盖率调度、反事实分叉器、导出器、泄漏检查、质量报告 |

## 快速开始

```bash
git clone https://github.com/Crqse/worldloop.git
python -m pip install -e ./worldloop-kernel -e ./worldloop-scenarios \
    -e ./worldloop-data -e "worldloop-adapters[dev]"
pytest worldloop-kernel/tests worldloop-scenarios/tests \
       worldloop-adapters/tests worldloop-data/tests -q
```

---

*如果你在做「会行动的 Agent」，最被低估的工程决策是：谁来做裁判。我们觉得
应该是环境（世界）。欢迎提交 issue，或发邮件至 1148395497@qq.com —— 也愿意
聊聊付费集成、技术支持订阅与定制开发。*
