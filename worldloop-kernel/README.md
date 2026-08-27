# worldloop-kernel


WorldLoop v2 独立转移微内核。把一次世界变化变成可执行、可验证、可重放、可分支的标准过程。

## 这个包是什么

`worldloop-kernel` 是 WorldLoop v2 的协议层和运行时骨架。它定义一次状态转移从候选行动、合法性检查、环境裁决、状态差分到 checkpoint / replay / counterfactual branch 的标准协议，并把所有可执行、可验证、可重放的工件以版本化、可序列化的方式记录下来。

它不是第六层，不取得任何世界的状态所有权，不调用 LLM，不写业务规则，不做训练。

## 它解决什么问题

WorldLoop v1 的五层 Native 主线（an earlier five-layer native line）已经具备 WST、ECS、WorldGraph、注册表、种群、动作执行、能量、死亡、繁殖、WorldTransition V1 协议、checkpoint 和确定性重放能力。但所有这些能力都嵌在五层内部，外部环境（PettingZoo / Gymnasium / OpenEnv / MCP / learned world）无法复用同一套协议。

`worldloop-kernel` 把"包围并记录一次完整转移"的最小协议抽出来，让任何世界（Native 五层、外部环境、learned simulator）都通过同一接口被运行、记录、验证和重放。

## 它不解决什么

- 不解决数字生命、开放式演化、真实社会预测或通用世界模型
- 不解决在线 RLVR 训练闭环
- 不解决 LLM 调用、提示词或 agent 记忆
- 不解决具体世界规则（WST 动力学、WorldGraph 更新、L4 选择繁殖等仍由各 world 实现持有）
- 不解决可视化、分布式调度、训练算法、特定领域 reward

## 当前能力

M0/M1 Gates 全部闭合：

- 世界协议（`WorldProtocol`）：`step()`, `validate_action()`, `legal_actions()`, `checkpoint()`, `restore()`, `capabilities()`, `reset()`
- 状态转移（`StateView`, `StateDelta`, `canonical_encode`, `hash_state`, `diff_state`, `apply_delta`）
- 7 条硬不变量校验 + quarantine
- 完整 checkpoint / deterministic replay / counterfactual branch（含 frozen-action replay）
- ToyWorld 1000-step diff/apply + exact replay 一致
- `pip install -e .` / wheel / sdist build 成功，0 v1 imports，0 third-party deps

0.1.2 新增（Beta correction Phase 1 / Phase 5）：

- **Agent-local observation 契约**：`AgentObservationView` + `ObservationProjector` + `hash_observation`（每-agent 授权观测投影，不直接暴露全局 `StateView`，hidden-state non-leak tests PASS）
- **Joint action 提交**（multi-agent 同 tick）：`JointAction` / `JointReceipt` / `JointActionWorld` 协议 + `supports_joint_actions()` + missing-agent 策略（NOOP/STAY/ERROR）。边界：kernel 只定义协议；实际 joint 执行由 world/adapter 实现，当前经机械验证的外部环境仅限 adapters 包的 exact-restore verified allowlist（2 个 env families）

## 设计原则

1. **协议一等、实现外置**：kernel 原生认识 fields/entities/relations/registries/population 五类状态接口，但不实现具体动力学（WST 通道、WorldGraph 边类型、ObjectRegistry 投影规则、L4 选择繁殖等留在各 world 实现内）。
2. **StateView 与 Checkpoint 分离**：StateView 是给模型、数据集、评估器看的可交换状态；Checkpoint 是给执行器恢复世界用的完整状态（含隐藏变量、内部缓存、RNG）。
3. **不伪造能力**：环境不具备的能力通过 `CapabilityProfile` 显式声明 `false`，缺失字段通过 `missing_mask=True` 表达，二者不得混用。
4. **authority 标注**：learned/neural simulator 必须声明 `authority=learned`、`ground_truth=false`，不能伪装成规则真值。
5. **核心依赖最小化**：核心包仅依赖 Python 标准库；JSON Schema 等可选依赖放 extras。

## 包边界与依赖方向

- `worldloop-kernel` 不得 `import current.worldloop.core.*`
- `worldloop-kernel` 核心依赖优先保持 Python 标准库
- 现有五层通过 `WorldLoopNativeAdapter` 实现 kernel 协议（M1 阶段）
- `worldloop-adapters` 可以依赖 kernel 和对应外部框架
- `worldloop-scenarios` 可以依赖 kernel，不依赖完整五层

完整架构决策见各包 `README.md` 与 `docs/CLAIMS.md`（公开仓库根目录）。

## 使用

```python
from worldloop_kernel import (
    WorldProtocol, StateView, ActionProposal, TransitionRecord,
    ToyWorld, canonical_encode, hash_state
)

# 创建内置验证世界
world = ToyWorld(grid_length=10)
world.reset(seed=42)

# 提议-校验-执行闭环
space = world.legal_actions(agent_id="agent_0")
proposal = ActionProposal(          # 实际使用中可由 LLM/Policy 生成
    agent_id=space.agent_id,
    action_type=space.legal_actions[0].action_type,
    params=space.legal_actions[0].params,
    proposed_at_tick=0, proposer="readme-example",
)
executed, receipt = world.validate_action(proposal)
assert receipt.success

transition = world.step(executed)
print(transition.tick, transition.state_after_hash)

# checkpoint → restore → replay
ckpt = world.checkpoint()
world2 = ToyWorld(grid_length=10)
world2.restore(ckpt)
```

## 安装

```bash
pip install -e .      # 开发模式（源码可编辑）
pip install build && python -m build   # 构建 wheel/sdist
```

## 测试

```bash
pytest tests -q        # 254 passed (kernel + conformance + dual-run + gate)
```

## License

MIT
