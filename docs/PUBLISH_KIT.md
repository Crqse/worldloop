# WorldLoop 发布操作手册（对外发布物 & 渠道指南）

> 本文档是"公开之后怎么让人知道"的操作手册。所有文案都是最终稿，可直接复制粘贴。
> 唯一需要做的准备：在每个平台登录你自己的账号（发帖需要账号，无法代办）。

---

## 0. 我们有什么可发的（全部已就绪）

| 发布物 | 地址 | 用途 |
|---|---|---|
| 仓库 | https://github.com/Crqse/worldloop | 一切内容的主入口 |
| 主页（GitHub Pages） | https://crqse.github.io/worldloop/ | 给不想读代码的人看的门面 |
| 交互 demo | https://crqse.github.io/worldloop/examples/assets/emergency_scheduling.html | 浏览器直接玩，最有说服力 |
| 40 行最小示例 Gist | https://gist.github.com/Crqse/0ad906b3aa956133bd1391b5c7404796 | 给开发者 30 秒看懂核心 |
| Beta Release | https://github.com/Crqse/worldloop/releases/tag/v0.1.3-beta.1 | 正式版本入口 |
| 英文长文 | 仓库 `docs/BLOG_en.md` | 深度阅读版（HN/Reddit 用） |
| 中文长文 | 仓库 `docs/BLOG_zh.md` | 深度阅读版（知乎/掘金用） |

**核心一句话（所有帖子的主轴，别改）**：

> Agents/LLMs only **propose**; a deterministic world **adjudicates** every consequence — replayable, branchable, counterfactually comparable.
> （Agent/LLM 只负责提议，确定性世界负责裁决——可重放、可分支、可反事实比较。）

---

## 1. 发布顺序（重要：先英文后中文，先小后大）

```
第1天  Show HN（英文，先发）
第1天  Reddit r/MachineLearning（英文，HN 发完 2-3 小时后）
第2天  V2EX / 掘金（中文，攒反馈后调整话术）
第3天  知乎专栏（中文，长文完整版）
随时   Twitter/X、朋友圈等（可选，一条链接+一句话即可）
```

为什么这个顺序：HN 是最容易出"第一批真实用户"的地方（技术人群密集、不需要粉丝）；中文平台反馈能帮你快速改进英文话术里没讲清的点。

---

## 2. 各平台发布操作 + 最终文案

### 2.1 Hacker News（Show HN）— 最重要的一个

**地址**：https://news.ycombinator.com/submit
**要求**：需要账号（https://news.ycombinator.com/login?goto=news 注册，邮箱即可，无需手机号）
**规则要点**：
- 标题必须以 `Show HN:` 开头（Show 规则）
- 不要自己顶帖、不要刷评论（会被 shadow ban）
- 发布后前 1-2 小时守着，认真回复每条评论（HN 最看重作者响应）

**标题（直接复制）**：

```
Show HN: WorldLoop – Agents propose, a deterministic world adjudicates
```

**正文/首评（提交后自己发的第一条评论，直接复制）**：

```
Hi HN, I built an environment-authoritative multi-agent simulation substrate.

The core idea: in most agent frameworks the LLM is proposer, judge, and
record-keeper at once — state lives in the chat history. That makes three
questions unanswerable: what actually happened, can I replay it, can I trace
a number back to a transition?

WorldLoop inverts it: policies/LLMs only submit candidate actions. A
deterministic world performs legality checks, conflict resolution, cost
settlement, and state write-back. Every transition is recorded as a
hash-chained state diff, so any trajectory can be replayed, branched, and
compared counterfactually.

A 40-line runnable taste (deterministic replay + counterfactual branch on
PettingZoo Simple Spread):
https://gist.github.com/Crqse/0ad906b3aa956133bd1391b5c7404796

Interactive browser demo (no install):
https://crqse.github.io/worldloop/examples/assets/emergency_scheduling.html

Four packages: kernel (protocol/replay/branch), scenarios (YAML → compiled
world), adapters (PettingZoo/Gymnasium), data (trajectory export + leakage
checks). All green on Ubuntu+Windows × Py3.10/3.12.

Honest limits: this is a research-grade beta; our M8 counterfactual
data-value result was a NEGATIVE result, which we report openly. No
training-gain claims anywhere.

Happy to answer anything — especially happy to hear what you'd use a
replayable, branchable world for.
```

**链接（submit 表单的 url 字段）**：https://github.com/Crqse/worldloop

---

### 2.2 Reddit r/MachineLearning

**地址**：https://www.reddit.com/r/MachineLearning/submit
**要求**：需要 Reddit 账号（https://www.reddit.com/register，邮箱即可）
**规则要点**：
- r/MachineLearning 发帖选 flair：[D]（Discussion）或 [R]（Research）
- 版规禁止纯 self-promo，所以文案要"讨论导向"，别写成广告
- 同样：认真回复评论，别只发不回

**标题**：

```
[D] Why LLM agents should propose, not adjudicate — I built a deterministic world substrate where every transition is replayable/branchable
```

**正文**：

```
Most agent frameworks carry state in chat history — the model is proposer,
judge, and record-keeper at once. For workflow orchestration that's fine.
For state-transition research it means you can't separate what the model
wanted from what the system did, and you can't deterministically branch a
trajectory.

I built WorldLoop as the opposite: agents only submit candidate actions; a
deterministic world does legality checks, conflict resolution, settlement,
and state write-back. Every transition is a hash-chained state diff, so
trajectories can be replayed byte-identically, forked from any checkpoint,
and compared counterfactually.

Runnable 40-line example (PettingZoo Simple Spread, deterministic replay +
counterfactual branch):
https://gist.github.com/Crqse/0ad906b3aa956133bd1391b5c7404796

Use cases it's aimed at:
1. Reproducible multi-agent runs (two runs = byte-identical hashes)
2. Counterfactual experiments (fork at tick N, change one action, compare)
3. Trajectory datasets with leakage reports (paths/keys/PII flagged)
4. Wrapping PettingZoo/Gymnasium envs under the same authority contract

Honest limits: research-grade beta, exact-restore only verified for 2 MPE
env families, and our counterfactual data-value evaluation came back
negative — reported openly, no training-gain claims.

Curious what people here think: is "environment-authoritative" the right
frame, or do you solve reproducibility differently in your MARL work?

Repo: https://github.com/Crqse/worldloop
```

**备选 subreddit**（若 r/MachineLearning 被自动过滤，可试）：
- r/LocalLLaMA（更活跃，对 LLM agent 基建友好）
- r/reinforcementlearning（MARL 受众精准）

---

### 2.3 掘金（中文，第 2 天）

**地址**：https://juejin.cn/editor/drafts/new
**要求**：掘金账号（手机号/微信登录）
**操作**：编辑器里直接贴 `docs/BLOG_zh.md` 全文 → 标题用下面这条 → 标签选"人工智能、开源、Python"

**标题**：

```
Agent 只负责提议，确定性世界负责裁决：我写了一个可重放、可反事实的多智能体仿真基座
```

**封面**：直接用仓库里的 `examples/assets/single_step_chain.svg`（截图保存为 png 上传）

---

### 2.4 知乎专栏（中文，第 3 天，长文完整版）

**地址**：https://zhuanlan.zhihu.com/write
**要求**：知乎账号
**操作**：贴 `docs/BLOG_zh.md` 全文（知乎支持 markdown 粘贴），标题同掘金，可在开头补一句：

```
开源地址：https://github.com/Crqse/worldloop （欢迎 issue 和场景 YAML 贡献）
```

**发布后动作**：知乎回答区可以搜"多智能体仿真""LLM Agent 可复现性"相关问题，
在高质量相关问题下写简短回答并附链接（比硬广有效，但要真的回答问题）。

---

### 2.5 V2EX（中文，轻量，第 2 天任意时间）

**地址**：https://www.v2ex.com/new/tech
**节点**：/t/tech 或 /go/create（分享创造节点，最合适：https://www.v2ex.com/go/create）

**标题**：

```
[分享] 写了一个环境权威的多智能体仿真基座 WorldLoop，Agent 只提议、世界裁决
```

**正文（短版）**：

```
最近把做了一段时间的项目开源了：WorldLoop —— 一个环境权威的多智能体
仿真与轨迹数据生成系统。

核心思路：LLM/策略只提交候选动作，一个确定性世界负责合法性检查、冲突
处理、数值结算和状态写入。每步转移都是带哈希链的状态差分，所以任何
轨迹都可以重放、分叉、反事实比较。

- 仓库：https://github.com/Crqse/worldloop
- 浏览器直接玩的 demo：https://crqse.github.io/worldloop/examples/assets/emergency_scheduling.html
- 40 行最小示例：https://gist.github.com/Crqse/0ad906b3aa956133bd1391b5c7404796

四个包（kernel/scenarios/adapters/data），Ubuntu+Windows × Py3.10/3.12 CI
全绿。研究级 beta，反事实数据价值的评估结果是负结果，我们如实公开。
欢迎来提 issue 或者贡献一个场景 YAML。
```

---

## 3. 通用注意事项（避免踩坑）

1. **同一个帖子别在多个平台互相抄评论区链接**——各平台都反感引流
2. **HN 发帖后前 2 小时必须在线回评论**，这是 Show HN 出成绩的最大变量
3. **别用营销词**（revolutionary/game-changing/颠覆），HN 和 Reddit 社区对浮夸零容忍；我们已经准备好的"诚实负结果"话术反而是加分项
4. **发完不要自己转发到自己小号顶帖**，被发现就是永封
5. **有人在评论里指出问题 = 好事**。哪怕被挑刺，认真回应"你说得对，我记下了"远好于辩护
6. 若 HN 帖子 24 小时只有 <50 浏览，正常（没上首页就是这水平）；上了首页也别慌，准备好随时答问题

---

## 4. 发布后第一周盯什么（不看 Star）

每周固定看三个数（`gh` 命令可用）：

```powershell
# 外部 issue / 讨论数（目标 ≥3）
gh issue list -R Crqse/worldloop --state all

# 有没有人提交 PR / 新场景 YAML（目标 ≥2）
gh pr list -R Crqse/worldloop --state all

# Release 下载量 / 仓库访问（Settings → Traffic 看克隆数）
gh api repos/Crqse/worldloop/traffic/clones
gh api repos/Crqse/worldloop/traffic/views
```

**第一阶段成功标准**（比 Star 重要得多）：
- ≥10 个陌生人成功 clone + 跑通 4 包测试
- ≥3 个外部 issue（哪怕是问问题的）
- ≥2 个社区贡献的场景 YAML 被 merge

---

## 5. 可选加强项（不急）

- **30–60s GIF 挂 Release**：录屏 `crqse.github.io/worldloop` 的交互 demo →
  上传到 https://github.com/Crqse/worldloop/releases/tag/v0.1.3-beta.1 的 assets
- **Tweet/X**：一句话 + 仓库链接 + demo 截图，@ PettingZoo 官方（他们转发过
  很多基于 PZ 的项目）
- **PyPI 发布**：等有 5-10 个真实用户、且有人提出"想 pip install"时再做
  （名字已确认可用；届时需要你注册 pypi.org 账号 + API token）
