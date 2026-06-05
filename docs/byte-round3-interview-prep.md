# 医疗智能助手 — 字节三面面试准备

> 字节三面是交叉面/总监面，风格和二面完全不同：**不追问代码细节，考的是技术视野、系统思维、业务判断和软素质。**

---

## 目录

- [一、三面面试风格与核心差异](#一三面面试风格与核心差异)
- [二、项目宏观理解](#二项目宏观理解)
- [三、技术深度与广度](#三技术深度与广度)
- [四、系统设计能力](#四系统设计能力)
- [五、业务理解与产品思维](#五业务理解与产品思维)
- [六、工程判断与权衡](#六工程判断与权衡)
- [七、软素质与成长](#七软素质与成长)
- [八、反问环节](#八反问环节)

---

## 一、三面面试风格与核心差异

| 维度 | 二面 | 三面 |
|------|------|------|
| 关注点 | 怎么实现的 | 为什么这么做、还能怎么做 |
| 提问风格 | 连环追问挖细节 | 宏观问题看视野 |
| 代码深度 | 要求到函数级 | 关注架构级和决策级 |
| 新问题 | 少，围绕项目 | 多，开放性场景题 |
| 软素质 | 不怎么考 | 重点考察 |

**三面核心考察维度**：
1. 技术视野——知不知道行业在做什么、方案在什么位置
2. 系统思维——能不能从全局看问题，不是只盯着一个模块
3. 业务判断——技术决策有没有业务意识
4. 成长潜力——遇到问题怎么反思、怎么改进

---

## 二、项目宏观理解

### Q1：用一句话说清楚你的项目解决了什么问题？

> 面试官想看你会不会抽象，能不能站在用户视角而不是技术视角描述。

**参考回答**：

解决了医疗场景下"问诊、分诊、挂号"三个核心需求在一轮对话中无缝切换的问题。用户不需要分别打开三个功能，说一句话系统自动识别意图走不同链路，而且中途可以交叉——挂号途中问个医学问题，回来还能继续挂号。

---

### Q2：如果让你给这个项目打分，满分 10 分你打几分？扣分在哪？

> 考察你有没有自省能力，能不能看到自己系统的不足。

**参考回答**：

7 分。扣掉的 3 分在三个地方：

**第一，评估体系不完整**（扣 1 分）。意图路由准确率、检索精度、回答质量这些指标有离线测试集，但样本量小（意图 21 个、检索 10 个），没有线上 A/B 测试，也没有消融实验量化每个优化的贡献。一个真正可度量的系统应该有完整的评估 pipeline。

**第二，生产级可靠性不足**（扣 1 分）。没有 LLM 熔断和降级链，没有 Redis 高可用集群，没有分布式限流，没有压测数据。系统"能跑"和"能扛"是两回事。

**第三，用户反馈闭环缺失**（扣 1 分）。没有收集用户对回答质量的反馈，也没有 badcase 自动发现和回归机制。系统上线后靠什么持续优化？现在靠人看，不可持续。

---

### Q3：你项目中最大的技术风险是什么？如果上线了你会最担心什么？

**参考回答**：

最担心 LLM 的不可控性。虽然我做了很多兜底——规则层、关键词校验、grounding check——但本质上 LLM 的输出是概率性的，没有 100% 的保证。上线后最可能出问题的场景是：

1. **prompt injection**——用户刻意绕过安全机制，虽然三层防御能挡住大部分，但不是绝对的
2. **LLM 幻觉漏过 grounding check**——grounding check 本身也是 LLM 做的，有可能漏判
3. **长尾医学问题**——知识库没覆盖的罕见病，系统给了通用建议但用户当成专业诊断

如果真的上线，我会加三层监控：回答质量抽样人工审核、高风险关键词自动告警、用户反馈收集渠道。

---

## 三、技术深度与广度

### Q4：你觉得 RAG 这个方向未来会怎么演进？你现在用的方案有什么局限？

> 考察技术视野，不只是"做了什么"，还要知道"行业在往哪走"。

**参考回答**：

当前 RAG 的核心局限是**检索和生成是割裂的**——先检索再生成，检索错了生成不可能对。未来的演进方向我关注三个：

**第一，检索和生成的深度耦合**。比如 FLARE（Active Retrieval Augmented Generation），不是一次性检索，而是生成过程中遇到不确定的地方动态检索。还有 Self-RAG，模型自己决定要不要检索、检索结果要不要用。这比当前"检索完就不管了"的方案更智能。

**第二，更长的上下文窗口可能改变 RAG 的形态**。现在 Gemini 支持 100 万 token，如果上下文够长，是不是可以直接把整个知识库塞进去？我认为在医疗场景不太行——知识库会持续增长，全塞进去成本太高，而且长上下文的注意力衰减问题还没解决。RAG 的"按需检索"在成本和精度上仍然有优势。

**第三，多模态 RAG**。医疗场景有影像、表格、图表，当前只处理了文本。未来需要把影像检索和文本检索融合，这对 embedding 模型和检索架构都有新要求。

我当前方案的局限：检索是单轮的（虽然 check_sufficiency 会重试一次，但不是真正的多轮检索）、只处理文本、rerank 模型是通用的不是医疗专用的。

---

### Q5：你对 Agent 这个方向怎么看？你的项目算不算 Agent？

> 考察你对 Agent 概念的理解，而不是只会用框架。

**参考回答**：

Agent 的核心是**自主决策 + 工具调用 + 环境交互**。我的项目算半个 Agent——有工具调用（检索工具、预约工具），但决策不是完全自主的。意图路由是规则+LLM 分类的，不是模型自己决定下一步做什么；预约流程是半受控的，不是模型自由规划执行路径。

这是有意的设计。在医疗场景下，完全自主的 Agent 风险太高——模型可能走错路径、误判意图、直接执行不该执行的操作。我选择的是"受控 Agent"：LLM 负责理解和生成，关键路径由代码控制。这和 Anthropic 提出的"Constitutional AI"思路一致——给 AI 设边界，而不是给它完全自由。

如果未来场景更复杂（比如多步骤诊断推理、跨系统数据查询），可能需要更自主的 Agent，但安全机制必须同步升级。

---

### Q6：如果你要选一个 LLM provider，你会怎么选？考虑哪些因素？

**参考回答**：

五个维度：

1. **效果**——中文医学场景下的理解能力，需要实际测试，不能只看 benchmark
2. **延迟**——对话场景用户对首 token 延迟敏感，流式输出的 TTFT（Time To First Token）很关键
3. **成本**——意图分类和 query rewrite 这种高频调用要用便宜模型，回答生成可以用更强模型
4. **稳定性**——SLA 保障、降级方案、多 provider 支持
5. **合规**——医疗数据出境问题，如果用海外 provider 需要考虑数据合规

当前项目支持多 provider 切换（deepseek/openai/anthropic/google/ollama），就是为了灵活应对这些因素的变化。实际选择时会做分级：高频低价值路径（意图分类、摘要）用便宜模型，低频高价值路径（回答生成、科室推荐）用强模型。

---

## 四、系统设计能力

### Q7：如果让你从零设计一个日活 10 万的医疗对话系统，你会怎么设计？和你的项目有什么不同？

> 考察系统设计能力，看你能不能从"项目级"跳到"产品级"思维。

**参考回答**：

和当前项目三个核心差异：

**第一，架构要拆服务**。当前是单体 FastAPI，10 万日活需要拆成：API 网关层（限流、鉴权）、对话服务（无状态，可水平扩展）、检索服务（独立部署，缓存热 query）、LLM 调度服务（多 provider、降级、熔断）、预约服务（独立事务）。当前的单体是因为开发阶段优先简单，但上线前必须拆。

**第二，存储要分层**。Redis 集群替代单点（会话状态）、PostgreSQL 读写分离（业务数据）、向量库可能要迁移到专业服务（Milvus/Qdrant，支持分布式）、CDN 缓存热门文档的 embedding 结果。当前 pgvector 够用是因为数据量小，10 万日活意味着知识库和检索日志量会大幅增长。

**第三，可观测性要完整**。当前只有日志，需要加：链路追踪（OpenTelemetry）、指标监控（LLM 调用延迟/成功率/成本、检索延迟/召回率、端到端延迟）、告警（LLM 超时、Redis 连接异常、检索空结果率突增）、质量监控（回答质量抽样评分、用户反馈收集）。

**不变的是核心架构**——LangGraph 状态机 + 规则/LLM 两阶段路由 + 三链路分离，这个设计是可扩展的，不需要推倒重来。

---

### Q8：如果让你设计一个多租户版本（不同医院各自独立知识库），你会怎么改？

**参考回答**：

三个核心改动：

**第一，知识库隔离**。当前知识库是全局的，多租户需要每个租户独立的向量空间。方案是在 `child_chunks` 表加 `tenant_id` 字段，检索时强制过滤。更彻底的方案是每个租户独立的 pgvector schema 或者独立的向量库实例，物理隔离更安全但成本更高。

**第二，配置可定制**。不同医院的科室列表、号源规则、预约流程可能不同。当前这些是代码里定义的，需要抽成配置，按租户加载。意图路由的规则也可能需要租户级别覆盖——比如有的医院不支持在线取消预约。

**第三，权限模型升级**。当前是 user/admin 两级，多租户需要 user/tenant_admin/platform_admin 三级。tenant_admin 管自己医院的知识库和配置，platform_admin 管全局。数据访问必须带 tenant_id 过滤，防止跨租户数据泄露。

---

## 五、业务理解与产品思维

### Q9：你觉得这个系统真正的用户价值在哪？技术上做了这么多，用户感知到的是什么？

> 考察你有没有产品意识，不是纯技术自嗨。

**参考回答**：

用户感知到的是三件事：

1. **不用切换功能**——问诊、分诊、挂号在一个对话里完成，不用分别找入口
2. **中途可以岔开**——挂号途中问个问题不会被强制留在预约流程，回来还能继续
3. **回答有来源**——不是 AI 随口说的，是来自权威医学资料，有参考链接

技术上的很多工作（混合检索、rerank、grounding check、规则+LLM 路由）用户是感知不到的，但这些是保证"回答靠谱"的底层支撑。用户不会说"你的 RRF 融合做得好"，但会说"这个回答比别的 AI 靠谱"——这就是技术转化成用户价值的方式。

如果让我排优先级，用户最在意的是第三点——回答的可信度。医疗场景下用户的核心焦虑是"AI 给的建议能不能信"，所以 grounding check、免责声明、来源引用这些安全机制，从产品角度看和功能同等重要。

---

### Q10：如果你的系统上线后，用户反馈"回答不够智能"，你怎么分析这个问题？

**参考回答**：

先定义"不够智能"是什么——可能是三种完全不同的问题：

1. **检索不到**——query rewrite 没改好，或者知识库没覆盖 → 看检索日志的空结果率和 NO_EVIDENCE 比例
2. **检索到了但回答不好**——生成模型的 prompt 或上下文拼装有问题 → 抽样看 grounding check 的结果
3. **用户期望不合理**——用户问了知识库没有的问题（比如个人用药建议），系统给了 fallback → 这不是 bug，是产品预期管理的问题

分析路径：先看数据（检索日志、LLM 调用日志、用户反馈标签），定位是哪个环节的问题，再针对性优化。不是上来就换模型或者加数据，那可能是浪费。

---

## 六、工程判断与权衡

### Q11：你项目里最满意的一个技术决策是什么？为什么？

**参考回答**：

规则 + LLM 两阶段意图路由。

不是因为这个决策最复杂，而是因为它最好地体现了"工程权衡"的思路。全用 LLM 最简单但延迟高成本高确定性差，全用规则最可控但覆盖不了模糊输入。两阶段不是中庸，是把两种方案的优势对准了不同场景——规则对准高频确定性意图（80% 的 query，微秒级，零成本，100% 确定性），LLM 对准模糊意图（20% 的 query，需要语义理解）。

这个决策的难点不在实现，在边界定义——规则和 LLM 的分界线在哪？规则太宽会误路由，太窄 LLM 压力大。我的策略是规则宁可窄一点（高精确率），漏的交给 LLM（高召回率），因为规则误路由的代价远大于 LLM 多处理一个 query。

---

### Q12：你项目里最后悔的一个技术决策是什么？如果重来会怎么做？

**参考回答**：

没有做消融实验和量化评估。

当前所有优化——query rewrite、混合检索、rerank、规则+LLM 路由——都是叠加在一起做的，每个优化各自贡献多少我不知道。Precision@5 从 0.68 到 0.83 是三个优化的叠加效果，但如果去掉 rerank 只剩 0.75，那 rerank 的性价比可能不高（延迟增加显著但精度提升有限）。

如果重来，我会在每个优化上线前后跑同一组测试集，记录每个优化的单独贡献。这样在做工程权衡时（比如要不要为了 5% 的精度提升加 200ms 延迟），有数据支撑而不是凭感觉。

---

### Q13：你怎么判断一个技术方案"够用"还是"需要优化"？

**参考回答**：

看两个维度：**当前指标是否满足业务要求**，和**指标恶化的速度是否可控**。

如果当前指标够，但数据量增长后指标会快速恶化（比如 pgvector 在百万级数据下性能下降），那"够用"只是暂时的，需要提前规划优化方案。

如果当前指标不够，但优化成本极高（比如把 rerank 模型换成医疗专用模型，训练成本和推理成本都高），需要先评估"不够"对业务的影响有多大——如果 5% 的 case 回答质量差但不是安全问题，可能优先级低于其他改进。

核心原则：**不为优化而优化，为业务目标而优化**。医疗场景下安全和准确性的优先级高于延迟和成本，但通用场景可能反过来。

---

### Q14：项目可扩展性有什么不足？如果要加一个新功能你会怎么优化？

> 考察你能不能发现架构层面的瓶颈，以及有没有可落地的改进方案。

**当前不足**：

现在加一个新意图要改四五个文件——`graph_state.py` 加状态字段、`routing_nodes.py` 加规则、`edges.py` 加路由、`graph.py` 加节点。加"取消预约"的时候我就踩过坑：取消规则插到预约规则后面了，"取消预约"被预约规则先吞掉，测试才抓出来。

问题的根源是**意图的规则、状态、路由散落在不同文件里**，不属于同一个功能单元。改一个意图要同时改四五个地方，容易漏，而且没有编译期保护——插错位置只有跑测试才知道。

**优化方案：Skill 注册机制**

项目里 `appointment_skill/` 已经是一个独立目录的雏形——规则、状态、节点都在一起。但它的状态字段还是写死在 `AgentState` 里的，路由也是硬编码在 `edges.py` 里的。我的思路是把这种模式抽象成通用的 Skill 注册机制，核心三件事：

**第一，规则配置化。** 现在的 `_classify_query_by_rules` 是一个 if-elif 链，加新意图要往里插。改成每个 Skill 声明自己的优先级和 `match` 方法，系统启动时按优先级自动排序遍历。加新意图不改已有代码，只加新文件。

**第二，状态命名空间化。** 现在每个功能的状态字段写死在 `AgentState` 里。改成 `AgentState` 加一个通用 `skill_data: dict` 字段，每个 Skill 读写自己的 key，不用改 `graph_state.py`。

**第三，路由自注册。** 现在路由条件硬编码在 `edges.py` 里。改成每个 Skill 自己声明路由条件，图构建时自动装配。

**具体例子：预约和取消预约做成两个 Skill 的完整流程**

**两个 Skill 的定义**：

`skills/appointment/skill.py`：

```python
class AppointmentSkill(BaseSkill):
    name = "appointment"
    priority = 40  # 排在 cancel(20) 后面

    state_schema = {
        "department": (str, ""),
        "doctor_name": (str, ""),
        "date": (str, ""),
        "time_slot": (str, ""),
        "step": (str, "idle"),     # idle → discover_dept → discover_doc → discover_slot → prepare → confirm
        "candidates": (list, []),
    }

    def match(self, query, context):
        if _looks_like_explicit_appointment_intent(query) and not _looks_like_explicit_cancel_intent(query):
            return True
        if _looks_like_appointment_discovery_query(query):
            return True
        return False

    def register_nodes(self, builder):
        builder.add_node("handle_appointment", self._handle)
        builder.add_conditional_edge("intent_router", "handle_appointment",
            condition=lambda s: s["intent"] == "appointment")
        builder.add_edge("handle_appointment", "route_after_action")

    def on_enter(self, state):
        if "appointment" not in state.get("skill_data", {}):
            state["skill_data"]["appointment"] = {"step": "idle"}
        return state

    def on_exit(self, state):
        state["skill_data"]["appointment"] = {"step": "idle"}
        return state
```

`skills/cancel_appointment/skill.py`：

```python
class CancelAppointmentSkill(BaseSkill):
    name = "cancel_appointment"
    priority = 20  # 排在 appointment(40) 前面，先匹配

    state_schema = {
        "cancel_target": (str, ""),
        "cancel_reason": (str, ""),
        "step": (str, "idle"),     # idle → find → prepare → confirm
        "candidates": (list, []),
    }

    def match(self, query, context):
        return _looks_like_explicit_cancel_intent(query)

    def register_nodes(self, builder):
        builder.add_node("handle_cancel", self._handle)
        builder.add_conditional_edge("intent_router", "handle_cancel",
            condition=lambda s: s["intent"] == "cancel_appointment")
        builder.add_edge("handle_cancel", "route_after_action")

    def on_enter(self, state):
        if "cancel_appointment" not in state.get("skill_data", {}):
            state["skill_data"]["cancel_appointment"] = {"step": "idle"}
        return state

    def on_exit(self, state):
        state["skill_data"]["cancel_appointment"] = {"step": "idle"}
        return state
```

**系统启动时**：

```
SkillRegistry.discover() 扫描 skills/ 目录
  ↓
找到 6 个 Skill（加上已有的 greeting、triage、medical_rag）：
  GreetingSkill          priority=10
  CancelAppointmentSkill priority=20
  AppointmentSkill       priority=40
  TriageSkill            priority=50
  MedicalRagSkill        priority=60
  ↓
按 priority 排序，存入 self._skills
  ↓
合并 state_schema，生成 skill_data 默认值：
  skill_data = {
      "appointment":        {"department": "", "doctor_name": "", "date": "", "time_slot": "", "step": "idle", "candidates": []},
      "cancel_appointment": {"cancel_target": "", "cancel_reason": "", "step": "idle", "candidates": []},
  }
  ↓
逐个调用 register_nodes()：
  CancelAppointmentSkill → 注册 "handle_cancel" 节点 + 条件边
  AppointmentSkill       → 注册 "handle_appointment" 节点 + 条件边
  TriageSkill            → 注册 "recommend_department" 节点 + 条件边
  MedicalRagSkill        → 注册 RAG 链路节点 + 条件边
  ↓
图构建完成，和现在一样的 LangGraph 状态机
```

**第 1 轮：用户说"帮我挂号"**

```
用户消息 "帮我挂号" 进入 analyze_turn
  ↓
检查 pending_action_type → 空，跳过
检查 pending_clarification → 空，跳过
进入 registry.classify("帮我挂号")
  ↓
  priority=10: GreetingSkill.match("帮我挂号") → False
  priority=20: CancelAppointmentSkill.match("帮我挂号") → False
  priority=40: AppointmentSkill.match("帮我挂号") → True ✓ 命中
  ↓
返回 "appointment"，primary_intent = "appointment"
  ↓
intent_router 确认意图，路由到 handle_appointment
  ↓
AppointmentSkill.on_enter() 初始化：
  skill_data["appointment"] = {"step": "idle"}
  ↓
handle_appointment 节点执行：
  step="idle" → 查询科室列表
  返回科室列表给用户
  更新 skill_data["appointment"]["step"] = "discover_dept"
  设置 pending_action_type = "appointment"
  ↓
走到 END，checkpoint 保存状态，Redis 写回
```

**第 2 轮：用户说"呼吸内科"**

```
用户消息 "呼吸内科" 进入 analyze_turn
  ↓
检查 pending_action_type = "appointment" ✓
_should_continue_pending_action → True（"呼吸内科"匹配科室候选列表）
  ↓
primary_intent = "appointment"，直接路由，不走 registry.classify
  ↓
handle_appointment 节点执行：
  读取 skill_data["appointment"]["step"] = "discover_dept"
  "呼吸内科" 匹配科室 → 查医生列表
  更新 skill_data["appointment"]["department"] = "呼吸内科"
  更新 skill_data["appointment"]["step"] = "discover_doc"
  ↓
走到 END
```

**第 3 轮：用户说"取消预约"**

```
用户消息 "取消预约" 进入 analyze_turn
  ↓
检查 pending_action_type = "appointment" ✓
_should_continue_pending_action：
  _is_explicit_confirmation("取消预约", "appointment") → False（不是确认预约词）
  _looks_like_appointment_update("取消预约") → False
  返回 False → pending 不续接
  ↓
进入 registry.classify("取消预约")
  ↓
  priority=10: GreetingSkill.match("取消预约") → False
  priority=20: CancelAppointmentSkill.match("取消预约") → True ✓ 命中
  ↓
返回 "cancel_appointment"，primary_intent = "cancel_appointment"
  ↓
intent_router 路由到 handle_cancel
  ↓
CancelAppointmentSkill.on_enter() 初始化：
  skill_data["cancel_appointment"] = {"step": "idle"}
  ↓
handle_cancel 节点执行：
  step="idle" → 查用户已有预约
  找到预约：呼吸内科 李医生 明天上午
  更新 skill_data["cancel_appointment"]["cancel_target"] = "APT001"
  更新 skill_data["cancel_appointment"]["step"] = "prepare"
  返回确认预览："确认取消呼吸内科李医生明天上午的预约？"
  设置 pending_action_type = "cancel_appointment"
  ↓
走到 END
```

此时 skill_data 里有两组数据，互不影响：

```python
skill_data = {
    "appointment":        {"department": "呼吸内科", "step": "discover_doc", ...},
    "cancel_appointment": {"cancel_target": "APT001", "step": "prepare", ...},
}
```

**第 4 轮：用户说"确认取消"**

```
用户消息 "确认取消" 进入 analyze_turn
  ↓
检查 pending_action_type = "cancel_appointment" ✓
_should_continue_pending_action：
  _is_explicit_confirmation("确认取消", "cancel_appointment") → True ✓
  ↓
primary_intent = "cancel_appointment"，续接
  ↓
handle_cancel 节点执行：
  读取 skill_data["cancel_appointment"]["step"] = "prepare"
  读取 skill_data["cancel_appointment"]["cancel_target"] = "APT001"
  执行取消 → DB 删除预约
  ↓
CancelAppointmentSkill.on_exit()：
  skill_data["cancel_appointment"] = {"step": "idle"}
  清空 pending_action_type
  ↓
走到 END
```

**第 5 轮：用户说"继续预约"**

```
用户消息 "继续预约" 进入 analyze_turn
  ↓
检查 pending_action_type → 空（上一轮清空了）
检查 pending_clarification → 空
进入 registry.classify("继续预约")
  ↓
  AppointmentSkill.match("继续预约") → True ✓
  ↓
handle_appointment 节点执行：
  读取 skill_data["appointment"]["step"] = "discover_doc"
  读取 skill_data["appointment"]["department"] = "呼吸内科"
  → 从选医生这一步继续！之前的数据还在
  ↓
返回呼吸内科的医生列表给用户
```

**核心细节**：

1. **优先级决定谁先匹配** — cancel(20) 在 appointment(40) 前面，所以"取消预约"不会被预约规则吞掉
2. **skill_data 是命名空间隔离的** — 预约和取消预约各自的数据在 `skill_data["appointment"]` 和 `skill_data["cancel_appointment"]` 里，互不影响
3. **pending 续接和之前一样** — `_should_continue_pending_action` 的逻辑不变，只是状态的读写从 `state["appointment_context"]` 变成了 `state["skill_data"]["appointment"]`
4. **on_exit 清理各自的命名空间** — 预约完成只清 `skill_data["appointment"]`，不影响 `skill_data["cancel_appointment"]`

**对现有代码的影响**：

老代码不改也能跑——`skill_data` 是新加的字段不影响现有逻辑，注册机制建好后新功能走新机制，老功能渐进迁移。不是一把全改，是先建机制，新功能用新机制，老功能等有空再迁。

**为什么现在没做**：当前只有 5 个意图，硬编码的维护成本还能接受。但每加一个意图的边际成本在上升，7-8 个以上就必须做了。重构的前提是先有稳定的测试覆盖，不然改了也不知道对不对。

---

## 七、软素质与成长

### Q15：项目中遇到过最大的困难是什么？怎么克服的？

> 三面必考，要讲一个真实的困难，不是"技术难"而是"选择难"。

**参考回答**：

最大的困难是意图冲突的边界定义。不是匹配难，是定义难——"预约前注意什么"有预约关键词却是医学咨询，"取消对药物依赖的方法"有"取消"却不是取消预约。每解决一个边界 case，可能引入新的边界 case。

克服的方式是转变思路：从"规则覆盖一切"转向"规则保核心、LLM 兜边界"。规则只覆盖高频确定性场景，边界 case 不硬写规则，交给 LLM 分类。这个转变的关键认知是——**规则误路由的代价远大于 LLM 多处理一个 query 的代价**，所以规则宁可窄一点，宁可漏一点，也不能误路由。这个认知不是一开始就有的，是在反复调规则、发现调了 A 又坏了 B 之后才想通的。

---

### Q16：你在团队中是怎么协作的？有没有和别人的方案冲突过？怎么解决的？

**参考回答**：

前后端协作是通过 API 契约先行——FastAPI 自动生成 OpenAPI 文档，前后端并行开发。SSE 的事件类型（session/status/message/final/app-error）是提前约定的，避免后端改了前端不知道。

方案冲突有过一次：SSE 流式接口，前端一开始用 `EventSource`，我后来要求改成 `fetch + ReadableStream`。原因是 Bearer Token 鉴权 + 医疗问题不能暴露到 URL。前端觉得改动大，我的做法是先解释为什么（安全原因，不是技术偏好），再给出具体的改动方案（前端只改 hook 层，业务层不用动），最后一起联调确认。解决冲突的关键是**说清为什么 + 降低对方的改动成本**。

---

### Q17：你觉得你相比其他候选人，核心差异在哪？

> 不要说"我更努力"这种虚的，要说具体的能力差异。

**参考回答**：

两个差异：

**第一，我有端到端的系统思维。** 很多人做 RAG 只做了检索+生成，做 Agent 只做了工具调用。我的项目从意图路由、检索工程、状态管理、安全兜底到前后端交互，是完整链路。每个模块不是孤立的——意图路由错了后面全错，检索不到 grounding check 没用，状态丢了预约续接不了。理解模块间的依赖关系比做好单个模块更重要。

**第二，我在技术选型上有明确的判断标准。** 用 pgvector 不用 Milvus、用 SSE 不用 WebSocket、用规则+LLM 不用纯 LLM、Redis 挂了 fail fast 不降级——每个决策都有"为什么这么选"和"不这么选会怎样"。不是哪个方案新用哪个，是哪个方案适合当前场景用哪个。

---

### Q18：你未来想往什么方向发展？

**参考回答**：

AI 应用工程方向——把 LLM 的能力转化为可靠的产品。不是做模型训练，是做模型落地。这个方向的核心挑战不是"模型能不能做到"，是"做到了怎么保证可靠、可控、可观测"。我的项目就是在做这件事，未来想在更大规模、更复杂场景下继续做。

具体关注两个子方向：一是 RAG 的工程化（检索质量评估、知识库运维、多模态检索），二是 Agent 的安全可控（关键路径代码门控、执行审计、权限边界）。这两个方向在医疗、金融等高合规场景有大量需求。

---

### Q19：做这个项目你学到了什么？

> 三面高频题。不要列举技术栈，要讲认知层面的转变。

**参考回答**：

三个层次的转变：

**第一层：技术能力——从"会用"到"会选"**

做之前我对 RAG 的理解就是"检索+生成"，做完之后理解了检索是一个完整的工程链——query rewrite、混合检索、RRF 融合、rerank、充足性检查、grounding check，每一步都有兜底。做之前我觉得 LLM 什么都能做，做完之后理解了关键路径不能交给 LLM——预约确认是代码校验不是模型判断，意图路由有规则层兜底不是全靠 LLM。从"会调 API"到"知道什么该交给模型什么不该交"。

**第二层：工程思维——从"能不能跑"到"出了问题怎么办"**

做之前我只关注功能能不能实现，做完之后理解了每个设计决策都要问"如果这个环节失败了怎么办"——LLM 超时走 fallback、检索不到走 no_evidence 兜底、Redis 挂了生产环境 fail fast 不降级。这不是事后补救，是设计时就要想清楚的。另外一个教训是评估——没有做消融实验，所有优化叠加在一起不知道各自贡献多少，有数据才能做权衡，凭感觉不行。

**第三层：认知转变——从"追求最优"到"追求合适"**

做之前我会想"哪个方案最强"，做完之后理解了没有最优方案只有最合适的方案——用 pgvector 不用 Milvus 是因为当前数据量够用、运维简单；用 SSE 不用 WebSocket 是因为场景匹配；规则+LLM 不是中庸是各有分工。技术选型的本质不是选最强的，是选约束条件下最合适的。

如果用一句话总结：从"把功能做出来"变成了"把功能做可靠"。做出来靠技术能力，做可靠靠工程判断。

---

## 八、反问环节

### Q20：你有什么想问我的？

三面反问建议更"高层"：

1. **问业务方向**："字节医疗/健康业务接下来的技术重点是什么？是做深垂类还是做通用平台？"
2. **问团队挑战**："团队目前在规模化落地时遇到的最大技术瓶颈是什么？"
3. **问成长**："对于校招生，团队在'从项目到产品'这个能力上是怎么培养的？"

三面反问的核心是展示你关注的是"业务和产品"，不只是"技术实现"。

---

> **三面核心提醒**：二面考"你做了什么"，三面考"你理解什么"。每个问题都要能从具体项目跳到一般性思考，再回到项目验证。回答结构：**观点 → 项目中的实例 → 一般性规律/方法论**。祝三面顺利！
