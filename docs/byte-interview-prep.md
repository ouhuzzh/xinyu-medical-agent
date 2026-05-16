# 医疗智能助手 — 字节二面面试准备

> 基于项目源码深度分析，模拟字节二面风格：**连环追问 + 场景题 + 架构设计 + 生产问题 + 软素质**

---

## 目录

- [一、字节二面面试风格与应对策略](#一字节二面面试风格与应对策略)
- [阶段一：项目介绍](#阶段一项目介绍)
- [二、项目整体架构与核心链路](#二项目整体架构与核心链路)
- [三、工作内容 1：意图路由框架](#三工作内容-1意图路由框架)
- [四、工作内容 2：混合记忆机制](#四工作内容-2混合记忆机制)
- [五、工作内容 3：医疗知识库检索](#五工作内容-3医疗知识库检索)
- [六、工作内容 4：预约 Skill 模块](#六工作内容-4预约-skill-模块)
- [七、工作内容 5：医疗安全与可信度](#七工作内容-5医疗安全与可信度)
- [八、跨模块综合场景题](#八跨模块综合场景题)
- [九、生产故障与应急处理](#九生产故障与应急处理)
- [十、设计权衡与复盘反思](#十设计权衡与复盘反思)
- [十一、软素质与团队协作](#十一软素质与团队协作)
- [附录：核心代码速查表](#附录核心代码速查表)
- [附录：高频数字速查](#附录高频数字速查)
- [附录：手写高频题](#附录手写高频题)
- [高危追问速答](#高危追问速答)

---

## 一、字节二面面试风格与应对策略

### 字节二面核心特征

| 特征 | 具体表现 |
|------|---------|
| **连环追问** | 一个问题剥 3-5 层，从 What → Why → What if → Trade-off |
| **场景题驱动** | "如果线上出现了 XXX 问题你怎么排查？" |
| **深度优先** | 不追广度追深度，一个点挖到极致 |
| **架构画图** | 可能要求画出模块间数据流 |
| **反问环节** | "你还有什么想问的？"——考察技术视野 |

### 应对框架：STAR + 深度追问法

```
回答结构：结论先行 → 实现细节 → 为什么这么选 → 如果不这样会怎样 → 还能怎么优化
```

---

## 阶段一：项目介绍（3-5分钟）

### 面试官

> 先简单介绍一下你的项目吧。

### 参考回答

> 这是一个医疗智能助手，核心能力是医学问答、科室推荐和预约挂号。用户多轮对话，系统自动识别意图走不同处理链路。
>
> 整体是前后端分离，前端 React，后端 FastAPI，核心是一个 LangGraph 驱动的对话状态机。用户消息进来先过意图路由，路由到三条链路：医学问答走 RAG 检索+生成，科室推荐走 LLM 分诊，预约挂号走结构化的预约技能。三条链路共享会话记忆和上下文管理。
>
> 整个系统最核心的设计是意图路由——因为三条链路的处理方式完全不同，意图判错了后面全白做。我的方案是两阶段：规则快速通道 + LLM 兜底。规则层覆盖 80% 的高频确定性意图，微秒级响应；剩余模糊的交给 LLM 结构化分类。中间还处理了三类中断续接——用户上次没做完的事优先恢复，复合请求自动拆分顺序执行。
>
> 在记忆方面，我用 Redis 做短期上下文、LLM 做长程摘要、PostgreSQL 做状态持久化，三层混合记忆，长对话 prompt token 降低了 27.4%。
>
> 检索方面，基于 pgvector 实现混合检索，结合 query rewrite、RRF 融合和 rerank，在 NHC/WHO 指南文档集上 Precision@5 从 0.68 提升到 0.83。
>
> 预约流程抽象成独立 Skill，采用"预览→确认→执行"的半受控模式，防止误触发。还有医疗安全兜底——高风险症状给就医提醒，知识库没证据时给带免责声明的回答。
>
> 最近一轮工程化我又补了几层边界：API 统一 Bearer 鉴权，`thread_id` 绑定 owner_user_id 做会话隔离，普通用户只能访问自己的聊天接口，文档上传和官方同步只开放给管理员。聊天流也从 `GET + EventSource` 改成了带认证头的 `POST + fetch/ReadableStream`，避免把医疗问题暴露到 URL 和代理日志里。

---

## 二、项目整体架构与核心链路

### Q1：请画出你们系统的整体架构图，并讲清一个完整的用户问诊链路

**参考回答：**

系统分为前后端两层，后端核心是一个 LangGraph 驱动的对话状态机：

```
用户端 (React/Vite :5173)
    ↓
FastAPI (:8000) — Bearer 鉴权 + 会话 owner 校验
    ↓
ChatInterface — SSE 流式输出
    ↓
LangGraph 状态机
    ↓
┌─────────────┼─────────────┬─────────────┐
▼             ▼             ▼             ▼
意图路由    医学问答 RAG   科室推荐      预约 Skill
(Rule+LLM)  (检索+生成)   (LLM 分诊)    (半受控流程)
    ↓         ↓             ↓             ↓
Redis       pgvector      PostgreSQL    PostgreSQL
(Pending)   (向量+全文)   (文档/预约)   (预约记录)
```

**完整用户问诊链路（医学问答场景）：**

1. 用户消息到达 `POST /api/chat/stream`，经过 Bearer Token 鉴权和 `thread_id` owner 校验
2. `ChatInterface.chat()` 构建图输入：从 Redis 读 `session_state`，从 session memory 读历史消息，组装成 `HumanMessage`
3. LangGraph 从 `START` 进入 `summarize_history` → `analyze_turn`
4. `analyze_turn` 判断是否有 pending 状态（预约待确认/澄清待回答），有则续接，无则走规则分类
5. 规则命中 `medical_rag` → `intent_router` 确认意图 → 路由到 `rewrite_query`
6. `rewrite_query` 做 query 改写和意图澄清判断
7. `plan_retrieval_queries` 规划多路检索查询
8. 子图 `agent` 执行检索（pgvector 向量检索 + tsvector 全文检索，RRF 融合）
9. `grounded_answer_generation` 基于检索结果生成回答
10. `answer_grounding_check` 检查回答是否基于检索证据
11. 走到 `END`，ChatInterface 把最终回答写回 Redis，发 SSE `final` 事件给前端

**追问：如果用户在预约流程中途问了个医学问题，链路怎么变？**

答：预约流程中 `pending_action_type = "appointment"`，用户说"哮喘能跑步吗"，`analyze_turn` 调用 `_should_continue_pending_action` 判断——用户消息不匹配预约相关关键词，返回 False，走正常规则分类，命中 `medical_rag`，走 RAG 链路回答。但 `pending_action_type` 不清除，保留在 Redis 中。用户回答完医学问题后，下一轮说"继续预约"或医生名，`_should_continue_pending_action` 返回 True，直接续接回 `handle_appointment_skill`。

---

### Q2：模块间怎么通信的？为什么这么设计？

**参考回答：**

- **同步调用**：`ChatInterface` 内部调用图节点，都是本地函数调用，无网络开销
- **数据持久化**：Redis（会话状态、短期上下文）、PostgreSQL（文档、预约、检索日志）、LangGraph Checkpoint（图状态持久化到 `runtime/langgraph_checkpoints.pkl`）
- **SSE 流式**：后端通过 `text/event-stream` 向前端推送 token 和状态，前端用 `fetch + ReadableStream` 消费

**为什么用 SSE 而不是 WebSocket？**
- 对话场景本质上是"客户端发一次请求，服务端持续单向回流 token"，SSE 语义更贴合
- SSE 比 WebSocket 更简单，代理层配置友好，落地成本低
- 后来前端从 `EventSource` 改成 `fetch + ReadableStream`，主要是为了 Bearer Token 鉴权和避免医疗问题暴露到 URL

---

### Q3：PostgreSQL 在项目中承担了哪些角色？

**参考回答：**

PostgreSQL 是三角色合一：

1. **业务数据库**：存储文档、预约记录、检索日志、路由日志
2. **向量数据库**：pgvector 插件提供向量存储和相似度检索
3. **全文检索引擎**：tsvector + tsquery 提供关键词检索

向量检索和全文检索都在同一个数据库里，所以混合检索不需要跨系统查询，RRF 融合可以在应用层完成。如果换成 Milvus + Elasticsearch，虽然性能更好，但需要维护三个系统，数据一致性也麻烦。当前文档量在万级别，pgvector 够用。

---

## 三、工作内容 1：意图路由框架

### Q4：你说规则+LLM两层，为什么不全用LLM？

**参考回答：**

三个原因。

**第一延迟**——规则是纯函数微秒级，LLM 一个请求几百毫秒。意图分类是每个用户消息必经路径，这个延迟用户能感知。

**第二成本**——80% 的查询是高频模式（打招呼、明确预约、明确取消），规则零成本就解决了，没必要每次都调模型。

**第三确定性**——"取消预约"这种关键意图，规则 100% 命中。LLM 可能误分类，医疗场景不能赌概率。一个误路由的代价是整个后续流程全错，这不是准确率 95% 能接受的。

**追问：那为什么不全用规则？**

答：规则覆盖不了语义模糊的输入。比如"我最近老觉得不太舒服，想了解一下"，没有关键词命中任何规则，但 LLM 能结合上下文判断这是医学咨询。规则擅长确定性场景，LLM 擅长模糊场景，两者互补。

---

### Q5：规则匹配具体怎么做的？优先级怎么定的？

**参考回答：**

优先级是 if-elif 链的顺序：`greeting` → `cancel_appointment` → `appointment_discovery` → `explicit_appointment` → `triage` → `medical_rag` → `general_chat`，最后规则未命中走 LLM。

这个顺序的核心原则是**窄意图优先、防冲突**：

- 取消必须在预约前面，因为"取消预约"同时含两个关键词，cancel 先命中才对
- 分诊必须在医疗前面，因为"挂什么科"同时有挂号词和问句特征，不先截走就会被医疗意图吞掉
- 每条规则不是简单关键词，是多层组合。医疗问题必须医学术语和问句特征同时出现；预约有排除条件——"预约前注意什么"有预约关键词但被排除掉

**追问：规则越来越多会不会维护不了？**

答：规则只覆盖高频确定性场景，模糊的交给 LLM。规则的数量上限是意图类别 × 冲突 case，不会无限膨胀。实际上规则函数大概 7-8 个，对应 5 个意图类别加排除条件。如果规则超过 50 条，说明意图类别太多或者规则粒度太细，该考虑分类模型了。

---

### Q6："取消预约"同时命中取消和预约，具体怎么处理的？

**参考回答：**

两个层面配合。

**第一层是优先级**——取消排在预约前面，if-elif 先命中取消。

**第二层是排除条件**——在 `_looks_like_explicit_cancel_intent` 里，如果检测到"取消"但后面跟的是医学知识问句（比如"取消对药物依赖的方法"），就排除取消意图，让它走医学。反过来在预约规则里也有排除——如果检测到"预约前"模式，不算预约。

单靠优先级解决不了所有冲突。比如"取消对药物依赖的方法"里"取消"不是取消预约的意思，优先级会错误命中。所以必须每条规则加排除条件，定义自己的边界。优先级解决规则之间的顺序，排除条件解决单条规则的边界。

---

### Q7：复合请求拆分怎么做？拆完怎么执行？

**参考回答：**

按"另外、然后、顺便、同时"这些分隔词把用户输入拆成两段，每段独立跑规则匹配，得出 primary 和 secondary intent。不是所有意图组合都有意义——支持的有效组合是：取消+医学、预约+医学、分诊+预约、分诊+医学，其他组合会被过滤。

执行顺序是交互型优先。挂号需要用户多轮交互（选科室、选医生、确认），先锁定用户注意力；医学咨询是查询型，可以 defer。挂号完成后，后路由检查到有 `secondary_intent` + `deferred_user_question`，自动触发 `prepare_secondary_turn`，把 secondary 提升为 primary 执行。

**追问：如果用户一句话里三个意图呢？**

答：当前只支持两段拆分。三个意图的场景实际很少，而且拆多了执行顺序的语义很难定义——用户不会期望系统自动决定三个任务的执行顺序。如果真遇到，两段拆分后第二段如果还包含复合结构，会走 LLM 分类，LLM 可以把它识别成一个更复杂的意图，交给澄清流程。

---

### Q8：中断续接具体怎么判断"用户是在回应上次的问题"还是"开了新话题"？

**参考回答：**

比如用户正在选号，说"内科"——这是选科室还是新开了一个分诊？

判断在 `_should_continue_pending_action` 里：

1. 先看是否有 `pending_action_type` 或 `pending_candidates`，没有直接不续接
2. 再看是否显式确认/放弃，是则续接
3. 再看 pending_candidates 里是否有匹配项，有则续接
4. 最后看消息是否跟 pending 动作相关——`pending_action_type == "appointment"` 时检查消息是否含预约相关词（"改""换""医生""时间"）

所以"内科"匹配了 `pending_candidates` 里的科室列表 → 续接。"头疼怎么办"不匹配任何 pending 相关词 → 新话题，走正常意图路由。

**但有个细节**——新话题处理完后，pending 状态不能丢。用户问完头疼回来还要能继续选号。所以 pending 状态只在用户显式完成或取消时清除，新话题不会覆盖它。

---

### Q9：LLM 分类用的什么 prompt？怎么保证输出稳定？

**参考回答：**

用结构化输出，定义了 `IntentAnalysis` 的 schema：`intent`、`is_clear`、`clarification_needed` 三个字段。分类链路用低温度配置，当前项目默认是 0，目标就是尽量提高同样输入下的一致性。

prompt 里给了意图类别定义和当前对话上下文。关键是不给 few-shot 示例——因为 few-shot 会引导模型偏向示例的模式，而规则层已经覆盖了标准 case，LLM 需要处理的是规则没覆盖的模糊 case，给示例反而限制它的泛化能力。

**追问：如果 LLM 分类错了怎么办？**

答：有三层兜底。

第一，LLM 输出的 intent 值必须是预定义的 5 种之一，否则 fallback 到 `medical_rag`。

第二，澄清熔断——如果 LLM 说需要澄清，最多追问 1 次，超过直接 fallback。

第三，异常兜底——LLM 调用失败直接走 `medical_rag`，不会卡死。医疗 RAG 本身有检索兜底，即使意图判错了，检索不到相关文档也会给 fallback 回答，不会给错误信息。

---

### Q10：`interrupt_before` 是什么？澄清流程怎么工作的？

**参考回答：**

`interrupt_before=["request_clarification"]` 意味着图执行到 `request_clarification` 节点之前会暂停，等用户下一轮输入后再继续。

这是澄清流程的核心机制。比如 `recommend_department` 发现信息不够，设置 `clarification_target = "recommend_department"`，发追问消息，然后图走到 `request_clarification` 前暂停。用户回答后，ChatInterface 检测到 `current_state.next` 非空，把用户消息 `update_state` 进去，`stream_input = None`，图从断点继续。

`request_clarification` 节点本身返回空，但它的条件边 `route_after_clarification` 根据 `clarification_target` 决定下一步——`"recommend_department"` 就回到科室推荐，`"handle_appointment_skill"` 就回到预约节点。`clarification_target` 就是个"回执地址"。

**如果不用 interrupt**，追问后用户回答会走一遍完整的 `summarize_history → analyze_turn → intent_router`，可能被路由到错误的地方。

---

## 四、工作内容 2：混合记忆机制

### Q11：你说 token 降了 27.4%，怎么算的？

**参考回答：**

4 个测试样本，每个是一段多轮对话。对比两种方式：一是把完整历史塞进 prompt，二是用摘要+最近 N 轮。统计两种方式下 prompt 的 token 数，计算 `(full - compressed) / full`。平均降低 27.4%，p95 降低率也大于 0。

摘要不是每轮都做，是消息数 >= 4 时才触发。取最近 6 条相关消息生成摘要，替换掉更早的历史。这样摘要的频率低，LLM 调用成本低，但上下文信息不丢。

---

### Q12：为什么用三层记忆？Redis、LLM 摘要、PG 分别存什么？

**参考回答：**

Redis 存短期上下文——最近 N 轮消息、摘要、topic focus、pending 状态。读写频繁、要求低延迟，Redis 的内存操作微秒级。

LLM 摘要存长程语义——当对话超过窗口大小时，把早期历史压缩成一段摘要。摘要本身也存在 Redis 里，但生成过程靠 LLM。

PostgreSQL 存持久化数据——文档、预约记录、检索日志。这些数据量比会话上下文大得多，需要持久化和复杂查询，Redis 不适合。

三层的核心逻辑是：短期高频放内存，长期语义靠压缩，持久业务放数据库。不是每层独立工作，是 Redis 是会话的入口，摘要和 PG 是 Redis 的后备。

**追问：Redis 挂了怎么办？**

答：现在分环境处理。开发环境下允许降级到进程内字典，方便本地调试，不会因为 Redis 没起就完全跑不起来；但生产思路相反，如果 `REDIS_ENABLED=true` 且不是 development，Redis 不可用会直接启动失败，不再静默降级。

这样做是因为多实例场景下，进程内 fallback 会导致会话不共享、pending 状态错乱。开发时优先可用性，生产时优先一致性和可观测性。

---

### Q13：topic focus 是什么？怎么用的？

**参考回答：**

topic focus 是从对话历史中提取的当前话题焦点，比如"高血压""流感预防"。用途是辅助追问识别——用户说"那应该注意什么"，没有主题词，但 topic focus 是"高血压"，系统就知道是在问高血压的注意事项，而不是随便问的。

生成方式是在每次对话轮次后，从 recent context 和 summary 中提取。它和摘要的区别是：摘要是历史回顾，topic focus 是当前关注点，更短更聚焦。

---

### Q14：图走到 END 后状态会丢吗？Redis 会不会把旧状态灌回去？

**参考回答：**

不会。

关键在 `_resolved_session_state`——它优先取 `latest_values`（图走完后的 checkpoint 状态），而不是用旧值。当预约完成走到 END 时，节点已经调用了 `_clear_pending_action_state()`，把 `pending_action_type`、`pending_confirmation_id`、`pending_candidates` 全清空了。所以：

1. 图走完 → checkpoint 里的 `pending_action_type` 已经是 `""`
2. `_resolved_session_state` 优先取 `latest_values`
3. 写回 Redis 的就是清空后的状态

反过来说，`pending_action_type` 这个字段**只有在预约还没确认时才不为空**，它存在的意义恰恰就是为了跨轮保持。预约一旦确认或放弃，节点清空它，Redis 也就跟着清空了。

**追问：清空状态是节点做的，不是 END 自动清空？**

答：对。走到 END 不会自动清空任何状态。LangGraph 的 END 只是表示"这轮图执行完毕"，checkpoint 会把当前状态原样保存。清空是节点自己显式做的——比如预约确认成功时返回里调用了 `_clear_pending_action_state()`。如果忘记调这个函数，走到 END 之后 `pending_action_type` 还会是 `"appointment"`，下一轮从 Redis 读回来就会误判。

---

### Q15：消息从节点到前端具体是怎么传的？

**参考回答：**

三步。

**第一**，节点返回 `{"messages": [AIMessage(content="已预约成功...")]}`，LangGraph 把 AIMessage 追加到 State 的 `messages` 字段（`MessagesState` 内置 `add_messages` reducer，追加而非覆盖）。

**第二**，ChatInterface 用 `stream_mode="messages"` 遍历图的输出，捕获 `AIMessageChunk` 拼到 `response_messages` 列表。`handle_appointment_skill` 不在 `SILENT_NODES` 里，所以它的内容会被捕获。

**第三**，SSE 层从 `response_messages` 提取 assistant 文本，发 `event: message` 给前端。

```
节点返回 AIMessage
  ↓
LangGraph 流式吐出 AIMessageChunk
  ↓
ChatInterface 拼到 response_messages
  ↓
SSE 层提取文本发事件
  ↓
前端渲染
```

**追问：`messages` 也是 State 的字段吗？**

答：是的。`State` 继承自 LangGraph 的 `MessagesState`，内置了一个 `messages` 字段，类型是 `Annotated[list, add_messages]`。每次节点返回新的 `messages`，LangGraph 会自动追加到已有列表里。

---

## 五、工作内容 3：医疗知识库检索

### Q16：混合检索具体怎么做的？RRF 融合是什么？

**参考回答：**

混合检索是向量检索 + 关键词检索并行跑。向量检索用 pgvector 的余弦相似度，关键词检索用 PostgreSQL 的 tsvector 全文索引。两路结果用 RRF（Reciprocal Rank Fusion）融合。

RRF 的公式是 `score = Σ 1/(k + rank_i)`，k 一般取 60。比如一个文档在向量检索排第 1，在关键词检索排第 5，它的 RRF 分 = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318。

RRF 的好处是不需要两路分数归一化——向量检索的余弦相似度和全文检索的 ts_rank 分数量级不同，直接加权融合会有偏向，RRF 用排名代替分数，天然消除了量纲问题。

---

### Q17：Precision@5 从 0.68 到 0.83，具体做了什么优化？

**参考回答：**

三个优化叠加的效果。

**第一，query rewrite**——把用户的口语化问题改写成更适合检索的查询，比如"那应该注意什么"改写成"高血压平时应该注意什么"。

**第二，混合检索 + RRF**——原来只有向量检索，加了全文检索后对关键词匹配的 case 提升明显。

**第三，rerank**——用 bge-reranker-v2-m3 对 top-12 候选重排序，取 top-5。

测试集是 10 个医疗 RAG 样本，覆盖患者教育、公卫、临床指南三类文档。每个样本标注了期望的 source_type 和关键词，precision@5 算的是 top-5 结果中包含期望关键词的比例。

**追问：三个优化各自的贡献你知道吗？**

答：没有做严格的消融实验，但从设计上看：query rewrite 对追问类 case 贡献最大，因为追问本身没有检索关键词；混合检索对关键词匹配类 case 贡献最大，因为纯向量检索对专有名词不敏感；rerank 对所有 case 都有提升，因为它是在候选集里做精排。如果要严格量化，应该做消融实验——去掉一个优化跑一遍测试集，看精度变化。

---

### Q18：父子块分层索引是什么？为什么这样做？

**参考回答：**

文档切分时生成两层：父块是较大的段落（2k-4k 字符，按 Markdown 标题层级切），子块是较小的片段（500 字符，100 overlap）。

检索用子块（粒度细，精度高），返回用父块（上下文完整，给 LLM 的信息更充分）。

如果只用子块检索，返回的片段可能缺少上下文——比如子块只讲了"低盐饮食"，但没说这是高血压的注意事项。用父块返回，上下文完整。如果只用父块检索，粒度太粗，可能一个父块里包含多个主题，检索精度下降。父子块分层是精度和上下文的平衡。

**追问：overlap 是什么？为什么需要？**

答：切分时相邻块之间有 100 字符的重叠区域。防止关键信息正好被切断在两个块的边界上——比如一句话"高血压患者应避免高盐饮食，每日钠摄入不超过2g"，如果切分点落在"高盐饮食"后面，前一个块有"避免高盐饮食"但没剂量，后一个块有剂量但没上下文。overlap 保证了边界信息两边都有。

---

### Q19：为什么用 pgvector 不用 Milvus/Qdrant？

**参考回答：**

三个原因。

**第一，数据量**——当前知识库文档量在万级别，pgvector 完全够用，专业向量库的优势在亿级数据上才体现。

**第二，运维复杂度**——pgvector 和业务数据同库，不需要额外部署和维护一个向量服务。

**第三，事务一致性**——文档的元数据和向量在同一个事务里，插入和删除不需要跨系统同步。

如果未来数据量到百万级或者需要分布式检索，会考虑迁移到专业向量库。但当前阶段，pgvector 的简单性比性能更重要。

---

### Q20：rerank 和向量检索的区别是什么？为什么还需要 rerank？

**参考回答：**

向量检索是双塔模型——query 和 document 分别编码成向量，算余弦相似度。优点是快，缺点是 query 和 document 没有交互，细粒度语义匹配弱。

rerank 是交叉编码器——query 和 document 拼接后一起过模型，有深层交互，精度高但慢。

所以用向量检索做粗排（从全量里取 top-12），用 rerank 做精排（从 12 个里取 top-5）。这是经典的漏斗结构，粗排保证召回，精排保证精度。

---

### Q21：知识库同步怎么做增量更新？怎么避免重复索引？

**参考回答：**

用 content_hash 做增量。每次同步时，先对文档内容算 hash，再和数据库里存的 content_hash 对比：hash 一致且已有 chunks → 跳过（unchanged）；hash 不一致 → 删旧 chunks，重新切分索引（updated）；不存在 → 新建（added）。

对比全量重建，增量同步避免了对未变化文档的重复切分和 embedding 计算，同步时间从全量的 O(全部文档) 降到 O(变更文档)。

**追问：源站下架了一篇文档怎么办？**

答：用 soft delete。同步时开启 `soft_delete_missing=True`，对本次同步中没出现的源文档，把 `is_active` 标为 False 并记录 `deleted_at`。但注意——soft delete 时子块是被**物理删除**的（`_clear_chunks` 做 DELETE），不是软删除。`is_active = False` 只标记在 `documents` 表上，是给同步逻辑看的。检索时查的是 `child_chunks` 表，行都删了，自然搜不到。

---

## 六、工作内容 4：预约 Skill 模块

### Q22："预览→确认→执行"的半受控模式是什么意思？

**参考回答：**

和全自主的 agent 不同，预约流程的每一步都有明确的边界。用户说"帮我挂号"，系统不是直接执行，而是先展示预览——查到哪些科室、哪些医生有号，让用户选择。用户选完后，系统展示确认信息——"您确认预约XX医生XX时间吗？"用户明确说"确认"才执行。任何一步用户都可以退出或改条件。

"半受控"是指流程是预定义的（发现→选择→确认→执行），但每一步的输入是用户自由提供的。比如用户可以说"呼吸内科明天下午"，系统一次提取多个参数跳过多步，也可以说"我想挂号"让系统引导。

---

### Q23：怎么防止 LLM 判断错误就直接预约了？

**参考回答：**

不会。确认环节走的是代码硬校验，不是 LLM 判断。

确认词是硬编码的元组：`"确认预约"`、`"确认挂号"`、`"现在预约"` 等。执行入口在 `handle_appointment_skill` 里，先检查 `pending_action_type == "appointment"`，再走 `_is_explicit_confirmation()`——这个函数是**纯字符串包含匹配**，不经过 LLM。

整个流程是三道门：

1. **没有 pending 状态 → 不会执行**：必须先经过 `prepare_appointment` 设置 `pending_action_type`，否则代码根本不进这个分支
2. **用户说"好的""可以" → 不匹配确认词 → 不执行**：只会回一句"请回复确认预约"
3. **即使 LLM 幻觉想直接预约 → 代码拦住**：`prepare_*` 只返回预览不写 DB，`confirm_*` 是唯一写 DB 的路径，且必须经过关键词校验

**一句话总结：LLM 只负责理解和路由，写库的钥匙在代码手里，不在模型手里。**

---

### Q24：怎么防止重复预约？

**参考回答：**

两层保护。

第一层是幂等性——执行预约前先查是否已存在同一天同一科室同一时段的预约，存在就拒绝。

第二层是确认门控——用户必须显式说"确认预约"才算确认。`_is_explicit_confirmation()` 是纯关键词匹配，不经过模型。`prepare_*` 方法只返回预览不写 DB，`confirm_*` 是唯一写 DB 的路径且必须过关键词校验。

---

### Q25：用户在预约中途问了个医学问题怎么办？

**参考回答：**

pending 状态保留。用户说"帮我挂呼吸内科"，系统进入预约流程，用户接着说"哮喘能不能运动"，系统识别这是新意图（不是预约相关），走 RAG 回答。但预约的 pending 状态不清除——`pending_action_type`、`pending_candidates`、`appointment_context` 都保留。

判断"续接还是新话题"靠的是 `_should_continue_pending_action`，它不是排除条件而是**准入条件**——用户说的话跟预约相关才续接，无关就走正常路由。比如 pending 是预约确认，用户说"换个时间"→ 续接；用户说"高血压能跑步吗"→ 不续接，走正常分类，pending 不清。

---

### Q26：预约过程中查询可用号源返回了，这个走到 END 了吗？

**参考回答：**

对，查询可用号源**走完了 END**。

`handle_appointment_skill` 之后走 `route_after_action`：如果 `pending_clarification` 为空、`pending_action_type` 也为空，直接走到 `__end__`。

**但走到 END 不等于状态丢失。** 走到 END 只是这轮图执行结束了，LangGraph 的 checkpoint 会保存最后一帧的完整状态。下一轮用户发消息时，ChatInterface 从 Redis 读取 `session_state`，再通过 `_graph_state_from_session` 写入图的初始状态。

所以跨轮状态保持靠的是 **Redis + checkpoint**，不是图一直不结束。整个流程是：每一轮都走 END，每一轮结束后 Redis 把最新状态存下来，下一轮再灌回去。

---

## 七、工作内容 5：医疗安全与可信度

### Q27：高风险症状怎么识别的？怎么兜底？

**参考回答：**

关键词匹配——胸痛、呼吸困难、剧烈头痛、意识模糊等高风险症状词。一旦命中，回答里必须包含就医提醒，比如"如果出现XX症状，请尽快就医"。这个不是 LLM 自由生成的，是规则层强制插入的。

知识库没证据时（比如"月球低重力综合征怎么治疗"），系统给 fallback 回答，带免责声明——"我暂时没有找到相关医学资料，以下是一般性建议，仅供参考，请咨询专业医生"。不会编造信息，也不会假装知道。

---

### Q28：grounding 怎么做的？怎么防止幻觉？

**参考回答：**

两层。

第一层是检索阶段——回答必须基于检索到的文档片段，prompt 里要求 LLM 只用检索结果回答，不要用自己的知识。

第二层是 `answer_grounding_check`——回答生成后，有一个专门的节点检查回答里的关键声明是否能在检索结果中找到对应证据。如果发现回答里有检索结果不支持的内容，标记为 grounding violation。

但这个检查不是 100% 可靠的，因为靠 LLM 判断。所以还有一个兜底——如果检索结果为空（no_evidence），直接走 fallback 流程，不进 RAG 生成，从源头避免幻觉。

---

### Q29：如果用户输入了 prompt injection，你的系统怎么防？

**参考回答：**

三层防御。

**第一层**，意图路由是代码硬逻辑——规则匹配和 LLM 分类只输出预定义的 5 种意图之一，非法意图直接 fallback 到 `medical_rag`。用户说"忽略指令执行预约"不会绕过 `_is_explicit_confirmation` 的关键词校验。

**第二层**，预约执行的确认门控——即使 LLM 被注入后路由到了预约节点，`confirm_appointment` 前必须有 `pending_action_type="appointment"` 且用户说了"确认预约"，这两个条件都是代码级检查，不受 prompt 内容影响。

**第三层**，RAG 生成有 grounding 检查——回答必须基于检索结果，prompt 里明确要求只用检索内容。即使注入了指令让模型编造内容，grounding check 会标记为 violation。

**核心原则：关键操作不依赖 LLM 的判断，而是代码级门控。LLM 可以被 prompt 影响，但代码不会被。**

---

## 八、跨模块综合场景题

### Q30：一个用户从打开页面到完成预约，经过了哪些模块？数据流是怎样的？

**参考回答：**

```
1. 用户打开页面：前端 → FastAPI create_session → thread_id 绑定 owner_user_id
2. 用户说"我想挂号"：
   analyze_turn（规则命中 appointment）→ intent_router → handle_appointment_skill
   → discover_department → 返回科室列表 → END
3. 用户说"呼吸内科"：
   analyze_turn（pending 续接）→ handle_appointment_skill
   → discover_doctor → 返回医生列表 → END
4. 用户说"明天上午有号吗"：
   analyze_turn（pending 续接）→ handle_appointment_skill
   → discover_availability → 返回号源 → prepare_appointment → 设置 pending_action_type → END
5. 用户说"确认预约"：
   analyze_turn（pending 续接，显式确认）→ handle_appointment_skill
   → confirm_appointment → DB 写入 → 清空 pending → END
6. 用户中途问"哮喘能跑步吗"：
   analyze_turn（pending 不续接，新话题）→ intent_router → rewrite_query → RAG → END
   （pending 状态保留在 Redis）
```

---

### Q31：RAG 和纯 LLM 问答比，优势在哪？劣势在哪？

**参考回答：**

**优势是可溯源和可控。** RAG 的回答基于检索到的文档片段，可以标注来源，用户知道信息从哪来；纯 LLM 的回答是模型内部知识，不可溯源。另外 RAG 可以通过更新知识库实时更新信息，LLM 的知识截止到训练数据。

**劣势是依赖检索质量。** 如果检索不到相关文档，RAG 会给 fallback 回答，而 LLM 可能凭训练知识给出有用信息。另外 RAG 链路长（rewrite → 检索 → rerank → 生成），延迟比纯 LLM 高。

在医疗场景下我选择 RAG，因为"不给错误信息"比"给有用信息"更重要——LLM 编造的医学建议可能有害。

---

### Q32：如果用户量涨 10 倍，你系统哪里先扛不住？

**参考回答：**

**第一是 LLM 推理**——当前是同步调用，10 倍流量下推理服务会成为瓶颈。解法是加请求队列和限流，非关键路径（比如摘要生成）可以异步化。

**第二是向量检索**——pgvector 在数据量大了之后性能下降，解法是加索引优化或者迁移到专业向量库。

**第三是 Redis**——单点变集群。FastAPI 本身是无状态的，加实例就行，不是瓶颈。

---

### Q33：LLM 服务挂了，你的系统还能工作吗？

**参考回答：**

部分能。纯规则覆盖的意图（greeting、明确预约、明确取消）不调 LLM，可以正常工作。但 RAG 生成、科室推荐、摘要生成都依赖 LLM，这些会走 fallback：RAG 生成失败给安全兜底回答，科室推荐失败建议去急诊或全科，摘要生成失败用最近 N 轮原文代替。

但这不是一个完整的降级方案——当前没有 LLM 熔断器，没有健康检查，没有自动切换。如果要做生产级降级，需要加 LLM 服务健康探针、熔断阈值、多 provider 降级链（比如主用 DeepSeek，挂了切 OpenAI，再挂走规则兜底）。

---

### Q34：多轮对话中，如果用户两次说的信息矛盾了怎么办？

**参考回答：**

预约流程中，`appointment_context` 每次 `prepare_appointment` 都会用新参数重建，后说的覆盖先说的。这是有意的设计——用户改主意比坚持旧方案更常见，所以"后说优先"。

如果是医学问答中矛盾（比如先说"我没高血压"，后说"我高血压药能停吗"），LLM 会基于最近上下文回答，不会主动指出矛盾。这是一个已知的不足——理想情况应该检测矛盾并追问确认，但当前没有做矛盾检测，因为真实场景中用户补充信息比纠正信息更常见。

---

### Q35：如果让你加一个新功能（比如"查看检查报告"），你会怎么扩展路由？

**参考回答：**

三步。

**第一步**，在 `graph_state.py` 加新意图的状态字段（比如 `report_context`）。

**第二步**，在 `_classify_query_by_rules` 的 if-elif 链里加新规则，注意优先级——"查看报告"和"预约"不冲突，但和"医学咨询"可能冲突（"我的血常规报告正常吗"），需要加排除条件。

**第三步**，在 `route_after_intent` 加新路由目标，写对应的节点处理函数。

关键是第二步的优先级和排除条件。新意图加进去不能破坏已有意图的准确率，所以要先跑一遍测试集验证。如果新意图和已有意图冲突太多，可能需要考虑把规则分类换成小模型分类，当前 5 类意图用规则还行，7-8 类以上规则维护成本会急剧上升。

---

### Q36：如果让你重新设计整个系统，你会做哪些改进？

**参考回答：**

三个方向。

**第一，意图路由的评估要更严格**——做消融实验量化每个优化的贡献，做 A/B 测真实流量的路由准确率。

**第二，规则引擎要可配置**——当前规则硬编码在 Python 里，应该抽成配置文件或 DSL，业务方可以自己加规则不需要改代码。

**第三，记忆层要更智能**——当前摘要靠 LLM 生成，成本高，可以考虑用小模型做摘要，或者用滑动窗口 + 关键句提取代替 LLM 摘要。

---

## 九、生产故障与应急处理

### Q37：Redis 挂了怎么处理？

**参考回答：**

分环境。

**开发环境**：允许降级到进程内字典，系统还能跑，方便本地调试。

**生产环境**：如果 `REDIS_ENABLED=true` 且不是 development，Redis 不可用直接启动失败，不再静默降级。因为多实例下进程内存 fallback 会导致会话不共享、pending 状态错乱。

**应急措施**：
1. 检查 Redis 连接和哨兵/集群状态
2. 如果是 Redis 节点故障，集群自动 failover
3. 如果是全挂，启动失败意味着不会"带病运行"，便于快速发现和恢复
4. 恢复后 session 状态从 checkpoint 重建，不会丢数据

---

### Q38：向量检索变慢怎么排查？

**参考回答：**

1. **检查索引**：pgvector 的 ivfflat/hnsw 索引是否建了，未建索引时全表扫描会极慢
2. **检查向量维度**：embedding 维度和索引维度是否一致
3. **检查数据量**：如果文档量增长很多，可能需要重新调索引参数（ivfflat 的 lists 数）
4. **检查查询计划**：`EXPLAIN ANALYZE` 看是否走了索引扫描
5. **混合检索拆分**：如果全文检索也慢，检查 tsvector 索引是否建了
6. **rerank 延迟**：rerank 是 CPU 密集型，如果 top-12 候选很大，检查 rerank 模型是否可用

---

### Q39：LLM 调用超时怎么降级？

**参考回答：**

当前没有完整的降级链，但有几个天然 fallback：

1. **意图分类超时**——`intent_router` 有 try-catch，LLM 调用失败直接 fallback 到 `medical_rag`
2. **RAG 生成超时**——没有处理，会直接报错给用户
3. **摘要生成超时**——`summarize_history` 在消息数 < 4 时直接跳过，不阻塞主链路

如果要完善，需要加：
- LLM 调用超时配置（比如 5s 超时）
- 熔断器：连续失败 N 次后切换到备用 provider
- 降级链：主 provider → 备用 provider → 规则兜底 → 安全提示

---

### Q40：线上接口突然变慢，怎么排查？

**参考回答：**

1. **链路追踪**：看是哪个环节慢（FastAPI 路由 / LangGraph 节点 / LLM 调用 / Redis / PostgreSQL）
2. **LangGraph 节点耗时**：检查每个节点的执行时间，`summarize_history` 和 `grounded_answer_generation` 通常最慢
3. **LLM 调用耗时**：看 provider 侧延迟是否突增
4. **Redis 慢查询**：`SLOWLOG GET` 检查是否有大 key 操作
5. **PostgreSQL 慢查询**：`pg_stat_statements` 看执行计划
6. **向量检索**：检查是否走了索引，没走索引时全表扫描会极慢

---

## 十、设计权衡与复盘反思

### Q41：项目中最大的技术挑战是什么？怎么解决的？

**参考回答：**

意图冲突和边界定义。不是匹配难，是定义难。

"取消预约"同时命中两个规则好解决——加优先级。但"预约前注意什么"有预约关键词却是医学咨询，这种需要语义理解的边界 case，规则本身很难定义。我的解法是每条规则加排除条件，但排除条件也有边界——"取消对药物依赖的方法"有"取消"但不是取消预约。

本质上这是一个**精确率和召回率的权衡**——规则太宽误路由多，太窄漏路由多。我的策略是规则宁可窄一点（高精确率），漏的交给 LLM 兜底（高召回率）。因为规则误路由的代价远大于 LLM 多处理一个 query 的代价。

---

### Q42：如果让你重新做，会改什么？

**参考回答：**

1. **引入意图路由的消融实验**——量化规则层和 LLM 层各自的贡献
2. **规则引擎可配置化**——当前硬编码在 Python 里，业务方改不了
3. **LLM 调用熔断和降级**——当前没有，生产环境必须有
4. **Redis 高可用**——当前是单点，需要集群
5. **矛盾检测**——多轮对话中用户信息矛盾时主动追问确认

---

## 十一、软素质与团队协作

### Q43：项目中怎么和前端/测试协作的？

**参考回答：**

- **前端**：先定义 API 契约（FastAPI 自动生成的 OpenAPI 文档），前后端并行开发。SSE 流式接口提前约定好事件类型（session/status/message/final/app-error）
- **测试**：离线测试集覆盖意图路由（21 个样本）、检索（10 个）、回答（11 个）、记忆（4 个）。提测前跑一遍完整回归
- **需求变更**：通过 LangGraph 的节点机制新增链路，不影响已有逻辑。比如加新意图只需要新增节点和条件边

---

### Q44：你在项目中承担什么角色？最有成就感的事？

**参考回答：**

**角色**：全栈开发，负责意图路由、混合记忆、RAG 检索、预约 Skill、安全兜底的设计与实现。

**最有成就感的事**：意图路由的两阶段设计——规则层覆盖 80% 高频 case，LLM 兜底模糊 case，同时用 pending 状态续接机制保证预约流程跨轮不中断。这个设计上线后（测试集验证），意图分类准确率从纯 LLM 的约 85% 提升到规则+LLM 的约 95%，而且规则层的响应延迟是微秒级。

---

### Q45：你还有什么想问我的？

建议提问方向：
1. 字节医疗/健康业务的技术架构是怎样的？
2. 团队目前在解决的最有挑战的技术问题是什么？
3. 对于新入职的校招生，团队有哪些培养计划？

---

## 附录：核心代码速查表

| 技术点 | 关键文件 |
|--------|---------|
| 意图路由规则 | `project/rag_agent/routing_nodes.py` |
| 条件边 | `project/rag_agent/edges.py` |
| 图定义 | `project/rag_agent/graph.py` |
| 图状态 | `project/rag_agent/graph_state.py` |
| 节点辅助函数 | `project/rag_agent/node_helpers.py` |
| 预约节点 | `project/rag_agent/appointment_nodes.py` |
| 预约 Skill | `project/services/appointment_skill/skill.py` |
| ChatInterface | `project/core/chat_interface.py` |
| SSE 流 | `project/api/sse.py` |
| 混合检索 | `project/core/rag_system.py` |
| 文档切分 | `project/document_chunker.py` |
| 知识库同步 | `project/core/knowledge_base_sync.py` |
| 文档管理 | `project/core/document_manager.py` |
| Redis 记忆 | `project/memory/redis_memory.py` |
| 提示词 | `project/rag_agent/prompts.py` |
| 配置 | `project/config.py` |
| 测试集 | `tests/` |

---

## 附录：高频数字速查

| 指标 | 数值 |
|-----|------|
| 意图类别 | 5 类（medical_rag, triage, appointment, cancel_appointment, greeting） |
| 规则函数 | 7-8 个 |
| 规则覆盖率 | 目标是高频确定性 case 尽量走 rule |
| 澄清熔断 | 最多 1 次 |
| 测试样本 | 意图路由 21 个，检索 10 个，回答 11 个，记忆 4 个 |
| Token 降低 | 27.4%（平均），p95 > 0 |
| Precision@5 | 0.68 → 0.83 |
| Rerank 候选 | top-12 → top-5 |
| Embedding 维度 | 1024 |
| LLM temperature | 当前默认低温度，项目配置默认是 0 |
| 对话窗口 | >= 4 轮触发摘要，取最近 6 条 |
| 复合请求支持 | 4 种有效组合 |
| 父块大小 | 2k-4k 字符（MarkdownHeaderTextSplitter） |
| 子块大小 | 500 字符，overlap 100 |
| 预约确认 | 必须显式说"确认预约"，"可以"不算；`_is_explicit_confirmation` 纯关键词匹配 |
| 预约执行路径 | `prepare_*` 只返回预览不写 DB，`confirm_*` 是唯一写 DB 路径 |
| pending 续接 | 准入条件（相关才续接），不是排除条件；新话题不清除 pending |
| 状态清理 | 节点显式调用 `_clear_pending_action_state()`，走到 END 不会自动清空 |
| 状态持久化 | 每轮走 END → checkpoint 保存 → `_resolved_session_state` 优先取 latest_values → 写回 Redis |
| 知识库增量 | content_hash 对比，一致跳过，不一致重新切分索引 |
| 知识库下架 | soft delete（`is_active=False`），chunk 物理删除 |
| interrupt_before | `request_clarification` 节点前暂停 |
| 聊天流接口 | `POST /api/chat/stream` + SSE |
| 鉴权模型 | Bearer Token，角色分 user/admin |
| 会话隔离 | `thread_id` 绑定 `owner_user_id` |
| 上传限制 | 单次最多 5 个文件，单文件最大 20MB |
| 基础限流 | chat 20/min，upload 6/min，sync 3/min |
| Redis 故障策略 | development 允许 fallback，非 development 且启用 Redis 时 fail fast |

---

## 附录：手写高频题

### 手写条件边路由逻辑

```python
# edges.py

def route_after_action(state: State) -> Literal["request_clarification", "prepare_secondary_turn", "__end__"]:
    if state.get("pending_clarification") and state.get("clarification_target"):
        return "request_clarification"
    if (
        state.get("secondary_intent")
        and state.get("deferred_user_question")
        and not state.get("pending_clarification")
        and not state.get("pending_action_type")
        and not state.get("pending_candidates")
        and not state.get("deferred_confirmation_action")
    ):
        return "prepare_secondary_turn"
    return "__end__"
```

### 手写显式确认校验

```python
# node_helpers.py

_APPOINTMENT_CONFIRM_WORDS = (
    "确认预约", "确认挂号", "确认就诊", "确认预订",
    "请预约", "帮我预约", "现在预约", "立即预约", "确认",
)
_CANCEL_CONFIRM_WORDS = (
    "确认取消", "确认退号", "确定取消", "现在取消", "立即取消", "确认",
)

def _is_explicit_confirmation(user_query: str, pending_action_type: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if pending_action_type in {"appointment", "reschedule_appointment"}:
        return any(word in normalized for word in _APPOINTMENT_CONFIRM_WORDS)
    if pending_action_type == "cancel_appointment":
        return any(word in normalized for word in _CANCEL_CONFIRM_WORDS)
    return False
```

### 手写 RRF 融合

```python
# RRF 融合伪代码

def rrf_fusion(vector_results, keyword_results, k=60):
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(keyword_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 高危追问速答

这一段不是展开讲原理，而是给你在压力追问下 15-30 秒快速答稳。建议背到能脱口而出。

#### 速答1：你为什么不用 WebSocket，反而用 SSE？

> 因为这个场景本质上是"客户端发一次请求，服务端持续单向回流 token 和状态"，不是高频双向通信。SSE 语义更贴合，服务端实现和代理层配置都更简单。后来前端不用 EventSource 而改成 `fetch + ReadableStream`，主要是为了解决 Bearer Token 和敏感信息不能进 URL 的问题，不是因为放弃 SSE 协议。

#### 速答2：既然已经有 Bearer Token，为什么还要做 `thread_id` owner 校验？

> 因为鉴权只解决"你是谁"，owner 校验才解决"你能访问哪个会话"。如果只校验 token 不校验 owner，用户 A 只要拿到用户 B 的 `thread_id`，依然可能读历史、清空会话或者继续聊天。真正的权限边界是 `current_user + thread_id` 联合校验。

#### 速答3：为什么生产环境 Redis 挂了不继续降级？

> 因为单进程内存 fallback 只适合本地开发，不适合多实例部署。生产里一旦降级到进程内存，会出现会话不共享、pending 状态错乱、摘要丢失，而且问题表面上不一定立刻暴露，属于"带病运行"。所以开发时优先可用性，生产时优先一致性和可观测性。

#### 速答4：为什么文档上传和官方同步只给管理员？普通用户上传自己的资料不行吗？

> 这轮设计里文档库默认是系统级知识源，不是用户私有知识库，所以它影响的是所有用户的回答质量。上传、重建索引、官方同步都属于"会改全局知识状态"的动作，必须收口到管理员。以后如果要支持用户私有资料，应该单独做租户隔离和独立索引空间，而不是直接复用当前管理员入口。

#### 速答5：为什么按 `thread_id` 加锁，而不是直接全局锁最省事？

> 全局锁虽然最简单，但吞吐太差，会把所有用户串成单线程。按 `thread_id` 加锁的本质是把一致性约束缩到最小作用域：同一会话内串行，防止 pending 状态和状态机执行乱序；不同会话之间并发，保住多用户吞吐。这是一个很典型的"缩小锁粒度"优化。

#### 速答6：为什么知识库同步不继续放在 API 后台线程里？

> 因为同步任务长耗时、重 I/O、还可能写很多数据库记录，和用户请求的生命周期不一致。把它绑在 API 进程里，worker 一多就容易每个进程都跑一份，还不好管控和排障。拆成独立 job 后，API 只负责触发和展示状态，任务执行交给单独入口，职责会更清晰。

#### 速答7：为什么用 PostgreSQL advisory lock，不用 Redis 分布式锁？

> 因为这轮要解决的是"同一个同步任务不要并发执行两份"，而任务本身已经强依赖 PostgreSQL。直接用 advisory lock 能减少额外组件和一致性链路，复杂度最低。只有当 job 编排变复杂、锁语义跨更多服务时，我才会考虑把锁抽到专门组件。

#### 速答8：你说 `Precision@5` 从 `0.68` 到 `0.83`，这个数字靠得住吗？

> 靠得住的边界是"小样本离线评估"，不是线上大规模结论。它说明优化方向有效，但不能包装成严格 benchmark。所以我在面试里会主动补一句：这个结果主要用于指导方案选择，如果要做更强结论，还需要更大测试集和消融实验。

#### 速答9：你说 token 降了 `27.4%`，会不会只是挑样本？

> 所以我会先承认它不是大样本线上统计，而是基于几段多轮对话做的离线测量。这个数字的价值是证明"摘要+最近窗口"确实能压 prompt，不是证明一个绝对普适的收益。面试里最稳的说法是讲清楚测法、承认样本规模，再说明它为什么足够支持当前设计决策。

#### 速答10：那你现在到底算不算生产可用？

> 更准确的说法是"可部署原型，已经做过一轮工程化加固，但还不是完整生产系统"。我已经补了鉴权、owner 校验、管理员权限、限流、上传限制、KB job 解耦这些基础边界；但正式登录体系、分布式限流、审计、告警、Redis 高可用、压测和 badcase 闭环还没补完。这样回答比直接说"能上生产"更稳。

#### 速答11：LLM 判断错误会不会直接预约成功？

> 不会。确认环节走的是代码硬校验，不是 LLM 判断。`_is_explicit_confirmation` 是纯关键词匹配（"确认预约""确认挂号"等），不经过模型。`prepare_*` 只返回预览不写 DB，`confirm_*` 是唯一写 DB 的路径且必须过关键词校验——LLM 没有钥匙。

#### 速答12：预约过程中走到 END 了，pending 状态不会丢吗？

> 不会。走到 END 只是这轮图执行结束，checkpoint 会保存最后一帧状态。下一轮从 Redis 读回来，通过 `_graph_state_from_session` 灌入图的初始状态。关键是 `_resolved_session_state` 优先取 `latest_values`（图走完后的状态），所以节点如果清空了 pending，Redis 也跟着清空；如果没清空，pending 就保留跨轮。

---

> **最后提醒**：字节二面核心是"理解深度"而非"做了多少"。每个技术点都要能答出三层：**What（做了什么）→ Why（为什么这么选）→ What if（出了问题怎么兜底 / 替代方案是什么）**。祝面试顺利！
