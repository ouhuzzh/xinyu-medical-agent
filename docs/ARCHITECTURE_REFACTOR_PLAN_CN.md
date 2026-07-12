# 架构改造路线图

更新时间：2026-07-12

本文档是当前仍有参考价值的架构路线图，不是逐条待办清单。已经完成的阶段用于说明系统为什么变成现在的结构；未完成的阶段用于指导后续演进。

本文档记录从“单机可运行 demo”走向“可演进、可部署产品原型”的改造路线。目标不是一次性大重构，而是用小步、可验证的切片逐步收紧边界。

## 当前状态摘要

- 已完成运行态边界外置的第一批工作：Redis-backed session lock、Redis-backed auth/runtime guard、运行态 backend 状态暴露。
- 已完成聊天主链路拆分：`ChatTurnInputService` 和 `ChatTurnService` 已从 `ChatInterface` 中抽出。
- 已完成 `RAGSystem` 瘦身第一步：服务启动、图编译、知识库监督和 skill 注册已经开始拆分。
- 已完成 embedding schema guard：启动期检查 `VECTOR_DIMENSION` 与 pgvector 列定义。
- 已完成多医院 MCP 挂号选择闭环：多医院未明确时澄清，确认阶段锁定 `hospital_code`。
- 已完成部署拓扑第一步：Compose 增加独立知识库 Worker，自动 bootstrap 和定时同步不再由 API 容器执行。
- 当前正在推进 P4 多智能体监督/任务分解方向，详细设计记录放在 `docs/superpowers/`。

## 当前主要问题

1. `RAGSystem` 职责过重：同时承担服务实例化、状态中心、后台线程、图编译、技能注册和健康检查。
2. 多处运行态依赖进程内内存：会话锁、限流、登录锁定、MCP 工具池和熔断状态不适合多 worker/多实例。
3. `ChatInterface` 混合了传输适配、会话编排、状态同步、流式渲染、记忆抽取和日志记录。
4. 数据迁移体系不够正式，embedding 维度在配置和 schema 之间存在隐性不一致风险。
5. Skill 插件方向正确，但注册、路由和状态 schema 仍分散在多个模块。
6. 部署形态仍偏本地开发脚本，生产拓扑没有被一等建模。

## 改造原则

- 每次只移动一个边界，避免业务行为和架构整理混在一个提交里。
- 先抽象运行态接口，再替换具体实现。
- 保留开发模式的低门槛 fallback，但让生产模式优先使用 Redis/PostgreSQL 等外部一致性组件。
- 新增能力必须有清楚的 owner 模块，避免继续扩大 `RAGSystem` 和 `ChatInterface`。
- 对医疗、预约、凭证相关链路保持保守变更，先补保护层，再做内部瘦身。

## 阶段 1：运行态边界外置（已完成第一批）

目标：消除最明显的单进程假设。

任务：

- 已完成：抽象会话并发锁，开发模式可用进程内锁，Redis 可用时使用 Redis lock。
- 已完成：将 API rate limit 和登录锁定迁移到 Redis-backed 实现，保留内存 fallback。
- 已完成：MCP 工具池暴露 `backend_name()`，状态不再在 API 路由里硬编码。
- 已完成：给 `/api/system/status` 增加运行态组件状态：lock backend、rate-limit backend、login-lockout backend、MCP pool backend、schema guard backend。
- 后续：将 MCP 工具池健康、熔断和工具清单迁移到可共享的持久/缓存后端。

验收：

- 单 worker 测试行为不变。
- 两个 API 进程并发请求同一 `thread_id` 时，只允许一个 chat turn 执行。
- Redis 不可用时，development 能降级；production 明确失败或进入 degraded 状态。

## 阶段 2：聊天应用服务拆分（进行中）

目标：让 FastAPI、Gradio 和未来 CLI/API 复用同一个 chat turn 应用服务。

任务：

- 已完成：新增 `ChatTurnInputService`，负责 thread/config/state/user memory/stream input 组装。
- 已完成：新增 `ChatTurnService`，负责最终答案兜底、会话落库、summary、记忆抽取、session state 和 route log。
- 将 SSE event formatting 留在 `api/sse.py`。
- 将 Gradio message formatting 留在 `ui/`。
- 将记忆注入、摘要更新、route log 记录拆成可测试的 collaborators。

验收：

- `ChatInterface` 不再直接承担 FastAPI SSE 语义。
- 单元测试可以绕过 FastAPI，直接测试一次 chat turn 的状态变化。
- Gradio 和 React API 使用同一业务服务，不再复制状态逻辑。

## 阶段 3：RAGSystem 瘦身（已开始）

目标：把系统启动、依赖组装、图编译和后台任务分离。

任务：

- 已完成第一步：新增 `ServiceBootstrapper`，集中创建核心服务、派生服务并注册到 `ServiceContainer`。
- 已完成第一步：新增 `AgentGraphFactory`，负责 LLM runtime、tools、skill 注册和 LangGraph 编译。
- 已完成第一步：新增 `KnowledgeBaseSupervisor`，负责 KB 状态、bootstrap 和 sync scheduler；`RAGSystem` 保留兼容代理方法。
- 已完成第一步：将 Skill 注册移到 `SkillBootstrapper`，并按 skill name 做幂等注册；后续可继续演进为 manifest loader。

验收：

- `RAGSystem` 只保留兼容 facade，核心逻辑迁移到独立组件。
- 图编译可以在测试中注入 fake LLM/fake tools。
- 知识库任务可以独立于 API 进程运行。

## 阶段 4：数据迁移和知识库版本化

目标：让 embedding/model/schema 的变更可控。

任务：

- 引入 Alembic 或等价 migration 工具。
- 将 embedding model、dimension、chunk strategy、source version 写入 KB metadata。
- 已完成第一步：新增 embedding schema guard，检查 `VECTOR_DIMENSION` 与 pgvector 列定义是否一致；development 标记 degraded，production 启动失败。
- 当 `VECTOR_DIMENSION` 与数据库 schema 不一致时，启动时明确报错。
- 为重建索引提供离线 job，而不是隐式依赖 API 进程。

验收：

- 更换 embedding 模型时，有明确的迁移/重建流程。
- 线上启动不会因为维度不一致在第一次写入时才失败。
- 文档同步、软删除和重建过程可观测。

## 阶段 5：Skill 插件收束

目标：让新增业务意图不需要改 graph 主干。

任务：

- 为每个 skill 定义 manifest：intent、keywords、utterances、route target、state schema、required services、permissions。
- 图构建从 registry 自动读取节点和边。
- 删除 legacy route fallback 中已经被 skill 覆盖的重复规则。
- 将 intent 分类 prompt 从 skill manifest 自动生成。

验收：

- 新增一个简单 skill 只需要新增 skill 文件和 manifest。
- intent 标签、路由、状态字段只有一个事实来源。
- 路由测试覆盖 L1/L2/L3 分类路径。

## 阶段 6：部署拓扑产品化

目标：让本地 demo、测试、生产部署都有明确拓扑。

任务：

- 已完成第一步：`docker-compose.yml` 和生产 Compose 已增加独立 `worker` 服务。
- 将 Gradio 标记为 admin/debug profile。
- 增加 readiness/liveness 分离：`/api/health` 只测进程存活，`/api/system/status` 测依赖。
- 已完成第一步：增加 `project/worker.py` 入口运行 KB bootstrap 和定时同步，复用 PostgreSQL advisory lock。
- 后续：将 memory extraction 等非交互任务迁移到持久化任务队列。

验收：

- 新人可以一条命令启动完整本地栈。
- API 进程不再承担所有后台维护任务。
- 部署文档能说明单机、多 worker、多实例的限制和推荐配置。

## 已完成的第一批改动

- 新增会话锁组件，将锁从 `api.dependencies` 中拆出。
- Redis 可用时使用 Redis-backed lock；否则开发模式回退到进程内锁。
- 新增 Redis-backed API rate limiter 和登录锁定组件；开发模式保留内存 fallback。
- `/api/system/status` 暴露 `runtime_backends` 和 `schema_health`。
- `ChatInterface` 已拆出 `ChatTurnInputService` 和 `ChatTurnService`，下一步可继续瘦身为纯 adapter。

## 下一阶段建议

- 继续拆 `RAGSystem`：下一步可以推进 Skill manifest loader，并将 memory extraction 接入持久化任务队列。
- 为 schema guard 后续接入正式迁移工具（如 Alembic）预留 metadata/version 表。
- 将 MCP pool 的健康状态和熔断状态从进程内迁移到 Redis/PostgreSQL。
