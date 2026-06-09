# 项目结构说明

这份文档只描述当前主线代码，避免旧文档里的 Qdrant/纯 Gradio 版本信息干扰阅读。

## 一句话定位

这是一个医疗场景的 Agentic RAG 产品原型：

```text
React 前端
  + FastAPI API
  + LangGraph 多意图智能体
  + PostgreSQL/pgvector 知识库
  + Redis 会话记忆
  + 用户长期记忆
  + MCP 多医院预约工具
```

## 根目录

| 路径 | 作用 |
| --- | --- |
| `README.md` | 对外总览、快速启动、架构摘要。 |
| `project/` | Python 后端主包。 |
| `frontend/` | React/Vite 用户端。 |
| `tests/` | 单元、回归、路由、RAG、MCP、预约和安全测试。 |
| `docs/` | 中文说明、架构文档、评测说明、面试材料和设计记录。 |
| `scripts/` | 压测、mock MCP server、demo、评测和维护脚本。 |
| `markdown_docs/` | 本地导入后的 Markdown 文档，运行时数据，不提交。 |
| `runtime/` | 日志、上传缓存、checkpoint、demo profile，运行时数据，不提交。 |
| `assets/` | README 演示图片等仓库级静态资源。 |

## 后端 `project/`

| 路径 | 作用 |
| --- | --- |
| `api_app.py` | FastAPI 启动入口。 |
| `app.py` | Gradio 管理/调试台入口。 |
| `config.py` | 读取 `.env`，集中暴露模型、数据库、Redis、MCP、JWT、RAG 阈值等配置。 |
| `api/` | React 调用的 HTTP/SSE 协议层。只做鉴权、DTO、路由和请求转换。 |
| `core/` | 系统初始化、文档管理、知识库同步、chunking、评测、观测。 |
| `rag_agent/` | LangGraph 图、节点、边、状态、提示词、检索节点和预约节点。 |
| `skills/` | 技能插件：问候、导诊、预约、取消、医疗 RAG、MCP 等。 |
| `services/` | 业务服务，当前重点是预约服务和 appointment skill 包。 |
| `mcp_integration/` | 医院 MCP 注册、用户凭证、工具池、token 加密、MCP skill。 |
| `db/` | PostgreSQL 连接、schema、pgvector 检索、会话、日志、导入任务等 store。 |
| `memory/` | Redis 短期记忆、对话摘要、长期用户记忆和记忆抽取。 |
| `ui/` | Gradio 管理/调试界面。 |
| `benchmarks/` | 路由、检索、记忆、问答质量等离线评测。 |

## 前端 `frontend/`

| 路径 | 作用 |
| --- | --- |
| `src/App.jsx` | 前端总装配：登录态、侧边栏、页面切换、主题、全局快捷键。 |
| `src/pages/` | 聊天、文档管理、医院绑定、登录页面。 |
| `src/components/` | 消息、输入框、侧边栏、状态、按钮等复用组件。 |
| `src/hooks/` | 聊天会话、系统状态、文档、搜索、主题等状态逻辑。 |
| `src/lib/` | API、SSE、导出工具。 |
| `src/i18n/` | 中英文文案。 |
| `src/styles/` | 拆分后的 CSS。 |

## 核心请求链路

```text
用户发消息
  -> frontend/src/lib/api.js 构造 /api/chat/stream
  -> project/api/routes/chat.py 鉴权、限流、会话归属校验
  -> project/api/sse.py 输出 SSE
  -> project/core/chat_interface.py 管理流式响应和会话状态
  -> project/rag_agent/graph.py 编排 LangGraph
  -> skills / rag_agent / services / memory / mcp_integration
  -> 返回前端可见消息
```

## 可以清理但不建议直接提交删除的内容

这些通常是本地运行产物，已经在 `.gitignore` 中，不影响仓库主线：

- `runtime/*.log`
- `runtime/langgraph_checkpoints.pkl`
- `runtime/langgraph_checkpoints.pkl.bak`
- `runtime/api_uploads/`
- `runtime/test_tmp/`
- `runtime/chrome-demo-profile/`
- `runtime/demo_frames/`
- `markdown_docs/`
- `parent_store/`
- `qdrant_db/`

这些是个人工具或 IDE/agent 配置，删除前最好确认使用者：

- `.claude/`
- `.claude-plugin/`
- `.qoder/`
- 顶层 `skills/` 下的本地 agent skill 链接

## 当前应保留的“看起来像旧东西”的模块

- `project/ui/` 和 `project/app.py`：仍作为 Gradio 管理/调试台存在。
- `project/db/parent_store_manager.py`：RAG 工具仍会读取 parent chunk 内容。
- `project/config.py` 里的 `QDRANT_DB_PATH`：目前是兼容遗留配置名，不代表当前主存储仍是 Qdrant。
- `project/README.md`：现在已改为后端包说明，不再重复旧版架构。
