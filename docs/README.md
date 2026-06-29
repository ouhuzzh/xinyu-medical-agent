# 文档中心

这份索引用来快速判断“该看哪份文档”。项目文档按用途分成五类：项目总览、开发运行、架构设计、上线运维、面试展示。

## 推荐阅读顺序

如果是第一次接手项目，建议按这个顺序看：

1. [根目录 README](../README.md)：对外项目介绍、能力概览和快速启动。
2. [项目结构说明](PROJECT_STRUCTURE_CN.md)：当前代码目录和核心链路。
3. [项目导读](PROJECT_GUIDE_CN.md)：从入口文件到核心模块的走读。
4. [Docker 部署说明](DOCKER_DEPLOY_CN.md)：本机和服务器部署方式。
5. [生产上线检查清单](PRODUCTION_ROLLOUT_CHECKLIST_CN.md)：上线前需要确认的事项。

如果是准备面试，优先看：

1. [面试架构说明](INTERVIEW_PROJECT_ARCHITECTURE_CN.md)
2. [面试架构图展示页](INTERVIEW_PROJECT_ARCHITECTURE_GALLERY.html)
3. [项目专项面试准备](medical-agent-project-interview-prep.md)

## 项目总览

| 文档 | 作用 |
| --- | --- |
| [根目录 README](../README.md) | 对外介绍、能力概览、快速启动、API 和测试入口。 |
| [项目结构说明](PROJECT_STRUCTURE_CN.md) | 当前目录职责、核心请求链路、保留模块说明。 |
| [项目导读](PROJECT_GUIDE_CN.md) | 面向代码阅读的中文导览。 |
| [用户使用指南](USER_GUIDE.md) | 前端使用、医院绑定、知识库、聊天和常见问题。 |

## 开发运行

| 文档 | 作用 |
| --- | --- |
| [贡献与开发流程](../CONTRIBUTING.md) | 分支、提交、测试、代码变更和文档约定。 |
| [PostgreSQL 配置](POSTGRES_SETUP_CN.md) | PostgreSQL + pgvector 本地初始化。 |
| [Docker 部署说明](DOCKER_DEPLOY_CN.md) | Docker Compose、本地部署和服务器部署说明。 |
| [医疗资料导入](MEDICAL_IMPORT.md) | 本地与官方医疗资料导入流程。 |
| [医疗资料来源](MEDICAL_SOURCES.md) | 医疗资料来源分层和可信度建议。 |
| [QA 评测](QA_EVAL.md) | 问答评测入口、指标和输出说明。 |

## 架构设计

| 文档 | 作用 |
| --- | --- |
| [架构重构计划](ARCHITECTURE_REFACTOR_PLAN_CN.md) | 系统从复杂单体向清晰边界演进的路线。 |
| [MCP 工具契约](MCP_TOOL_CONTRACT_CN.md) | 医院 MCP 接入、工具命名、映射和安全边界。 |
| [MCP 连接池重构计划](mcp-pool-refactor-plan.md) | MCP pool 的专项设计记录。 |
| [前后端拆分说明](architecture/frontend_backend_split.md) | React/FastAPI 拆分后的边界说明。 |
| [Superpowers 设计文档](superpowers/specs/) | 较大的阶段性设计规格。 |
| [Superpowers 实施计划](superpowers/plans/) | 具体阶段的执行计划和检查点。 |

## 上线运维与安全

| 文档 | 作用 |
| --- | --- |
| [生产上线检查清单](PRODUCTION_ROLLOUT_CHECKLIST_CN.md) | 环境变量、数据库、Redis、模型、备份、安全检查。 |
| [安全策略](../SECURITY.md) | 敏感信息、鉴权、MCP 高风险动作、漏洞处理。 |
| [变更记录](../CHANGELOG.md) | 重要版本和阶段性变更记录。 |

## 面试展示

| 文档/资源 | 作用 |
| --- | --- |
| [面试架构说明](INTERVIEW_PROJECT_ARCHITECTURE_CN.md) | 可直接照着讲的项目定位、架构图、RAG 和 MCP 安全说明。 |
| [面试架构图展示页](INTERVIEW_PROJECT_ARCHITECTURE_GALLERY.html) | 浏览器打开即可展示 4 张架构图。 |
| [项目专项面试准备](medical-agent-project-interview-prep.md) | 更完整的面试问答、项目亮点和表达材料。 |
| [系统分层架构图](architecture/interview-system-architecture.svg) | SVG 图片，可单独打开或放入 PPT。 |
| [核心请求时序图](architecture/interview-request-sequence.svg) | SVG 图片，可单独打开或放入 PPT。 |
| [Agentic RAG 链路图](architecture/interview-agentic-rag-loop.svg) | SVG 图片，可单独打开或放入 PPT。 |
| [MCP 安全边界图](architecture/interview-mcp-safety-boundary.svg) | SVG 图片，可单独打开或放入 PPT。 |

## 文档维护约定

- 面向使用者的文档放在 `docs/` 根目录。
- 架构图片和专项架构说明放在 `docs/architecture/`。
- 大型设计规格和执行计划放在 `docs/superpowers/specs/` 与 `docs/superpowers/plans/`。
- 微信分享用的 PNG/PDF/zip 是本地导出产物，默认不提交到仓库。
- 新增文档后请同步更新本索引，避免资料越积越乱。
