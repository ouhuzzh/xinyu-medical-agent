# Docs Index

这份索引用来说明 `docs/` 目录里哪些文档是当前主线材料，哪些是设计记录，避免阅读时被旧资料带偏。

## 当前主线

- [medical-agent-project-interview-prep.md](/D:/nageoffer/agentic-rag-for-dummies/docs/medical-agent-project-interview-prep.md)
  项目专项面试准备。现在面试复习优先看这份。

- [PROJECT_STRUCTURE_CN.md](/D:/nageoffer/agentic-rag-for-dummies/docs/PROJECT_STRUCTURE_CN.md)
  当前代码结构说明，适合快速建立目录级认知。

- [PROJECT_GUIDE_CN.md](/D:/nageoffer/agentic-rag-for-dummies/docs/PROJECT_GUIDE_CN.md)
  面向项目走读的中文导读，适合从入口文件一路往下看。

- [USER_GUIDE.md](/D:/nageoffer/agentic-rag-for-dummies/docs/USER_GUIDE.md)
  运行和使用说明，偏“怎么启动和怎么玩”。

- [QA_EVAL.md](/D:/nageoffer/agentic-rag-for-dummies/docs/QA_EVAL.md)
  QA 评测入口和输出说明。

- [MEDICAL_IMPORT.md](/D:/nageoffer/agentic-rag-for-dummies/docs/MEDICAL_IMPORT.md)
  官方医疗资料导入链路说明。

- [MEDICAL_SOURCES.md](/D:/nageoffer/agentic-rag-for-dummies/docs/MEDICAL_SOURCES.md)
  医疗资料来源分层建议。

- [POSTGRES_SETUP_CN.md](/D:/nageoffer/agentic-rag-for-dummies/docs/POSTGRES_SETUP_CN.md)
  PostgreSQL + pgvector 本地初始化说明。

## 设计记录

- [ARCHITECTURE_REFACTOR_PLAN_CN.md](/D:/nageoffer/agentic-rag-for-dummies/docs/ARCHITECTURE_REFACTOR_PLAN_CN.md)
  架构演进计划，适合了解系统后续重构方向。

- [mcp-pool-refactor-plan.md](/D:/nageoffer/agentic-rag-for-dummies/docs/mcp-pool-refactor-plan.md)
  MCP 连接池重构方案，属于专项设计文档。

## 目录约定

当前 `docs/` 以“主线说明 + 设计记录”为主，不再保留旧版面试稿和明显过时的时序导读。

如果后续新增文档，建议优先放到下面两类之一：

1. 主线说明：和当前实现、README、启动方式直接对应。
2. 设计记录：明确说明是 proposal、refactor plan 或专项分析。
