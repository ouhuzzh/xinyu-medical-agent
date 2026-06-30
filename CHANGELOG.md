# Changelog

本文件记录项目的重要阶段性变化。日期按本地开发记录整理。

## 2026-06-29

- 整理项目文档中心，补充贡献、安全和变更记录文档。
- 新增面试架构说明和 4 张可直接展示的 SVG 架构图。
- 增加面试架构图 HTML 展示页，方便浏览器直接打开。
- 明确微信分享导出产物不进入仓库，只保留源文档和源图片。

## 2026-06

- 引入更完整的 Agentic RAG 证据闭环：query planning、工具化检索、证据强度评估、回答 grounding 和修正循环。
- 完成多医院 MCP 挂号选择改造，避免多医院绑定时按工具匹配顺序误选医院。
- 增强会话管理：多会话列表、会话隔离、会话重命名和删除归档。
- 增加 Redis-backed session lock 和运行态 guard，为多 worker 部署做准备。
- 增加 embedding dimension schema guard，在启动期发现 pgvector 维度不一致。
- 完善 Docker Compose 本地部署资料和生产上线检查清单。
