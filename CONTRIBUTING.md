# Contributing Guide

这份文档说明本项目的日常开发方式。目标是让代码、文档、测试和部署资料保持可维护。

## 开发分支

- 主分支：`main`
- 功能分支建议：`feat/<topic>` 或 `codex/<topic>`
- 修复分支建议：`fix/<topic>`
- 文档分支建议：`docs/<topic>`

不要在一个提交里混合无关主题。例如“修改 RAG 路由”和“整理面试文档”应该分开提交。

## 本地启动

后端：

```powershell
.\venv\Scripts\python.exe project\api_app.py
```

前端：

```powershell
cd frontend
npm run dev
```

Docker：

```powershell
docker compose up -d --build
```

更详细说明见 [docs/DOCKER_DEPLOY_CN.md](docs/DOCKER_DEPLOY_CN.md)。

## 推荐检查

后端快速检查：

```powershell
.\venv\Scripts\python.exe -m compileall project tests
.\venv\Scripts\python.exe -m unittest tests.test_api_app -v
```

前端检查：

```powershell
cd frontend
npm test
npm run build
```

提交前至少运行与本次改动相关的测试；如果没有运行完整测试，需要在提交说明或 PR 说明里写清楚。

## 代码约定

- API 层只做路由、鉴权、DTO 和协议转换，业务逻辑尽量放到 service 或 core 模块。
- LangGraph 节点保持小而明确，跨节点状态写入要有测试覆盖。
- 医疗预约、取消、改约等高风险动作必须经过预览和用户确认。
- 多医院 MCP 场景必须锁定 `hospital_code`，不能按工具顺序默认选择医院。
- 数据库 schema 变更要同步更新初始化 SQL、schema manager 和测试。

## 文档约定

- 新增主线文档后同步更新 [docs/README.md](docs/README.md)。
- 面试、演示、一次性分享产物不要混进主线文档；图片源文件可以提交，导出的 zip/PDF/PNG 默认不提交。
- 设计计划和阶段性实施记录放到 `docs/superpowers/`，避免和当前使用说明混在一起。

## 提交建议

提交信息用简短动词开头，例如：

```text
docs: organize project documentation
feat: add session rename and delete
fix: lock MCP appointment hospital selection
test: cover chat session ownership
```

提交前确认：

```powershell
git status --short
git diff --check
```
