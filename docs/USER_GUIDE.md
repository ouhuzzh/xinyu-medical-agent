# 心语医疗助手 - 使用文档

## 项目简介

基于 LangGraph 的医疗智能助手，支持：

- 医学知识问答（RAG 检索 + 答案可信度控制）
- 用户级跨会话记忆（高血压、过敏史等自动召回）
- 智能科室推荐与挂号
- 预约/取消半受控对话流程
- LLM 分级路由 + 三态熔断降级
- JWT 用户注册登录
- MCP 协议接入外部医院（如协和、仁济）

---

## 一、环境准备

### 1.1 系统要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 推荐 3.11 |
| Node.js | 18+ | 前端构建 |
| PostgreSQL | 14+ | 含 pgvector 扩展 |
| Redis | 6+ | 可选，无 Redis 时降级到内存 |

### 1.2 数据库初始化

```bash
# 安装 pgvector 扩展
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 启动后端时会自动跑数据库迁移（13 张表）
```

### 1.3 Python 依赖

```bash
cd D:\nageoffer\agentic-rag-for-dummies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1.4 前端依赖

```bash
cd frontend
npm install
cd ..
```

### 1.5 配置 .env

复制 `project/.env.example` 到 `project/.env`，关键配置：

```env
# LLM 配置
ACTIVE_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx                       # 硅基流动 / OpenAI 兼容平台
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash

# 分级路由（Light=路由分类，Strong=回答生成）
LLM_TIERS_JSON=[{"name":"light","provider":"openai","model":"Qwen/Qwen2.5-7B-Instruct","temperature":0,"max_tokens":256,"timeout_seconds":30},{"name":"strong","provider":"openai","model":"deepseek-ai/DeepSeek-V4-Flash","temperature":0,"max_tokens":2048,"timeout_seconds":60}]

# 数据库
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ai_companion
POSTGRES_USER=postgres
POSTGRES_PASSWORD=你的密码

# Redis（可选）
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379

# Embedding 模型
EMBEDDING_MODEL=BAAI/bge-m3
VECTOR_DIMENSION=1024
```

---

## 二、启动服务

需要启动 3-4 个进程，按顺序操作：

### 终端 1：Mock 协和医院（可选，演示 MCP 用）

```bash
python scripts\mock_hospital_mcp_server.py --port=8001
```

显示 `[北京协和医院] Mock MCP Server` 即启动成功。

### 终端 2：后端 API

```bash
.\venv\Scripts\python.exe project\api_app.py
```

显示 `Uvicorn running on http://127.0.0.1:8000` 即启动成功。
等待 30 秒左右系统完全初始化（加载知识库、连接 Redis 等）。

### 终端 3：前端

```bash
cd frontend
npm run dev
```

显示 `Local: http://localhost:5173` 即启动成功。

### 终端 4：注册 Mock 协和到数据库（首次启动需要）

```bash
python scripts\seed_mock_hospital.py
```

显示 `[OK] 北京协和医院 registered` 即成功。

如果你使用 Docker 版后端，mock MCP 服务跑在宿主机上，注册时要用：

```bash
python scripts\seed_mock_hospital.py --docker-desktop
```

这样 API 容器会通过 `host.docker.internal` 访问宿主机上的 mock MCP 服务。

---

## 三、使用流程

### 3.1 注册和登录

打开浏览器访问 `http://localhost:5173`

- 第一次：点「立即注册」 → 输入用户名密码 → 自动登录
- 之后：直接登录

### 3.2 聊天问诊

左侧菜单选「聊天咨询」，直接输入问题即可。

**示例对话**：

```
你：我有高血压，最近头晕
AI：（基于医疗知识库回答，告知可能原因和建议）

你：那能吃止痛药吗
AI：（基于"高血压"记忆，推荐对血压影响小的止痛药）

你：帮我挂个心内科的号
AI：（自动进入挂号流程，询问时间/医生）
```

系统会自动记住用户的医疗信息（病史、过敏、用药），下次开新会话也能召回。

### 3.3 绑定外部医院（MCP）

左侧菜单选「医院绑定」：

1. 点「绑定新医院」
2. 选「北京协和医院」
3. Token 填 `demo-xiehe-token-12345`（Mock server 不校验任何 token）
4. 点「保存」
5. 点「测试连接」，显示绿色「在线」即成功

之后回到聊天页对 AI 说：
```
帮我查协和医院有没有明天的心内科号
```
AI 会调 Mock 协和的 MCP server 返回真实可预约号源。

### 3.4 退出登录

左侧底部点「退出登录」按钮。

---

## 四、模型配置说明

### 4.1 分级路由

| 层 | 模型 | 用途 |
|---|---|---|
| Light | Qwen2.5-7B-Instruct | 意图分类、查询改写、摘要 |
| Strong | DeepSeek-V4-Flash | 回答生成、科室推荐 |

**为什么分级**：路由判断不需要重型模型，7B 够用且免费。回答生成才需要 V4 Flash 这种质量更好的模型。

### 4.2 切换模型

修改 `project/.env` 中的 `LLM_TIERS_JSON`，重启后端即生效。

**推荐模型**：

| 模型 | 速度 | 质量 | 价格 |
|------|------|------|------|
| Qwen2.5-7B-Instruct | 极快 | 一般 | 免费 |
| DeepSeek-V4-Flash | 中等 | 好 | ¥1入/¥2出 |
| Qwen3.5-35B-A3B | 快(MoE) | 好 | ¥1.7入/¥13出 |

### 4.3 熔断降级

当主接口（硅基流动）连续 3 次失败，自动切换到备用 provider（`LLM_FALLBACK_PROVIDER=deepseek`）。60 秒后自动尝试恢复。

---

## 五、记忆系统

三层混合记忆：

| 层 | 存储 | 内容 | 生命周期 |
|---|------|------|---------|
| L1 短期 | Redis 滑动窗口 | 最近 12 条消息原文 | 24h TTL |
| L2 摘要 | PostgreSQL | 对话主题摘要（按线程） | 永久 |
| L3 语义 | PostgreSQL + pgvector | 用户病史、过敏、偏好（按用户） | 永久 |

### 5.1 自动提取

每次对话结束后，系统会用 LLM 自动从对话中提取：

- **medical**: 病史、过敏、用药（如"对青霉素过敏"）
- **fact**: 个人事实（如"58岁退休教师"）
- **preference**: 偏好（如"喜欢详细解释"）
- **decision**: 决策（如"决定去心内科就诊"）

每条记忆 LLM 评分 1-10，低于 4 分自动丢弃。重复记忆 embedding 相似度 > 0.9 自动合并。

### 5.2 智能召回

每次新对话开头，根据当前问题用三因子加权排序检索 top 5 相关记忆：

```
score = 0.3 * 时效 + 0.4 * 重要性 + 0.3 * 相关性
```

重要性权重最大，确保过敏史等关键信息不被新近闲聊覆盖。

### 5.3 查看自己的记忆

API 端点（需 token）：

```
GET /api/memory/my-memories
```

---

## 六、常见问题

### Q1：后端启动失败 `ModuleNotFoundError: No module named 'pymupdf.layout'`

```bash
pip install pymupdf
```

### Q2：聊天请求超时

后端初始化需要 30 秒（加载知识库、连接服务）。等待 `state: ready` 后再聊天。

```bash
# 查状态
curl http://127.0.0.1:8000/api/system/status -H "Authorization: Bearer 你的token"
```

### Q3：登录页面进不去

确认后端在跑：
```bash
curl http://127.0.0.1:8000/api/health
```

### Q4：医院绑定页面 "Failed to fetch"

确认运行了种子脚本：
```bash
python scripts\seed_mock_hospital.py
```

Docker 版后端使用：

```bash
python scripts\seed_mock_hospital.py --docker-desktop
```

### Q5：starlette 版本冲突

```bash
pip install "starlette<0.51,>=0.40"
```

### Q6：怎么停掉所有服务

Windows: 关闭三个终端窗口；或者

```powershell
taskkill /F /IM python.exe
```

### Q7：测试压力

```bash
# 单线程跑 50 个真实场景
python scripts\stress_test_realistic.py --count=50 --workers=1

# 只跑某个用户画像
python scripts\stress_test_realistic.py --persona "张阿姨"
```

---

## 七、目录结构

```
project/
  api/              FastAPI 接口层
  core/             RAG 系统、Chat 接口、文档管理
  rag_agent/        LangGraph 图、节点、工具
  skills/           Skill 插件框架
  memory/           Redis 短期 + PostgreSQL 长期记忆
  mcp_integration/  MCP 集成（多医院支持）
  db/               数据库 Store + Schema 管理
  services/         预约服务

frontend/
  src/
    pages/          ChatPage / DocumentsPage / HospitalPage / LoginPage
    components/    Sidebar / MessageBubble / Composer 等
    hooks/         useChatSession / useSystemStatus 等

scripts/
  mock_hospital_mcp_server.py   Mock MCP 医院服务器
  seed_mock_hospital.py          注册 Mock 医院到 DB
  stress_test_realistic.py       200场景压力测试
```

---

## 八、API 快速参考

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/auth/register` | POST | 注册 |
| `/api/auth/login` | POST | 登录 |
| `/api/auth/refresh` | POST | 刷新 token |
| `/api/auth/profile` | GET | 用户信息 |
| `/api/chat/session` | POST | 创建会话 |
| `/api/chat/history` | GET | 历史消息 |
| `/api/chat/stream` | POST | SSE 流式对话 |
| `/api/chat/clear` | POST | 清空会话 |
| `/api/system/status` | GET | 系统状态 |
| `/api/hospitals/list` | GET | 平台支持的医院 |
| `/api/hospitals/credentials` | GET | 我的绑定 |
| `/api/hospitals/credentials/add` | POST | 绑定新医院 |
| `/api/hospitals/credentials/delete` | POST | 解绑 |
| `/api/hospitals/credentials/test` | POST | 测试连接 |
| `/api/memory/my-memories` | GET | 我的记忆 |

---

## 九、故障排查命令速查

```bash
# 检查 Mock 协和
curl -X POST http://127.0.0.1:8001/mcp -H "Content-Type: application/json" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}"

# 检查后端
curl http://127.0.0.1:8000/api/health

# 清 Python 缓存（修改代码后重启之前用）
find . -name "__pycache__" -type d -exec rm -rf {} +

# 看后端日志
tail -50 runtime\api_server.err.log

# 看 Mock 协和日志
tail -20 runtime\mock_hospital.log
```

---

更新于 2026-06-07
```
