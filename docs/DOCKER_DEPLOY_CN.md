# Docker 本机部署指南

这套 Docker 配置用于本机或内网 staging 试部署，包含：

- PostgreSQL + pgvector
- Redis
- FastAPI 后端
- Vite 前端静态站点（nginx）

## 1. 准备配置

复制示例环境文件：

```powershell
Copy-Item .env.docker.example .env.docker.local
```

编辑 `.env.docker.local`，至少按需填写：

```text
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
```

第一次只是验证容器、数据库、Redis 和页面能跑起来，可以先不填模型 key，但真实聊天会失败或不可用。

## 2. 启动

```powershell
docker compose --env-file .env.docker.local up --build
```

启动后访问：

```text
前端：http://localhost:8080
后端：http://localhost:8000
```

默认前端会使用 `demo-admin-token` 调后端。这个配置只适合本地开发和 staging。
如果你改了 `API_AUTH_TOKENS_JSON`，也要同步设置 `API_HEALTHCHECK_TOKEN` 和 `VITE_API_AUTH_TOKEN`。

## 3. 健康检查

```powershell
curl -H "Authorization: Bearer demo-admin-token" http://localhost:8000/api/health
curl -H "Authorization: Bearer demo-admin-token" http://localhost:8000/api/system/status
```

也可以查看容器状态：

```powershell
docker compose ps
docker compose logs -f api
```

## 4. 停止与清理

停止容器：

```powershell
docker compose down
```

如果要连数据库数据也清掉：

```powershell
docker compose down -v
```

## 5. 生产上线前必须修改

不要直接把默认配置暴露到公网。生产环境至少要改：

- `APP_ENV=production`
- 强随机 `JWT_SECRET_KEY`
- 强随机 `MCP_TOKEN_ENCRYPTION_KEYS`
- 真实 `API_CORS_ORIGINS`
- 移除 demo `API_AUTH_TOKENS_JSON`
- 使用托管 PostgreSQL/Redis 或加持久化备份
- 前面接 HTTPS 反向代理
- 明确 MCP 医院服务地址、工具 mapping、医院 alias 审核流程

当前 compose 更适合“先跑起来用一用”，不是最终生产拓扑。

## 6. 生产部署骨架

生产环境使用单独的 compose 文件：

```bash
cp .env.docker.prod.example .env.docker.prod.local
```

编辑 `.env.docker.prod.local`，至少替换：

```text
APP_DOMAIN=你的前端域名
API_DOMAIN=你的 API 域名
PUBLIC_API_BASE_URL=https://你的 API 域名
JWT_SECRET_KEY=强随机字符串
CHECKPOINT_SIGNING_KEY=另一个强随机字符串
POSTGRES_PASSWORD=强数据库密码
MCP_TOKEN_ENCRYPTION_KEYS=生产 Fernet key
DEEPSEEK_API_KEY=真实 key
OPENAI_API_KEY=真实 key
```

启动生产拓扑：

```bash
docker compose --env-file .env.docker.prod.local -f docker-compose.prod.yml up --build -d
```

生产拓扑和本机拓扑的区别：

- 只暴露 `80/443`，由 Caddy 自动处理 HTTPS。
- PostgreSQL 和 Redis 不映射到公网端口。
- 前端生产构建不再默认使用 `demo-admin-token`。
- API healthcheck 使用公开的 `/api/healthz`，不依赖 demo token。
- 后端镜像默认不安装 `torch/sentence-transformers/transformers`；只有本地 HuggingFace embedding 才设置 `INSTALL_LOCAL_ML=true`。

建议域名：

```text
https://medical.example.com      前端
https://api.medical.example.com  后端 API
```

## 7. 数据库备份

生产服务器上可以执行：

```bash
sh scripts/backup_postgres.sh .env.docker.prod.local
```

备份会写入：

```text
backups/postgres/postgres-YYYYMMDD-HHMMSS.sql.gz
```

`backups/` 已加入 `.gitignore`，不要把备份文件提交到仓库。第一次正式上线前，务必做一次恢复演练。
