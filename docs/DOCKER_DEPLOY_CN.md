# Docker 本机部署指南

这套 Docker 配置用于本机或内网 staging 试部署，包含：

- PostgreSQL + pgvector
- Redis
- FastAPI 后端
- 知识库后台 Worker
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
docker compose logs -f worker
```

如需自动补建知识库并按计划同步官方来源，在环境文件中设置：

```text
AUTO_BOOTSTRAP_KNOWLEDGE_BASE=true
ENABLE_KB_SYNC_SCHEDULER=true
KB_SYNC_INTERVAL_HOURS=24
```

这些开关只交给 `worker` 容器执行；API 容器不会启动同类后台线程。因此增加 API
副本不会重复触发定时同步。多个 Worker 意外同时运行时，PostgreSQL advisory lock
会阻止同一知识库任务并发执行。

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

生成生产密钥：

```bash
python scripts/generate_prod_secrets.py
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

启动前校验配置：

```bash
python scripts/validate_prod_env.py .env.docker.prod.local
```

再做一次生产主机预检：

```bash
python scripts/check_prod_host.py .env.docker.prod.local
```

它会额外检查这些上线前常见坑：

- `docker` / `docker compose` 是否可用
- Docker daemon 是否正常
- `docker-compose.prod.yml` 和 `deploy/Caddyfile` 是否存在
- `80/443` 端口是否已被别的进程占用
- `APP_DOMAIN` / `API_DOMAIN` 是否已经能解析
- 当前部署盘的剩余空间是否过低

如果你只是提前在本机演练，也可以跳过部分项：

```bash
python scripts/check_prod_host.py .env.docker.prod.local --skip-dns --skip-ports
```

启动生产拓扑：

```bash
docker compose --env-file .env.docker.prod.local -f docker-compose.prod.yml up --build -d
```

或者直接用一键脚本：

```bash
python scripts/deploy_prod_stack.py .env.docker.prod.local
```

它会按顺序执行：

- 校验 `.env.docker.prod.local`
- 生产主机预检
- `docker compose ... up --build -d`
- 对前端和 `/api/healthz` 做冒烟检查

如果你已经确认域名还没切 DNS，或者服务器上 80/443 端口检查不适合当前场景，也可以临时跳过：

```bash
python scripts/deploy_prod_stack.py .env.docker.prod.local --skip-dns
python scripts/deploy_prod_stack.py .env.docker.prod.local --skip-ports
python scripts/deploy_prod_stack.py .env.docker.prod.local --skip-preflight
```

启动后冒烟检查：

```bash
FRONTEND_URL=https://medical.example.com \
API_BASE_URL=https://api.medical.example.com \
python scripts/smoke_docker_deploy.py
```

如果你保留了管理员 API token，也可以检查系统状态：

```bash
API_AUTH_TOKEN=你的管理员token python scripts/smoke_docker_deploy.py
```

如果你要做更接近真实上线的验收，可以用：

```bash
python scripts/prod_acceptance_check.py \
  --frontend-url https://medical.example.com \
  --api-base-url https://api.medical.example.com
```

如果要连带鉴权接口和真实聊天链路一起验收：

```bash
API_AUTH_TOKEN=你的token \
python scripts/prod_acceptance_check.py \
  --frontend-url https://medical.example.com \
  --api-base-url https://api.medical.example.com \
  --chat-smoke
```

生产拓扑和本机拓扑的区别：

- 只暴露 `80/443`，由 Caddy 自动处理 HTTPS。
- PostgreSQL 和 Redis 不映射到公网端口。
- 前端生产构建不再默认使用 `demo-admin-token`。
- API healthcheck 使用公开的 `/api/healthz`，不依赖 demo token。
- 自动知识库补建和定时同步由独立 `worker` 容器执行，API 只处理交互请求。
- 后端镜像默认安装 `requirements-api.txt`，不安装 `requirements-ml-local.txt` 里的 `torch/sentence-transformers/transformers`；只有本地 HuggingFace embedding 才设置 `INSTALL_LOCAL_ML=true`。

建议域名：

```text
https://medical.example.com      前端
https://api.medical.example.com  后端 API
```

建议首次正式上线前做一次最小生产演练：

1. 在服务器上复制 `.env.docker.prod.example` 为 `.env.docker.prod.local`
2. 用 `python scripts/generate_prod_secrets.py` 生成密钥并填入
3. 用 `python scripts/validate_prod_env.py .env.docker.prod.local` 校验
4. 用 `python scripts/deploy_prod_stack.py .env.docker.prod.local` 启动并冒烟
5. 登录页面、发一条真实聊天、查看 `/api/system/status`

更完整的真实服务器演练步骤，可以看：

- [真实服务器上线清单](./PRODUCTION_ROLLOUT_CHECKLIST_CN.md)

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

恢复时不要直接覆盖生产库。建议先在临时服务器或临时数据库验证：

```bash
gunzip -c backups/postgres/postgres-YYYYMMDD-HHMMSS.sql.gz | \
  docker compose --env-file .env.docker.prod.local -f docker-compose.prod.yml exec -T postgres \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

也可以使用脚本恢复：

```bash
sh scripts/restore_postgres.sh backups/postgres/postgres-YYYYMMDD-HHMMSS.sql.gz .env.docker.prod.local --yes
```

真实生产恢复前，要先停止 API 写入流量，并确认当前库已经另存一份备份。

## 8. 用户记忆加密巡检

如果你之前在缺少 `cryptography` 或错误加密 key 的环境里运行过服务，可能会留下历史坏密文或哨兵值。现在可以用下面的脚本先做只读巡检：

```bash
python scripts/repair_user_memory_encryption.py
```

如果输出里存在：

- `plaintext`
- `encrypted_invalid_format`
- `encrypted_unreadable`

说明库里还有需要处理的历史记录。

只把旧明文重写为加密内容：

```bash
python scripts/repair_user_memory_encryption.py --apply --reencrypt-plaintext
```

把坏格式密文重写成哨兵值（避免持续刷日志）：

```bash
python scripts/repair_user_memory_encryption.py --apply
```

`--rewrite-unreadable` 风险更高，因为它会把“当前 key 解不开”的密文直接改成哨兵值，只有在你确认那部分内容已经不可恢复时再用。
