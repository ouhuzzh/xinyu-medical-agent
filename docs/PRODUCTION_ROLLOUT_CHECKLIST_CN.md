# 真实服务器上线清单

这份清单不是讲 Docker 原理，而是帮你把第一次真实上线拆成一串可执行动作。目标是两件事：

- 先把服务稳定地跑起来
- 再把“出了问题怎么回滚、怎么恢复”也提前准备好

建议第一次正式上线时，按下面顺序逐项勾掉，不要跳步。

## 1. 服务器准备

推荐最小配置：

- Linux 云服务器 2 vCPU / 4 GB RAM 起步
- 系统盘至少 40 GB
- 公网 IP 1 个
- Ubuntu 22.04 LTS 或同级别常见发行版

上线前要确认：

- 安全组或防火墙放行 `22`、`80`、`443`
- 机器上没有别的 Web 服务占用 `80/443`
- 时区、系统时间正常
- 已安装 Docker 和 Docker Compose

可以先在服务器仓库目录执行：

```bash
python scripts/check_prod_host.py .env.docker.prod.local
```

## 2. 域名和 DNS

建议准备两个域名：

- `medical.your-domain.com` 给前端
- `api.your-domain.com` 给后端

DNS 至少要做：

- `medical.your-domain.com -> 服务器公网 IP`
- `api.your-domain.com -> 服务器公网 IP`

等 DNS 生效后，再跑：

```bash
python scripts/check_prod_host.py .env.docker.prod.local
```

如果 DNS 还没切，但你想先在服务器内网做演练，可以临时：

```bash
python scripts/check_prod_host.py .env.docker.prod.local --skip-dns
```

## 3. 仓库与配置

服务器上准备代码：

```bash
git clone 你的仓库地址
cd agentic-rag-for-dummies
```

复制生产环境文件：

```bash
cp .env.docker.prod.example .env.docker.prod.local
```

生成生产密钥：

```bash
python scripts/generate_prod_secrets.py
```

必须替换这些关键值：

- `APP_DOMAIN`
- `API_DOMAIN`
- `PUBLIC_API_BASE_URL`
- `JWT_SECRET_KEY`
- `CHECKPOINT_SIGNING_KEY`
- `POSTGRES_PASSWORD`
- `MCP_TOKEN_ENCRYPTION_KEYS`
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`

上线前再跑一次：

```bash
python scripts/validate_prod_env.py .env.docker.prod.local
```

## 4. 首次启动

推荐直接用一键脚本：

```bash
python scripts/deploy_prod_stack.py .env.docker.prod.local
```

它会自动做：

1. env 校验
2. 主机预检
3. `docker compose up -d --build`
4. 基础冒烟检查

如果只是演练、域名还没切好，可以暂时：

```bash
python scripts/deploy_prod_stack.py .env.docker.prod.local --skip-dns
```

## 5. 上线验收

容器起来以后，先做最基本的站点验收：

```bash
python scripts/prod_acceptance_check.py \
  --frontend-url https://medical.your-domain.com \
  --api-base-url https://api.your-domain.com
```

如果你手里有管理员或用户 token，还可以做带鉴权验收：

```bash
API_AUTH_TOKEN=你的token \
python scripts/prod_acceptance_check.py \
  --frontend-url https://medical.your-domain.com \
  --api-base-url https://api.your-domain.com
```

如果你想让它真的走一遍聊天链路，再加：

```bash
API_AUTH_TOKEN=你的token \
python scripts/prod_acceptance_check.py \
  --frontend-url https://medical.your-domain.com \
  --api-base-url https://api.your-domain.com \
  --chat-smoke
```

这个聊天冒烟会：

- 创建会话
- 调 `/api/chat/stream`
- 检查 SSE 里是否有 `final`
- 检查是否出现 `app-error`

## 6. 数据与恢复演练

正式开放给用户前，至少做一次备份和恢复演练。

备份：

```bash
sh scripts/backup_postgres.sh .env.docker.prod.local
```

恢复到测试环境或临时库：

```bash
sh scripts/restore_postgres.sh backups/postgres/你的备份.sql.gz .env.docker.prod.local --yes
```

注意：

- 不要第一次恢复就直接打生产库
- 恢复前先确认当前生产库已经有新备份
- 恢复期间要停掉 API 写入流量

## 7. 上线后第一天重点盯什么

至少盯这些：

- `docker compose ps`
- `docker compose logs -f api`
- `/api/healthz`
- `/api/system/status`
- 首条真实聊天的耗时和是否报错
- 知识库状态是否 `ready`
- `schema_health.status` 是否 `ok`

如果启用了 MCP 医院挂号，还要额外确认：

- 只会调用用户已绑定医院
- 多医院时会先澄清医院
- 不会因为工具名顺序误挂别家医院

## 8. 回滚思路

这套项目当前更适合用“代码版本回滚 + 容器重启”的方式快速止损。

推荐做法：

1. 保留上一版可运行 commit
2. 每次上线前先做数据库备份
3. 如果新版本出问题，先回滚代码版本
4. `docker compose ... up -d --build` 重起上一版
5. 如果数据也被破坏，再走恢复流程

不要把“代码回滚”和“数据库恢复”混成一步。多数线上问题先回滚代码就够了，数据库恢复是更重的动作。

## 9. 一次完整演练的最低标准

如果下面这几项都完成了，才算真正具备第一次上线条件：

- 生产 env 已替换所有 placeholder
- `check_prod_host.py` 通过
- `deploy_prod_stack.py` 通过
- `prod_acceptance_check.py` 通过
- 做过一次备份
- 做过一次恢复演练
- 至少做过一次真实聊天冒烟
- 明确谁来盯首日日志和告警
