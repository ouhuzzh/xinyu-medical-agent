# MCP 连接池重构方案

## 目标

将 per-user per-hospital 的连接管理，改为 hospital 维度的连接池共享。

## 现状 vs 目标

```
【现状】每个用户独立管理
  UserA → [协和连接 + 熔断器] [仁济连接 + 熔断器]
  UserB → [协和连接 + 熔断器] [仁济连接 + 熔断器]
  UserC → [协和连接 + 熔断器] [仁济连接 + 熔断器]
  问题：1万个用户 = 1万个协和熔断器 + 1万个连接槽位

【目标】医院维度统一管理
  协和 → [连接池(50并发槽位) + 全局熔断器]
           ↑       ↑        ↑
         UserA   UserB    UserC  (共享)
  仁济 → [连接池(50并发槽位) + 全局熔断器]
           ↑       ↑        ↑
         UserA   UserB    UserC  (共享)
```

---

## 新增模块

### 1. `HospitalConnPool` — 医院级连接池

新建文件：`project/mcp_integration/hospital_conn_pool.py`

```python
class HospitalConnPool:
    """一家医院的共享连接池。控制并发数、记录健康状态。"""

    def __init__(self, hospital_code: str, hospital_info: dict,
                 max_concurrency: int = 50):
        self.code = hospital_code
        self.url = hospital_info["mcp_url"]
        self.name = hospital_info.get("name", hospital_code)
        self.breaker = CircuitBreaker(failure_threshold=3,
                                       recovery_timeout=120)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tool_cache: List[Any] = []    # 医院提供的工具列表（缓存）
        self._tools_loaded_at: float = 0.0  # 工具列表加载时间

    async def load_tools(self) -> List[Any]:
        """加载该医院提供的工具列表（只执行一次，结果缓存）。"""
        # TODO: 实现
        # 1. 检查缓存是否在有效期内
        # 2. 如果过期，用 client.session(code) 重新加载
        # 3. 返回工具列表（不带 namespace，由上层加）
        pass

    async def call_tool(self, token: str, tool_name: str,
                        arguments: dict) -> Any:
        """用指定用户的 token 调用该医院的某个工具。"""
        # TODO: 实现
        # 1. 获取 semaphore 槽位
        # 2. 创建 client.session(code)，headers 里带用户 token
        # 3. 调用工具
        # 4. 返回结果 + 释放槽位
        pass
```

**接口设计：**

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `load_tools()` | 无 | `List[Any]` | 加载医院工具描述，全局缓存。参数：`MCP_TOOLS_CACHE_TTL`（新增配置，默认 300s） |
| `call_tool(token, tool_name, arguments)` | 用户token + 工具名 + 参数 | `Any` | 真正执行工具调用。每个调用走独立 MCP session（带用户 token） |

---

### 2. `GlobalMCPManager` — 全局协调器

新建文件：`project/mcp_integration/global_mcp_manager.py`

```python
class GlobalMCPManager:
    """全局 MCP 管理：维护医院连接池 + 用户工具视图。"""

    def __init__(self, registry: MCPServerRegistry,
                 credential_store: UserMCPCredentialStore,
                 max_concurrency_per_hospital: int = 50):
        self._registry = registry
        self._store = credential_store
        self._max_concurrency = max_concurrency_per_hospital
        self._pools: Dict[str, HospitalConnPool] = {}  # code -> pool
        self._lock = threading.Lock()

    def get_tools_for_user(self, user_id: str) -> List[Any]:
        """返回该用户绑定的医院能用的所有工具列表（带 namespace）。"""
        # TODO: 实现
        # 1. 从 credential_store 获取用户绑定的医院列表
        # 2. 对每家医院，检查熔断器状态（跳过熔断的）
        # 3. 调用 pool.load_tools() 获取工具列表（利用缓存）
        # 4. 给每个工具加 namespace 前缀 + 医院名描述
        # 5. 返回汇总的工具列表

        pass

    def call_tool_for_user(self, user_id: str, tool_name: str,
                           arguments: dict) -> Any:
        """调用工具。自动解析 namespace 找到对应医院。"""
        # TODO: 实现
        # 1. 从 tool_name 解析出 hospital_code（按 _NAMESPACE_SEP 分割）
        # 2. 检查熔断器状态
        # 3. 获取用户在该医院的 token
        # 4. 调用 pool.call_tool(token, raw_tool_name, arguments)
        # 5. 根据结果更新熔断器状态
        pass

    def invalidate_user(self, user_id: str):
        """用户凭证变更时调用，清理用户缓存。"""
        # TODO: 实现
        # 只需清理用户维度的工具缓存即可
        # 医院维度的工具列表和连接池不受影响
        pass
```

**与现有 `UserMCPPool` 的 API 兼容性：**

| 现有方法 | 新方法 | 兼容？ |
|----------|--------|--------|
| `get_tools_for_user(user_id)` | `get_tools_for_user(user_id)` | ✅ 签名一致 |
| `get_connected_hospitals(user_id)` | `get_tools_for_user` 内部可推导 | ⚠️ 需新增适配 |
| `get_failed_hospitals(user_id)` | 从熔断器状态获取 | ⚠️ 需新增适配 |
| `invalidate(user_id)` | `invalidate_user(user_id)` | ✅ |
| `get_status_summary(user_id)` | `get_status_summary(user_id)` | ⚠️ 需新增适配 |

> 建议：新 `GlobalMCPManager` 作为 `UserMCPPool` 的**内部替换**，对外 API 保持一致。这样可以不改调用方代码。

---

## 改造步骤（建议按顺序）

### Step 1：新增配置项

文件：`project/config.py`

```python
# 新增配置
MCP_HOSPITAL_MAX_CONCURRENCY = int(os.environ.get("MCP_HOSPITAL_MAX_CONCURRENCY", "50"))
MCP_TOOLS_CACHE_TTL = int(os.environ.get("MCP_TOOLS_CACHE_TTL", "300"))  # 秒
```

### Step 2：实现 `HospitalConnPool`

新文件：`project/mcp_integration/hospital_conn_pool.py`

要处理的问题：
- **工具加载**：第一次 `load_tools()` 时加载并缓存，后续直接返回缓存
- **并发控制**：`asyncio.Semaphore` 控制同时执行的工具调用数
- **熔断**：`CircuitBreaker` 实例在此层管理，调用成功/失败后更新
- **超时**：每次 `call_tool` 用 `asyncio.wait_for` 或 `asyncio.timeout` 设超时

### Step 3：实现 `GlobalMCPManager`

新文件：`project/mcp_integration/global_mcp_manager.py`

要处理的问题：
- **工具 namespace**：在 `get_tools_for_user()` 里给工具名加 `{hospital_code}__` 前缀
- **熔断检查**：在加载工具和调用工具前检查熔断器
- **用户 token**：从 `credential_store` 获取，每次调用时传入 `call_tool()`
- **线程安全**：`self._lock` 保护 `self._pools` 字典的并发访问

### Step 4：替换 `UserMCPPool`

文件：`project/mcp_integration/user_mcp_pool.py`

两种策略任选：

**策略 A（推荐）**：不改 `UserMCPPool` 的公开 API，把内部实现替换为调用 `GlobalMCPManager`。

```python
class UserMCPPool:
    def __init__(self, ...):
        self._global = GlobalMCPManager(registry, store)
    
    def get_tools_for_user(self, user_id):
        return self._global.get_tools_for_user(user_id)
    
    # ... 其他方法同理
```

**策略 B**：直接删掉 `UserMCPPool`，全局用 `GlobalMCPManager` 替换，改所有调用方。

> 推荐策略 A，改动面最小。

### Step 5：测试

- **单元测试**：`HospitalConnPool` 的并发上限（用 asyncio.gather 发 60 个请求，验证第 51 个排队）
- **单元测试**：熔断器独立工作（A 医院挂不影响 B 医院）
- **集成测试**：用户 A 和用户 B 绑定同一家医院，验证连接池复用
- **回归测试**：用现有的 `test_user_mcp_pool.py` 跑一遍，确保旧功能不受影响

---

## 关键设计决策

### Q1：工具列表缓存多久刷新一次？

**答**：`MCP_TOOLS_CACHE_TTL` 秒（默认 300s）。每次 `load_tools()` 检查时间戳，过期重新加载。这样医院新增工具 5 分钟后就能被用户发现。

### Q2：用户 A 的 token 和用户 B 的 token 不同，怎么共享连接？

**答**：不共享 TCP 连接，只共享并发槽位（Semaphore）。`MultiServerMCPClient` 的 `session()` 每次调用都是独立 session，token 在 header 里区别。

```
协和连接池（Semaphore: 50）
  UserA.call_tool(token_A, "book", {...}) → 占 1 个槽位 → 完释放
  UserB.call_tool(token_B, "query", {...}) → 占 1 个槽位 → 完释放
  两个请求并行执行，token 互不干扰
```

### Q3：熔断器下沉到全局后，用户 A 的失败会影响用户 B 吗？

**答**：会。协和 MCP 如果挂了，用户 A 的调用失败 → 全局熔断器记 1 次失败，用户 B 再调也失败 → 累积到 3 次 → 熔断。**这是合理的**——协和是同一台服务器，如果挂了，所有用户的请求都会失败。早点熔断避免浪费。

### Q4：`_UserPool` 还要不要保留？

**答**：可以精简。`_UserPool` 里现在存的东西：
- `breakers` → 移到 `HospitalConnPool`（全局）
- `tools` → 由 `GlobalMCPManager.get_tools_for_user()` 按需计算
- `connected_hospitals` / `failed_hospitals` → 从熔断器状态推导
- `built_at` → 工具缓存 TTL 替代

如果用户维度的数据不再需要持久化，`_UserPool` 可以删掉。

---

## 边界情况

1. **用户绑定 0 家医院**：`get_tools_for_user()` 返回空列表
2. **医院在 registry 里但 inactive**：跳过该医院
3. **熔断器 open**：`get_tools_for_user()` 不返回该医院的工具，`call_tool()` 直接抛异常
4. **用户 token 过期**：MCP 调用返回 401 → 记入失败次数 → 可能触发熔断
5. **Semaphore 全部占用**：新请求在 `await semaphore.acquire()` 处排队，不会报错
6. **embedding 服务 / LLM 服务挂**：不受影响，MCP 连接池不依赖这些服务
