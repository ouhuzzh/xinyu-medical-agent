# 医院 MCP 工具接入契约

本文档定义医院 MCP server 接入本项目时的工具命名、参数和安全元数据约定。

## 命名规则

医院 MCP server 暴露原始工具名，不要自行添加医院前缀。平台加载工具时会自动加上 `hospital_code__` 前缀，例如：

```text
search_doctors -> xiehe__search_doctors
book_appointment -> xiehe__book_appointment
```

`hospital_code` 来自平台 `hospitals.code`，工具命名空间分隔符由 `MCP_TOOL_NAMESPACE_SEPARATOR` 配置，默认是 `__`。

## 推荐标准工具

预约域推荐实现以下工具名。平台保留 alias 适配能力，但新接入医院应优先使用标准名。

| 标准工具 | 类型 | 说明 |
| --- | --- | --- |
| `search_doctors` | read | 查询科室医生 |
| `search_schedules` | read | 查询号源/排班 |
| `book_appointment` | write | 提交挂号 |
| `list_appointments` | read | 查询用户已有预约 |
| `cancel_appointment` | write | 取消预约 |
| `reschedule_appointment` | write | 改约 |

## 非标准工具名映射

如果医院已有 MCP server 无法使用标准工具名，不要在业务代码里继续增加硬编码分支。应通过 `MCP_APPOINTMENT_TOOL_MAPPING` 配置声明“平台标准动作 -> 医院实际工具名”。

示例：

```json
{
  "xiehe": {
    "search_doctors": "query_doctor",
    "search_schedules": "get_availability",
    "book_appointment": "create_registration_order",
    "cancel_appointment": "cancel_registration"
  },
  "ruijin": {
    "search_schedules": "query_registration_slots",
    "book_appointment": "submit_visit_order"
  }
}
```

平台会在工具加载后自动添加医院命名空间，所以配置里推荐填写医院 MCP server 的原始工具名，例如 `create_registration_order`，不需要写 `xiehe__create_registration_order`。

解析优先级：

1. 每家医院的显式 mapping。
2. 标准工具名精确匹配。
3. 旧 alias 精确匹配。

第三步只是兼容兜底，不再使用 substring 模糊匹配，避免 `cancel` 误匹配到无关工具。

## 多医院选择规则

用户同时绑定多家医院时，预约流程必须先确定医院，再查询号源或提交挂号。

选择规则：

1. 用户明确说医院 code、完整医院名或配置 alias 时，使用该医院。
2. 用户只绑定一家可用医院时，自动使用该医院。
3. 用户绑定多家医院但没有明确说明时，先向用户澄清，不查询号源、不提交挂号。
4. 预约预览生成后，`hospital_code` 和 `hospital_name` 会写入待确认 payload。用户确认时继续使用同一家医院。
5. 如果用户选择的医院没有对应工具，直接提示该医院暂不支持当前操作，不 fallback 到其他医院。

匹配规则刻意保守：

- 英文 `code` 或英文 alias 必须有词边界，例如 `xiehe` 可以匹配，`xiehe2` 不匹配。
- 完整中文医院名、带 `医院` / `院区` / `门诊部` / `医学中心` 的 alias，或至少 4 个汉字的中文 alias，可以在用户句子中匹配。
- 1-3 个汉字的短 alias 如果出现在长句里，只能进入医院确认，例如“你说的医院我理解为：协和医院，请回复确认医院”；用户回复内容完全等于该 alias 时，可视为明确选择。
- 不会自动把“协和医院”裁剪成“协和”参与匹配；短 alias 必须由平台显式配置。
- 同一句里命中多家医院时，必须继续澄清，不自动选择第一家。

医院别名可通过 `MCP_HOSPITAL_ALIASES` 配置：

```json
{
  "xiehe": ["协和", "北京协和", "PUMCH"],
  "ruijin": ["瑞金", "上海瑞金医院"]
}
```

这些 alias 只用于把用户自然语言映射到已绑定医院 code。最终执行前仍必须得到确定的 `hospital_code`。

## 参数约定

```text
search_doctors(department, date?, time_slot?)
search_schedules(department?, date?, time_slot?, doctor_name?)
book_appointment(department, date, time_slot, doctor_name?, idempotency_key?)
list_appointments()
cancel_appointment(appointment_id?, appointment_no?, idempotency_key?)
reschedule_appointment(appointment_id?, appointment_no?, date, time_slot, doctor_name?, idempotency_key?)
```

日期使用 `YYYY-MM-DD`，时段建议使用 `morning`、`afternoon`、`evening` 或医院明确返回的稳定枚举。

## 安全元数据

每个 MCP 工具应声明安全元数据。平台会优先读取 `annotations` 或 `metadata` 中的字段；没有元数据时才退回工具名规则。

读操作示例：

```json
{
  "name": "search_schedules",
  "annotations": {
    "readOnlyHint": true,
    "domain": "appointment",
    "effect": "read"
  }
}
```

写操作示例：

```json
{
  "name": "book_appointment",
  "annotations": {
    "readOnlyHint": false,
    "destructiveHint": false,
    "domain": "appointment",
    "effect": "write",
    "requires_confirmation": true
  }
}
```

平台通用 MCP skill 不会执行 `requires_confirmation=true`、`destructiveHint=true`、或预约域写操作工具。这类工具只能通过专门业务流程执行，例如预约状态机先生成预览，再等待用户明确回复“确认预约”或“确认取消”。

## 幂等要求

`book_appointment`、`cancel_appointment`、`reschedule_appointment` 必须支持 `idempotency_key`。同一个 key 的重复请求应返回同一个结果，不能重复挂号、重复取消或重复改约。

## 接入校验

每家医院上线前至少通过以下 contract test：

1. 工具能被 MCP client 发现。
2. 标准工具或映射后的工具齐全。
3. `inputSchema` 和返回结构稳定。
4. 查询类工具标记为 read-only。
5. 写操作标记 `requires_confirmation=true`。
6. 写操作支持 `idempotency_key`，重复提交不会产生重复预约。
