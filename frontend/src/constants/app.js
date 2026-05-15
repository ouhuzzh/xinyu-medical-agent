export const THREAD_KEY = "medical_assistant_thread_id";
export const AUTH_TOKEN_KEY = "medical_assistant_auth_token";

export const STARTER_PROMPTS = [
  { key: "hypertension", text: "高血压应该注意什么？" },
  { key: "triage", text: "我咳嗽三天了，挂什么科？" },
  { key: "booking", text: "我想挂呼吸内科的号" },
  { key: "cancel", text: "取消刚才的预约" },
];

export const EMPTY_STATE_CAPABILITIES = [
  "解答医学常识与健康咨询",
  "智能分诊推荐科室",
  "预约挂号与取消",
  "查询医院信息与就医指引",
];

export function statusTone(value) {
  if (["ready", "completed"].includes(value)) return "good";
  if (["failed", "error"].includes(value)) return "bad";
  if (["no_documents", "pending_rebuild"].includes(value)) return "warn";
  return "info";
}
