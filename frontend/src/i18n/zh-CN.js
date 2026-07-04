/**
 * Chinese (Simplified) locale
 */
export default {
  // App
  "app.title": "心语医疗小助手",
  "app.subtitle": "医疗咨询与预约挂号",

  // Sidebar
  "sidebar.nav.chat": "聊天咨询",
  "sidebar.nav.documents": "知识库文档",
  "sidebar.status.system": "系统状态",
  "sidebar.status.knowledgeBase": "知识库",
  "sidebar.status.refresh": "刷新状态",
  "sidebar.gradioAdmin": "Gradio 后台",
  "sidebar.clearChat": "清空会话",
  "sidebar.closeMenu": "关闭侧边栏",
  "sidebar.theme.light": "亮色",
  "sidebar.theme.dark": "暗色",

  // Chat Header
  "chat.header.eyebrow": "心语医疗 AI",
  "chat.header.title": "直接说你的问题",
  "chat.header.connected": "后端已连接",
  "chat.header.disconnected": "后端连接失败",

  // Stream states
  "stream.connecting": "连接中",
  "stream.thinking": "思考中",
  "stream.generating": "生成中",
  "stream.stopped": "已停止",
  "stream.error": "需重试",
  "stream.done": "已完成",
  "stream.idle": "待命",

  // Composer
  "composer.placeholder": "输入症状、医学问题或挂号需求… (Shift+Enter 换行)",
  "composer.placeholder.thinking": "AI 正在思考…",
  "composer.placeholder.generating": "AI 正在回复…",
  "composer.send": "发送",
  "composer.stop": "停止生成",

  // Messages
  "message.ai.label": "心语医疗 AI",
  "message.user.label": "我",
  "message.copy": "复制",
  "message.copied": "已复制",
  "message.copyReply": "复制回复",
  "message.interrupted": "已停止",
  "message.thinking": "AI 正在思考",

  // Actions
  "action.confirmBooking": "确认预约",
  "action.confirmCancel": "确认取消",

  // Empty state
  "empty.title": "你好，我是心语医疗小助手",
  "empty.subtitle": "专业的 AI 医疗咨询助手，随时为您提供健康指导与就医帮助",
  "empty.disclaimer": "本助手仅提供健康参考，不能替代专业医生的诊断与治疗",

  // Capabilities
  "cap.medicalQA": "解答医学常识与健康咨询",
  "cap.triage": "智能分诊推荐科室",
  "cap.booking": "预约挂号与取消",
  "cap.hospitalInfo": "查询医院信息与就医指引",

  // Starter prompts
  "prompt.hypertension": "高血压应该注意什么？",
  "prompt.triage": "我咳嗽三天了，挂什么科？",
  "prompt.booking": "我想挂呼吸内科的号",
  "prompt.cancel": "取消刚才的预约",

  // Documents page
  "docs.header.eyebrow": "心语医疗 · 知识库",
  "docs.header.title": "知识库文档",
  "docs.header.desc": "这里先提供用户友好的状态、上传和官方同步入口；高级诊断继续放在 Gradio 后台。",
  "docs.refresh": "刷新",
  "docs.status.label": "知识库状态",
  "docs.status.defaultMsg": "正在读取知识库状态。",
  "docs.upload.title": "上传本地资料",
  "docs.upload.desc": "支持 Markdown、PDF、Office、HTML 等格式。上传后会自动转换并同步到知识库。",
  "docs.upload.button": "选择文件",
  "docs.upload.dragHint": "拖拽文件到此处",
  "docs.upload.dragActive": "松开即可上传",
  "docs.sync.title": "同步官方资料",
  "docs.sync.desc": "手动拉取官方来源的当前资料，适合面试演示「知识库可更新化」。",
  "docs.sync.button": "同步",
  "docs.sources.title": "官方来源覆盖度",
  "docs.sources.count": "{count} 个来源",
  "docs.documents.title": "已同步文档",
  "docs.documents.count": "{count} 个文件",
  "docs.documents.empty": "还没有可展示的本地 Markdown 文档，可以先上传或同步官方资料。",
  "docs.tasks.title": "最近同步任务",
  "docs.tasks.count": "{count} 条",
  "docs.tasks.empty": "暂无同步任务记录。",
  "docs.sources.empty": "暂时无法读取官方来源覆盖度。",

  // Search
  "search.placeholder": "搜索聊天记录…",
  "search.noResults": "无结果",
  "search.prev": "上一个",
  "search.next": "下一个",
  "search.close": "关闭",

  // Export
  "export.markdown": "导出 Markdown",
  "export.json": "导出 JSON",
  "export.title": "心语医疗小助手 · 对话记录",

  // Dialog
  "dialog.clear.title": "清空会话",
  "dialog.clear.body": "确定要清空当前所有对话记录吗？此操作无法撤销。",
  "dialog.clear.cancel": "取消",
  "dialog.clear.confirm": "确认清空",
  "dialog.compress.title": "压缩上下文",
  "dialog.compress.body": "将较早的消息压缩成摘要吗？最近的几轮对话仍会保留可见。",
  "dialog.compress.cancel": "取消",
  "dialog.compress.confirm": "压缩",
  "chat.header.compress": "压缩上下文",

  // Error
  "error.connection": "无法连接后端服务，请确认 FastAPI 已启动。",
  "error.history": "历史会话暂时无法读取。",
  "error.chatUnavailable": "聊天服务暂时不可用。",
  "error.chatDisconnected": "聊天连接中断，请稍后重试。",
  "error.clearFailed": "清空会话失败，请稍后再试。",
  "error.docsRead": "知识库信息暂时无法读取。",
  "error.docsUpload": "文档上传失败。",
  "error.docsSync": "官方资料同步失败。",
  "error.boundary.title": "页面出了点问题",
  "error.boundary.desc": "应用遇到了一个意外错误，请尝试重新加载。如果问题持续，请刷新页面。",
  "error.boundary.retry": "重试",
  "error.boundary.reload": "刷新页面",

  // Loading
  "loading": "加载中…",

  // Status tones
  "status.ready": "就绪",
  "status.completed": "已完成",
  "status.failed": "失败",
  "status.error": "错误",
  "status.no_documents": "无文档",
  "status.pending_rebuild": "等待重建",
  "status.preparing": "准备中",
  "status.not_checked": "未检查",

  // Keyboard shortcuts
  "shortcuts.focusInput": "聚焦输入框",
  "shortcuts.copyReply": "复制最后回复",
  "shortcuts.search": "搜索",
  "shortcuts.export": "导出",
  "shortcuts.stop": "停止生成",
};
