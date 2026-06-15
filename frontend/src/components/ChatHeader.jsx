import React from "react";
import { Menu, Search, Download, ShieldCheck, Sparkles } from "lucide-react";

const STATE_MAP = {
  connecting: { text: "连接中", variant: "info" },
  thinking: { text: "思考中", variant: "thinking" },
  generating: { text: "生成中", variant: "generating" },
  stopped: { text: "已停止", variant: "warn" },
  error: { text: "需重试", variant: "error" },
  done: { text: "已完成", variant: "done" },
  idle: { text: "待命", variant: "idle" },
};

const ChatHeader = React.memo(function ChatHeader({
  threadId,
  isConnected,
  streamState,
  onMenuClick,
  onOpenSearch,
  onExport,
}) {
  const { text: stateText, variant: stateVariant } =
    STATE_MAP[streamState || "idle"] ?? STATE_MAP.idle;
  const isActive = streamState === "thinking" || streamState === "generating";

  return (
    <header className="chat-header">
      <button
        type="button"
        className="icon-button chat-header__menu"
        onClick={onMenuClick}
        aria-label="打开菜单"
      >
        <Menu size={20} />
      </button>

      <div className="chat-header__title">
        <span className="eyebrow">心语医疗 AI</span>
        <h2>直接说你的问题</h2>
        <div className="chat-header__trust" aria-label="医疗助手能力说明">
          <span><ShieldCheck size={13} /> 安全分诊</span>
          <span><Sparkles size={13} /> RAG 知识增强</span>
        </div>
      </div>

      <div className="chat-header__meta">
        <span
          className={`conn-dot ${isConnected ? "conn-dot--on" : "conn-dot--off"}`}
          title={isConnected ? "后端已连接" : "后端连接失败"}
        />
        <div className={`stream-chip stream-chip--${stateVariant}`}>
          {isActive && <span className="stream-chip__pulse" />}
          {stateText}
        </div>
        <div className="thread-chip" title={threadId}>
          {threadId ? `#${threadId.slice(0, 6)}` : "…"}
        </div>
      </div>

      <div className="chat-header__actions">
        <button
          type="button"
          className="icon-button"
          onClick={onOpenSearch}
          title="搜索 (Ctrl+F)"
          aria-label="搜索聊天记录"
        >
          <Search size={16} />
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={onExport}
          title="导出 (Ctrl+E)"
          aria-label="导出对话"
        >
          <Download size={16} />
        </button>
      </div>
    </header>
  );
});

export default ChatHeader;
