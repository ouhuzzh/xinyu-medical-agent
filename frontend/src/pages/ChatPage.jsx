import { useState } from "react";
import ChatHeader from "../components/ChatHeader";
import MessageList from "../components/MessageList";
import Composer from "../components/Composer";
import SearchBar from "../components/SearchBar";
import ConfirmDialog from "../components/ConfirmDialog";

export default function ChatPage({
  chat,
  isConnected,
  onMenuClick,
}) {
  const { search, onExport, compressChat } = chat;
  const [compressOpen, setCompressOpen] = useState(false);

  const handleCompress = async () => {
    setCompressOpen(false);
    await compressChat();
  };

  return (
    <section className="chat-shell">
      <ChatHeader
        threadId={chat.threadId}
        isConnected={isConnected}
        streamState={chat.streamState}
        onMenuClick={onMenuClick}
        onOpenSearch={search.openSearch}
        onExport={onExport}
        onCompress={() => setCompressOpen(true)}
      />

      {search.isOpen && (
        <SearchBar
          query={search.query}
          matchCount={search.matchCount}
          currentIndex={search.currentIndex}
          onQueryChange={search.setQuery}
          onClose={search.closeSearch}
          onNext={search.goNext}
          onPrev={search.goPrev}
        />
      )}

      <MessageList
        messages={chat.messages}
        isStreaming={chat.isStreaming}
        isLoadingHistory={chat.isLoadingHistory}
        onSendMessage={chat.sendMessage}
        searchQuery={search.query}
        currentMatchId={search.currentMatch?.messageId}
      />

      {chat.error && (
        <div className="error-bar" role="alert">
          <span>{chat.error}</span>
          <div className="error-bar__actions">
            {chat.lastUserMessage && (
              <button
                type="button"
                className="error-bar__retry"
                onClick={chat.retryLastMessage}
                disabled={chat.isStreaming}
              >
                重试
              </button>
            )}
            <button
              type="button"
              className="error-bar__close"
              onClick={() => chat.setError("")}
              aria-label="关闭错误提示"
            >
              ×
            </button>
          </div>
        </div>
      )}

      <Composer
        ref={chat.composerRef}
        input={chat.input}
        onChange={chat.setInput}
        onSubmit={() => chat.sendMessage(chat.input)}
        onStop={chat.stopStreaming}
        isStreaming={chat.isStreaming}
        disabled={!chat.threadId}
        streamState={chat.streamState}
      />

      <ConfirmDialog
        open={compressOpen}
        title="压缩上下文"
        body="将较早的消息压缩成摘要吗？最近的几轮对话仍会保留可见。"
        confirmText="压缩"
        cancelText="取消"
        onConfirm={handleCompress}
        onCancel={() => setCompressOpen(false)}
      />
    </section>
  );
}
