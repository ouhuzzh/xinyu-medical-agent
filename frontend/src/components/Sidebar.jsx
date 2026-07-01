import React from "react";
import { Activity, Building2, Database, MessageCircle, Trash2, ExternalLink, X, LogOut, User, ChevronRight, Plus, MessagesSquare, Pencil, Check } from "lucide-react";
import StatusIndicator from "./StatusIndicator";
import XinyuLogo from "./XinyuLogo";
import ThemeToggle from "./ThemeToggle";
import ConfirmDialog from "./ConfirmDialog";

const Sidebar = React.memo(function Sidebar({
  status,
  activeView,
  onNavigate,
  onClear,
  onRefresh,
  mobileOpen,
  onMobileClose,
  theme,
  onToggleTheme,
  currentUser,
  canManageDocuments,
  onLogout,
  sessions = [],
  activeThreadId = "",
  isChatStreaming = false,
  onNewSession,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
}) {
  const [editingThreadId, setEditingThreadId] = React.useState("");
  const [draftTitle, setDraftTitle] = React.useState("");
  const [sessionPendingDelete, setSessionPendingDelete] = React.useState(null);
  const systemState = status?.state || "preparing";
  const kbState = status?.knowledge_base?.status || "not_checked";
  const stats = status?.knowledge_base?.stats || {};
  const gradioUrl = "http://127.0.0.1:7860";

  function openGradioAdmin() {
    const opened = window.open(gradioUrl, "_blank", "noopener,noreferrer");
    if (!opened) {
      window.location.href = gradioUrl;
    }
  }

  const systemMetrics = [
    { label: "状态", value: systemState },
  ];
  const kbMetrics = [
    { label: "文档", value: stats.documents ?? 0 },
    { label: "片段", value: stats.child_chunks ?? 0 },
  ];

  function startRename(session) {
    setEditingThreadId(session.thread_id);
    setDraftTitle(session.title || "新会话");
  }

  async function submitRename() {
    const title = draftTitle.trim();
    if (!editingThreadId || !title) return;
    const ok = await onRenameSession?.(editingThreadId, title);
    if (ok !== false) {
      setEditingThreadId("");
      setDraftTitle("");
    }
  }

  async function confirmDeleteSession() {
    if (!sessionPendingDelete) return;
    await onDeleteSession?.(sessionPendingDelete.thread_id);
    if (editingThreadId === sessionPendingDelete.thread_id) {
      setEditingThreadId("");
      setDraftTitle("");
    }
    setSessionPendingDelete(null);
  }

  const pendingDeleteTitle = sessionPendingDelete?.title || "新会话";

  return (
    <>
      {mobileOpen && (
        <div className="sidebar-backdrop" onClick={onMobileClose} aria-hidden="true" />
      )}
      <aside className={`sidebar${mobileOpen ? " sidebar--open" : ""}`}>
        <div className="sidebar__top">
          <div className="brand">
            <div className="brand-mark">
              <XinyuLogo size={34} />
            </div>
            <div className="brand-text">
              <h1 className="brand-title">心语医疗小助手</h1>
              <p className="brand-sub">医疗咨询与预约挂号</p>
            </div>
          </div>
          <button
            type="button"
            className="sidebar-close icon-button"
            onClick={onMobileClose}
            aria-label="关闭侧边栏"
          >
            <X size={18} />
          </button>
        </div>

        <div className="sidebar__status-group">
          <nav className="sidebar-nav" aria-label="主导航">
            <button
              type="button"
              className={`sidebar-nav__item${activeView === "chat" ? " sidebar-nav__item--active" : ""}`}
              onClick={() => onNavigate("chat")}
            >
              <MessageCircle size={16} />
              聊天咨询
            </button>
            <button
              type="button"
              className={`sidebar-nav__item${activeView === "hospitals" ? " sidebar-nav__item--active" : ""}`}
              onClick={() => onNavigate("hospitals")}
            >
              <Building2 size={16} />
              医院绑定
            </button>
            {canManageDocuments && (
              <button
                type="button"
                className={`sidebar-nav__item${activeView === "documents" ? " sidebar-nav__item--active" : ""}`}
                onClick={() => onNavigate("documents")}
              >
                <Database size={16} />
                知识库文档
              </button>
            )}
          </nav>

          <div className="sidebar-sessions" aria-label="聊天会话">
            <div className="sidebar-sessions__header">
              <span>最近会话</span>
              <button
                type="button"
                className="icon-button sidebar-sessions__new"
                onClick={onNewSession}
                disabled={isChatStreaming}
                title={isChatStreaming ? "生成中暂不能新建会话" : "新建会话"}
                aria-label="新建会话"
              >
                <Plus size={15} />
              </button>
            </div>
            <div className="sidebar-sessions__list">
              {(sessions || []).slice(0, 8).map((session) => {
                const isActive = session.thread_id === activeThreadId;
                const isEditing = editingThreadId === session.thread_id;
                return (
                  <div
                    key={session.thread_id}
                    className={`sidebar-session${isActive ? " sidebar-session--active" : ""}`}
                    title={session.title || session.thread_id}
                  >
                    {isEditing ? (
                      <input
                        className="sidebar-session__input"
                        value={draftTitle}
                        onChange={(event) => setDraftTitle(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") submitRename();
                          if (event.key === "Escape") {
                            setEditingThreadId("");
                            setDraftTitle("");
                          }
                        }}
                        autoFocus
                        maxLength={80}
                      />
                    ) : (
                      <button
                        type="button"
                        className="sidebar-session__main"
                        onClick={() => onSelectSession?.(session.thread_id)}
                        disabled={isChatStreaming || isActive}
                      >
                        <MessagesSquare size={14} />
                        <span>{session.title || "新会话"}</span>
                      </button>
                    )}
                    <div className="sidebar-session__actions">
                      {isEditing ? (
                        <button
                          type="button"
                          className="icon-button sidebar-session__action"
                          onClick={submitRename}
                          disabled={isChatStreaming || !draftTitle.trim()}
                          title="保存"
                          aria-label="保存会话名称"
                        >
                          <Check size={13} />
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="icon-button sidebar-session__action"
                          onClick={() => startRename(session)}
                          disabled={isChatStreaming}
                          title="重命名"
                          aria-label="重命名会话"
                        >
                          <Pencil size={13} />
                        </button>
                      )}
                      <button
                        type="button"
                        className="icon-button sidebar-session__action sidebar-session__action--danger"
                        onClick={() => setSessionPendingDelete(session)}
                        disabled={isChatStreaming}
                        title="删除"
                        aria-label="删除会话"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>
                );
              })}
              {!sessions?.length && (
                <div className="sidebar-sessions__empty">暂无会话</div>
              )}
            </div>
          </div>

          <StatusIndicator
            icon={Activity}
            label="系统状态"
            value={systemState}
            message={status?.message || "正在读取系统状态。"}
            metrics={systemMetrics}
            onRefresh={onRefresh}
          />
          <StatusIndicator
            icon={Database}
            label="知识库"
            value={kbState}
            message={status?.knowledge_base?.message || "知识库状态读取中。"}
            metrics={kbMetrics}
          />
        </div>

        <div className="sidebar-actions">
          <div className="sidebar-auth-card">
            <div className="sidebar-auth-card__user">
              <div className="sidebar-auth-card__avatar">
                <User size={16} strokeWidth={2} />
              </div>
              <div className="sidebar-auth-card__identity">
                <strong>{currentUser?.username || currentUser?.display_name || currentUser?.user_id || "已登录"}</strong>
                <span className="role-badge">{currentUser?.role === "admin" ? "管理员" : "用户"}</span>
              </div>
            </div>
            <button
              type="button"
              className="sidebar-logout-btn"
              onClick={onLogout}
            >
              <LogOut size={14} />
              退出登录
              <ChevronRight size={13} className="sidebar-logout-btn__arrow" />
            </button>
          </div>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
          <button
            type="button"
            onClick={openGradioAdmin}
            className="sidebar-link-icon"
            title="打开 Gradio 后台。若打不开，请先运行：python project/app.py"
          >
            <ExternalLink size={16} />
            <span>Gradio 后台</span>
          </button>
          <button type="button" className="sidebar-clear-btn" onClick={onClear}>
            <Trash2 size={15} />
            清空会话
          </button>
        </div>
      </aside>
      <ConfirmDialog
        open={Boolean(sessionPendingDelete)}
        title="删除会话"
        body={`确定要删除“${pendingDeleteTitle}”吗？删除后不会再显示在最近会话中。`}
        confirmText="确认删除"
        onConfirm={confirmDeleteSession}
        onCancel={() => setSessionPendingDelete(null)}
      />
    </>
  );
});

export default Sidebar;
