import React from "react";
import { Activity, Database, MessageCircle, Trash2, ExternalLink, X, LogOut } from "lucide-react";
import StatusIndicator from "./StatusIndicator";
import XinyuLogo from "./XinyuLogo";
import ThemeToggle from "./ThemeToggle";

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
}) {
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
            <div className="sidebar-auth-card__head">
              <strong>{currentUser ? (currentUser.username || currentUser.user_id) : "已登录"}</strong>
              <span className="role-badge">{currentUser?.role === "admin" ? "管理员" : "用户"}</span>
            </div>
            <button
              type="button"
              className="sidebar-logout-btn"
              onClick={onLogout}
            >
              <LogOut size={14} />
              退出登录
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
    </>
  );
});

export default Sidebar;
