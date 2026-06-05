import React, { useState, useRef, useCallback, useEffect, lazy, Suspense } from "react";
import Sidebar from "./components/Sidebar";
import ClearConfirmDialog from "./components/ClearConfirmDialog";
import LoginPage from "./pages/LoginPage";
import { I18nProvider } from "./i18n";
import { useTheme } from "./hooks/useTheme";
import { useChatSession } from "./hooks/useChatSession";
import { useDocuments } from "./hooks/useDocuments";
import { useSystemStatus } from "./hooks/useSystemStatus";
import { useSearch } from "./hooks/useSearch";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { exportChat } from "./lib/export";
import { AUTH_TOKEN_KEY, REFRESH_TOKEN_KEY } from "./constants/app";

const ChatPage = lazy(() => import("./pages/ChatPage"));
const DocumentsPage = lazy(() => import("./pages/DocumentsPage"));

function PageLoader() {
  return (
    <div className="page-loader">
      <div className="page-loader__spinner" />
      <span>Loading…</span>
    </div>
  );
}

function AppInner() {
  const [activeView, setActiveView] = useState("chat");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [loggedIn, setLoggedIn] = useState(false);
  const [checkingAuth, setCheckingAuth] = useState(true);
  const composerRef = useRef(null);
  const { theme, toggleTheme } = useTheme();
  const system = useSystemStatus({ onAuthExpired: () => setLoggedIn(false) });

  const handleLogin = useCallback((accessToken, refreshToken) => {
    if (typeof window !== "undefined") {
      localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
      if (refreshToken) localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
    }
    system.setAuthToken(accessToken);
    setLoggedIn(true);
  }, [system]);

  const handleLogout = useCallback(() => {
    if (typeof window !== "undefined") {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);
    }
    system.setAuthToken("");
    system.setCurrentUser(null);
    setLoggedIn(false);
  }, [system]);

  // Auto-login when token is valid (e.g. from localStorage or env var)
  useEffect(() => {
    if (system.currentUser && system.authToken) {
      setLoggedIn(true);
      setCheckingAuth(false);
    } else if (!system.isConnected && system.authToken) {
      // Token exists but API failed — could be network issue, keep showing login
      setCheckingAuth(false);
    } else if (!system.authToken) {
      // No token at all — show login
      setCheckingAuth(false);
    }
  }, [system.currentUser, system.authToken, system.isConnected]);

  const chat = useChatSession({
    apiBaseUrl: system.apiBaseUrl,
    setApiBaseUrl: system.setApiBaseUrl,
    authToken: system.authToken,
    refreshStatus: system.refreshStatus,
    setIsConnected: system.setIsConnected,
    enabled: loggedIn,
  });
  const documents = useDocuments({
    apiBaseUrl: system.apiBaseUrl,
    setApiBaseUrl: system.setApiBaseUrl,
    authToken: system.authToken,
    refreshStatus: system.refreshStatus,
    enabled: system.isAdmin,
  });
  const search = useSearch(chat.messages);
  const lastAssistantMessage = [...chat.messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.content);

  function navigate(view) {
    setActiveView(view);
    setSidebarOpen(false);
  }

  function handleExport() {
    exportChat(chat.messages, chat.threadId, "markdown");
  }

  // Register global keyboard shortcuts
  useKeyboardShortcuts({
    isStreaming: chat.isStreaming,
    onStop: chat.stopStreaming,
    composerRef,
    lastAssistantMessage,
    onOpenSearch: search.openSearch,
    onExport: handleExport,
  });

  // Expose composerRef to ChatPage via chat object
  const chatWithRef = { ...chat, composerRef, search, onExport: handleExport };

  // Show brief loader while checking existing token
  if (checkingAuth && system.authToken && !system.currentUser) {
    return <PageLoader />;
  }

  // Login gate — show login page if not authenticated
  if (!loggedIn) {
    return (
      <LoginPage
        apiBaseUrl={system.apiBaseUrl}
        onLogin={handleLogin}
      />
    );
  }

  return (
    <div className="app">
      <Sidebar
        status={system.status}
        activeView={activeView}
        onNavigate={navigate}
        onClear={() => setClearDialogOpen(true)}
        onRefresh={system.refreshStatus}
        mobileOpen={sidebarOpen}
        onMobileClose={() => setSidebarOpen(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
        currentUser={system.currentUser}
        canManageDocuments={system.isAdmin}
        onLogout={handleLogout}
      />

      <Suspense fallback={<PageLoader />}>
        {activeView === "documents" && system.isAdmin ? (
          <DocumentsPage
            documentsState={documents}
            onMenuClick={() => setSidebarOpen(true)}
          />
        ) : (
          <ChatPage
            chat={chatWithRef}
            isConnected={system.isConnected}
            onMenuClick={() => setSidebarOpen(true)}
          />
        )}
      </Suspense>

      <ClearConfirmDialog
        open={clearDialogOpen}
        onConfirm={async () => {
          setClearDialogOpen(false);
          await chat.clearChat();
        }}
        onCancel={() => setClearDialogOpen(false)}
      />
    </div>
  );
}

export default function App() {
  return (
    <I18nProvider>
      <AppInner />
    </I18nProvider>
  );
}
