import { useState, useRef, useCallback, lazy, Suspense } from "react";
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
import { initialAuthToken } from "./lib/api";
import { AUTH_TOKEN_KEY } from "./constants/app";

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
  // If there's already a stored token, start with loggedIn=true so we show the app
  // immediately. useSystemStatus will gate with onAuthExpired if the token is bad.
  const [loggedIn, setLoggedIn] = useState(() => {
    if (typeof window === "undefined") return false;
    return !!(localStorage.getItem(AUTH_TOKEN_KEY) || initialAuthToken());
  });
  const composerRef = useRef(null);
  const { theme, toggleTheme } = useTheme();
  const system = useSystemStatus({ onAuthExpired: () => setLoggedIn(false) });

  const handleLogin = useCallback((accessToken, refreshToken) => {
    if (typeof window !== "undefined") {
      localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
    }
    system.setAuthToken(accessToken);
    setLoggedIn(true);
  }, [system]);

  const handleLogout = useCallback(() => {
    if (typeof window !== "undefined") {
      localStorage.removeItem(AUTH_TOKEN_KEY);
    }
    system.setAuthToken("");
    system.setCurrentUser(null);
    setLoggedIn(false);
  }, [system]);

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
