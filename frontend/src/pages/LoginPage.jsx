import React, { useState } from "react";
import { loginUser, registerUser } from "../lib/api";
import XinyuLogo from "../components/XinyuLogo";

const S = {
  page: {
    position: "fixed",
    top: 0, left: 0, right: 0, bottom: 0,
    width: "100vw", height: "100vh",
    background: "linear-gradient(160deg, #f0f4ff 0%, #e8f0fe 30%, #fdf2f8 70%, #fef3c7 100%)",
    zIndex: 9999,
  },
  card: {
    position: "fixed",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: 400,
    maxWidth: "calc(100vw - 32px)",
    background: "#fff",
    borderRadius: 20,
    padding: "44px 40px 36px",
    boxShadow: "0 4px 6px rgba(0,0,0,0.02), 0 12px 40px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.03)",
  },
  brand: {
    textAlign: "center",
    marginBottom: 36,
  },
  brandTitle: {
    fontSize: 24,
    fontWeight: 700,
    color: "#1e293b",
    margin: "16px 0 6px",
    letterSpacing: "-0.3px",
  },
  brandSub: {
    fontSize: 13.5,
    color: "#94a3b8",
    fontWeight: 400,
    margin: 0,
  },
  formTitle: {
    fontSize: 17,
    fontWeight: 650,
    color: "#334155",
    marginBottom: 22,
    textAlign: "center",
    margin: "0 0 22px 0",
  },
  field: {
    marginBottom: 18,
  },
  label: {
    display: "block",
    fontSize: 12.5,
    fontWeight: 600,
    color: "#64748b",
    marginBottom: 6,
    letterSpacing: "0.2px",
  },
  input: {
    width: "100%",
    padding: "11px 14px",
    fontSize: 14.5,
    border: "1.5px solid #e2e8f0",
    borderRadius: 10,
    background: "#f8fafc",
    color: "#1e293b",
    outline: "none",
    boxSizing: "border-box",
  },
  error: {
    background: "#fff1f2",
    border: "1px solid #fecdd3",
    color: "#be123c",
    padding: "10px 14px",
    borderRadius: 10,
    fontSize: 13,
    marginBottom: 16,
  },
  submit: {
    width: "100%",
    padding: 12,
    fontSize: 15,
    fontWeight: 650,
    color: "#fff",
    background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
    border: "none",
    borderRadius: 10,
    cursor: "pointer",
    marginTop: 6,
    letterSpacing: "0.3px",
  },
  switch: {
    textAlign: "center",
    marginTop: 22,
    fontSize: 13.5,
    color: "#94a3b8",
  },
  link: {
    background: "none",
    border: "none",
    color: "#6366f1",
    fontSize: 13.5,
    fontWeight: 600,
    cursor: "pointer",
    padding: "0 0 0 4px",
  },
};

export default function LoginPage({ apiBaseUrl, onLogin }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function switchMode() {
    setMode(mode === "login" ? "register" : "login");
    setError("");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    if (!username.trim() || !password.trim()) {
      setError("请填写用户名和密码。");
      return;
    }
    setLoading(true);
    try {
      let result;
      if (mode === "login") {
        result = await loginUser(apiBaseUrl, username.trim(), password);
      } else {
        result = await registerUser(apiBaseUrl, username.trim(), password, displayName.trim());
      }
      onLogin(result.access_token, result.refresh_token);
    } catch (err) {
      setError(err.message || "操作失败，请稍后再试。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={S.page}>
      <div style={S.card}>
        <div style={S.brand}>
          <XinyuLogo size={48} />
          <h1 style={S.brandTitle}>心语医疗小助手</h1>
          <p style={S.brandSub}>医疗咨询与预约挂号</p>
        </div>

        <form onSubmit={handleSubmit}>
          <h2 style={S.formTitle}>{mode === "login" ? "登录" : "注册"}</h2>

          <div style={S.field}>
            <label style={S.label} htmlFor="login-username">用户名</label>
            <input
              id="login-username"
              style={S.input}
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="输入用户名"
              autoComplete="username"
              autoFocus
            />
          </div>

          {mode === "register" && (
            <div style={S.field}>
              <label style={S.label} htmlFor="login-display-name">显示名称（可选）</label>
              <input
                id="login-display-name"
                style={S.input}
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="输入显示名称"
              />
            </div>
          )}

          <div style={S.field}>
            <label style={S.label} htmlFor="login-password">密码</label>
            <input
              id="login-password"
              style={S.input}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="输入密码"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
            />
          </div>

          {error && <div style={S.error}>⚠ {error}</div>}

          <button type="submit" style={S.submit} disabled={loading}>
            {loading ? "请稍候…" : mode === "login" ? "登录" : "注册"}
          </button>
        </form>

        <div style={S.switch}>
          {mode === "login" ? "还没有账号？" : "已有账号？"}
          <button type="button" style={S.link} onClick={switchMode}>
            {mode === "login" ? "立即注册" : "去登录"}
          </button>
        </div>
      </div>
    </div>
  );
}
