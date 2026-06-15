import React, { useState } from "react";
import { loginUser, registerUser } from "../lib/api";
import XinyuLogo from "../components/XinyuLogo";

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
    <div className="login-page">
      <div className="login-card">
        <div className="login-card__brand">
          <XinyuLogo size={52} />
          <h1>心语医疗小助手</h1>
          <p>Medical AI · 智能咨询</p>
        </div>

        <form className="login-card__form" onSubmit={handleSubmit}>
          <h2>{mode === "login" ? "欢迎回来" : "创建账号"}</h2>

          <div className="login-card__field">
            <label htmlFor="login-username">用户名</label>
            <input
              id="login-username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="输入用户名"
              autoComplete="username"
              autoFocus
            />
          </div>

          {mode === "register" && (
            <div className="login-card__field">
              <label htmlFor="login-display-name">显示名称（可选）</label>
              <input
                id="login-display-name"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="输入显示名称"
              />
            </div>
          )}

          <div className="login-card__field">
            <label htmlFor="login-password">密码</label>
            <input
              id="login-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="输入密码"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
            />
          </div>

          {error && (
            <div className="login-card__error" role="alert" aria-live="polite">
              {error}
            </div>
          )}

          <button type="submit" className="login-card__submit" disabled={loading}>
            {loading ? "请稍候…" : mode === "login" ? "登录" : "注册"}
          </button>
        </form>

        <div className="login-card__switch">
          {mode === "login" ? "还没有账号？" : "已有账号？"}
          <button type="button" className="link-btn" onClick={switchMode}>
            {mode === "login" ? "立即注册" : "去登录"}
          </button>
        </div>
      </div>
    </div>
  );
}
