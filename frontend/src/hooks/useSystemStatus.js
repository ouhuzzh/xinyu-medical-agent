import { useCallback, useEffect, useState } from "react";
import { AUTH_TOKEN_KEY } from "../constants/app";
import { fetchSystemStatus, initialApiBaseUrl, initialAuthToken } from "../lib/api";

export function useSystemStatus({ onAuthExpired } = {}) {
  const [status, setStatus] = useState(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(initialApiBaseUrl);
  const [authToken, setAuthTokenState] = useState(initialAuthToken);
  const [currentUser, setCurrentUser] = useState(null);
  const [isConnected, setIsConnected] = useState(true);
  const [statusError, setStatusError] = useState("");

  const refreshStatus = useCallback(async () => {
    if (!authToken) {
      setIsConnected(false);
      return null;
    }
    try {
      const data = await fetchSystemStatus(apiBaseUrl, setApiBaseUrl, authToken);
      setStatus(data);
      setCurrentUser(data?.current_user || null);
      setIsConnected(true);
      setStatusError("");
      return data;
    } catch (err) {
      setStatus(null);
      setIsConnected(false);
      // If 401, token expired — trigger logout
      if (err?.message && (err.message.includes("401") || err.message.includes("Token"))) {
        onAuthExpired?.();
      }
      setStatusError("系统状态暂时无法读取。");
      return null;
    }
  }, [apiBaseUrl, authToken, onAuthExpired]);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  const setAuthToken = useCallback((value) => {
    const next = (value || "").trim();
    setAuthTokenState(next);
    if (typeof window !== "undefined") {
      if (next) {
        localStorage.setItem(AUTH_TOKEN_KEY, next);
      } else {
        localStorage.removeItem(AUTH_TOKEN_KEY);
      }
    }
  }, []);

  return {
    status,
    apiBaseUrl,
    setApiBaseUrl,
    authToken,
    setAuthToken,
    currentUser,
    setCurrentUser,
    isAdmin: currentUser?.role === "admin",
    isConnected,
    setIsConnected,
    statusError,
    clearStatusError: () => setStatusError(""),
    refreshStatus,
  };
}
