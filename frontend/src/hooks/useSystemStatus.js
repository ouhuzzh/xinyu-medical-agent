import { useCallback, useEffect, useState } from "react";
import { AUTH_TOKEN_KEY } from "../constants/app";
import { fetchSystemStatus, initialApiBaseUrl, initialAuthToken } from "../lib/api";

export function useSystemStatus() {
  const [status, setStatus] = useState(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(initialApiBaseUrl);
  const [authToken, setAuthTokenState] = useState(initialAuthToken);
  const [isConnected, setIsConnected] = useState(true);
  const [statusError, setStatusError] = useState("");

  const refreshStatus = useCallback(async () => {
    try {
      const data = await fetchSystemStatus(apiBaseUrl, setApiBaseUrl, authToken);
      setStatus(data);
      setIsConnected(true);
      setStatusError("");
      return data;
    } catch {
      setStatus(null);
      setIsConnected(false);
      setStatusError("系统状态暂时无法读取。");
      return null;
    }
  }, [apiBaseUrl, authToken]);

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
    currentUser: status?.current_user || null,
    isAdmin: status?.current_user?.role === "admin",
    isConnected,
    setIsConnected,
    statusError,
    clearStatusError: () => setStatusError(""),
    refreshStatus,
  };
}
