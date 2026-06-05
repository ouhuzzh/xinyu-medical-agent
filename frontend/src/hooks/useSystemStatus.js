import { useCallback, useEffect, useRef, useState } from "react";
import { AUTH_TOKEN_KEY, REFRESH_TOKEN_KEY } from "../constants/app";
import { fetchSystemStatus, initialApiBaseUrl, initialAuthToken, refreshAccessToken } from "../lib/api";

export function useSystemStatus({ onAuthExpired } = {}) {
  const [status, setStatus] = useState(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(initialApiBaseUrl);
  const [authToken, setAuthTokenState] = useState(initialAuthToken);
  const [currentUser, setCurrentUser] = useState(null);
  const [isConnected, setIsConnected] = useState(true);
  const [statusError, setStatusError] = useState("");
  const refreshingRef = useRef(false);

  const tryRefresh = useCallback(async () => {
    if (refreshingRef.current) return null;
    const refreshToken = typeof window !== "undefined"
      ? localStorage.getItem(REFRESH_TOKEN_KEY)
      : "";
    if (!refreshToken) return null;

    refreshingRef.current = true;
    try {
      const data = await refreshAccessToken(apiBaseUrl, refreshToken);
      const newAccess = data.access_token;
      const newRefresh = data.refresh_token;
      if (newAccess) {
        setAuthTokenState(newAccess);
        if (typeof window !== "undefined") {
          localStorage.setItem(AUTH_TOKEN_KEY, newAccess);
          if (newRefresh) localStorage.setItem(REFRESH_TOKEN_KEY, newRefresh);
        }
        return newAccess;
      }
    } catch {
      // Refresh failed — token truly expired
    } finally {
      refreshingRef.current = false;
    }
    return null;
  }, [apiBaseUrl]);

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
      const isAuthError = err?.message && (
        err.message.includes("401") ||
        err.message.includes("Token") ||
        err.message.includes("token")
      );
      if (isAuthError) {
        // Try to refresh before giving up
        const newToken = await tryRefresh();
        if (newToken) {
          // Retry status with new token
          try {
            const data = await fetchSystemStatus(apiBaseUrl, setApiBaseUrl, newToken);
            setStatus(data);
            setCurrentUser(data?.current_user || null);
            setIsConnected(true);
            setStatusError("");
            return data;
          } catch {
            // Still failed after refresh
          }
        }
        onAuthExpired?.();
      }
      setStatus(null);
      setIsConnected(false);
      setStatusError("系统状态暂时无法读取。");
      return null;
    }
  }, [apiBaseUrl, authToken, tryRefresh, onAuthExpired]);

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
