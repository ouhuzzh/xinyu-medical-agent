import { AUTH_TOKEN_KEY } from "../constants/app";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
const configuredApiAuthToken = import.meta.env.VITE_API_AUTH_TOKEN;
const browserApiBaseUrl =
  typeof window !== "undefined" && window.location.hostname
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://127.0.0.1:8000";
export const fallbackApiBaseUrl = "http://127.0.0.1:8000";

export function initialApiBaseUrl() {
  return configuredApiBaseUrl || browserApiBaseUrl;
}

export function initialAuthToken() {
  if (typeof window === "undefined") {
    return configuredApiAuthToken || "demo-admin-token";
  }
  return localStorage.getItem(AUTH_TOKEN_KEY) || configuredApiAuthToken || "demo-admin-token";
}

function buildHeaders(authToken, headers = {}) {
  const nextHeaders = new Headers(headers);
  if (authToken) {
    nextHeaders.set("Authorization", `Bearer ${authToken}`);
  }
  return nextHeaders;
}

async function readJson(response) {
  if (!response.ok) {
    const text = await response.text();
    let payload = null;
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
    throw new Error(payload?.detail || payload?.message || text || `HTTP ${response.status}`);
  }
  return response.json();
}

function withAuth(options = {}, authToken) {
  return {
    ...options,
    headers: buildHeaders(authToken, options.headers),
  };
}

export async function apiFetchJson(path, options, apiBaseUrl, onFallback, authToken) {
  const firstUrl = `${apiBaseUrl}${path}`;
  try {
    return await readJson(await fetch(firstUrl, withAuth(options, authToken)));
  } catch (err) {
    if (configuredApiBaseUrl || apiBaseUrl === fallbackApiBaseUrl) {
      throw err;
    }
    const fallbackUrl = `${fallbackApiBaseUrl}${path}`;
    const data = await readJson(await fetch(fallbackUrl, withAuth(options, authToken)));
    onFallback?.(fallbackApiBaseUrl);
    return data;
  }
}

export function createSession(apiBaseUrl, onFallback, authToken, threadId) {
  return apiFetchJson(
    "/api/chat/session",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(threadId ? { thread_id: threadId } : {}),
    },
    apiBaseUrl,
    onFallback,
    authToken,
  );
}

export function fetchChatHistory(apiBaseUrl, onFallback, authToken, threadId) {
  return apiFetchJson(
    `/api/chat/history?thread_id=${encodeURIComponent(threadId)}`,
    undefined,
    apiBaseUrl,
    onFallback,
    authToken,
  );
}

export function clearChatSession(apiBaseUrl, onFallback, authToken, threadId) {
  return apiFetchJson(
    "/api/chat/clear",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: threadId }),
    },
    apiBaseUrl,
    onFallback,
    authToken,
  );
}

export function fetchSystemStatus(apiBaseUrl, onFallback, authToken) {
  return apiFetchJson("/api/system/status", undefined, apiBaseUrl, onFallback, authToken);
}

export function fetchDocumentsStatus(apiBaseUrl, onFallback, authToken) {
  return apiFetchJson("/api/documents/status", undefined, apiBaseUrl, onFallback, authToken);
}

export function fetchDocumentList(apiBaseUrl, onFallback, authToken) {
  return apiFetchJson("/api/documents/list", undefined, apiBaseUrl, onFallback, authToken);
}

export function fetchDocumentTasks(apiBaseUrl, onFallback, authToken) {
  return apiFetchJson("/api/documents/tasks", undefined, apiBaseUrl, onFallback, authToken);
}

export function fetchDocumentSources(apiBaseUrl, onFallback, authToken) {
  return apiFetchJson("/api/documents/sources", undefined, apiBaseUrl, onFallback, authToken);
}

export function uploadDocuments(apiBaseUrl, onFallback, authToken, files) {
  const formData = new FormData();
  Array.from(files || []).forEach((file) => formData.append("files", file));
  return apiFetchJson(
    "/api/documents/upload",
    {
      method: "POST",
      body: formData,
    },
    apiBaseUrl,
    onFallback,
    authToken,
  );
}

export function syncOfficialDocuments(apiBaseUrl, onFallback, authToken, source, limit) {
  return apiFetchJson(
    "/api/documents/sync-official",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, limit }),
    },
    apiBaseUrl,
    onFallback,
    authToken,
  );
}

export function buildStreamRequest(apiBaseUrl, authToken, threadId, message) {
  return {
    url: `${apiBaseUrl}/api/chat/stream`,
    options: withAuth(
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId, message }),
      },
      authToken,
    ),
  };
}

// --- Auth API ---

export function loginUser(apiBaseUrl, username, password) {
  return apiFetchJson(
    "/api/auth/login",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    },
    apiBaseUrl,
    () => {},
    "",
  );
}

export function registerUser(apiBaseUrl, username, password, displayName) {
  return apiFetchJson(
    "/api/auth/register",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, display_name: displayName }),
    },
    apiBaseUrl,
    () => {},
    "",
  );
}

export function fetchUserProfile(apiBaseUrl, authToken) {
  return apiFetchJson("/api/auth/profile", undefined, apiBaseUrl, () => {}, authToken);
}

export function refreshAccessToken(apiBaseUrl, refreshToken) {
  return apiFetchJson(
    "/api/auth/refresh",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    },
    apiBaseUrl,
    () => {},
    "",
  );
}

// --- Hospital MCP API ---

export function fetchHospitalList(apiBaseUrl, onFallback, authToken) {
  return apiFetchJson("/api/hospitals/list", undefined, apiBaseUrl, onFallback, authToken);
}

export function fetchHospitalCredentials(apiBaseUrl, onFallback, authToken) {
  return apiFetchJson("/api/hospitals/credentials", undefined, apiBaseUrl, onFallback, authToken);
}

export function addHospitalCredential(apiBaseUrl, onFallback, authToken, hospitalCode, token, label) {
  return apiFetchJson(
    "/api/hospitals/credentials/add",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hospital_code: hospitalCode, token, label: label || "" }),
    },
    apiBaseUrl,
    onFallback,
    authToken,
  );
}

export function deleteHospitalCredential(apiBaseUrl, onFallback, authToken, hospitalCode) {
  return apiFetchJson(
    "/api/hospitals/credentials/delete",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hospital_code: hospitalCode }),
    },
    apiBaseUrl,
    onFallback,
    authToken,
  );
}

export function testHospitalConnection(apiBaseUrl, onFallback, authToken, hospitalCode) {
  return apiFetchJson(
    "/api/hospitals/credentials/test",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hospital_code: hospitalCode }),
    },
    apiBaseUrl,
    onFallback,
    authToken,
  );
}
