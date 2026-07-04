import { useCallback, useEffect, useRef, useReducer } from "react";
import { THREAD_KEY } from "../constants/app";
import {
  clearChatSession,
  compressChatSession,
  createSession,
  deleteChatSession,
  fetchChatHistory,
  fetchChatSessions,
  renameChatSession,
} from "../lib/api";
import { openChatStream } from "../lib/sse";

const STREAMING_STATES = new Set(["connecting", "thinking", "generating"]);

const initialState = {
  threadId: localStorage.getItem(THREAD_KEY) || "",
  sessions: [],
  messages: [],
  input: "",
  streamState: "idle",
  error: "",
  lastUserMessage: "",
  isLoadingHistory: false,
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_THREAD_ID":
      return { ...state, threadId: action.payload };
    case "SET_SESSIONS":
      return { ...state, sessions: action.payload };
    case "SET_MESSAGES":
      return { ...state, messages: action.payload };
    case "ADD_MESSAGES":
      return { ...state, messages: [...state.messages, ...action.payload] };
    case "UPDATE_LAST_ASSISTANT": {
      const next = [...state.messages];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = { ...last, ...action.payload };
      }
      return { ...state, messages: next };
    }
    case "SET_INPUT":
      return { ...state, input: action.payload };
    case "SET_STREAM_STATE":
      return { ...state, streamState: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "SET_LAST_USER_MESSAGE":
      return { ...state, lastUserMessage: action.payload };
    case "SET_LOADING_HISTORY":
      return { ...state, isLoadingHistory: action.payload };
    case "CLEAR_MESSAGES":
      return { ...state, messages: [], error: "" };
    case "RESET_TRANSIENT_CHAT_STATE":
      return {
        ...state,
        sessions: [],
        messages: [],
        input: "",
        streamState: "idle",
        error: "",
        lastUserMessage: "",
      };
    case "STOP_STREAMING": {
      const next = [...state.messages];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = {
          ...last,
          content: last.content || "已停止生成。",
          interrupted: true,
        };
      }
      return { ...state, messages: next, streamState: "stopped" };
    }
    default:
      return state;
  }
}

export function useChatSession({
  apiBaseUrl,
  setApiBaseUrl,
  authToken,
  refreshStatus,
  setIsConnected,
  enabled = true,
}) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const streamRef = useRef(null);
  const streamDoneRef = useRef(false);
  const isStreamingRef = useRef(false);
  const inputRef = useRef(state.input);
  const threadIdRef = useRef(state.threadId);
  const enabledRef = useRef(enabled);
  const historyRequestRef = useRef(0);
  inputRef.current = state.input;
  threadIdRef.current = state.threadId;
  enabledRef.current = enabled;

  const isStreaming = STREAMING_STATES.has(state.streamState);
  isStreamingRef.current = isStreaming;

  useEffect(() => {
    if (enabled) return;
    historyRequestRef.current += 1;
    streamRef.current?.close();
    dispatch({ type: "RESET_TRANSIENT_CHAT_STATE" });
  }, [enabled]);

  const loadHistory = useCallback(async (activeThreadId = state.threadId) => {
    if (!activeThreadId) return;
    const requestId = historyRequestRef.current + 1;
    historyRequestRef.current = requestId;
    dispatch({ type: "SET_LOADING_HISTORY", payload: true });
    try {
      const data = await fetchChatHistory(apiBaseUrl, setApiBaseUrl, authToken, activeThreadId);
      if (
        requestId !== historyRequestRef.current ||
        !enabledRef.current ||
        activeThreadId !== threadIdRef.current
      ) {
        return;
      }
      dispatch({
        type: "SET_MESSAGES",
        payload: (data.messages || []).map((m, i) => ({
          id: m.id ?? `hist-${i}`,
          timestamp: m.timestamp ?? Date.now() + i,
          ...m,
        })),
      });
      dispatch({ type: "SET_ERROR", payload: "" });
    } catch (err) {
      if (
        requestId !== historyRequestRef.current ||
        !enabledRef.current ||
        activeThreadId !== threadIdRef.current
      ) {
        return;
      }
      dispatch({ type: "SET_MESSAGES", payload: [] });
      dispatch({ type: "SET_ERROR", payload: err.message || "历史会话暂时无法读取。" });
    } finally {
      if (requestId === historyRequestRef.current) {
        dispatch({ type: "SET_LOADING_HISTORY", payload: false });
      }
    }
  }, [apiBaseUrl, authToken, setApiBaseUrl, state.threadId]);

  const refreshSessions = useCallback(async () => {
    const data = await fetchChatSessions(apiBaseUrl, setApiBaseUrl, authToken);
    const sessions = Array.isArray(data.sessions) ? data.sessions : [];
    dispatch({ type: "SET_SESSIONS", payload: sessions });
    return sessions;
  }, [apiBaseUrl, authToken, setApiBaseUrl]);

  const createNewSession = useCallback(async () => {
    if (!enabled || isStreamingRef.current) return null;
    historyRequestRef.current += 1;
    streamRef.current?.close();
    dispatch({ type: "SET_STREAM_STATE", payload: "idle" });
    dispatch({ type: "SET_MESSAGES", payload: [] });
    try {
      const data = await createSession(apiBaseUrl, setApiBaseUrl, authToken);
      dispatch({ type: "SET_THREAD_ID", payload: data.thread_id });
      localStorage.setItem(THREAD_KEY, data.thread_id);
      await refreshSessions();
      setIsConnected(true);
      dispatch({ type: "SET_ERROR", payload: "" });
      return data.thread_id;
    } catch (err) {
      setIsConnected(false);
      dispatch({ type: "SET_MESSAGES", payload: [] });
      dispatch({ type: "SET_ERROR", payload: err.message || "无法连接后端服务，请确认 Bearer Token 和 FastAPI 状态。" });
      return null;
    }
  }, [apiBaseUrl, authToken, enabled, refreshSessions, setApiBaseUrl, setIsConnected]);

  const ensureSession = useCallback(async () => {
    if (!enabled) return;
    try {
      const sessions = await refreshSessions();
      const currentThread = threadIdRef.current;
      const selected =
        (currentThread && sessions.find((item) => item.thread_id === currentThread)?.thread_id) ||
        sessions[0]?.thread_id ||
        "";
      if (selected) {
        if (selected !== currentThread) {
          historyRequestRef.current += 1;
          dispatch({ type: "SET_MESSAGES", payload: [] });
        }
        dispatch({ type: "SET_THREAD_ID", payload: selected });
        localStorage.setItem(THREAD_KEY, selected);
        setIsConnected(true);
        dispatch({ type: "SET_ERROR", payload: "" });
        return;
      }
      await createNewSession();
    } catch (err) {
      setIsConnected(false);
      dispatch({ type: "SET_MESSAGES", payload: [] });
      dispatch({ type: "SET_ERROR", payload: err.message || "无法连接后端服务，请确认 Bearer Token 和 FastAPI 状态。" });
    }
  }, [createNewSession, enabled, refreshSessions, setIsConnected]);

  useEffect(() => {
    ensureSession();
    return () => streamRef.current?.close();
  }, [ensureSession]);

  useEffect(() => {
    if (!state.threadId) return;
    localStorage.setItem(THREAD_KEY, state.threadId);
    loadHistory(state.threadId);
  }, [state.threadId, loadHistory]);

  const clearChat = useCallback(async () => {
    if (!state.threadId) return;
    streamRef.current?.close();
    dispatch({ type: "SET_STREAM_STATE", payload: "idle" });
    try {
      await clearChatSession(apiBaseUrl, setApiBaseUrl, authToken, state.threadId);
      dispatch({ type: "CLEAR_MESSAGES" });
      await refreshSessions();
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: err.message || "清空会话失败，请稍后再试。" });
    }
  }, [apiBaseUrl, authToken, refreshSessions, setApiBaseUrl, state.threadId]);

  const compressChat = useCallback(async () => {
    if (!state.threadId || isStreamingRef.current) return null;
    try {
      const result = await compressChatSession(
        apiBaseUrl,
        setApiBaseUrl,
        authToken,
        state.threadId,
      );
      await loadHistory(state.threadId);
      dispatch({ type: "SET_ERROR", payload: "" });
      return result;
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        payload: err.message || "压缩会话失败，请稍后再试。",
      });
      return null;
    }
  }, [apiBaseUrl, authToken, loadHistory, setApiBaseUrl, state.threadId]);

  const renameSession = useCallback(async (threadId, title) => {
    const nextTitle = String(title || "").trim();
    if (!threadId || !nextTitle || isStreamingRef.current) return false;
    try {
      await renameChatSession(apiBaseUrl, setApiBaseUrl, authToken, threadId, nextTitle);
      await refreshSessions();
      dispatch({ type: "SET_ERROR", payload: "" });
      return true;
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: err.message || "重命名会话失败，请稍后再试。" });
      return false;
    }
  }, [apiBaseUrl, authToken, refreshSessions, setApiBaseUrl]);

  const deleteSession = useCallback(async (threadId) => {
    const targetThreadId = String(threadId || "").trim();
    if (!targetThreadId || isStreamingRef.current) return false;
    try {
      await deleteChatSession(apiBaseUrl, setApiBaseUrl, authToken, targetThreadId);
      const sessions = await refreshSessions();
      if (targetThreadId === threadIdRef.current) {
        historyRequestRef.current += 1;
        streamRef.current?.close();
        dispatch({ type: "SET_STREAM_STATE", payload: "idle" });
        dispatch({ type: "SET_MESSAGES", payload: [] });
        const nextThreadId = sessions.find((item) => item.thread_id !== targetThreadId)?.thread_id || "";
        if (nextThreadId) {
          dispatch({ type: "SET_THREAD_ID", payload: nextThreadId });
          localStorage.setItem(THREAD_KEY, nextThreadId);
        } else {
          await createNewSession();
        }
      }
      dispatch({ type: "SET_ERROR", payload: "" });
      return true;
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: err.message || "删除会话失败，请稍后再试。" });
      return false;
    }
  }, [apiBaseUrl, authToken, createNewSession, refreshSessions, setApiBaseUrl]);

  const selectSession = useCallback((threadId) => {
    const nextThreadId = String(threadId || "").trim();
    if (!nextThreadId || nextThreadId === state.threadId || isStreamingRef.current) return;
    historyRequestRef.current += 1;
    streamRef.current?.close();
    dispatch({ type: "SET_STREAM_STATE", payload: "idle" });
    dispatch({ type: "SET_ERROR", payload: "" });
    dispatch({ type: "SET_MESSAGES", payload: [] });
    dispatch({ type: "SET_THREAD_ID", payload: nextThreadId });
    localStorage.setItem(THREAD_KEY, nextThreadId);
  }, [state.threadId]);

  const stopStreaming = useCallback(() => {
    streamRef.current?.close();
    streamDoneRef.current = true;
    dispatch({ type: "STOP_STREAMING" });
  }, []);

  const sendMessage = useCallback((text) => {
    const content = (text ?? inputRef.current).trim();
    if (!content || !state.threadId || isStreamingRef.current) return;

    streamRef.current?.close();
    streamDoneRef.current = false;
    dispatch({ type: "SET_ERROR", payload: "" });
    dispatch({ type: "SET_INPUT", payload: "" });
    dispatch({ type: "SET_LAST_USER_MESSAGE", payload: content });
    dispatch({ type: "SET_STREAM_STATE", payload: "connecting" });

    const now = Date.now();
    dispatch({
      type: "ADD_MESSAGES",
      payload: [
        { id: `u-${now}`, role: "user", content, timestamp: now },
        { id: `a-${now}`, role: "assistant", content: "", timestamp: now + 1 },
      ],
    });

    const stream = openChatStream({
      apiBaseUrl,
      authToken,
      threadId: state.threadId,
      message: content,
      onFallback: setApiBaseUrl,
      doneRef: streamDoneRef,
      onStatus: () => dispatch({ type: "SET_STREAM_STATE", payload: "thinking" }),
      onMessage: (payload) => {
        dispatch({ type: "SET_STREAM_STATE", payload: "generating" });
        dispatch({ type: "UPDATE_LAST_ASSISTANT", payload: { content: payload.content } });
      },
      onFinal: (payload) => {
        dispatch({ type: "UPDATE_LAST_ASSISTANT", payload: { content: payload.content } });
        dispatch({ type: "SET_STREAM_STATE", payload: "done" });
        setIsConnected(true);
        refreshStatus?.();
        refreshSessions().catch(() => {});
      },
      onAppError: (payload) => {
        dispatch({ type: "SET_ERROR", payload: payload.content || "聊天服务暂时不可用。" });
        dispatch({ type: "SET_STREAM_STATE", payload: "error" });
      },
      onConnectionError: (error) => {
        dispatch({ type: "SET_ERROR", payload: error?.message || "聊天连接中断，请稍后重试。" });
        dispatch({ type: "SET_STREAM_STATE", payload: "error" });
        setIsConnected(false);
      },
    });
    streamRef.current = stream;
  }, [apiBaseUrl, authToken, refreshSessions, refreshStatus, setApiBaseUrl, setIsConnected, state.threadId]);

  const retryLastMessage = useCallback(() => {
    if (state.lastUserMessage && !isStreamingRef.current) {
      sendMessage(state.lastUserMessage);
    }
  }, [state.lastUserMessage, sendMessage]);

  return {
    threadId: state.threadId,
    sessions: state.sessions,
    messages: state.messages,
    input: state.input,
    setInput: (value) => dispatch({ type: "SET_INPUT", payload: value }),
    streamState: state.streamState,
    isStreaming,
    isLoadingHistory: state.isLoadingHistory,
    error: state.error,
    setError: (value) => dispatch({ type: "SET_ERROR", payload: value }),
    lastUserMessage: state.lastUserMessage,
    sendMessage,
    retryLastMessage,
    stopStreaming,
    clearChat,
    compressChat,
    newSession: createNewSession,
    selectSession,
    renameSession,
    deleteSession,
    refreshSessions,
  };
}
