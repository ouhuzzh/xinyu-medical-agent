import { describe, it, expect, vi } from "vitest";
import { useSearch } from "../hooks/useSearch";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChatSession } from "../hooks/useChatSession";
import { exportAsMarkdown, exportAsJSON } from "../lib/export";
import { buildStreamRequest, deleteChatSession, fetchChatSessions, renameChatSession } from "../lib/api";
import zhCN from "../i18n/zh-CN.js";
import enUS from "../i18n/en-US.js";

// ==========================================
// useSearch
// ==========================================
describe("useSearch", () => {
  const messages = [
    { id: "1", role: "user", content: "高血压应该注意什么？" },
    { id: "2", role: "assistant", content: "高血压患者应注意低盐饮食。" },
    { id: "3", role: "user", content: "挂什么科？" },
    { id: "4", role: "assistant", content: "建议挂心内科。" },
  ];

  it("starts closed with empty query", () => {
    const { result } = renderHook(() => useSearch(messages));
    expect(result.current.isOpen).toBe(false);
    expect(result.current.query).toBe("");
    expect(result.current.matchCount).toBe(0);
  });

  it("finds matching messages", () => {
    const { result } = renderHook(() => useSearch(messages));
    act(() => {
      result.current.openSearch();
    });
    act(() => {
      result.current.setQuery("高血压");
    });
    expect(result.current.matchCount).toBe(2);
    expect(result.current.isOpen).toBe(true);
  });

  it("navigates through matches", () => {
    const { result } = renderHook(() => useSearch(messages));
    act(() => {
      result.current.openSearch();
    });
    act(() => {
      result.current.setQuery("高血压");
    });
    expect(result.current.currentIndex).toBe(0);
    act(() => {
      result.current.goNext();
    });
    expect(result.current.currentIndex).toBe(1);
    act(() => {
      result.current.goNext();
    });
    // Wraps around
    expect(result.current.currentIndex).toBe(0);
  });

  it("goes to previous match", () => {
    const { result } = renderHook(() => useSearch(messages));
    act(() => {
      result.current.openSearch();
    });
    act(() => {
      result.current.setQuery("高血压");
    });
    act(() => {
      result.current.goPrev();
    });
    // Wraps from 0 to last
    expect(result.current.currentIndex).toBe(1);
  });

  it("resets on close", () => {
    const { result } = renderHook(() => useSearch(messages));
    act(() => {
      result.current.openSearch();
    });
    act(() => {
      result.current.setQuery("高血压");
    });
    act(() => {
      result.current.closeSearch();
    });
    expect(result.current.isOpen).toBe(false);
    expect(result.current.query).toBe("");
  });

  it("returns 0 matches for empty query", () => {
    const { result } = renderHook(() => useSearch(messages));
    act(() => {
      result.current.openSearch();
    });
    expect(result.current.matchCount).toBe(0);
  });
});

// ==========================================
// Export Utils
// ==========================================
describe("exportAsMarkdown", () => {
  const messages = [
    { id: "1", role: "user", content: "你好", timestamp: 1700000000000 },
    { id: "2", role: "assistant", content: "你好，有什么可以帮你？", timestamp: 1700000001000 },
  ];

  it("includes title and thread ID", () => {
    const md = exportAsMarkdown(messages, "test-thread-123");
    expect(md).toContain("心语医疗小助手");
    expect(md).toContain("test-thread-123");
  });

  it("includes user and assistant messages", () => {
    const md = exportAsMarkdown(messages);
    expect(md).toContain("你好");
    expect(md).toContain("你好，有什么可以帮你？");
  });

  it("marks interrupted messages", () => {
    const msgs = [{ ...messages[1], interrupted: true }];
    const md = exportAsMarkdown(msgs);
    expect(md).toContain("生成已中断");
  });
});

describe("exportAsJSON", () => {
  const messages = [
    { id: "1", role: "user", content: "hello", timestamp: 1700000000000 },
  ];

  it("produces valid JSON with correct structure", () => {
    const json = exportAsJSON(messages, "thread-1");
    const data = JSON.parse(json);
    expect(data.threadId).toBe("thread-1");
    expect(data.messageCount).toBe(1);
    expect(data.messages).toHaveLength(1);
    expect(data.messages[0].role).toBe("user");
  });
});

// ==========================================
// buildStreamRequest (POST instead of GET)
// ==========================================
describe("buildStreamRequest", () => {
  it("uses POST body instead of leaking message in the URL", () => {
    const request = buildStreamRequest("http://api.test", null, "thread-1", "高血压要注意什么");

    expect(request.url).toBe("http://api.test/api/chat/stream");
    expect(request.options.method).toBe("POST");
    expect(request.options.headers.get("Content-Type")).toBe("application/json");
    expect(request.url).not.toContain("高血压");
    expect(JSON.parse(request.options.body)).toEqual({
      thread_id: "thread-1",
      message: "高血压要注意什么",
    });
  });
});

describe("fetchChatSessions", () => {
  it("loads the authenticated session list", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ sessions: [{ thread_id: "thread-1", title: "你好" }] }),
    }));
    try {
      const data = await fetchChatSessions("http://api.test", undefined, "token-1");

      expect(globalThis.fetch).toHaveBeenCalledWith(
        "http://api.test/api/chat/sessions",
        expect.objectContaining({
          headers: expect.any(Headers),
        }),
      );
      const headers = globalThis.fetch.mock.calls[0][1].headers;
      expect(headers.get("Authorization")).toBe("Bearer token-1");
      expect(data.sessions[0].thread_id).toBe("thread-1");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("renames a chat session", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ thread_id: "thread-1", title: "复诊咨询" }),
    }));
    try {
      const data = await renameChatSession("http://api.test", undefined, "token-1", "thread-1", "复诊咨询");

      expect(globalThis.fetch).toHaveBeenCalledWith(
        "http://api.test/api/chat/session/rename",
        expect.objectContaining({
          method: "POST",
          headers: expect.any(Headers),
          body: JSON.stringify({ thread_id: "thread-1", title: "复诊咨询" }),
        }),
      );
      expect(data.title).toBe("复诊咨询");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("deletes a chat session", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ thread_id: "thread-1", deleted: true }),
    }));
    try {
      const data = await deleteChatSession("http://api.test", undefined, "token-1", "thread-1");

      expect(globalThis.fetch).toHaveBeenCalledWith(
        "http://api.test/api/chat/session/delete",
        expect.objectContaining({
          method: "POST",
          headers: expect.any(Headers),
          body: JSON.stringify({ thread_id: "thread-1" }),
        }),
      );
      expect(data.deleted).toBe(true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe("useChatSession", () => {
  it("clears in-memory messages when chat is disabled", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (url) => {
      const value = String(url);
      if (value.includes("/api/chat/sessions")) {
        return {
          ok: true,
          json: async () => ({ sessions: [{ thread_id: "thread-1", title: "上一位用户的问题" }] }),
        };
      }
      if (value.includes("/api/chat/history")) {
        return {
          ok: true,
          json: async () => ({
            thread_id: "thread-1",
            messages: [{ role: "user", content: "上一位用户的问题" }],
          }),
        };
      }
      return { ok: false, text: async () => "unexpected request" };
    });

    const props = {
      apiBaseUrl: "http://api.test",
      setApiBaseUrl: vi.fn(),
      authToken: "token-1",
      refreshStatus: vi.fn(),
      setIsConnected: vi.fn(),
      enabled: true,
    };

    try {
      const { result, rerender } = renderHook((hookProps) => useChatSession(hookProps), {
        initialProps: props,
      });

      await waitFor(() => {
        expect(result.current.messages[0]?.content).toBe("上一位用户的问题");
      });

      rerender({ ...props, enabled: false });

      await waitFor(() => {
        expect(result.current.messages).toEqual([]);
        expect(result.current.sessions).toEqual([]);
      });
    } finally {
      globalThis.fetch = originalFetch;
      localStorage.clear();
    }
  });

  it("ignores late history responses after chat is disabled", async () => {
    const originalFetch = globalThis.fetch;
    let resolveHistory;
    const historyPromise = new Promise((resolve) => {
      resolveHistory = resolve;
    });
    globalThis.fetch = vi.fn(async (url) => {
      const value = String(url);
      if (value.includes("/api/chat/sessions")) {
        return {
          ok: true,
          json: async () => ({ sessions: [{ thread_id: "thread-1", title: "上一位用户的问题" }] }),
        };
      }
      if (value.includes("/api/chat/history")) {
        return historyPromise;
      }
      return { ok: false, text: async () => "unexpected request" };
    });

    const props = {
      apiBaseUrl: "http://api.test",
      setApiBaseUrl: vi.fn(),
      authToken: "token-1",
      refreshStatus: vi.fn(),
      setIsConnected: vi.fn(),
      enabled: true,
    };

    try {
      const { result, rerender } = renderHook((hookProps) => useChatSession(hookProps), {
        initialProps: props,
      });

      await waitFor(() => {
        expect(globalThis.fetch.mock.calls.some(([url]) => String(url).includes("/api/chat/history"))).toBe(true);
      });

      rerender({ ...props, enabled: false });
      await waitFor(() => {
        expect(result.current.messages).toEqual([]);
      });

      await act(async () => {
        resolveHistory({
          ok: true,
          json: async () => ({
            thread_id: "thread-1",
            messages: [{ role: "user", content: "迟到的旧消息" }],
          }),
        });
        await historyPromise;
      });

      expect(result.current.messages).toEqual([]);
    } finally {
      globalThis.fetch = originalFetch;
      localStorage.clear();
    }
  });
});

// ==========================================
// i18n locale files
// ==========================================
describe("i18n locale files", () => {
  it("zh-CN has all required keys", () => {
    expect(zhCN["app.title"]).toBe("心语医疗小助手");
    expect(zhCN["composer.placeholder"]).toBeTruthy();
    expect(zhCN["search.placeholder"]).toBeTruthy();
    expect(zhCN["export.markdown"]).toBeTruthy();
  });

  it("en-US has all required keys", () => {
    expect(enUS["app.title"]).toBe("Xinyu Medical Assistant");
    expect(enUS["composer.placeholder"]).toBeTruthy();
    expect(enUS["search.placeholder"]).toBeTruthy();
    expect(enUS["export.markdown"]).toBeTruthy();
  });

  it("both locales have matching keys", () => {
    const zhKeys = Object.keys(zhCN).sort();
    const enKeys = Object.keys(enUS).sort();
    expect(zhKeys).toEqual(enKeys);
  });
});
