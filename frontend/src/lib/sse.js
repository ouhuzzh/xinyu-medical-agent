import { buildStreamRequest, fallbackApiBaseUrl } from "./api";

async function readErrorMessage(response) {
  const text = await response.text();
  try {
    const payload = JSON.parse(text);
    return payload.detail || payload.message || `HTTP ${response.status}`;
  } catch {
    return text || `HTTP ${response.status}`;
  }
}

function dispatchEventBlock(block, handlers) {
  if (!block.trim()) return;
  let eventType = "message";
  const dataLines = [];
  block.split("\n").forEach((line) => {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim() || eventType;
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  });
  if (!dataLines.length) return;
  const payload = JSON.parse(dataLines.join("\n"));
  handlers[eventType]?.(payload);
}

async function consumeSseResponse(response, handlers, signal) {
  if (!response.body) {
    throw new Error("Streaming response body is empty.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    if (signal.aborted) return;
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundaryIndex = buffer.indexOf("\n\n");
    while (boundaryIndex >= 0) {
      const block = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);
      dispatchEventBlock(block, handlers);
      boundaryIndex = buffer.indexOf("\n\n");
    }
  }

  if (buffer.trim()) {
    dispatchEventBlock(buffer, handlers);
  }
}

export function openChatStream({
  apiBaseUrl,
  authToken,
  threadId,
  message,
  onMessage,
  onStatus,
  onFinal,
  onAppError,
  onConnectionError,
  onFallback,
  doneRef,
}) {
  const controller = new AbortController();
  const signal = controller.signal;

  async function start(urlBase, allowFallback) {
    try {
      const { url, options } = buildStreamRequest(urlBase, authToken, threadId, message);
      const response = await fetch(url, { ...options, signal });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      await consumeSseResponse(
        response,
        {
          status: (payload) => onStatus?.(payload),
          message: (payload) => onMessage?.(payload),
          final: (payload) => {
            doneRef.current = true;
            onFinal?.(payload);
          },
          "app-error": (payload) => {
            doneRef.current = true;
            onAppError?.(payload);
          },
        },
        signal,
      );
    } catch (error) {
      if (signal.aborted) {
        return;
      }
      if (allowFallback && urlBase !== fallbackApiBaseUrl) {
        onFallback?.(fallbackApiBaseUrl);
        await start(fallbackApiBaseUrl, false);
        return;
      }
      onConnectionError?.(error);
    }
  }

  start(apiBaseUrl, apiBaseUrl !== fallbackApiBaseUrl);

  return {
    close() {
      controller.abort();
    },
  };
}
