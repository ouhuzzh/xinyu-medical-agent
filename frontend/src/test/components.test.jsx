import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import TypingDots from "../components/TypingDots";
import SkeletonLoader from "../components/SkeletonLoader";
import Composer from "../components/Composer";
import ClearConfirmDialog from "../components/ClearConfirmDialog";
import ActionButtons from "../components/ActionButtons";
import Sidebar from "../components/Sidebar";

describe("TypingDots", () => {
  it("renders without crashing", () => {
    render(<TypingDots />);
    expect(screen.getByLabelText("AI 正在思考")).toBeInTheDocument();
  });
});

describe("SkeletonLoader", () => {
  it("renders default 3 rows", () => {
    const { container } = render(<SkeletonLoader />);
    const rows = container.querySelectorAll(".skeleton-row");
    expect(rows.length).toBe(3);
  });

  it("renders custom row count", () => {
    const { container } = render(<SkeletonLoader rows={5} />);
    const rows = container.querySelectorAll(".skeleton-row");
    expect(rows.length).toBe(5);
  });
});

describe("Composer", () => {
  it("renders with send button when not streaming", () => {
    render(
      <Composer
        input="hello"
        onChange={() => {}}
        onSubmit={() => {}}
        onStop={() => {}}
        isStreaming={false}
        disabled={false}
        streamState="idle"
      />,
    );
    expect(screen.getByLabelText("发送消息")).toBeInTheDocument();
  });

  it("renders with stop button when streaming", () => {
    render(
      <Composer
        input=""
        onChange={() => {}}
        onSubmit={() => {}}
        onStop={() => {}}
        isStreaming={true}
        disabled={false}
        streamState="generating"
      />,
    );
    expect(screen.getByLabelText("停止 AI 生成")).toBeInTheDocument();
  });

  it("disables send button when input is empty", () => {
    render(
      <Composer
        input=""
        onChange={() => {}}
        onSubmit={() => {}}
        onStop={() => {}}
        isStreaming={false}
        disabled={false}
        streamState="idle"
      />,
    );
    expect(screen.getByLabelText("发送消息")).toBeDisabled();
  });
});

describe("ClearConfirmDialog", () => {
  it("renders when open", () => {
    // jsdom doesn't implement dialog.showModal, so we mock it
    HTMLDialogElement.prototype.showModal = () => {};
    HTMLDialogElement.prototype.close = () => {};
    render(
      <ClearConfirmDialog
        open={true}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );
    expect(screen.getByText("清空会话")).toBeInTheDocument();
  });
});

describe("Sidebar", () => {
  it("uses an in-app confirmation dialog when deleting a session", () => {
    HTMLDialogElement.prototype.showModal = function showModal() {
      this.open = true;
    };
    HTMLDialogElement.prototype.close = function close() {
      this.open = false;
    };
    const browserConfirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const onDeleteSession = vi.fn();

    render(
      <Sidebar
        status={{ state: "ready", knowledge_base: { status: "ready", stats: {} } }}
        activeView="chat"
        onNavigate={() => {}}
        onClear={() => {}}
        onRefresh={() => {}}
        mobileOpen={false}
        onMobileClose={() => {}}
        theme="light"
        onToggleTheme={() => {}}
        currentUser={{ username: "demo" }}
        canManageDocuments={false}
        onLogout={() => {}}
        sessions={[{ thread_id: "thread-1", title: "问诊记录" }]}
        activeThreadId="thread-1"
        onDeleteSession={onDeleteSession}
      />,
    );

    fireEvent.click(screen.getByLabelText("删除会话"));

    expect(browserConfirm).not.toHaveBeenCalled();
    expect(screen.getByText("删除会话")).toBeInTheDocument();
    expect(screen.getByText("确定要删除“问诊记录”吗？删除后不会再显示在最近会话中。")).toBeInTheDocument();

    fireEvent.click(screen.getByText("确认删除"));

    expect(onDeleteSession).toHaveBeenCalledWith("thread-1");
    browserConfirm.mockRestore();
  });
});

describe("ActionButtons", () => {
  it("renders nothing when content has no action pattern", () => {
    const { container } = render(
      <ActionButtons content="这是一段普通文本" onAction={() => {}} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders confirm button for booking confirmation", () => {
    render(
      <ActionButtons content="请**确认预约**以继续" onAction={() => {}} />,
    );
    expect(screen.getByText("确认预约")).toBeInTheDocument();
  });

  it("renders cancel button for cancellation confirmation", () => {
    render(
      <ActionButtons content="请**确认取消**以继续" onAction={() => {}} />,
    );
    expect(screen.getByText("确认取消")).toBeInTheDocument();
  });
});
