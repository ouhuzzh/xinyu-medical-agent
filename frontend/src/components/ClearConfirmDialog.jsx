import React from "react";
import ConfirmDialog from "./ConfirmDialog";

const ClearConfirmDialog = React.memo(function ClearConfirmDialog({ open, onConfirm, onCancel }) {
  return (
    <ConfirmDialog
      open={open}
      title="清空会话"
      body="确定要清空当前所有对话记录吗？此操作无法撤销。"
      confirmText="确认清空"
      onConfirm={onConfirm}
      onCancel={onCancel}
    />
  );
});

export default ClearConfirmDialog;
