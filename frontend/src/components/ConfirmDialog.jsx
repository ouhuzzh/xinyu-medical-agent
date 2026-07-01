import React, { useEffect, useRef } from "react";
import { AlertTriangle } from "lucide-react";

const ConfirmDialog = React.memo(function ConfirmDialog({
  open,
  title,
  body,
  confirmText = "确认",
  cancelText = "取消",
  onConfirm,
  onCancel,
}) {
  const dialogRef = useRef(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open) {
      if (!dialog.open) dialog.showModal();
    } else {
      if (dialog.open) dialog.close();
    }
  }, [open]);

  function handleBackdropClick(e) {
    const rect = dialogRef.current?.getBoundingClientRect();
    if (!rect) return;
    const isOutside =
      e.clientX < rect.left ||
      e.clientX > rect.right ||
      e.clientY < rect.top ||
      e.clientY > rect.bottom;
    if (isOutside) onCancel?.();
  }

  return (
    <dialog ref={dialogRef} className="confirm-dialog" onClick={handleBackdropClick}>
      <div className="confirm-dialog__inner">
        <div className="confirm-dialog__icon">
          <AlertTriangle size={24} />
        </div>
        <div className="confirm-dialog__body">
          <h3>{title}</h3>
          <p>{body}</p>
        </div>
        <div className="confirm-dialog__actions">
          <button type="button" className="confirm-dialog__btn--cancel" onClick={onCancel}>
            {cancelText}
          </button>
          <button type="button" className="confirm-dialog__btn--confirm" onClick={onConfirm}>
            {confirmText}
          </button>
        </div>
      </div>
    </dialog>
  );
});

export default ConfirmDialog;
