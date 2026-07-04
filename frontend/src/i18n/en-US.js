/**
 * English (US) locale
 */
export default {
  // App
  "app.title": "Xinyu Medical Assistant",
  "app.subtitle": "Medical Consultation & Appointment Booking",

  // Sidebar
  "sidebar.nav.chat": "Chat Consultation",
  "sidebar.nav.documents": "Knowledge Base",
  "sidebar.status.system": "System Status",
  "sidebar.status.knowledgeBase": "Knowledge Base",
  "sidebar.status.refresh": "Refresh Status",
  "sidebar.gradioAdmin": "Gradio Admin",
  "sidebar.clearChat": "Clear Chat",
  "sidebar.closeMenu": "Close Sidebar",
  "sidebar.theme.light": "Light",
  "sidebar.theme.dark": "Dark",

  // Chat Header
  "chat.header.eyebrow": "Xinyu Medical AI",
  "chat.header.title": "Ask your question",
  "chat.header.connected": "Backend connected",
  "chat.header.disconnected": "Backend connection failed",

  // Stream states
  "stream.connecting": "Connecting",
  "stream.thinking": "Thinking",
  "stream.generating": "Generating",
  "stream.stopped": "Stopped",
  "stream.error": "Retry needed",
  "stream.done": "Done",
  "stream.idle": "Ready",

  // Composer
  "composer.placeholder": "Enter symptoms, medical questions, or booking needs… (Shift+Enter for newline)",
  "composer.placeholder.thinking": "AI is thinking…",
  "composer.placeholder.generating": "AI is responding…",
  "composer.send": "Send",
  "composer.stop": "Stop",

  // Messages
  "message.ai.label": "Xinyu Medical AI",
  "message.user.label": "Me",
  "message.copy": "Copy",
  "message.copied": "Copied",
  "message.copyReply": "Copy reply",
  "message.interrupted": "Stopped",
  "message.thinking": "AI is thinking",

  // Actions
  "action.confirmBooking": "Confirm Booking",
  "action.confirmCancel": "Confirm Cancellation",

  // Empty state
  "empty.title": "Hello, I'm Xinyu Medical Assistant",
  "empty.subtitle": "Your professional AI medical consultant for health guidance and appointment help",
  "empty.disclaimer": "This assistant provides health references only and cannot replace professional medical diagnosis or treatment",

  // Capabilities
  "cap.medicalQA": "Medical Q&A & Health Consultation",
  "cap.triage": "Smart Triage & Department Recommendation",
  "cap.booking": "Appointment Booking & Cancellation",
  "cap.hospitalInfo": "Hospital Info & Visit Guidance",

  // Starter prompts
  "prompt.hypertension": "What should I watch for with hypertension?",
  "prompt.triage": "I've been coughing for 3 days, which department?",
  "prompt.booking": "I want to book respiratory medicine",
  "prompt.cancel": "Cancel my last appointment",

  // Documents page
  "docs.header.eyebrow": "Xinyu Medical · Knowledge Base",
  "docs.header.title": "Knowledge Base Documents",
  "docs.header.desc": "User-friendly status, upload, and official sync entry. Advanced diagnostics remain in the Gradio admin panel.",
  "docs.refresh": "Refresh",
  "docs.status.label": "Knowledge Base Status",
  "docs.status.defaultMsg": "Reading knowledge base status…",
  "docs.upload.title": "Upload Local Documents",
  "docs.upload.desc": "Supports Markdown, PDF, Office, HTML, etc. Files are auto-converted and synced to the knowledge base.",
  "docs.upload.button": "Choose Files",
  "docs.upload.dragHint": "Drag files here",
  "docs.upload.dragActive": "Drop to upload",
  "docs.sync.title": "Sync Official Sources",
  "docs.sync.desc": "Manually pull current official source data. Great for demoing an updatable knowledge base.",
  "docs.sync.button": "Sync",
  "docs.sources.title": "Official Source Coverage",
  "docs.sources.count": "{count} sources",
  "docs.documents.title": "Synced Documents",
  "docs.documents.count": "{count} files",
  "docs.documents.empty": "No local Markdown documents yet. Upload or sync official data first.",
  "docs.tasks.title": "Recent Sync Tasks",
  "docs.tasks.count": "{count} tasks",
  "docs.tasks.empty": "No sync task records yet.",
  "docs.sources.empty": "Unable to read official source coverage.",

  // Search
  "search.placeholder": "Search chat history…",
  "search.noResults": "No results",
  "search.prev": "Previous",
  "search.next": "Next",
  "search.close": "Close",

  // Export
  "export.markdown": "Export Markdown",
  "export.json": "Export JSON",
  "export.title": "Xinyu Medical Assistant · Chat Log",

  // Dialog
  "dialog.clear.title": "Clear Chat",
  "dialog.clear.body": "Are you sure you want to clear all chat history? This action cannot be undone.",
  "dialog.clear.cancel": "Cancel",
  "dialog.clear.confirm": "Confirm Clear",
  "dialog.compress.title": "Compress Context",
  "dialog.compress.body": "Compress older messages into a summary? The most recent exchanges will remain visible.",
  "dialog.compress.cancel": "Cancel",
  "dialog.compress.confirm": "Compress",
  "chat.header.compress": "Compress context",

  // Error
  "error.connection": "Cannot connect to backend. Please ensure FastAPI is running.",
  "error.history": "Chat history temporarily unavailable.",
  "error.chatUnavailable": "Chat service temporarily unavailable.",
  "error.chatDisconnected": "Chat connection lost. Please retry later.",
  "error.clearFailed": "Failed to clear chat. Please try again.",
  "error.docsRead": "Knowledge base info temporarily unavailable.",
  "error.docsUpload": "Document upload failed.",
  "error.docsSync": "Official source sync failed.",
  "error.boundary.title": "Something went wrong",
  "error.boundary.desc": "The app encountered an unexpected error. Try reloading. If the problem persists, refresh the page.",
  "error.boundary.retry": "Retry",
  "error.boundary.reload": "Refresh Page",

  // Loading
  "loading": "Loading…",

  // Status tones
  "status.ready": "Ready",
  "status.completed": "Completed",
  "status.failed": "Failed",
  "status.error": "Error",
  "status.no_documents": "No Documents",
  "status.pending_rebuild": "Pending Rebuild",
  "status.preparing": "Preparing",
  "status.not_checked": "Not Checked",

  // Keyboard shortcuts
  "shortcuts.focusInput": "Focus Input",
  "shortcuts.copyReply": "Copy Last Reply",
  "shortcuts.search": "Search",
  "shortcuts.export": "Export",
  "shortcuts.stop": "Stop Generation",
};
