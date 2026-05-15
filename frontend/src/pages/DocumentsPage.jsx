import { useMemo, useRef, useState } from "react";
import { Database, FileText, Menu, RefreshCw, UploadCloud } from "lucide-react";
import { statusTone } from "../constants/app";
import StatusIndicator from "../components/StatusIndicator";

const FALLBACK_OFFICIAL_SOURCES = [
  { value: "medlineplus", label: "MedlinePlus" },
  { value: "nhc", label: "国家卫健委" },
  { value: "who", label: "WHO" },
];

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function sourceLabel(doc) {
  if (doc.source_name) return doc.source_name;
  if (doc.source_key?.startsWith("official:medlineplus:")) return "MedlinePlus";
  if (doc.source_key?.startsWith("official:who:")) return "WHO";
  if (doc.source_key?.startsWith("official:nhc:")) return "国家卫健委";
  return "本地文档";
}

function formatDuration(ms) {
  const value = Number(ms || 0);
  if (!value) return "";
  if (value < 1000) return `${Math.round(value)}ms`;
  return `${(value / 1000).toFixed(1)}s`;
}

const ACCEPTED_EXTENSIONS = ".md,.pdf,.txt,.html,.htm,.doc,.docx,.ppt,.pptx,.xls,.xlsx";

export default function DocumentsPage({
  documentsState,
  onMenuClick,
}) {
  const fileRef = useRef(null);
  const [source, setSource] = useState("nhc");
  const [limit, setLimit] = useState(5);
  const [isDragOver, setIsDragOver] = useState(false);
  const {
    documents,
    tasks,
    sourceCoverage,
    documentStatus,
    isLoading,
    isWorking,
    message,
    error,
    setMessage,
    setError,
    refreshDocuments,
    upload,
    syncOfficial,
  } = documentsState;

  const stats = documentStatus?.stats || {};
  const statusValue = documentStatus?.status || "not_checked";
  const metrics = [
    { label: "文档", value: stats.documents ?? documents.length },
    { label: "片段", value: stats.child_chunks ?? 0 },
    { label: "本地文件", value: stats.local_markdown_files ?? documents.length },
  ];
  const sourceOptions = useMemo(() => {
    if (!sourceCoverage.length) return FALLBACK_OFFICIAL_SOURCES;
    return sourceCoverage.map((item) => ({
      value: item.source,
      label: item.label,
      defaultLimit: item.default_limit,
      maxLimit: item.max_limit,
    }));
  }, [sourceCoverage]);
  const selectedCoverage = sourceCoverage.find((item) => item.source === source);
  const sourceDistribution = useMemo(() => {
    const counts = new Map();
    documents.forEach((doc) => {
      const label = sourceLabel(doc);
      counts.set(label, (counts.get(label) || 0) + 1);
    });
    return Array.from(counts.entries()).map(([label, count]) => ({ label, count }));
  }, [documents]);

  function handleSourceChange(event) {
    const nextSource = event.target.value;
    setSource(nextSource);
    const option = sourceCoverage.find((item) => item.source === nextSource);
    if (option?.default_limit) {
      setLimit(option.default_limit);
    }
  }

  async function handleUpload(event) {
    const files = event.target.files;
    await upload(files);
    event.target.value = "";
  }

  function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(false);
  }

  async function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(false);
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      await upload(files);
    }
  }

  return (
    <section className="documents-shell">
      <header className="documents-header">
        <button
          type="button"
          className="icon-button chat-header__menu"
          onClick={onMenuClick}
          aria-label="打开菜单"
        >
          <Menu size={20} />
        </button>
        <div>
          <span className="eyebrow">心语医疗 · 知识库</span>
          <h2>知识库文档</h2>
          <p>这里先提供用户友好的状态、上传和官方同步入口；高级诊断继续放在 Gradio 后台。</p>
        </div>
        <button
          type="button"
          className="secondary-btn"
          onClick={refreshDocuments}
          disabled={isLoading || isWorking}
        >
          <RefreshCw size={16} />
          刷新
        </button>
      </header>

      <div className="documents-grid">
        <div className="document-card document-card--status">
          <StatusIndicator
            icon={Database}
            label="知识库状态"
            value={statusValue}
            message={documentStatus?.message || "正在读取知识库状态。"}
            metrics={metrics}
          />
          <div className={`status-banner status-banner--${statusTone(statusValue)}`}>
            {documentStatus?.message || "知识库状态读取中。"}
          </div>
        </div>

        <div className="document-card">
          <div className="document-card__title">
            <UploadCloud size={18} />
            <h3>上传本地资料</h3>
          </div>
          <p className="document-card__desc">
            支持 Markdown、PDF、Office、HTML 等格式。单次最多 5 个文件，单文件最大 20 MB。
          </p>
          <input
            ref={fileRef}
            type="file"
            multiple
            className="visually-hidden"
            onChange={handleUpload}
            accept={ACCEPTED_EXTENSIONS}
          />
          <div
            className={`drop-zone${isDragOver ? " drop-zone--active" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <UploadCloud size={28} className="drop-zone__icon" />
            <p className="drop-zone__text">
              {isDragOver ? "松开即可上传" : "拖拽文件到此处"}
            </p>
            <button
              type="button"
              className="primary-btn"
              onClick={() => fileRef.current?.click()}
              disabled={isWorking}
            >
              <UploadCloud size={17} />
              选择文件
            </button>
          </div>
        </div>

        <div className="document-card">
          <div className="document-card__title">
            <RefreshCw size={18} />
            <h3>同步官方资料</h3>
          </div>
          <p className="document-card__desc">
            手动拉取官方来源的当前资料，适合面试演示“知识库可更新化”。
          </p>
          <div className="sync-form">
            <select value={source} onChange={handleSourceChange}>
              {sourceOptions.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
            <input
              type="number"
              min="1"
              max={selectedCoverage?.max_limit ?? 50}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
            <button
              type="button"
              className="secondary-btn"
              onClick={() => syncOfficial(source, limit)}
              disabled={isWorking}
            >
              同步
            </button>
          </div>
          {selectedCoverage?.scope_note && (
            <p className="sync-hint">{selectedCoverage.scope_note}</p>
          )}
        </div>
      </div>

      <div className="document-section">
        <div className="document-section__head">
          <h3>官方来源覆盖度</h3>
          <span>{sourceCoverage.length} 个来源</span>
        </div>
        {sourceCoverage.length === 0 ? (
          <div className="empty-panel">暂时无法读取官方来源覆盖度。</div>
        ) : (
          <div className="source-coverage-list">
            {sourceCoverage.map((item) => (
              <article className="source-coverage-card" key={item.source}>
                <div className="source-coverage-card__head">
                  <strong>{item.label}</strong>
                  <span>{item.language} · {item.source_type}</span>
                </div>
                <div className="source-coverage-card__metrics">
                  <span>本地 {item.local_file_count ?? 0}</span>
                  <span>
                    {item.manifest_count == null
                      ? `可批量同步，单次上限 ${item.max_limit ?? 50}`
                      : `内置清单 ${item.manifest_count} 条`}
                  </span>
                </div>
                <p>{item.coverage_note || item.scope_note}</p>
                {item.next_step && <p className="source-coverage-card__next">下一步：{item.next_step}</p>}
              </article>
            ))}
          </div>
        )}
      </div>

      {(message || error) && (
        <div className={`document-alert ${error ? "document-alert--error" : "document-alert--ok"}`}>
          <span>{error || message}</span>
          <button type="button" onClick={() => { setError(""); setMessage(""); }}>×</button>
        </div>
      )}

      <div className="document-section">
        <div className="document-section__head">
          <h3>已同步文档</h3>
          <span>{documents.length} 个文件</span>
        </div>
        {sourceDistribution.length > 0 && (
          <div className="source-summary-row">
            {sourceDistribution.map((item) => (
              <span key={item.label}>{item.label} {item.count}</span>
            ))}
          </div>
        )}
        {documents.length === 0 ? (
          <div className="empty-panel">还没有可展示的本地 Markdown 文档，可以先上传或同步官方资料。</div>
        ) : (
          <div className="document-list">
            {documents.map((doc) => (
              <article className="document-row" key={doc.name}>
                <div className="document-row__icon"><FileText size={17} /></div>
                <div>
                  <strong>{doc.title || doc.name}</strong>
                  <p>
                    {sourceLabel(doc)} · {doc.source_type || "document"} ·
                    {doc.freshness_bucket || doc.sync_status || "active"}
                  </p>
                  <p>{doc.name} · {doc.file_type.toUpperCase()} · {formatBytes(doc.size_bytes)} · {doc.modified_at || "未知时间"}</p>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>

      <div className="document-section">
        <div className="document-section__head">
          <h3>最近同步任务</h3>
          <span>{tasks.length} 条</span>
        </div>
        {tasks.length === 0 ? (
          <div className="empty-panel">暂无同步任务记录。</div>
        ) : (
          <div className="task-list">
            {tasks.slice(0, 8).map((task, index) => (
              <article className="task-row" key={`${task.timestamp || "task"}-${index}`}>
                <div className="task-row__head">
                  <strong>{task.label || task.source || "同步任务"}</strong>
                  <span className={`task-status task-status--${task.failed ? "bad" : "good"}`}>
                    {task.failed ? "部分失败" : task.status || "完成"}
                  </span>
                </div>
                <p>
                  {task.timestamp || ""} · 新增 {task.written ?? 0} · 更新 {task.updated ?? 0} ·
                  下线 {task.deactivated ?? 0} · 未变化 {task.unchanged ?? 0} ·
                  失败 {task.failed ?? 0}
                </p>
                {(task.trigger_type || task.scope || task.duration_ms) && (
                  <p>
                    {task.trigger_type || "manual"} · {task.scope || task.source || "knowledge_base"}
                    {formatDuration(task.duration_ms) ? ` · ${formatDuration(task.duration_ms)}` : ""}
                  </p>
                )}
                {task.failure_details?.length > 0 && (
                  <details className="task-row__details">
                    <summary>查看失败原因</summary>
                    <ul>
                      {task.failure_details.slice(0, 4).map((detail, detailIndex) => (
                        <li key={`${index}-failure-${detailIndex}`}>{detail}</li>
                      ))}
                    </ul>
                  </details>
                )}
                {task.conversion_details?.length > 0 && (
                  <details className="task-row__details">
                    <summary>查看转换详情</summary>
                    <ul>
                      {task.conversion_details.slice(0, 4).map((detail, detailIndex) => (
                        <li key={`${index}-conversion-${detailIndex}`}>{detail}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
