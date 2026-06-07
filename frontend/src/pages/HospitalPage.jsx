import React, { useState, useCallback, useEffect } from "react";
import { ArrowLeft, Plus, Trash2, Wifi, ExternalLink } from "lucide-react";
import {
  fetchHospitalList,
  fetchHospitalCredentials,
  addHospitalCredential,
  deleteHospitalCredential,
  testHospitalConnection,
} from "../lib/api";

export default function HospitalPage({ apiBaseUrl, authToken, onMenuClick, onNavigate }) {
  const [hospitals, setHospitals] = useState([]);
  const [credentials, setCredentials] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [selectedCode, setSelectedCode] = useState("");
  const [tokenInput, setTokenInput] = useState("");
  const [labelInput, setLabelInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [testing, setTesting] = useState({});

  const noop = useCallback(() => {}, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [hlist, clist] = await Promise.all([
        fetchHospitalList(apiBaseUrl, noop, authToken),
        fetchHospitalCredentials(apiBaseUrl, noop, authToken),
      ]);
      setHospitals(hlist?.hospitals || []);
      setCredentials(clist?.credentials || []);
    } catch {
      setError("无法加载医院列表。");
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, authToken, noop]);

  useEffect(() => { load(); }, [load]);

  function getCredential(hospitalCode) {
    return credentials.find((c) => c.hospital_code === hospitalCode);
  }

  async function handleAdd(e) {
    e.preventDefault();
    if (!selectedCode || !tokenInput.trim()) {
      setError("请选择医院并输入 token。");
      return;
    }
    setSaving(true);
    setError("");
    setSuccess("");
    try {
      await addHospitalCredential(apiBaseUrl, noop, authToken, selectedCode, tokenInput.trim(), labelInput.trim());
      setSuccess("已绑定。");
      setShowAdd(false);
      setSelectedCode("");
      setTokenInput("");
      setLabelInput("");
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(hospitalCode) {
    setError("");
    setSuccess("");
    try {
      await deleteHospitalCredential(apiBaseUrl, noop, authToken, hospitalCode);
      setSuccess("已解除绑定。");
      await load();
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleTest(hospitalCode) {
    setTesting((prev) => ({ ...prev, [hospitalCode]: true }));
    setError("");
    try {
      const result = await testHospitalConnection(apiBaseUrl, noop, authToken, hospitalCode);
      if (result?.ok) {
        setSuccess(`连接 ${hospitalCode} 成功。`);
      } else {
        setError(result?.error || "连接失败。");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setTesting((prev) => ({ ...prev, [hospitalCode]: false }));
    }
  }

  const boundCodes = new Set(credentials.map((c) => c.hospital_code));
  const unboundHospitals = hospitals.filter((h) => !boundCodes.has(h.code));

  return (
    <div className="page">
      <header className="page-header">
        <button type="button" className="icon-button" onClick={onMenuClick} aria-label="菜单">
          <ArrowLeft size={20} />
        </button>
        <h2>医院绑定</h2>
      </header>

      <div className="page-body">
        {error && <div className="toast toast--error" onClick={() => setError("")}>{error}</div>}
        {success && <div className="toast toast--success" onClick={() => setSuccess("")}>{success}</div>}

        {loading ? (
          <div className="page-loader"><div className="page-loader__spinner" /><span>加载中…</span></div>
        ) : (
          <>
            {/* Bound credentials */}
            {credentials.length === 0 && (
              <div className="empty-state">
                <p>你还没有绑定任何外部医院。</p>
                <p className="empty-state__hint">绑定后即可在心语上挂该院的号。</p>
              </div>
            )}

            {credentials.map((c) => {
              const hospital = hospitals.find((h) => h.code === c.hospital_code);
              const name = hospital?.name || c.hospital_code;
              const healthStatus = c.last_health_status || "unknown";
              const healthColor =
                healthStatus === "healthy" ? "var(--green-500)"
                : healthStatus === "failed" ? "var(--red-600)"
                : "var(--slate-400)";

              return (
                <div key={c.hospital_code} className="hospital-card">
                  <div className="hospital-card__info">
                    <div className="hospital-card__head">
                      <strong>{name}</strong>
                      <span className="health-dot" style={{ color: healthColor }} title={healthStatus}>
                        <Wifi size={14} />
                      </span>
                    </div>
                    <span className="hospital-card__meta">
                      {c.label || c.hospital_code}
                      {c.last_used_at && ` · 上次调用 ${new Date(c.last_used_at).toLocaleDateString()}`}
                    </span>
                  </div>

                  <div className="hospital-card__actions">
                    <button
                      type="button"
                      className="secondary-btn"
                      disabled={testing[c.hospital_code]}
                      onClick={() => handleTest(c.hospital_code)}
                    >
                      {testing[c.hospital_code] ? "测试中…" : "测试连接"}
                    </button>
                    <button
                      type="button"
                      className="icon-button danger"
                      onClick={() => handleDelete(c.hospital_code)}
                      aria-label="解除绑定"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              );
            })}

            {/* Add new binding */}
            {!showAdd && unboundHospitals.length > 0 && (
              <button type="button" className="primary-btn" onClick={() => setShowAdd(true)} style={{ marginTop: 16 }}>
                <Plus size={16} /> 绑定新医院
              </button>
            )}

            {!showAdd && unboundHospitals.length === 0 && hospitals.length > 0 && (
              <p className="hint-text" style={{ marginTop: 16, textAlign: "center", color: "var(--slate-400)", fontSize: 13 }}>
                已绑定所有平台支持的医院。
              </p>
            )}

            {showAdd && (
              <form className="hospital-add-form" onSubmit={handleAdd}>
                <h3>绑定新医院</h3>

                <div className="hospital-add-form__field">
                  <label>选择医院</label>
                  <select value={selectedCode} onChange={(e) => setSelectedCode(e.target.value)}>
                    <option value="">请选择…</option>
                    {unboundHospitals.map((h) => (
                      <option key={h.code} value={h.code}>{h.name} ({h.code})</option>
                    ))}
                  </select>
                </div>

                {selectedCode && (
                  <>
                    <div className="hospital-add-form__field">
                      <label>Bearer Token</label>
                      <input
                        type="password"
                        value={tokenInput}
                        onChange={(e) => setTokenInput(e.target.value)}
                        placeholder={`粘贴 ${hospitals.find((h) => h.code === selectedCode)?.name || selectedCode} 的 token`}
                        autoFocus
                      />
                    </div>
                    <div className="hospital-add-form__field">
                      <label>备注（可选）</label>
                      <input
                        type="text"
                        value={labelInput}
                        onChange={(e) => setLabelInput(e.target.value)}
                        placeholder="例如：张阿姨的协和账号"
                      />
                    </div>
                    <div className="hospital-add-form__actions">
                      <button type="submit" className="primary-btn" disabled={saving}>
                        {saving ? "保存中…" : "绑定"}
                      </button>
                      <button type="button" className="secondary-btn" onClick={() => { setShowAdd(false); setError(""); }}>
                        取消
                      </button>
                    </div>
                  </>
                )}
              </form>
            )}
          </>
        )}
      </div>
    </div>
  );
}
