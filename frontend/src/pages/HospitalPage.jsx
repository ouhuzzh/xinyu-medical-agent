import React, { useState, useCallback, useEffect } from "react";
import { ArrowLeft, Plus, Trash2, Wifi, ExternalLink, Building2, CheckCircle2, XCircle, Loader2, Unlink, ShieldCheck, Clock } from "lucide-react";
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

  const healthConfig = {
    healthy: { color: "var(--green-500)", bg: "rgba(34,197,94,0.08)", label: "在线", icon: CheckCircle2 },
    failed: { color: "var(--red-600)", bg: "rgba(220,38,38,0.07)", label: "离线", icon: XCircle },
    unknown: { color: "var(--slate-400)", bg: "rgba(148,163,184,0.06)", label: "未知", icon: Wifi },
  };

  return (
    <div className="page">
      <header className="page-header">
        <button type="button" className="icon-button" onClick={onMenuClick} aria-label="菜单">
          <ArrowLeft size={20} />
        </button>
        <h2>医院绑定</h2>
        <span className="page-header__badge">{credentials.length}/{hospitals.length}</span>
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
              <div className="empty-state empty-state--hospital">
                <div className="empty-state__icon"><Building2 size={36} strokeWidth={1.4} /></div>
                <p className="empty-state__title">你还没有绑定任何外部医院</p>
                <p className="empty-state__subtitle">绑定后即可在心语上挂该院的号，享受一站式预约服务。</p>
              </div>
            )}

            {credentials.map((c) => {
              const hospital = hospitals.find((h) => h.code === c.hospital_code);
              const name = hospital?.name || c.hospital_code;
              const healthStatus = c.last_health_status || "unknown";
              const hc = healthConfig[healthStatus] || healthConfig.unknown;
              const HealthIcon = hc.icon;

              return (
                <div key={c.hospital_code} className="hospital-card">
                  <div className="hospital-card__icon-wrap">
                    <Building2 size={18} strokeWidth={1.6} />
                  </div>

                  <div className="hospital-card__info">
                    <div className="hospital-card__head">
                      <strong>{name}</strong>
                      <span
                        className="hospital-badge"
                        style={{ color: hc.color, background: hc.bg }}
                        title={`${hc.label}${healthStatus !== "unknown" ? ` (${healthStatus})` : ""}`}
                      >
                        <HealthIcon size={12} strokeWidth={2.2} />
                        {hc.label}
                      </span>
                    </div>
                    <div className="hospital-card__meta-row">
                      <span className="hospital-card__meta">
                        <ShieldCheck size={11} />
                        {c.label || c.hospital_code}
                      </span>
                      {c.last_used_at && (
                        <span className="hospital-card__meta">
                          <Clock size={11} />
                          上次调用 {new Date(c.last_used_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="hospital-card__actions">
                    <button
                      type="button"
                      className="secondary-btn secondary-btn--test"
                      disabled={testing[c.hospital_code]}
                      onClick={() => handleTest(c.hospital_code)}
                    >
                      {testing[c.hospital_code]
                        ? <><Loader2 size={13} className="spin" /> 测试中</>
                        : <><Wifi size={13} /> 连接测试</>
                      }
                    </button>
                    <button
                      type="button"
                      className="icon-btn-danger"
                      onClick={() => handleDelete(c.hospital_code)}
                      title="解除绑定"
                    >
                      <Unlink size={16} />
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
              <div className="hospital-all-bound">
                <CheckCircle2 size={18} strokeWidth={1.8} />
                <span>已绑定所有平台支持的医院</span>
              </div>
            )}

            {showAdd && (
              <form className="hospital-add-form" onSubmit={handleAdd}>
                <h3 className="hospital-add-form__title">
                  <Plus size={17} strokeWidth={2} /> 绑定新医院
                </h3>

                <div className="hospital-add-form__field">
                  <label><Building2 size={13} /> 选择医院</label>
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
                      <label><ShieldCheck size={13} /> Bearer Token</label>
                      <input
                        type="password"
                        value={tokenInput}
                        onChange={(e) => setTokenInput(e.target.value)}
                        placeholder={`粘贴 ${hospitals.find((h) => h.code === selectedCode)?.name || selectedCode} 的 token`}
                        autoFocus
                      />
                      <span className="field-hint">Token 仅用于与该院 MCP 服务通信，不会存储明文。</span>
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
                        {saving ? <><Loader2 size={14} className="spin" /> 绑定中</> : "确认绑定"}
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
