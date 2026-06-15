import React, { useState, useCallback, useEffect } from "react";
import { ArrowLeft, Wifi, Building2, CheckCircle2, XCircle, Loader2, Unlink, ShieldCheck, Clock, Lock, RefreshCw } from "lucide-react";
import {
  fetchHospitalList,
  fetchHospitalCredentials,
  addHospitalCredential,
  deleteHospitalCredential,
  testHospitalConnection,
} from "../lib/api";

export default function HospitalPage({ apiBaseUrl, authToken, onMenuClick }) {
  const [hospitals, setHospitals] = useState([]);
  const [credentials, setCredentials] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandingCode, setExpandingCode] = useState(null);
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

  async function handleAdd(hospitalCode) {
    if (!tokenInput.trim()) { setError("请输入 token。"); return; }
    setSaving(true); setError(""); setSuccess("");
    try {
      await addHospitalCredential(apiBaseUrl, noop, authToken, hospitalCode, tokenInput.trim(), labelInput.trim());
      setSuccess(`${hospitalCode} 已绑定。`);
      setExpandingCode(null);
      setTokenInput(""); setLabelInput("");
      await load();
    } catch (err) { setError(err.message); } finally { setSaving(false); }
  }

  async function handleDelete(hospitalCode) {
    setError(""); setSuccess("");
    try {
      await deleteHospitalCredential(apiBaseUrl, noop, authToken, hospitalCode);
      setSuccess("已解除绑定。"); await load();
    } catch (err) { setError(err.message); }
  }

  async function handleTest(hospitalCode) {
    setTesting((prev) => ({ ...prev, [hospitalCode]: true }));
    setError("");
    try {
      const result = await testHospitalConnection(apiBaseUrl, noop, authToken, hospitalCode);
      if (result?.ok) setSuccess(`连接 ${hospitalCode} 成功。`);
      else setError(result?.error || "连接失败。");
    } catch (err) { setError(err.message); }
    finally { setTesting((prev) => ({ ...prev, [hospitalCode]: false })); }
  }

  const boundMap = Object.fromEntries(credentials.map((c) => [c.hospital_code, c]));

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
        <span className="page-header__badge">{Object.keys(boundMap).length}/{hospitals.length}</span>
      </header>

      <div className="page-body page-body--hospitals">
        {error && <div className="toast toast--error" onClick={() => setError("")}>{error}</div>}
        {success && <div className="toast toast--success" onClick={() => setSuccess("")}>{success}</div>}

        {loading ? (
          <div className="page-loader"><div className="page-loader__spinner" /><span>加载中…</span></div>
        ) : hospitals.length === 0 ? (
          <div className="hospital-empty">
            <div className="hospital-empty__icon">
              <Building2 size={22} strokeWidth={1.8} />
            </div>
            <div className="hospital-empty__copy">
              <h3>当前没有可绑定医院</h3>
              <p>管理员添加医院 MCP 注册表后，这里会显示可绑定的医院。</p>
            </div>
            <button type="button" className="secondary-btn hospital-empty__refresh" onClick={load}>
              <RefreshCw size={14} />
              刷新
            </button>
          </div>
        ) : (
          <div className="hospital-grid">
            {hospitals.map((hospital) => {
              const cred = boundMap[hospital.code];
              const isBound = !!cred;
              const isExpanding = expandingCode === hospital.code;

              if (isBound && !isExpanding) {
                // ---- 已绑定卡片 ----
                const healthStatus = cred.last_health_status || "unknown";
                const hc = healthConfig[healthStatus] || healthConfig.unknown;
                const HealthIcon = hc.icon;
                return (
                  <div key={hospital.code} className="hospital-card hospital-card--bound">
                    <div className="hospital-card__top">
                      <div className="hospital-card__icon-wrap">
                        <Building2 size={18} strokeWidth={1.6} />
                      </div>
                      <div className="hospital-card__info">
                        <div className="hospital-card__head">
                          <strong>{hospital.name}</strong>
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
                          <span className="hospital-card__meta"><ShieldCheck size={11} />{cred.label || hospital.code}</span>
                          {cred.last_used_at && (
                            <span className="hospital-card__meta"><Clock size={11} />上次调用 {new Date(cred.last_used_at).toLocaleDateString()}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="hospital-card__actions">
                      <button type="button" className="secondary-btn secondary-btn--test"
                        disabled={testing[hospital.code]} onClick={() => handleTest(hospital.code)}>
                        {testing[hospital.code]
                          ? <><Loader2 size={13} className="spin" /> 测试中</>
                          : <><Wifi size={13} /> 连接测试</>}
                      </button>
                      <button type="button" className="icon-btn-danger" onClick={() => handleDelete(hospital.code)} title="解除绑定">
                        <Unlink size={16} />
                      </button>
                    </div>
                  </div>
                );
              }

              // ---- 未绑定 / 展开中 卡片 ----
              return (
                <div key={hospital.code}
                  className={`hospital-card hospital-card--unbound${isExpanding ? " hospital-card--expanding" : ""}`}>
                  {!isExpanding ? (
                    <>
                      <div className="hospital-card__top">
                        <div className="hospital-card__icon-wrap hospital-card__icon-wrap--dim">
                          <Building2 size={18} strokeWidth={1.6} />
                        </div>
                        <div className="hospital-card__info">
                          <div className="hospital-card__head">
                            <strong>{hospital.name}</strong>
                            <span className="hospital-badge hospital-badge--muted">未绑定</span>
                          </div>
                          <p className="hospital-card__code">{hospital.code}</p>
                        </div>
                      </div>
                      <button type="button" className="primary-btn primary-btn--sm"
                        onClick={() => { setExpandingCode(hospital.code); setError(""); }}>
                        <Lock size={14} /> 绑定此医院
                      </button>
                    </>
                  ) : (
                    <form onSubmit={(e) => { e.preventDefault(); handleAdd(hospital.code); }} className="hospital-bind-form">
                      <div className="hospital-bind-form__header">
                        <div className="hospital-card__icon-wrap">
                          <Building2 size={16} strokeWidth={1.6} />
                        </div>
                        <div>
                          <strong className="hospital-bind-form__name">{hospital.name}</strong>
                          <span className="hospital-card__code">{hospital.code}</span>
                        </div>
                      </div>
                      <div className="hospital-bind-form__field">
                        <label><ShieldCheck size={12} /> Bearer Token</label>
                        <input type="password" value={tokenInput}
                          onChange={(e) => setTokenInput(e.target.value)}
                          placeholder={`粘贴 ${hospital.name} 的 token`} autoFocus />
                        <span className="field-hint">Token 仅用于与该院 MCP 服务通信。</span>
                      </div>
                      <div className="hospital-bind-form__field">
                        <label>备注（可选）</label>
                        <input type="text" value={labelInput}
                          onChange={(e) => setLabelInput(e.target.value)}
                          placeholder="例如：张阿姨的协和账号" />
                      </div>
                      <div className="hospital-bind-form__actions">
                        <button type="submit" className="primary-btn" disabled={saving}>
                          {saving ? <><Loader2 size={13} className="spin" /> 绑定中</> : "确认绑定"}
                        </button>
                        <button type="button" className="secondary-btn"
                          onClick={() => { setExpandingCode(null); setError(""); }}>
                          取消
                        </button>
                      </div>
                    </form>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
