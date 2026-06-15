import React from "react";
import { Brain, CalendarCheck, HelpCircle, Activity, Heart, Shield, Stethoscope, ClipboardList } from "lucide-react";
import { EMPTY_STATE_CAPABILITIES, STARTER_PROMPTS } from "../constants/app";
import XinyuLogo from "./XinyuLogo";

const promptIcons = {
  hypertension: Activity,
  triage: HelpCircle,
  booking: CalendarCheck,
  cancel: Brain,
};

const capIcons = [Stethoscope, Shield, CalendarCheck, ClipboardList];

const EmptyState = React.memo(function EmptyState({ onSendMessage }) {
  return (
    <div className="empty-state">
      <div className="empty-state__bg-ornament" aria-hidden="true">
        <svg width="320" height="320" viewBox="0 0 320 320" fill="none">
          <circle cx="160" cy="160" r="120" stroke="rgba(13,148,136,0.06)" strokeWidth="1.5" />
          <circle cx="160" cy="160" r="80" stroke="rgba(13,148,136,0.04)" strokeWidth="1" />
          <circle cx="160" cy="160" r="40" stroke="rgba(13,148,136,0.03)" strokeWidth="1" />
          <line x1="160" y1="30" x2="160" y2="290" stroke="rgba(13,148,136,0.02)" strokeWidth="1" />
          <line x1="30" y1="160" x2="290" y2="160" stroke="rgba(13,148,136,0.02)" strokeWidth="1" />
        </svg>
      </div>

      <div className="empty-state__hero">
        <div className="empty-state__icon-wrap">
          <div className="empty-state__icon-ring" />
          <div className="empty-state__icon-ring empty-state__icon-ring--outer" />
          <XinyuLogo size={44} variant="glow" animated={true} className="empty-state__icon" />
        </div>
        <h2 className="empty-state__title">你好，我是心语医疗小助手</h2>
        <p className="empty-state__subtitle">
          专业的 AI 医疗咨询助手，随时为您提供健康指导与就医帮助
        </p>
        <ul className="empty-state__caps">
          {EMPTY_STATE_CAPABILITIES.map((cap, i) => {
            const CapIcon = capIcons[i] || HelpCircle;
            return (
              <li key={cap}>
                <CapIcon size={12} className="empty-state__cap-icon" />
                {cap}
              </li>
            );
          })}
        </ul>

        <div className="empty-state__metrics" aria-label="助手能力概览">
          <div>
            <strong>24/7</strong>
            <span>健康咨询</span>
          </div>
          <div>
            <strong>RAG</strong>
            <span>知识增强</span>
          </div>
          <div>
            <strong>3-Step</strong>
            <span>预约确认</span>
          </div>
        </div>
      </div>

      <div className="prompt-grid">
        {STARTER_PROMPTS.map(({ key, text }, i) => {
          const Icon = promptIcons[key] || HelpCircle;
          return (
          <button
            key={text}
            type="button"
            className="prompt-card"
            style={{ animationDelay: `${0.1 + i * 0.07}s` }}
            onClick={() => onSendMessage(text)}
          >
            <span className="prompt-card__icon">
              <Icon size={18} />
            </span>
            <span className="prompt-card__text">{text}</span>
          </button>
          );
        })}
      </div>

      <p className="empty-state__disclaimer">
        <Heart size={11} className="empty-state__disclaimer-icon" />
        本助手仅提供健康参考，不能替代专业医生的诊断与治疗
      </p>
    </div>
  );
});

export default EmptyState;
