import { useEffect, useMemo, useState } from "react";

import { fieldLabel, type CaseStatus, type ReviewCaseDetail, type RuleTrace } from "../types";

interface ReviewPanelProps {
  detail: ReviewCaseDetail;
  selectedField: string | null;
  busy: boolean;
  mutationError: string | null;
  onSelectField: (fieldName: string) => void;
  onSaveField: (fieldName: string, value: string | null) => Promise<void>;
  onDecision: (decision: Extract<CaseStatus, "APPROVE" | "REJECT">, comment: string) => Promise<void>;
}

const RULE_NAMES: Record<string, string> = {
  R001: "政策范围与个人自付比较",
  R002: "个人现金支付兜底",
  R003: "医保结算单差额计算",
  R004: "自付一优先",
  R005: "三项金额等式校验",
  R006: "药店价税合计",
};

const STATUS_LABEL: Record<ReviewCaseDetail["status"], string> = {
  PENDING_REVIEW: "待人工复核",
  APPROVE: "已批准",
  REJECT: "已驳回",
};

function fullTrace(trace: RuleTrace[]): RuleTrace[] {
  const byId = new Map(trace.map((item) => [item.rule_id, item]));
  return Object.keys(RULE_NAMES).map((ruleId) => byId.get(ruleId) ?? {
    rule_id: ruleId,
    outcome: "NOT_EVALUATED",
    reason: "前序规则已停止，未继续执行",
  });
}

function outcomeLabel(outcome: string): string {
  if (outcome === "MATCHED" || outcome === "CALCULATED") return "命中";
  if (outcome === "NEEDS_REVIEW") return "需复核";
  if (outcome === "SKIPPED" || outcome === "NOT_EVALUATED") return "未执行";
  return "未命中";
}

export function ReviewPanel({
  detail,
  selectedField,
  busy,
  mutationError,
  onSelectField,
  onSaveField,
  onDecision,
}: ReviewPanelProps) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [comment, setComment] = useState("");
  const terminal = detail.status !== "PENDING_REVIEW";

  useEffect(() => {
    setValues(Object.fromEntries(Object.entries(detail.fields).map(([name, field]) => [name, field.value ?? ""])));
    setComment("");
  }, [detail.case_id, detail.version, detail.fields]);

  const trace = useMemo(() => fullTrace(detail.rule_decision.trace), [detail.rule_decision.trace]);

  return (
    <aside className="panel review-panel" aria-label="字段审核区">
      <div className="review-topline">
        <div>
          <p className="eyebrow">STRUCTURED REVIEW</p>
          <h2>结构化审核</h2>
        </div>
        <span className={`decision-status status-${detail.status.toLowerCase()}`}>
          {STATUS_LABEL[detail.status]}
        </span>
      </div>

      {mutationError && <div className="inline-error" role="alert">{mutationError}</div>}

      <section className="review-section fields-section">
        <div className="section-title">
          <span className="section-number">01</span>
          <div><h3>关键字段</h3><p>修改后自动重跑规则</p></div>
        </div>

        <div className="field-table">
          {Object.entries(detail.fields).map(([name, field]) => {
            const label = fieldLabel(name, field);
            const dirty = values[name] !== (field.value ?? "");
            return (
              <div className={`field-row ${selectedField === name ? "is-selected" : ""}`} key={name}>
                <button type="button" className="field-focus" onClick={() => onSelectField(name)} aria-label={`查看${label}证据`}>
                  <span className="field-name">{label}</span>
                  <span className="confidence-copy">{Math.round(field.confidence * 100)}%</span>
                </button>
                <div className="field-input-line">
                  <label>
                    <span className="sr-only">{label}</span>
                    <input
                      type="text"
                      inputMode="decimal"
                      value={values[name] ?? ""}
                      onFocus={() => onSelectField(name)}
                      onChange={(event) => setValues((current) => ({ ...current, [name]: event.target.value }))}
                      disabled={terminal || busy}
                    />
                  </label>
                  <button
                    type="button"
                    className="save-field"
                    onClick={() => onSaveField(name, values[name]?.trim() || null)}
                    disabled={!dirty || terminal || busy}
                    aria-label={`保存${label}修正`}
                  >保存</button>
                </div>
                <div className="confidence-track" aria-hidden="true"><span style={{ width: `${field.confidence * 100}%` }} /></div>
                <progress className="sr-only" max="1" value={field.confidence} aria-label={`${label}置信度`} />
              </div>
            );
          })}
        </div>
      </section>

      <section className="review-section decision-section">
        <div className="section-title">
          <span className="section-number">02</span>
          <div><h3>规则结论</h3><p>规则版本 {detail.rule_decision.rule_version}</p></div>
        </div>
        <div className={`amount-result result-${detail.rule_decision.status.toLowerCase()}`}>
          <div>
            <small>{detail.rule_decision.status === "CALCULATED" ? "建议报销基数" : "当前结论"}</small>
            <strong>{detail.rule_decision.target_amount ? `目标金额 ¥${detail.rule_decision.target_amount}` : "等待人工确认"}</strong>
          </div>
          <span>{detail.rule_decision.selected_rule ? `命中 ${detail.rule_decision.selected_rule}` : "规则未定"}</span>
        </div>
        {detail.rule_decision.warnings.length > 0 && (
          <ul className="warning-list">
            {detail.rule_decision.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        )}
      </section>

      <section className="review-section trace-section" aria-label="六条规则执行轨迹">
        <div className="section-title">
          <span className="section-number">03</span>
          <div><h3>执行轨迹</h3><p>严格按 R001 → R006 顺序</p></div>
        </div>
        <ol className="trace-list">
          {trace.map((item) => (
            <li key={item.rule_id} className={`trace-${item.outcome.toLowerCase()}`}>
              <span className="trace-id">{item.rule_id}</span>
              <span className="trace-copy"><strong>{RULE_NAMES[item.rule_id]}</strong><small>{item.reason}</small></span>
              <span className="trace-outcome">{outcomeLabel(item.outcome)}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="review-actions">
        <label htmlFor="review-comment">审核意见</label>
        <textarea
          id="review-comment"
          value={comment}
          onChange={(event) => setComment(event.target.value)}
          placeholder="填写本次判断依据（必填）"
          maxLength={500}
          disabled={terminal || busy}
        />
        <div className="action-row">
          <button type="button" className="reject-button" disabled={terminal || busy || !comment.trim()} onClick={() => onDecision("REJECT", comment.trim())}>驳回复核</button>
          <button type="button" className="approve-button" disabled={terminal || busy || !comment.trim()} onClick={() => onDecision("APPROVE", comment.trim())}>批准通过</button>
        </div>
      </section>
    </aside>
  );
}
