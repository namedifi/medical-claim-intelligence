import type { CSSProperties } from "react";

import { fieldLabel, type ReviewCaseDetail } from "../types";

interface EvidenceViewerProps {
  detail: ReviewCaseDetail;
  selectedField: string | null;
  onSelectField: (fieldName: string) => void;
}

function boxStyle(bbox: number[][] | undefined): CSSProperties {
  if (!bbox || bbox.length === 0) return {};
  const xs = bbox.map((point) => point[0] ?? 0);
  const ys = bbox.map((point) => point[1] ?? 0);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  return {
    left: `${minX * 100}%`,
    top: `${minY * 100}%`,
    width: `${(Math.max(...xs) - minX) * 100}%`,
    height: `${(Math.max(...ys) - minY) * 100}%`,
  };
}

export function EvidenceViewer({ detail, selectedField, onSelectField }: EvidenceViewerProps) {
  const fieldEntries = Object.entries(detail.fields);
  const activeName = selectedField && detail.fields[selectedField] ? selectedField : fieldEntries[0]?.[0] ?? null;
  const activeField = activeName ? detail.fields[activeName] : undefined;
  const activeEvidence = activeField?.evidence[0];

  return (
    <main className="panel evidence-panel" aria-label="票据证据区">
      <div className="panel-heading evidence-heading">
        <div>
          <p className="eyebrow">SYNTHETIC DOCUMENT · PAGE 1/{detail.document.metadata.page_count}</p>
          <h2>{detail.document.metadata.label}</h2>
        </div>
        <div className="document-meta">
          <span>OCR + VLM</span>
          <strong>合成源</strong>
        </div>
      </div>

      <div className="document-stage">
        <div className="synthetic-ticket" aria-label="合成票据预览">
          <div className="ticket-watermark">DEMO · 非真实票据</div>
          <div className="ticket-head">
            <span>门诊</span>
            <div>
              <h3>示例医疗机构收费票据（电子）</h3>
              <p>SYNTHETIC MEDICAL RECEIPT</p>
            </div>
            <span>第 1 页</span>
          </div>
          <div className="ticket-identifiers">
            <span>票据代码：DEMO-000001</span>
            <span>开票日期：2026-08-26</span>
            <span>交款人：合成用户</span>
          </div>
          <div className="ticket-table">
            <div className="ticket-table-head"><span>项目名称</span><span>金额（元）</span><span>备注</span></div>
            <div><span>合成检查项目</span><span>80.00</span><span>演示</span></div>
            <div><span>合成诊疗项目</span><span>20.00</span><span>演示</span></div>
          </div>
          <div className="ticket-totals">
            {fieldEntries.map(([name, field]) => (
              <span key={name}><b>{fieldLabel(name, field)}</b> {field.value ?? "—"}</span>
            ))}
          </div>
          <div className="ticket-footer">
            <span>收费单位：示例医疗机构</span>
            <span>本票据仅用于界面演示</span>
          </div>

          {fieldEntries.map(([name, field]) => (
            <span
              key={name}
              className={`evidence-outline ${activeName === name ? "is-active" : ""}`}
              style={boxStyle(field.evidence[0]?.bbox)}
            >
              <button
                type="button"
                className="evidence-box"
                onClick={() => onSelectField(name)}
                aria-label={`定位证据 ${fieldLabel(name, field)}`}
                title={fieldLabel(name, field)}
              />
            </span>
          ))}
        </div>
        <span className="synthetic-ribbon">合成示例 · 非真实票据</span>
      </div>

      <section className="evidence-readout" aria-live="polite">
        <div className="readout-title">
          <span className="coordinate-icon" aria-hidden="true">⌖</span>
          <div>
            <small>当前证据</small>
            <strong>{activeName ? fieldLabel(activeName, activeField) : "未选择字段"}</strong>
          </div>
        </div>
        <blockquote>{activeEvidence?.ocr_text ?? "当前字段没有 OCR 证据"}</blockquote>
        <div className="evidence-stats">
          <span>页码 {activeEvidence ? activeEvidence.page_index + 1 : "—"}</span>
          <span>置信度 {activeField ? `${Math.round(activeField.confidence * 100)}%` : "—"}</span>
        </div>
      </section>
    </main>
  );
}
