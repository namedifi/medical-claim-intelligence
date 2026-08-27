import type { CaseSummary } from "../types";

interface DocumentRailProps {
  cases: CaseSummary[];
  selectedId: string | null;
  onSelect: (caseId: string) => void;
}

const STATUS_TEXT: Record<CaseSummary["status"], string> = {
  PENDING_REVIEW: "待复核",
  APPROVE: "已批准",
  REJECT: "已驳回",
};

function compactId(caseId: string): string {
  const suffix = caseId.match(/(\d+)$/)?.[1] ?? caseId.slice(-6);
  return `CASE · ${suffix}`;
}

export function DocumentRail({ cases, selectedId, onSelect }: DocumentRailProps) {
  return (
    <aside className="panel document-rail" aria-label="案例队列">
      <div className="panel-heading rail-heading">
        <div>
          <p className="eyebrow">REVIEW QUEUE</p>
          <h2>审核队列</h2>
        </div>
        <span className="count-chip" aria-label={`${cases.length} 个案例`}>{cases.length}</span>
      </div>

      <div className="queue-summary">
        <span className="summary-mark" aria-hidden="true" />
        <div>
          <strong>{cases.filter((item) => item.status === "PENDING_REVIEW").length} 份待处理</strong>
          <span>按规则风险优先排列</span>
        </div>
      </div>

      <nav className="case-list" aria-label="合成案例列表">
        {cases.map((item, index) => (
          <button
            type="button"
            className={`case-row ${selectedId === item.case_id ? "is-selected" : ""}`}
            key={item.case_id}
            onClick={() => onSelect(item.case_id)}
            aria-current={selectedId === item.case_id ? "true" : undefined}
            aria-label={`打开案例 ${item.document_label}`}
          >
            <span className="case-index">{String(index + 1).padStart(2, "0")}</span>
            <span className="case-copy">
              <strong>{compactId(item.case_id)}</strong>
              <span>{item.document_type === "medical_receipt" ? "医疗收费票据" : "待分类票据"}</span>
              <small className={`status-text status-${item.status.toLowerCase()}`}>
                {STATUS_TEXT[item.status]} · v{item.version}
              </small>
            </span>
            <span className="row-arrow" aria-hidden="true">›</span>
          </button>
        ))}
      </nav>

      <div className="demo-note">
        <span className="demo-note-mark" aria-hidden="true">D</span>
        <p><strong>DEMO 数据</strong><br />所有内容均为程序合成，不含真实个人信息。</p>
      </div>
    </aside>
  );
}
