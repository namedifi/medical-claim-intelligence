import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import { DocumentRail } from "./components/DocumentRail";
import { EvidenceViewer } from "./components/EvidenceViewer";
import { ReviewPanel } from "./components/ReviewPanel";
import type { CaseStatus, CaseSummary, ReviewCaseDetail } from "./types";

type PageState = "loading" | "ready" | "error";

function friendlyError(error: unknown): string {
  if (error instanceof api.ApiError && error.status === 409) return "案例已被更新，请刷新后重试";
  return "操作未完成，请检查服务后重试";
}

export default function App() {
  const [pageState, setPageState] = useState<PageState>("loading");
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ReviewCaseDetail | null>(null);
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const requestId = useRef(0);

  const loadDetail = useCallback(async (caseId: string) => {
    const currentRequest = ++requestId.current;
    setSelectedId(caseId);
    setDetail(null);
    setMutationError(null);
    try {
      const nextDetail = await api.getCase(caseId);
      if (currentRequest !== requestId.current) return;
      setDetail(nextDetail);
      setSelectedField(Object.keys(nextDetail.fields)[0] ?? null);
    } catch {
      if (currentRequest !== requestId.current) return;
      setMutationError("无法加载案例详情");
    }
  }, []);

  const loadCases = useCallback(async () => {
    const currentRequest = ++requestId.current;
    setPageState("loading");
    setMutationError(null);
    try {
      const nextCases = await api.listCases();
      if (currentRequest !== requestId.current) return;
      setCases(nextCases);
      setPageState("ready");
      if (nextCases.length > 0) await loadDetail(nextCases[0].case_id);
    } catch {
      if (currentRequest !== requestId.current) return;
      setPageState("error");
    }
  }, [loadDetail]);

  useEffect(() => { void loadCases(); }, [loadCases]);

  function syncSummary(nextDetail: ReviewCaseDetail): void {
    setCases((current) => current.map((item) => item.case_id === nextDetail.case_id ? {
      ...item,
      version: nextDetail.version,
      status: nextDetail.status,
      rule_status: nextDetail.rule_decision.status,
    } : item));
  }

  async function saveField(fieldName: string, value: string | null): Promise<void> {
    if (!detail) return;
    const caseId = detail.case_id;
    const currentRequest = requestId.current;
    setBusy(true);
    setMutationError(null);
    try {
      const nextDetail = await api.updateField(caseId, fieldName, value, detail.version, "人工复核修正");
      syncSummary(nextDetail);
      if (currentRequest !== requestId.current || nextDetail.case_id !== caseId) return;
      setDetail(nextDetail);
    } catch (error) {
      if (currentRequest !== requestId.current) return;
      setMutationError(friendlyError(error));
    } finally {
      setBusy(false);
    }
  }

  async function recordDecision(decision: Extract<CaseStatus, "APPROVE" | "REJECT">, comment: string): Promise<void> {
    if (!detail) return;
    const caseId = detail.case_id;
    const currentRequest = requestId.current;
    setBusy(true);
    setMutationError(null);
    try {
      const nextDetail = await api.decideCase(caseId, decision, detail.version, comment);
      syncSummary(nextDetail);
      if (currentRequest !== requestId.current || nextDetail.case_id !== caseId) return;
      setDetail(nextDetail);
    } catch (error) {
      if (currentRequest !== requestId.current) return;
      setMutationError(friendlyError(error));
    } finally {
      setBusy(false);
    }
  }

  function exportCase(): void {
    if (!detail) return;
    const blob = new Blob([JSON.stringify(detail, null, 2)], { type: "application/json" });
    const href = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = href;
    link.download = `${detail.case_id}.json`;
    document.body.append(link);
    try {
      link.click();
    } finally {
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(href), 0);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">审</span>
          <div>
            <h1>医疗票据智能预审</h1>
            <p>OpenCV · PaddleOCR · Qwen-VL · Sequential Rules</p>
          </div>
        </div>
        <div className="header-actions">
          <span className="demo-mode"><i aria-hidden="true" />本地 DEMO</span>
          <button type="button" className="secondary-button" onClick={() => void loadCases()}>刷新案例</button>
          <button type="button" className="secondary-button" onClick={exportCase} disabled={!detail}>导出 JSON</button>
        </div>
      </header>

      {pageState === "loading" && (
        <main className="state-page" role="status">
          <span className="loading-rule" aria-hidden="true" />
          <h2>正在加载审核案例</h2>
          <p>读取合成票据与规则结果</p>
        </main>
      )}

      {pageState === "error" && (
        <main className="state-page" role="alert">
          <span className="error-glyph" aria-hidden="true">!</span>
          <h2>无法加载审核案例</h2>
          <p>请确认本地 API 已启动。</p>
          <button type="button" className="primary-button" onClick={() => void loadCases()}>重新加载</button>
        </main>
      )}

      {pageState === "ready" && cases.length === 0 && (
        <main className="state-page">
          <span className="empty-glyph" aria-hidden="true">0</span>
          <h2>暂无待审核案例</h2>
          <p>队列为空，新的合成案例出现后会显示在这里。</p>
        </main>
      )}

      {pageState === "ready" && cases.length > 0 && (
        <div className="workspace-grid">
          <DocumentRail cases={cases} selectedId={selectedId} onSelect={(caseId) => void loadDetail(caseId)} />
          {detail ? (
            <>
              <EvidenceViewer detail={detail} selectedField={selectedField} onSelectField={setSelectedField} />
              <ReviewPanel
                detail={detail}
                selectedField={selectedField}
                busy={busy}
                mutationError={mutationError}
                onSelectField={setSelectedField}
                onSaveField={saveField}
                onDecision={recordDecision}
              />
            </>
          ) : (
            <section className="panel detail-loading" role="status">
              <span className="loading-rule" aria-hidden="true" />
              <p>{mutationError ?? "正在载入案例详情"}</p>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
