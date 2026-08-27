/// <reference types="vite/client" />

import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import * as api from "./api";
import styleSource from "./styles.css?raw";
import type { CaseSummary, ReviewCaseDetail, RuleTrace } from "./types";

vi.mock("./api", () => ({
  listCases: vi.fn(),
  getCase: vi.fn(),
  updateField: vi.fn(),
  decideCase: vi.fn(),
}));

const summary: CaseSummary = {
  case_id: "synthetic-medical-receipt-001",
  version: 1,
  status: "PENDING_REVIEW",
  document_type: "medical_receipt",
  document_label: "合成医疗收费票据 A",
  rule_status: "NEEDS_REVIEW",
};

const trace: RuleTrace[] = [
  { rule_id: "R001", outcome: "NEEDS_REVIEW", reason: "符合政策范围金额大于个人自付金额，等待人工确认" },
  { rule_id: "R002", outcome: "SKIPPED", reason: "前序规则已停止" },
];

const detail: ReviewCaseDetail = {
  case_id: summary.case_id,
  version: 1,
  status: "PENDING_REVIEW",
  document: {
    metadata: { label: summary.document_label, source: "synthetic-demo", page_count: 1 },
  },
  fields: {
    policy_scope_amount: {
      value: "100.00",
      confidence: 0.99,
      raw_name: "合成字段：符合政策范围金额",
      evidence: [{
        page_index: 0,
        ocr_text: "合成文字 符合政策范围金额 100.00",
        bbox: [[0.1, 0.1], [0.55, 0.1], [0.55, 0.18], [0.1, 0.18]],
        source_image: "synthetic-demo",
      }],
    },
    pooled_fund_payment: {
      value: "70.00",
      confidence: 0.92,
      raw_name: "合成字段：医保统筹基金支付",
      evidence: [{
        page_index: 0,
        ocr_text: "合成文字 医保统筹基金支付 70.00",
        bbox: [[0.1, 0.62], [0.55, 0.62], [0.55, 0.7], [0.1, 0.7]],
        source_image: "synthetic-demo",
      }],
    },
    personal_self_pay: {
      value: "20.00",
      confidence: 0.78,
      raw_name: "合成字段：个人自付",
      evidence: [{
        page_index: 0,
        ocr_text: "合成文字 个人自付 20.00",
        bbox: [[0.1, 0.72], [0.48, 0.72], [0.48, 0.8], [0.1, 0.8]],
        source_image: "synthetic-demo",
      }],
    },
  },
  rule_decision: {
    status: "NEEDS_REVIEW",
    selected_rule: null,
    target_amount: null,
    formula: null,
    inputs: {},
    trace,
    warnings: ["个人自付金额需要复核"],
    rule_version: "1.0.0",
  },
  audit_events: [],
};

const calculatedDetail: ReviewCaseDetail = {
  ...detail,
  version: 2,
  fields: {
    ...detail.fields,
    personal_self_pay: { ...detail.fields.personal_self_pay, value: "30.00", confidence: 1 },
  },
  rule_decision: {
    ...detail.rule_decision,
    status: "CALCULATED",
    selected_rule: "R001",
    target_amount: "100.00",
    formula: "符合政策范围金额",
    trace: [{ rule_id: "R001", outcome: "MATCHED", reason: "优先规则命中" }],
    warnings: [],
  },
};

const mocked = {
  listCases: vi.mocked(api.listCases),
  getCase: vi.mocked(api.getCase),
  updateField: vi.mocked(api.updateField),
  decideCase: vi.mocked(api.decideCase),
};

function loadDefault(): void {
  mocked.listCases.mockResolvedValue([summary]);
  mocked.getCase.mockResolvedValue(detail);
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

describe("medical claim review workspace", () => {
  beforeEach(() => vi.resetAllMocks());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("shows an explicit loading state", () => {
    mocked.listCases.mockReturnValue(new Promise(() => undefined));

    render(<App />);

    expect(screen.getByRole("status")).toHaveTextContent("正在加载审核案例");
  });

  it("shows a retryable error state", async () => {
    mocked.listCases.mockRejectedValue(new Error("offline"));

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("无法加载审核案例");
    expect(screen.getByRole("button", { name: "重新加载" })).toBeEnabled();
  });

  it("shows an empty queue without rendering the workspace", async () => {
    mocked.listCases.mockResolvedValue([]);

    render(<App />);

    expect(await screen.findByText("暂无待审核案例")).toBeInTheDocument();
    expect(screen.queryByLabelText("字段审核区")).not.toBeInTheDocument();
  });

  it("switches cases from the document rail", async () => {
    const second = { ...summary, case_id: "synthetic-medical-receipt-002", document_label: "合成医疗收费票据 B" };
    const secondDetail = { ...detail, case_id: second.case_id, document: { metadata: { ...detail.document.metadata, label: second.document_label } } };
    mocked.listCases.mockResolvedValue([summary, second]);
    mocked.getCase.mockImplementation(async (id) => id === second.case_id ? secondDetail : detail);

    render(<App />);
    await screen.findByText("合成医疗收费票据 A");
    await userEvent.click(screen.getByRole("button", { name: /打开案例 合成医疗收费票据 B/ }));

    expect(await screen.findByRole("heading", { name: "合成医疗收费票据 B" })).toBeInTheDocument();
    expect(mocked.getCase).toHaveBeenLastCalledWith(second.case_id);
  });

  it("renders evidence and the ordered rule trace", async () => {
    loadDefault();

    render(<App />);

    const rules = await screen.findByLabelText("六条规则执行轨迹");
    expect(within(rules).getByText("R001")).toBeInTheDocument();
    expect(within(rules).getByText(/等待人工确认/)).toBeInTheDocument();
    expect(within(rules).getByText("R002")).toBeInTheDocument();
    expect(await screen.findByText("合成文字 符合政策范围金额 100.00")).toBeInTheDocument();
    expect(screen.getByText("合成示例 · 非真实票据")).toBeInTheDocument();
  });

  it("saves a field correction and displays the recomputed decision", async () => {
    loadDefault();
    mocked.updateField.mockResolvedValue(calculatedDetail);
    render(<App />);
    const input = await screen.findByRole("textbox", { name: "个人自付" });

    await userEvent.clear(input);
    await userEvent.type(input, "30.00");
    await userEvent.click(screen.getByRole("button", { name: "保存个人自付修正" }));

    await waitFor(() => expect(mocked.updateField).toHaveBeenCalledWith(summary.case_id, "personal_self_pay", "30.00", 1, "人工复核修正"));
    expect(await screen.findByText("目标金额 ¥100.00")).toBeInTheDocument();
    expect(screen.getByText("命中 R001")).toBeInTheDocument();
  });

  it.each([
    ["批准通过", "APPROVE"],
    ["驳回复核", "REJECT"],
  ] as const)("records the %s decision", async (buttonName, decision) => {
    loadDefault();
    mocked.decideCase.mockResolvedValue({ ...detail, version: 2, status: decision });
    render(<App />);
    await screen.findByLabelText("字段审核区");

    await userEvent.type(screen.getByRole("textbox", { name: "审核意见" }), "合成案例审核完成");
    await userEvent.click(screen.getByRole("button", { name: buttonName }));

    await waitFor(() => expect(mocked.decideCase).toHaveBeenCalledWith(summary.case_id, decision, 1, "合成案例审核完成"));
    expect(await screen.findByText(decision === "APPROVE" ? "已批准" : "已驳回")).toBeInTheDocument();
  });

  it.each(["field", "decision"] as const)("keeps case B visible when case A %s mutation resolves late", async (mutation) => {
    const user = userEvent.setup();
    const secondSummary: CaseSummary = {
      ...summary,
      case_id: "synthetic-medical-receipt-002",
      document_label: "合成医疗收费票据 B",
    };
    const secondDetail: ReviewCaseDetail = {
      ...detail,
      case_id: secondSummary.case_id,
      document: { metadata: { ...detail.document.metadata, label: secondSummary.document_label } },
    };
    const pending = deferred<ReviewCaseDetail>();
    mocked.listCases.mockResolvedValue([summary, secondSummary]);
    mocked.getCase.mockImplementation(async (caseId) => caseId === secondSummary.case_id ? secondDetail : detail);
    if (mutation === "field") mocked.updateField.mockReturnValue(pending.promise);
    else mocked.decideCase.mockReturnValue(pending.promise);

    render(<App />);
    await screen.findByRole("heading", { name: summary.document_label });

    if (mutation === "field") {
      const input = screen.getByRole("textbox", { name: "个人自付" });
      await user.clear(input);
      await user.type(input, "30.00");
      await user.click(screen.getByRole("button", { name: "保存个人自付修正" }));
      await waitFor(() => expect(mocked.updateField).toHaveBeenCalledOnce());
    } else {
      await user.type(screen.getByRole("textbox", { name: "审核意见" }), "合成案例审核完成");
      await user.click(screen.getByRole("button", { name: "批准通过" }));
      await waitFor(() => expect(mocked.decideCase).toHaveBeenCalledOnce());
    }

    await user.click(screen.getByRole("button", { name: /打开案例 合成医疗收费票据 B/ }));
    expect(await screen.findByRole("heading", { name: secondSummary.document_label })).toBeInTheDocument();

    await act(async () => {
      pending.resolve(mutation === "field" ? calculatedDetail : { ...detail, version: 2, status: "APPROVE" });
      await pending.promise;
    });

    expect(screen.getByRole("heading", { name: secondSummary.document_label })).toBeInTheDocument();
    expect(screen.queryByText("命中 R001")).not.toBeInTheDocument();
    expect(screen.queryByText("已批准")).not.toBeInTheDocument();
  });

  it("contains no CSS gradients", () => {
    expect(styleSource).not.toMatch(/gradient\s*\(/i);
  });

  it("keeps a 44px evidence target separate from the exact bbox outline", async () => {
    loadDefault();
    render(<App />);

    const target = await screen.findByRole("button", { name: "定位证据 个人自付" });
    expect(target).toHaveClass("evidence-box");
    expect(target.parentElement).toHaveClass("evidence-outline");
    expect(styleSource).toMatch(/\.evidence-box\s*\{[^}]*min-width:\s*44px[^}]*min-height:\s*44px/is);
  });

  it("revokes exported JSON URL after the click task", async () => {
    loadDefault();
    render(<App />);
    await screen.findByLabelText("字段审核区");
    vi.useFakeTimers();
    const createUrl = vi.fn(() => "blob:synthetic-export");
    const revokeUrl = vi.fn();
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: createUrl },
      revokeObjectURL: { configurable: true, value: revokeUrl },
    });
    let connectedAtClick = false;
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      connectedAtClick = this.isConnected;
    });

    fireEvent.click(screen.getByRole("button", { name: "导出 JSON" }));

    expect(createUrl).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(connectedAtClick).toBe(true);
    expect(revokeUrl).not.toHaveBeenCalled();
    await vi.runOnlyPendingTimersAsync();
    expect(revokeUrl).toHaveBeenCalledWith("blob:synthetic-export");
  });
});
