export type CaseStatus = "PENDING_REVIEW" | "APPROVE" | "REJECT";
export type RuleStatus = "CALCULATED" | "NEEDS_REVIEW";

export interface CaseSummary {
  case_id: string;
  version: number;
  status: CaseStatus;
  document_type: string;
  document_label: string;
  rule_status: RuleStatus;
}

export interface Evidence {
  page_index: number;
  ocr_text: string;
  bbox: number[][];
  source_image: string;
}

export interface ExtractedField {
  value: string | null;
  confidence: number;
  raw_name: string;
  evidence: Evidence[];
}

export interface RuleTrace {
  rule_id: string;
  outcome: string;
  reason: string;
}

export interface RuleDecision {
  status: RuleStatus;
  selected_rule: string | null;
  target_amount: string | null;
  formula: string | null;
  inputs: Record<string, string | null>;
  trace: RuleTrace[];
  warnings: string[];
  rule_version: string;
}

export interface AuditEvent {
  event_type: string;
  version: number;
  timestamp: string;
  field_name?: string;
  decision?: "APPROVE" | "REJECT";
}

export interface ReviewCaseDetail {
  case_id: string;
  version: number;
  status: CaseStatus;
  document: {
    metadata: {
      label: string;
      source: string;
      page_count: number;
    };
  };
  fields: Record<string, ExtractedField>;
  rule_decision: RuleDecision;
  audit_events: AuditEvent[];
}

export const FIELD_LABELS: Record<string, string> = {
  policy_scope_amount: "符合政策范围金额",
  current_basic_medical_amount: "本次符合基本医疗金额",
  pooled_fund_payment: "医保统筹基金支付",
  personal_cash_payment: "个人现金支付",
  pooled_fund_scope_expense: "统筹基金支付范围内费用",
  fund_payment_total: "基金支付合计",
  self_pay_one: "自付一",
  self_pay_two: "自付二",
  personal_self_pay: "个人自付",
  personal_self_expense: "个人自费",
  total_amount: "金额合计",
  category_b_prepaid: "乙类先行自付费用",
  over_limit_self_pay: "超限价自付费用",
  tax_inclusive_total: "价税合计",
};

export function fieldLabel(name: string, field?: ExtractedField): string {
  return FIELD_LABELS[name] ?? field?.raw_name ?? name;
}
