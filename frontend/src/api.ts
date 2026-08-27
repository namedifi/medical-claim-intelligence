import type { CaseSummary, CaseStatus, ReviewCaseDetail } from "./types";

const REVIEW_ROOT = "/api/v1/review";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let message = "请求失败";
    try {
      const payload = await response.json() as { detail?: string };
      message = payload.detail ?? message;
    } catch {
      // Keep a stable public error when the response is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function listCases(): Promise<CaseSummary[]> {
  return requestJson<CaseSummary[]>(`${REVIEW_ROOT}/cases`);
}

export function getCase(caseId: string): Promise<ReviewCaseDetail> {
  return requestJson<ReviewCaseDetail>(`${REVIEW_ROOT}/cases/${encodeURIComponent(caseId)}`);
}

export function updateField(
  caseId: string,
  fieldName: string,
  value: string | null,
  expectedVersion: number,
  reason: string,
): Promise<ReviewCaseDetail> {
  return requestJson<ReviewCaseDetail>(
    `${REVIEW_ROOT}/cases/${encodeURIComponent(caseId)}/fields/${encodeURIComponent(fieldName)}`,
    {
      method: "PATCH",
      body: JSON.stringify({ value, expected_version: expectedVersion, reason }),
    },
  );
}

export function decideCase(
  caseId: string,
  decision: Extract<CaseStatus, "APPROVE" | "REJECT">,
  expectedVersion: number,
  comment: string,
): Promise<ReviewCaseDetail> {
  return requestJson<ReviewCaseDetail>(
    `${REVIEW_ROOT}/cases/${encodeURIComponent(caseId)}/decision`,
    {
      method: "POST",
      body: JSON.stringify({ decision, expected_version: expectedVersion, comment }),
    },
  );
}
