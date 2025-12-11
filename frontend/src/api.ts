import { EvaluateResponse, ModelInfo, PredictResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handleResponse<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }
  return resp.json() as Promise<T>;
}

export async function getModelInfo(): Promise<ModelInfo> {
  const resp = await fetch(`${API_URL}/model-info`);
  return handleResponse<ModelInfo>(resp);
}

export async function predictLog(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });
  return handleResponse<PredictResponse>(resp);
}

export async function evaluateLog(file: File): Promise<EvaluateResponse> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_URL}/evaluate-log`, {
    method: "POST",
    body: form,
  });
  return handleResponse<EvaluateResponse>(resp);
}
