export interface WindowPrediction {
  window_index: number;
  prediction: number;
  activity: string;
  proba: Record<string, number>;
}

export interface AggregatePrediction {
  fraction_per_activity: Record<string, number>;
  mean_proba: Record<string, number>;
}

export interface PredictResponse {
  per_window: WindowPrediction[];
  aggregate: AggregatePrediction;
}

export interface EvaluationMetrics {
  accuracy?: number | null;
  macro_f1?: number | null;
  confusion_matrix: number[][];
}

export interface EvaluateResponse {
  metrics: EvaluationMetrics;
  predictions?: number[];
  ground_truth?: number[];
}

export interface ModelInfo {
  version: string;
  model_type: string;
  random_seed: number;
  window_seconds: number;
  window_overlap_seconds: number;
  sample_rate_hz: number;
  excluded_subjects_demo: number[];
  splits: Record<string, any>;
  feature_columns: string[];
  activity_labels?: Record<string, string>;
  metrics?: Record<string, any>;
}
