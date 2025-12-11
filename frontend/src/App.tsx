import { useEffect, useMemo, useState } from "react";
import { evaluateLog, getModelInfo, predictLog } from "./api";
import { EvaluateResponse, ModelInfo, PredictResponse, WindowPrediction } from "./types";

type UploadMode = "predict" | "evaluate";

interface UploadCardProps {
  mode: UploadMode;
  loading: boolean;
  onUpload: (file: File) => Promise<void>;
  error?: string;
}

function UploadCard({ mode, loading, onUpload, error }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const label = mode === "predict" ? "Predicción de Actividades" : "Evaluación del Modelo";
  const isPredictMode = mode === "predict";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    await onUpload(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.name.endsWith('.log')) {
      setFile(droppedFile);
    }
  };

  return (
    <div className={`card upload-card ${isPredictMode ? 'upload-predict' : 'upload-evaluate'}`}>
      <div className="upload-card-header">
        <div className={`upload-icon-wrapper ${isPredictMode ? 'predict-icon' : 'evaluate-icon'}`}>
          <span className="upload-main-icon">{isPredictMode ? "🔮" : "📊"}</span>
        </div>
        <h3 className="upload-title">{label}</h3>
        <p className="upload-subtitle">
          {isPredictMode
            ? "Analiza datos de sensores y predice actividades humanas"
            : "Evalúa el rendimiento del modelo con métricas detalladas"}
        </p>
      </div>
      
      <form onSubmit={handleSubmit} className="upload-form">
        <div 
          className={`drop-zone ${isDragging ? 'drop-zone-active' : ''} ${file ? 'drop-zone-has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".log"
            id={`file-input-${mode}`}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            disabled={loading}
          />
          <label htmlFor={`file-input-${mode}`} className="drop-zone-content">
            {file ? (
              <>
                <div className="file-selected-icon">✓</div>
                <span className="file-name">{file.name}</span>
                <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
              </>
            ) : (
              <>
                <div className="drop-zone-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17,8 12,3 7,8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <span className="drop-zone-text">Arrastra tu archivo aquí</span>
                <span className="drop-zone-or">o</span>
                <span className="drop-zone-browse">Selecciona un archivo</span>
                <span className="drop-zone-hint">.log • Datos MHealth</span>
              </>
            )}
          </label>
        </div>
        
        <button 
          type="submit" 
          disabled={!file || loading} 
          className={`upload-btn ${isPredictMode ? 'btn-predict' : 'btn-evaluate'}`}
        >
          {loading ? (
            <>
              <span className="btn-spinner"></span>
              <span>Procesando...</span>
            </>
          ) : (
            <>
              <span className="btn-icon">{isPredictMode ? "🚀" : "✓"}</span>
              <span>{isPredictMode ? "Analizar Actividades" : "Evaluar Modelo"}</span>
            </>
          )}
        </button>
        
        {error && (
          <div className="upload-error">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
          </div>
        )}
      </form>
    </div>
  );
}

function StatsBadge({ label, value, icon }: { label: string; value: string; icon?: string }) {
  return (
    <div className="stats-badge">
      {icon && <span className="stats-icon">{icon}</span>}
      <div>
        <p className="stats-label">{label}</p>
        <p className="stats-value">{value}</p>
      </div>
    </div>
  );
}

function ComparisonTimelines({ predictions, groundTruth }: { predictions: number[]; groundTruth: number[] }) {
  const activityLabels: Record<number, string> = {
    1: 'De pie',
    2: 'Sentado',
    3: 'Acostado',
    4: 'Caminando',
    5: 'Escaleras',
    6: 'Flexión',
    7: 'Brazos',
    8: 'Rodillas',
    9: 'Ciclismo',
    10: 'Trote',
    11: 'Corriendo',
    12: 'Saltando',
  };

  const activityColors: Record<number, string> = {
    1: '#10b981',   // De pie
    2: '#3b82f6',   // Sentado
    3: '#8b5cf6',   // Acostado
    4: '#f59e0b',   // Caminando
    5: '#ef4444',   // Escaleras
    6: '#ec4899',   // Flexión
    7: '#06b6d4',   // Brazos
    8: '#84cc16',   // Rodillas
    9: '#f97316',   // Ciclismo
    10: '#14b8a6',  // Trote
    11: '#dc2626',  // Corriendo
    12: '#a855f7',  // Saltando
  };

  return (
    <div className="comparison-timelines">
      <h4>Comparación de Secuencias</h4>
      
      <div className="timeline-comparison-row">
        <div className="timeline-label">Real</div>
        <div className="comparison-timeline-bar">
          {groundTruth.map((activity, idx) => (
            <div
              key={`gt-${idx}`}
              className="comparison-segment"
              style={{
                background: activityColors[activity] || '#64748b',
                flex: 1,
              }}
              title={activityLabels[activity] || `Actividad ${activity}`}
            />
          ))}
        </div>
      </div>

      <div className="timeline-comparison-row">
        <div className="timeline-label">Predicho</div>
        <div className="comparison-timeline-bar">
          {predictions.map((activity, idx) => (
            <div
              key={`pred-${idx}`}
              className="comparison-segment"
              style={{
                background: activityColors[activity] || '#64748b',
                flex: 1,
              }}
              title={activityLabels[activity] || `Actividad ${activity}`}
            />
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="comparison-legend">
        {Array.from(new Set([...groundTruth, ...predictions])).sort((a, b) => a - b).map(activity => (
          <div key={activity} className="comparison-legend-item">
            <div className="comparison-legend-color" style={{ background: activityColors[activity] || '#64748b' }} />
            <span>{activityLabels[activity] || `Act. ${activity}`}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

interface MetricCardProps {
  metrics: EvaluateResponse["metrics"];
  predictions?: number[];
  groundTruth?: number[];
}

function MetricCard({ metrics, predictions, groundTruth }: MetricCardProps) {
  const accuracy = (metrics.accuracy ?? 0) * 100;
  const f1 = (metrics.macro_f1 ?? 0) * 100;

  return (
    <div className="card metric-card">
      <div className="card-header">
        <div className="icon-badge">📊</div>
        <div>
          <h3>Métricas de Rendimiento</h3>
          <p className="muted">Evaluación del modelo con datos etiquetados</p>
        </div>
      </div>
      
      <div className="metrics-summary">
        <div className="metric-box" title="Porcentaje de predicciones correctas sobre el total. Indica qué tan frecuentemente el modelo acierta en general.">
          <div className="metric-icon">🎯</div>
          <div className="metric-info">
            <span className="metric-name">Accuracy</span>
            <span className="metric-number">{accuracy.toFixed(1)}%</span>
          </div>
          <div className="metric-bar">
            <div className="metric-bar-fill accuracy-fill" style={{ width: `${accuracy}%` }} />
          </div>
        </div>
        <div className="metric-box" title="Es el promedio del F1-Score de cada clase. El F1-Score combina la precisión y el recall (cantidad de verdaderos positivos identificados).">
          <div className="metric-icon">⚡</div>
          <div className="metric-info">
            <span className="metric-name">Macro F1</span>
            <span className="metric-number">{f1.toFixed(1)}%</span>
          </div>
          <div className="metric-bar">
            <div className="metric-bar-fill f1-fill" style={{ width: `${f1}%` }} />
          </div>
        </div>
      </div>

      {metrics.confusion_matrix?.length > 0 && (
        <div className="confusion-section">
          <h4>Matriz de Confusión</h4>
          <ConfusionMatrix matrix={metrics.confusion_matrix} />
        </div>
      )}

      {predictions && groundTruth && predictions.length > 0 && (
        <ComparisonTimelines predictions={predictions} groundTruth={groundTruth} />
      )}
    </div>
  );
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const maxVal = Math.max(...matrix.flat());
  const activityLabels = ['De pie', 'Sentado', 'Acostado', 'Caminando', 'Escaleras', 'Flexión', 'Brazos', 'Rodillas', 'Ciclismo', 'Trote', 'Corriendo', 'Saltando'];
  
  // Calculate totals for percentages
  const rowTotals = matrix.map(row => row.reduce((sum, val) => sum + val, 0));
  
  return (
    <div className="confusion-matrix-wrapper">
      <div className="confusion-matrix-modern">
        {/* Row label "REAL" on left side */}
        <div className="cm-side-label cm-side-label-left">
          <span>REAL</span>
        </div>
        
        <div className="cm-main-content">
          {/* Column label "PREDICHO" on top */}
          <div className="cm-top-label">
            <span>PREDICHO</span>
          </div>
          
          {/* Header with column labels */}
          <div className="cm-header">
            <div className="cm-corner"></div>
            <div className="cm-col-labels">
              {matrix[0]?.map((_, i) => (
                <div key={i} className="cm-col-label">
                  <span className="cm-label-text">{activityLabels[i] || i + 1}</span>
                </div>
              ))}
            </div>
          </div>
          
          {/* Matrix Body */}
          <div className="cm-body">
            <div className="cm-row-labels">
              {matrix.map((_, i) => (
                <div key={i} className="cm-row-label">
                  <span className="cm-label-text">{activityLabels[i] || i + 1}</span>
                </div>
              ))}
            </div>
            
            <div className="cm-cells-grid">
              {matrix.map((row, i) => (
                <div className="cm-cells-row" key={i}>
                  {row.map((val, j) => {
                    const intensity = maxVal > 0 ? val / maxVal : 0;
                    const isDiagonal = i === j;
                    const percentage = rowTotals[i] > 0 ? (val / rowTotals[i] * 100) : 0;
                    
                    let bgColor;
                    if (val === 0) {
                      bgColor = '#f8fafc';
                    } else if (isDiagonal) {
                      bgColor = `rgba(16, 185, 129, ${0.15 + intensity * 0.85})`;
                    } else {
                      bgColor = `rgba(239, 68, 68, ${0.1 + intensity * 0.7})`;
                    }
                    
                    return (
                      <div 
                        key={`${i}-${j}`} 
                        className={`cm-cell-modern ${isDiagonal ? 'diagonal' : ''} ${val === 0 ? 'zero' : ''}`}
                        style={{ background: bgColor }}
                        title={`Real: ${activityLabels[i] || i + 1}, Predicho: ${activityLabels[j] || j + 1}\nValor: ${val} (${percentage.toFixed(1)}%)`}
                      >
                        <span className="cm-value">{val}</span>
                        {val > 0 && (
                          <span className="cm-percentage">{percentage.toFixed(0)}%</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Legend */}
        <div className="cm-legend">
          <div className="cm-legend-item">
            <div className="cm-legend-box diagonal-box"></div>
            <span>Predicciones Correctas</span>
          </div>
          <div className="cm-legend-item">
            <div className="cm-legend-box error-box"></div>
            <span>Errores de Clasificación</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function PredictionsList({ predictions }: { predictions: WindowPrediction[] }) {
  const activityColors: Record<string, string> = {
    'De pie': '#10b981',
    'Sentado': '#3b82f6',
    'Acostado': '#8b5cf6',
    'Caminando': '#f59e0b',
    'Subiendo escaleras': '#ef4444',
    'Flexión de cintura': '#ec4899',
    'Elevación frontal de brazos': '#06b6d4',
    'Flexión de rodillas': '#84cc16',
    'Ciclismo': '#f97316',
    'Trote': '#14b8a6',
    'Corriendo': '#dc2626',
    'Saltando': '#a855f7',
  };

  return (
    <div className="card timeline-card">
      <div className="card-header">
        <div className="icon-badge">⏱️</div>
        <div>
          <h3>Línea de Tiempo de Actividades</h3>
          <p className="muted">{predictions.length} ventanas detectadas</p>
        </div>
      </div>
      
      {/* Visual Timeline Bar */}
      <div className="visual-timeline-bar">
        {predictions.map((p, idx) => {
          const color = activityColors[p.activity] || '#64748b';
          const confidence = Math.max(...Object.values(p.proba));
          return (
            <div
              key={p.window_index}
              className="timeline-segment"
              style={{
                background: color,
                opacity: 0.6 + confidence * 0.4,
                flex: 1,
                cursor: 'pointer'
              }}
              title={`${p.activity} (${(confidence * 100).toFixed(0)}%)`}
              onClick={() => {
                const element = document.getElementById(`window-${p.window_index}`);
                element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
              }}
            />
          );
        })}
      </div>
      
      {/* Timeline Details */}
      <div className="timeline-container">
        <div className="timeline">
          {predictions.map((p, idx) => {
            const confidence = Math.max(...Object.values(p.proba));
            const color = activityColors[p.activity] || '#64748b';
            
            const sortedProba = Object.entries(p.proba)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 3);
            
            return (
              <div 
                key={p.window_index}
                id={`window-${p.window_index}`}
                className="timeline-item"
                style={{ 
                  '--item-color': color,
                } as React.CSSProperties}
              >
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <div className="timeline-header">
                    <span className="timeline-index">Ventana #{p.window_index}</span>
                    <span className="timeline-confidence" style={{ background: `${color}20`, color: color }}>
                      {(confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="timeline-activity" style={{ color: color }}>
                    {p.activity}
                  </div>
                  <div className="timeline-proba-bars">
                    {sortedProba.map(([activity, prob]) => {
                      const actColor = activityColors[activity] || '#64748b';
                      return (
                        <div key={activity} className="proba-bar-mini">
                          <span className="proba-label">{activity}</span>
                          <div className="proba-bar-container">
                            <div 
                              className="proba-fill" 
                              style={{ 
                                width: `${prob * 100}%`,
                                background: actColor
                              }}
                            >
                              <span className="proba-value">{(prob * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function AggregateCard({ aggregate }: { aggregate: PredictResponse["aggregate"] }) {
  const activityColors: Record<string, string> = {
    'De pie': '#10b981',
    'Sentado': '#3b82f6',
    'Acostado': '#8b5cf6',
    'Caminando': '#f59e0b',
    'Subiendo escaleras': '#ef4444',
    'Flexión de cintura': '#ec4899',
    'Elevación frontal de brazos': '#06b6d4',
    'Flexión de rodillas': '#84cc16',
    'Ciclismo': '#f97316',
    'Trote': '#14b8a6',
    'Corriendo': '#dc2626',
    'Saltando': '#a855f7',
  };

  const items = useMemo(
    () =>
      Object.entries(aggregate.fraction_per_activity || {})
        .map(([activity, fraction]) => ({
          activity,
          fraction,
          color: activityColors[activity] || '#64748b'
        }))
        .sort((a, b) => b.fraction - a.fraction),
    [aggregate]
  );

  const totalPercentage = items.reduce((sum, item) => sum + item.fraction, 0) * 100;

  return (
    <div className="card aggregate-card">
      <div className="card-header">
        <div className="icon-badge">📈</div>
        <div>
          <h3>Distribución de Actividades</h3>
          <p className="muted">Resumen general del análisis</p>
        </div>
      </div>

      <div className="donut-chart-wrapper">
        <svg viewBox="0 0 200 200" className="donut-chart">
          {items.reduce((acc, item, idx) => {
            const angle = item.fraction * 360;
            const startAngle = idx === 0 ? 0 : items.slice(0, idx).reduce((sum, i) => sum + i.fraction * 360, 0);
            const endAngle = startAngle + angle;
            
            const startRad = (startAngle - 90) * Math.PI / 180;
            const endRad = (endAngle - 90) * Math.PI / 180;
            
            const x1 = 100 + 70 * Math.cos(startRad);
            const y1 = 100 + 70 * Math.sin(startRad);
            const x2 = 100 + 70 * Math.cos(endRad);
            const y2 = 100 + 70 * Math.sin(endRad);
            
            const largeArc = angle > 180 ? 1 : 0;
            
            acc.push(
              <path
                key={item.activity}
                d={`M 100 100 L ${x1} ${y1} A 70 70 0 ${largeArc} 1 ${x2} ${y2} Z`}
                fill={item.color}
                opacity="0.9"
              />
            );
            return acc;
          }, [] as JSX.Element[])}
          <circle cx="100" cy="100" r="50" fill="white" />
          <text x="100" y="95" textAnchor="middle" fontSize="24" fontWeight="bold" fill="#0f172a">
            {totalPercentage.toFixed(0)}%
          </text>
          <text x="100" y="110" textAnchor="middle" fontSize="10" fill="#64748b">
            Total
          </text>
        </svg>
      </div>

      <div className="activity-legend">
        {items.map((item) => (
          <div key={item.activity} className="legend-item">
            <div className="legend-color" style={{ background: item.color }}></div>
            <div className="legend-content">
              <span className="legend-activity">{item.activity}</span>
              <span className="legend-percentage">{(item.fraction * 100).toFixed(1)}%</span>
            </div>
            <div className="legend-bar">
              <div style={{ width: `${item.fraction * 100}%`, background: item.color }}></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [evalResult, setEvalResult] = useState<EvaluateResponse | null>(null);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingEval, setLoadingEval] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [evalError, setEvalError] = useState("");

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null));
  }, []);

  const handlePredict = async (file: File) => {
    setPredictError("");
    setLoadingPredict(true);
    setPredictResult(null);
    setEvalResult(null); // Limpiar resultado de evaluación
    try {
      const result = await predictLog(file);
      setPredictResult(result);
    } catch (err: any) {
      setPredictError(err.message);
      setPredictResult(null);
    } finally {
      setLoadingPredict(false);
    }
  };

  const handleEvaluate = async (file: File) => {
    setEvalError("");
    setLoadingEval(true);
    setEvalResult(null);
    setPredictResult(null); // Limpiar resultado de predicción
    try {
      const result = await evaluateLog(file);
      setEvalResult(result);
    } catch (err: any) {
      setEvalError(err.message);
      setEvalResult(null);
    } finally {
      setLoadingEval(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-content">
          <h1>Human Activity Recognition</h1>
          <p className="hero-subtitle">
            Sistema profesional de reconocimiento de actividad humana basado en datos de sensores MHealth.
            Análisis en tiempo real con machine learning de última generación.
          </p>
          {modelInfo && (
            <div className="hero-stats">
              <StatsBadge label="Modelo" value={modelInfo.version} icon="🤖" />
              <StatsBadge label="Ventana" value={`${modelInfo.window_seconds}s`} icon="⏱️" />
              <StatsBadge label="Frecuencia" value={`${modelInfo.sample_rate_hz}Hz`} icon="📡" />
            </div>
          )}
        </div>
      </header>

      <section className="upload-section">
        <UploadCard mode="predict" loading={loadingPredict} onUpload={handlePredict} error={predictError} />
        <UploadCard mode="evaluate" loading={loadingEval} onUpload={handleEvaluate} error={evalError} />
      </section>

      {(predictResult || evalResult) && (
        <section className="results-section">
          {predictResult && (
            <>
              <PredictionsList predictions={predictResult.per_window} />
            </>
          )}
          {evalResult && (
            <MetricCard 
              metrics={evalResult.metrics} 
              predictions={evalResult.predictions} 
              groundTruth={evalResult.ground_truth} 
            />
          )}
        </section>
      )}

      {modelInfo && (
        <section className="card info-card">
          <div className="card-header">
            <div className="icon-badge">⚙️</div>
            <div>
              <h3>Configuración del Modelo</h3>
              <p className="muted">Detalles técnicos y parámetros de entrenamiento</p>
            </div>
          </div>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Algoritmo</span>
              <span className="info-value">{modelInfo.model_type}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Semilla Aleatoria</span>
              <span className="info-value">{modelInfo.random_seed}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Overlap</span>
              <span className="info-value">{modelInfo.window_overlap_seconds}s</span>
            </div>
            <div className="info-item">
              <span className="info-label">Features</span>
              <span className="info-value">{modelInfo.feature_columns?.length || 0}</span>
            </div>
          </div>
          {modelInfo.metrics && (
            <div className="model-metrics">
              <h4>Métricas de Entrenamiento</h4>
              <div className="metrics-breakdown">
                {Object.entries(modelInfo.metrics).map(([split, metrics]: [string, any]) => (
                  <div key={split} className="split-metric">
                    <span className="split-name">{split}</span>
                    <div className="split-values">
                      <span>Acc: {(metrics.accuracy * 100).toFixed(1)}%</span>
                      <span>F1: {(metrics.macro_f1 * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
