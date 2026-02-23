import { useState, useEffect, useCallback } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { engine, pollTask } from '@/lib/api';
import { Button, Card, Badge, Input, Textarea, Spinner, EmptyState } from '@/components/ui';
import {
  Target, Sparkles, Play, RotateCcw, Download, FileText,
  CheckCircle2, AlertTriangle, ChevronDown, ChevronUp, Clock,
  BarChart3, Lightbulb, BookOpen, Presentation, FilePieChart,
  TrendingUp, Eye, ChartPie
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';

/* ── Types ───────────────────────────────────────────────── */
interface Goal {
  problem: string;
  objective: string;
  target: string;
}

interface SessionInfo {
  dataframe_count: number;
  dataframe_keys: string[];
  has_active_df: boolean;
  active_df_shape: number[] | null;
}

/* ── Main Page ───────────────────────────────────────────── */
export default function GoalsPage() {
  const { sessionId } = useWorkspaceStore();

  // Goals
  const [goal, setGoal] = useState<Goal | null>(null);
  const [form, setForm] = useState<Goal>({ problem: '', objective: '', target: '' });
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);

  // Session info
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);

  // Analysis
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);
  const [analysisTimestamp, setAnalysisTimestamp] = useState<string | null>(null);
  const [analysisMeta, setAnalysisMeta] = useState<{ datasets: string[]; model: string } | null>(null);

  // Goals summary expanded
  const [goalsSummaryOpen, setGoalsSummaryOpen] = useState(true);

  // AI Recommendations
  const [vizRecommendations, setVizRecommendations] = useState<any[] | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [vizCharts, setVizCharts] = useState<any[] | null>(null);

  // Report export
  const [exportingPdf, setExportingPdf] = useState(false);
  const [exportingPptx, setExportingPptx] = useState(false);

  /* ── Load session info ─────────────────────────────────── */
  const loadSessionInfo = useCallback(async () => {
    if (!sessionId) return;
    try {
      const { data } = await engine.get(`/files/session-info?session_id=${sessionId}`);
      setSessionInfo(data.data as SessionInfo);
    } catch { /* silent */ }
  }, [sessionId]);

  /* ── Fetch goals ───────────────────────────────────────── */
  const fetchGoals = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const { data } = await engine.get(`/strategic/goals?session_id=${sessionId}`);
      const g = data.data?.goals;
      if (g && (g.problem || g.objective)) {
        setGoal(g);
        setForm(g);
      }
    } catch { /* no goals yet */ }
    finally { setLoading(false); }
  }, [sessionId]);

  useEffect(() => { fetchGoals(); loadSessionInfo(); }, [fetchGoals, loadSessionInfo]);

  /* ── Save goals ────────────────────────────────────────── */
  const saveGoal = async () => {
    if (!sessionId) return toast.error('Upload data first');
    if (!form.problem || !form.objective) return toast.error('Fill problem and objective');
    setSaving(true);
    try {
      await engine.post('/strategic/goals', {
        session_id: sessionId,
        problem: form.problem,
        objective: form.objective,
        target: form.target || '',
      });
      toast.success('Business goals saved');
      setEditing(false);
      setGoal({ ...form });
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save goals');
    } finally { setSaving(false); }
  };

  /* ── Run strategic analysis ────────────────────────────── */
  const runAnalysis = async () => {
    if (!sessionId || !goal) return;
    setAnalyzing(true);
    setProgress(0);
    setAnalysisResult(null);
    try {
      const { data } = await engine.post('/strategic/analyze', {
        session_id: sessionId,
        goals: {
          problem: goal.problem,
          objective: goal.objective,
          target: goal.target,
        },
      });
      const taskId = data.data?.task_id;
      if (!taskId) throw new Error('No task ID returned');

      const result = await pollTask(taskId, (p, _s) => setProgress(p));
      setAnalysisResult(result.analysis_markdown || JSON.stringify(result, null, 2));
      setAnalysisMeta({
        datasets: result.datasets_analyzed || [],
        model: result.model || 'qwen2.5:7b',
      });
      if (Array.isArray(result.recommendations)) {
        setVizRecommendations(result.recommendations);
      }
      if (Array.isArray(result.plotly_charts)) {
        setVizCharts(result.plotly_charts);
      }
      setAnalysisTimestamp(new Date().toLocaleString());
      toast.success('Strategic analysis complete!');
    } catch (err: any) {
      toast.error(err.message || 'Analysis failed');
    } finally { setAnalyzing(false); }
  };

  /* ── Download report (client-side) ───────────────────────── */
  const downloadReport = (format: 'md' | 'html' | 'txt') => {
    if (!analysisResult || !goal) return;
    const ts = analysisTimestamp || new Date().toISOString();
    const filename = `strategic_analysis_${new Date().toISOString().slice(0, 16).replace(/[:\-T]/g, '_')}`;

    let content: string;
    let mime: string;
    let ext: string;

    if (format === 'html') {
      content = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Strategic Analysis Report</title>
<style>
body{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:40px;line-height:1.6;color:#222}
h1{color:#1a1a2e;border-bottom:3px solid #4a90d9;padding-bottom:10px}
h2{color:#16213e;margin-top:30px}h3{color:#0f3460}
.meta{background:#f0f4f8;padding:15px;border-radius:8px;margin-bottom:20px}
.meta p{margin:5px 0}
@media print{body{padding:20px}}
</style></head><body>
<h1>Strategic Business Analysis Report</h1>
<div class="meta">
<p><strong>Generated:</strong> ${ts}</p>
<p><strong>Problem:</strong> ${goal.problem}</p>
<p><strong>Objective:</strong> ${goal.objective}</p>
<p><strong>Target Audience:</strong> ${goal.target || 'Not specified'}</p>
</div>
${analysisResult.replace(/\n/g, '<br>')}
<hr><p style="color:#777;font-size:0.9em">Report generated by Strategic AI Analyst</p>
</body></html>`;
      mime = 'text/html';
      ext = 'html';
    } else {
      content = `# Strategic Business Analysis Report\n\nGenerated: ${ts}\n\n## Business Context\n- **Problem:** ${goal.problem}\n- **Objective:** ${goal.objective}\n- **Target Audience:** ${goal.target || 'Not specified'}\n\n---\n\n${analysisResult}\n\n---\n*Report generated by Strategic AI Analyst*`;
      mime = format === 'md' ? 'text/markdown' : 'text/plain';
      ext = format;
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  /* ── Export as PDF or PPTX (server-side) ────────────────── */
  const exportStrategicReport = async (reportType: 'pdf' | 'pptx') => {
    if (!sessionId || !analysisResult) return toast.error('Run analysis first');
    const setLoading = reportType === 'pdf' ? setExportingPdf : setExportingPptx;
    setLoading(true);
    try {
      let exportRecommendations = vizRecommendations;
      let exportCharts = vizCharts;
      if ((!exportRecommendations || exportRecommendations.length === 0) && sessionId) {
        try {
          const { data: recData } = await engine.post('/strategic/viz-recommend', {
            session_id: sessionId,
            goals: goal ? { problem: goal.problem, objective: goal.objective, target: goal.target } : undefined,
            max_charts: 6,
          });
          const recResp = recData.data;
          exportRecommendations = recResp?.recommendations || [];
          exportCharts = recResp?.plotly_charts || [];
          setVizRecommendations(exportRecommendations);
          setVizCharts(exportCharts);
        } catch {
          // continue export even if recommendations fetch fails
        }
      }

      const { data } = await engine.post('/reports/strategic-report', {
        session_id: sessionId,
        report_type: reportType,
        analysis_markdown: analysisResult,
        goals: goal ? { problem: goal.problem, objective: goal.objective, target: goal.target } : null,
        recommendations: exportRecommendations || [],
        plotly_charts: exportCharts || [],
      });
      const taskId = data.data?.task_id;
      if (!taskId) throw new Error('No task ID returned');
      toast(`Generating ${reportType.toUpperCase()} report...`, { icon: '⏳' });
      const result = await pollTask(taskId);
      // Download the file
      const response = await engine.get(`/reports/download/${taskId}`, { responseType: 'blob' });
      const blob = new Blob([response.data]);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = result.filename || `strategic_report.${reportType}`;
      a.click();
      URL.revokeObjectURL(a.href);
      toast.success(`${reportType.toUpperCase()} report downloaded!`);
    } catch (err: any) {
      toast.error(err.message || `${reportType.toUpperCase()} export failed`);
    } finally { setLoading(false); }
  };

  /* ── Fetch AI visualization & KPI recommendations ──────── */
  const fetchVizRecommendations = async () => {
    if (!sessionId) return;
    setVizLoading(true);
    try {
      const { data } = await engine.post('/strategic/viz-recommend', {
        session_id: sessionId,
        goals: goal ? { problem: goal.problem, objective: goal.objective, target: goal.target } : undefined,
        max_charts: 6,
      });
      const resp = data.data;
      setVizRecommendations(resp?.recommendations || []);
      setVizCharts(resp?.plotly_charts || []);
      toast.success('AI recommendations generated!');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to get recommendations');
    } finally { setVizLoading(false); }
  };

  /* ── Derived state ─────────────────────────────────────── */
  const hasData = sessionInfo && sessionInfo.dataframe_count > 0;
  const hasGoals = !!(goal?.problem && goal?.objective);
  const canAnalyze = hasData && hasGoals && !analyzing;

  return (
    <div className="animate-fade-in">

      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Business Strategic Goals</h1>
          <p className="text-[var(--text-secondary)] mt-1">Define your business problem to guide AI analysis</p>
        </div>
        <div className="flex items-center gap-2">
          {sessionInfo && <Badge variant="success">{sessionInfo.dataframe_count} dataset(s)</Badge>}
          {hasGoals && <Badge variant="primary">Goals Set</Badge>}
        </div>
      </div>

      {!sessionId && (
        <EmptyState icon={<Target className="w-8 h-8" />} title="No data loaded" description="Upload your data first from the Upload page" />
      )}

      {sessionId && (
        <div className="space-y-6">

          {/* ═══════ SECTION 1: GOALS FORM ═══════ */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <Target className="w-4 h-4 text-[var(--accent)]" /> Define Business Goals
              </h3>
              {goal && !editing && (
                <Button size="sm" variant="secondary" onClick={() => setEditing(true)}
                  icon={<Sparkles className="w-3.5 h-3.5" />}>
                  Edit Goals
                </Button>
              )}
            </div>

            {loading && <div className="flex justify-center py-8"><Spinner /></div>}

            {/* Display saved goals */}
            {!loading && goal && !editing && (
              <div className="space-y-3">
                <div className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                  <div className="grid gap-4">
                    <div>
                      <label className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Business Problem</label>
                      <p className="text-sm mt-1">{goal.problem}</p>
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Primary Objective</label>
                      <p className="text-sm mt-1">{goal.objective}</p>
                    </div>
                    {goal.target && (
                      <div>
                        <label className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Target Audience</label>
                        <p className="text-sm mt-1">{goal.target}</p>
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 text-emerald-600 text-sm">
                  <CheckCircle2 className="w-4 h-4" /> Goals saved — AI analysis will be strategic
                </div>
              </div>
            )}

            {/* Edit / create form */}
            {!loading && (editing || (!goal && !loading)) && (
              <div className="space-y-4">
                <Textarea
                  label="What is the core business problem you're trying to solve? *"
                  value={form.problem}
                  onChange={(v: string) => setForm({ ...form, problem: v })}
                  placeholder="e.g., Sales have declined by 15% in the last quarter and we need to understand the root causes..."
                  rows={3}
                />
                <Textarea
                  label="What is your primary objective for this analysis? *"
                  value={form.objective}
                  onChange={(v: string) => setForm({ ...form, objective: v })}
                  placeholder="e.g., Identify the main drivers of sales decline and suggest recovery strategies."
                  rows={3}
                />
                <Input
                  label="Who is the target audience for this report?"
                  value={form.target}
                  onChange={(v: string) => setForm({ ...form, target: v })}
                  placeholder="e.g., Regional Sales Managers and Executive Board"
                />
                <div className="flex items-center gap-2 pt-2">
                  <Button onClick={saveGoal} loading={saving} icon={<Target className="w-4 h-4" />}>
                    Save Goals
                  </Button>
                  {goal && (
                    <Button variant="secondary" onClick={() => { setEditing(false); setForm(goal); }}>Cancel</Button>
                  )}
                </div>
              </div>
            )}
          </Card>

          {/* ═══════ SECTION 2: RUN ANALYSIS ═══════ */}
          <Card>
            <h3 className="font-semibold mb-2 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-indigo-600" /> Run Full Strategic Data Analysis
            </h3>
            <p className="text-sm text-[var(--text-secondary)] mb-4">
              After saving your business goals, run a comprehensive analysis that addresses your business problem using all loaded data.
            </p>

            {/* Warnings */}
            {!hasData && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-600 text-sm mb-4">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                Please load data first from the Upload page
              </div>
            )}
            {hasData && !hasGoals && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-600 text-sm mb-4">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                Please define and save your business goals above first
              </div>
            )}

            {/* Goals summary (collapsible) */}
            {hasGoals && (
              <button
                onClick={() => setGoalsSummaryOpen(!goalsSummaryOpen)}
                className="flex items-center gap-2 w-full p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] text-left text-sm mb-4 hover:bg-[var(--bg-tertiary)] transition cursor-pointer"
              >
                <BookOpen className="w-4 h-4 text-[var(--accent)] shrink-0" />
                <span className="font-medium flex-1">Current Business Goals</span>
                {goalsSummaryOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            )}
            {hasGoals && goalsSummaryOpen && (
              <div className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] mb-4 text-sm space-y-1">
                <p><strong className="text-[var(--text-muted)]">Problem:</strong> {goal!.problem}</p>
                <p><strong className="text-[var(--text-muted)]">Objective:</strong> {goal!.objective}</p>
                {goal!.target && <p><strong className="text-[var(--text-muted)]">Target Audience:</strong> {goal!.target}</p>}
              </div>
            )}

            {/* Run button */}
            <div className="flex justify-center">
              <Button
                onClick={runAnalysis}
                disabled={!canAnalyze}
                loading={analyzing}
                icon={<Play className="w-4 h-4" />}
                className="px-8"
              >
                {analyzing ? `Analyzing... ${Math.round(progress)}%` : '🚀 Run Full Strategic Analysis'}
              </Button>
            </div>

            {/* Progress bar */}
            {analyzing && (
              <div className="mt-4">
                <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
                  <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> Running comprehensive analysis...</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <div className="w-full h-2 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[var(--accent)] rounded-full transition-all duration-500"
                    style={{ width: `${Math.max(progress, 5)}%` }}
                  />
                </div>
                <p className="text-xs text-[var(--text-muted)] mt-2 text-center">This may take a few minutes — AI is analyzing all your data against your business goals</p>
              </div>
            )}
          </Card>

          {/* ═══════ SECTION 3: ANALYSIS RESULTS ═══════ */}
          {analysisResult && (
            <>
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold flex items-center gap-2">
                    <Lightbulb className="w-4 h-4 text-amber-600" /> Strategic Analysis Report
                  </h3>
                  <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
                    {analysisTimestamp && <span>📅 {analysisTimestamp}</span>}
                    {analysisMeta && <span>🤖 {analysisMeta.model}</span>}
                    {analysisMeta && <span>📊 {analysisMeta.datasets.length} dataset(s)</span>}
                  </div>
                </div>

                {/* Objective badge */}
                {goal && (
                  <div className="mb-4 p-2 rounded-lg bg-[var(--accent-bg)] text-sm">
                    <span className="font-medium text-[var(--accent)]">🎯 Objective:</span>{' '}
                    <span className="text-[var(--text-secondary)]">{goal.objective}</span>
                  </div>
                )}

                {/* Markdown report */}
                <div className="prose prose-invert prose-sm max-w-none schema-markdown">
                  <ReactMarkdown>{analysisResult}</ReactMarkdown>
                </div>
              </Card>

              {/* Download options */}
              <Card>
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Download className="w-4 h-4 text-emerald-600" /> Export Report
                </h3>

                {/* Primary exports: PDF & PowerPoint */}
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <Button onClick={() => exportStrategicReport('pdf')} loading={exportingPdf}
                    variant="primary" icon={<FilePieChart className="w-4 h-4" />} className="justify-center">
                    📄 Download PDF
                  </Button>
                  <Button onClick={() => exportStrategicReport('pptx')} loading={exportingPptx}
                    variant="primary" icon={<Presentation className="w-4 h-4" />} className="justify-center">
                    📊 Download PowerPoint
                  </Button>
                </div>

                {/* Secondary exports */}
                <div className="grid grid-cols-3 gap-3">
                  <Button variant="secondary" onClick={() => downloadReport('md')}
                    icon={<FileText className="w-4 h-4" />}>
                    📄 Markdown
                  </Button>
                  <Button variant="secondary" onClick={() => downloadReport('html')}
                    icon={<FileText className="w-4 h-4" />}>
                    🌐 HTML
                  </Button>
                  <Button variant="secondary" onClick={() => downloadReport('txt')}
                    icon={<FileText className="w-4 h-4" />}>
                    📝 Text
                  </Button>
                </div>
              </Card>

              {/* ═══════ SECTION 4: AI RECOMMENDATIONS ═══════ */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-violet-600" /> AI-Recommended Visualizations & KPIs
                    </h3>
                    <p className="text-sm text-[var(--text-secondary)] mt-1">
                      Based on your business objectives and data, AI recommends specific charts, KPIs, and visualizations to track.
                    </p>
                  </div>
                  <Button onClick={fetchVizRecommendations} loading={vizLoading}
                    icon={<TrendingUp className="w-4 h-4" />}>
                    {vizRecommendations ? 'Refresh' : 'Generate'} Recommendations
                  </Button>
                </div>

                {vizLoading && (
                  <div className="flex items-center justify-center gap-3 py-8">
                    <Spinner />
                    <span className="text-sm text-[var(--text-secondary)]">AI is analyzing your data and goals...</span>
                  </div>
                )}

                {!vizLoading && vizRecommendations && vizRecommendations.length > 0 && (
                  <div className="space-y-4">
                    {/* Recommendation cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {vizRecommendations.map((rec: any, i: number) => (
                        <div key={i} className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <div className="flex items-start gap-3">
                            <div className="w-8 h-8 rounded-lg bg-violet-50 flex items-center justify-center shrink-0">
                              {rec.chart_type?.includes('bar') ? <BarChart3 className="w-4 h-4 text-violet-600" /> :
                               rec.chart_type?.includes('pie') ? <ChartPie className="w-4 h-4 text-violet-600" /> :
                               rec.chart_type?.includes('line') ? <TrendingUp className="w-4 h-4 text-violet-600" /> :
                               <Eye className="w-4 h-4 text-violet-600" />}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-sm">{rec.title || `Visualization ${i + 1}`}</p>
                              {rec.chart_type && (
                                <Badge variant="primary" className="mt-1">{rec.chart_type}</Badge>
                              )}
                              <p className="text-xs text-[var(--text-secondary)] mt-1">{rec.description || rec.reasoning || ''}</p>
                              {rec.columns && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                  {(Array.isArray(rec.columns) ? rec.columns : []).map((col: string, j: number) => (
                                    <span key={j} className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-muted)]">{col}</span>
                                  ))}
                                </div>
                              )}
                              {rec.kpi_value != null && (
                                <div className="mt-2 flex items-center gap-2">
                                  <span className="text-lg font-bold text-[var(--accent)]">{rec.kpi_value}</span>
                                  {rec.kpi_label && <span className="text-xs text-[var(--text-muted)]">{rec.kpi_label}</span>}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Plotly charts */}
                    {vizCharts && vizCharts.length > 0 && (
                      <div className="space-y-4 mt-4">
                        <h4 className="font-semibold text-sm flex items-center gap-2">
                          <BarChart3 className="w-4 h-4 text-indigo-600" /> Generated Charts
                        </h4>
                        <div className="grid grid-cols-1 gap-4">
                          {vizCharts.map((chart: any, i: number) => (
                            <div key={i} className="rounded-lg border border-[var(--border)] overflow-hidden bg-white p-2">
                              <div
                                ref={(el) => {
                                  if (el && chart) {
                                    // @ts-ignore - plotly.js-dist-min has no types
                                    import('plotly.js-dist-min').then((Plotly) => {
                                      let parsed = chart;
                                      if (chart.plotly_json && typeof chart.plotly_json === 'string') {
                                        try { parsed = JSON.parse(chart.plotly_json); } catch { parsed = chart; }
                                      }
                                      Plotly.newPlot(el, parsed.data || chart.data || [], parsed.layout || chart.layout || {}, { responsive: true, displayModeBar: false });
                                    }).catch(() => {});
                                  }
                                }}
                                style={{ width: '100%', minHeight: 350 }}
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {!vizLoading && vizRecommendations && vizRecommendations.length === 0 && (
                  <p className="text-sm text-[var(--text-muted)] py-4">No specific recommendations generated. Try refining your business goals.</p>
                )}

                {!vizLoading && !vizRecommendations && (
                  <div className="text-center py-6 text-sm text-[var(--text-muted)]">
                    <Sparkles className="w-6 h-6 mx-auto mb-2 text-violet-400" />
                    Click "Generate Recommendations" to get AI-suggested KPIs, charts, and visualizations based on your goals.
                  </div>
                )}
              </Card>

              {/* Re-run */}
              <div className="flex justify-center">
                <Button variant="secondary" onClick={runAnalysis} disabled={analyzing}
                  icon={<RotateCcw className="w-4 h-4" />}>
                  Re-run Analysis
                </Button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
