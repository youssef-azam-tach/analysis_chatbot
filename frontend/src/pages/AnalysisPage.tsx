import { useCallback, useEffect, useState } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { useAuthStore } from '@/stores/auth-store';
import { engine, parsePlotly, pollTask } from '@/lib/api';
import { loadPageState, savePageState } from '@/lib/pagePersistence';
import { Button, Card, Badge, Spinner, EmptyState, Tabs } from '@/components/ui';
import { Brain, Zap, ShieldCheck, BarChart3, Download, FileText, FileSpreadsheet, Presentation, Pin } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';

interface DatasetInfo {
  key: string;
  rows: number;
  columns: number;
}

interface SessionInfo {
  dataframe_count: number;
  dataframe_keys: string[];
  has_active_df: boolean;
  active_df_shape: number[] | null;
}

export default function AnalysisPage() {
  const { sessionId, activeWorkspace, addPinnedCard, addPinnedKpi, addPinnedChart } = useWorkspaceStore();
  const { user } = useAuthStore();
  const [tab, setTab] = useState('strategic');
  const [autoTab, setAutoTab] = useState('metrics');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [trust, setTrust] = useState<any>(null);
  const [downloading, setDownloading] = useState('');
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo[]>([]);
  const [autoVisuals, setAutoVisuals] = useState<any>(null);
  const [cacheHydrated, setCacheHydrated] = useState(false);

  const totalRows = datasetInfo.reduce((acc, item) => acc + (item.rows || 0), 0);
  const totalColumns = datasetInfo.reduce((acc, item) => acc + (item.columns || 0), 0);

  const loadDataOverview = useCallback(async () => {
    if (!sessionId) {
      setSessionInfo(null);
      setDatasetInfo([]);
      return;
    }
    try {
      const { data } = await engine.get(`/files/session-info?session_id=${sessionId}`);
      const info = data.data as SessionInfo;
      setSessionInfo(info);

      const keys = info?.dataframe_keys || [];
      if (!keys.length) {
        setDatasetInfo([]);
        return;
      }

      const previews = await Promise.all(
        keys.map(async (key) => {
          try {
            const res = await engine.get(`/files/preview?session_id=${sessionId}&key=${encodeURIComponent(key)}&rows=1`);
            const payload = res.data?.data || {};
            const shape = Array.isArray(payload.shape) ? payload.shape : [0, 0];
            return {
              key: payload.key || key,
              rows: Number(shape[0] || 0),
              columns: Number(shape[1] || 0),
            };
          } catch {
            return { key, rows: 0, columns: 0 };
          }
        }),
      );
      setDatasetInfo(previews);
    } catch {
      setSessionInfo(null);
      setDatasetInfo([]);
    }
  }, [sessionId]);

  const fetchAutoVisuals = useCallback(async (sid: string) => {
    try {
      const { data } = await engine.post('/strategic/auto-visuals', { session_id: sid });
      setAutoVisuals(data.data || null);
    } catch {
      setAutoVisuals(null);
    }
  }, []);

  useEffect(() => {
    loadDataOverview();
  }, [loadDataOverview]);

  useEffect(() => {
    const cached = loadPageState<any>('analysis-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    });
    if (cached) {
      setTab(cached.tab || 'strategic');
      setAutoTab(cached.autoTab || 'metrics');
      setResult(cached.result || null);
      setTrust(cached.trust || null);
      setAutoVisuals(cached.autoVisuals || null);
    }
    setCacheHydrated(true);
  }, [user?.id, activeWorkspace?.id, sessionId]);

  useEffect(() => {
    if (!cacheHydrated) return;
    savePageState('analysis-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    }, {
      tab,
      autoTab,
      result,
      trust,
      autoVisuals,
    });
  }, [cacheHydrated, user?.id, activeWorkspace?.id, sessionId, tab, autoTab, result, trust, autoVisuals]);

  const runStrategic = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    setProgress(0);
    try {
      // POST /strategic/analyze returns TaskResponse
      const { data } = await engine.post('/strategic/analyze', {
        session_id: sessionId,
        goals: { problem: 'General analysis', objective: 'Discover insights', target: '' },
      });
      const taskId = data.data?.task_id;
      if (!taskId) { setResult(data.data); return; }
      toast('Running strategic analysis...', { icon: '⏳' });
      const taskResult = await pollTask(taskId, (p) => setProgress(p));
      setResult(taskResult);
      await fetchAutoVisuals(sessionId);
      toast.success('Strategic analysis complete');
    } catch (err: any) {
      toast.error(err.message || err.response?.data?.detail || 'Analysis failed');
    } finally { setLoading(false); setProgress(0); }
  };

  const runAutoEDA = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    setProgress(0);
    try {
      // POST /analysis/automatic returns TaskResponse
      const { data } = await engine.post('/analysis/automatic', { session_id: sessionId });
      const taskId = data.data?.task_id;
      if (!taskId) { setResult(data.data); return; }
      toast('Running automatic EDA...', { icon: '⏳' });
      const taskResult = await pollTask(taskId, (p) => setProgress(p));
      setResult(taskResult);
      toast.success('Auto EDA complete');
    } catch (err: any) {
      toast.error(err.message || err.response?.data?.detail || 'Analysis failed');
    } finally { setLoading(false); setProgress(0); }
  };

  const runTrust = async () => {
    if (!sessionId || !result) return toast.error('Run an analysis first');
    setLoading(true);
    try {
      // Build insights array — extract meaningful sentences from markdown
      let insights: string[] = [];
      const md = result.analysis_markdown || '';
      if (md) {
        // Split by lines, pick sentences that look like insights (non-headers, non-empty, >20 chars)
        const lines = md.split('\n').map((l: string) => l.trim()).filter((l: string) => l.length > 20 && !l.startsWith('#') && !l.startsWith('---') && !l.startsWith('|'));
        // Take up to 10 insight lines
        insights = lines.slice(0, 10);
      }
      if (insights.length === 0 && result.summary_report) {
        const report = typeof result.summary_report === 'string' ? result.summary_report : JSON.stringify(result.summary_report);
        insights = report.split('\n').filter((l: string) => l.trim().length > 20).slice(0, 10);
      }
      if (insights.length === 0) {
        insights = ['General data analysis completed'];
      }

      const { data } = await engine.post('/trust/verify-analysis', {
        session_id: sessionId,
        analysis_results: { insights },
      });
      setTrust(data.data);
      toast.success('Trust validation complete');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Trust validation failed');
    } finally { setLoading(false); }
  };

  const downloadReport = async (reportType: string) => {
    if (!sessionId) return toast.error('Upload data first');
    setDownloading(reportType);
    try {
      if (result?.analysis_markdown && (reportType === 'pdf' || reportType === 'pptx')) {
        const allCharts = collectAllReportCharts();
        // Use strategic-report endpoint for PDF/PPTX with markdown
        const { data } = await engine.post('/reports/strategic-report', {
          session_id: sessionId,
          report_type: reportType,
          analysis_markdown: result.analysis_markdown,
          goals: result.goals || { problem: 'General analysis', objective: 'Discover insights' },
          recommendations: result.recommendations || [],
          plotly_charts: allCharts,
        });
        const taskId = data.data?.task_id;
        if (taskId) {
          toast('Generating report...', { icon: '⏳' });
          await pollTask(taskId);
          // Download
          const response = await engine.get(`/reports/download/${taskId}`, { responseType: 'blob' });
          const ext = reportType === 'pptx' ? 'pptx' : 'pdf';
          triggerDownload(response.data, `analysis_report.${ext}`);
          toast.success(`${reportType.toUpperCase()} report downloaded`);
        }
      } else {
        // General report generation
        const { data } = await engine.post('/reports/generate', {
          session_id: sessionId,
          report_type: reportType,
        });
        const taskId = data.data?.task_id;
        if (taskId) {
          toast('Generating report...', { icon: '⏳' });
          await pollTask(taskId);
          const response = await engine.get(`/reports/download/${taskId}`, { responseType: 'blob' });
          const extMap: Record<string, string> = { pdf: 'pdf', markdown: 'md', excel: 'xlsx', pptx: 'pptx' };
          triggerDownload(response.data, `analysis_report.${extMap[reportType] || reportType}`);
          toast.success(`${reportType.toUpperCase()} report downloaded`);
        }
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || err.message || `Failed to download ${reportType}`);
    } finally { setDownloading(''); }
  };

  const downloadMarkdownDirect = () => {
    if (!result?.analysis_markdown) return toast.error('No analysis to download');
    const blob = new Blob([result.analysis_markdown], { type: 'text/markdown' });
    triggerDownload(blob, 'analysis_report.md');
    toast.success('Markdown downloaded');
  };

  const downloadHTMLDirect = () => {
    if (!result?.analysis_markdown) return toast.error('No analysis to download');
    const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Analysis Report</title>
<style>body{font-family:Inter,sans-serif;max-width:900px;margin:0 auto;padding:40px;line-height:1.7;color:#333}
h1,h2,h3{color:#1a1a2e}code{background:#f0f0f0;padding:2px 6px;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #e0e0e0;padding:8px 12px;text-align:left}
th{background:#f8f9fa}blockquote{border-left:4px solid #6366f1;margin:20px 0;padding:10px 20px;background:#f8f7ff}</style>
</head><body>${result.analysis_markdown.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</body></html>`;
    const blob = new Blob([html], { type: 'text/html' });
    triggerDownload(blob, 'analysis_report.html');
    toast.success('HTML downloaded');
  };

  const triggerDownload = (data: Blob, filename: string) => {
    const url = URL.createObjectURL(data);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const pinCard = (label: string, value: string, source = 'AI Analysis') => {
    addPinnedCard({
      id: `${sessionId || 'no-session'}::card::${label}`,
      sessionId,
      label,
      value,
      source,
    });
    toast.success(`Pinned ${label} to Dashboard`);
  };

  const pinKpi = (label: string, value: string, suffix?: string, source = 'AI Analysis') => {
    addPinnedKpi({
      id: `${sessionId || 'no-session'}::kpi::${source}::${label}`,
      sessionId,
      label,
      value,
      suffix,
      source,
    });
    toast.success(`Pinned KPI ${label} to Dashboard`);
  };

  const pinChart = (title: string, parsed: { data: any[]; layout: any } | null, source = 'AI Analysis', chartType?: string) => {
    if (!parsed) return;
    addPinnedChart({
      id: `${sessionId || 'no-session'}::chart::${source}::${title}`,
      sessionId,
      title,
      plotly_json: JSON.stringify({ data: parsed.data || [], layout: parsed.layout || {} }),
      source,
      chartType,
    });
    toast.success(`Pinned chart ${title} to Dashboard`);
  };

  const buildChartInsight = (title: string, chartType?: string) => {
    const t = (chartType || '').toLowerCase();
    if (t.includes('line') || title.toLowerCase().includes('trend')) return 'Business Insight: This chart highlights trend direction over time and helps forecast momentum.';
    if (t.includes('bar') || title.toLowerCase().includes('top')) return 'Business Insight: This chart compares performance segments and reveals top and underperforming categories.';
    if (t.includes('pie') || title.toLowerCase().includes('composition')) return 'Business Insight: This chart shows contribution share by segment to support prioritization.';
    if (t.includes('hist') || title.toLowerCase().includes('distribution')) return 'Business Insight: This chart reveals distribution spread and potential outliers impacting business stability.';
    if (t.includes('scatter')) return 'Business Insight: This chart indicates relationship patterns between key metrics for root-cause analysis.';
    return 'Business Insight: This chart provides actionable evidence for business decision-making.';
  };

  const collectAllReportCharts = () => {
    const out: any[] = [];

    const pushChart = (chart: any, fallbackTitle: string, chartType?: string) => {
      const parsed = parsePlotly(chart?.plotly_json || chart);
      if (!parsed) return;
      const title = chart?.title || fallbackTitle;
      const insight = chart?.business_insight || chart?.insight || chart?.reasoning || buildChartInsight(title, chartType || chart?.chart_type);
      out.push({
        title,
        chart_type: chartType || chart?.chart_type,
        plotly_json: JSON.stringify({ data: parsed.data || [], layout: parsed.layout || {} }),
        business_insight: insight,
      });
    };

    if (Array.isArray(result?.plotly_charts)) {
      result.plotly_charts.forEach((chart: any, idx: number) => pushChart(chart, `AI Chart ${idx + 1}`, chart?.chart_type));
    }

    const autoGroups = [
      ...(autoVisuals?.trend_charts || []).map((c: any) => ({ ...c, _t: 'line' })),
      ...(autoVisuals?.comparison_charts || []).map((c: any) => ({ ...c, _t: 'bar' })),
      ...(autoVisuals?.distribution_charts || []).map((c: any) => ({ ...c, _t: 'histogram' })),
    ];
    autoGroups.forEach((chart: any, idx: number) => pushChart(chart, `Auto Chart ${idx + 1}`, chart._t));

    return out;
  };

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">AI Analysis</h1>
          <p className="text-[var(--text-secondary)] mt-1">Comprehensive Business Intelligence & Strategic Analysis</p>
        </div>
        <div className="flex items-center gap-2">
          {result && (
            <Button onClick={runTrust} variant="secondary" icon={<ShieldCheck className="w-4 h-4" />} loading={loading && !downloading}>
              Validate Trust
            </Button>
          )}
        </div>
      </div>

      <Tabs
        tabs={[
          { key: 'strategic', label: 'Strategic Analysis', icon: <Brain className="w-4 h-4" /> },
          { key: 'auto', label: 'Auto EDA', icon: <Zap className="w-4 h-4" /> },
        ]}
        active={tab}
        onChange={(t) => { setTab(t); setResult(null); setTrust(null); }}
      />

      <div className="mt-6">
        {sessionId && tab === 'strategic' && (
          <Card>
            <h3 className="font-semibold mb-4 flex items-center gap-2"><BarChart3 className="w-4 h-4 text-[var(--accent)]" /> Data Overview</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <div className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                <p className="text-xs text-[var(--text-muted)]">Total Datasets</p>
                <p className="text-lg font-semibold">{sessionInfo?.dataframe_count || datasetInfo.length}</p>
                <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinCard('Total Datasets', String(sessionInfo?.dataframe_count || datasetInfo.length), 'Overview')}>Pin to Dashboard</Button>
              </div>
              <div className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                <p className="text-xs text-[var(--text-muted)]">Total Rows</p>
                <p className="text-lg font-semibold">{totalRows.toLocaleString()}</p>
                <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinCard('Total Rows', totalRows.toLocaleString(), 'Overview')}>Pin to Dashboard</Button>
              </div>
              <div className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                <p className="text-xs text-[var(--text-muted)]">Total Columns</p>
                <p className="text-lg font-semibold">{totalColumns}</p>
                <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinCard('Total Columns', String(totalColumns), 'Overview')}>Pin to Dashboard</Button>
              </div>
              <div className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                <p className="text-xs text-[var(--text-muted)]">Active Dataset</p>
                <p className="text-lg font-semibold">{sessionInfo?.has_active_df ? 'Yes' : 'No'}</p>
                <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinCard('Active Dataset', sessionInfo?.has_active_df ? 'Yes' : 'No', 'Overview')}>Pin to Dashboard</Button>
              </div>
            </div>

            {datasetInfo.length > 0 && (
              <div className="rounded-lg border border-[var(--border)] overflow-hidden">
                <div className="px-3 py-2 text-xs font-medium bg-[var(--bg-secondary)] border-b border-[var(--border)]">Loaded Datasets</div>
                <div className="divide-y divide-[var(--border)]">
                  {datasetInfo.map((ds) => (
                    <div key={ds.key} className="px-3 py-2 text-sm flex items-center justify-between">
                      <span className="text-[var(--text-secondary)]">{ds.key}</span>
                      <span className="text-[var(--text-muted)]">{ds.rows.toLocaleString()} rows × {ds.columns} cols</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Spinner size="lg" />
            <p className="text-sm text-[var(--text-muted)] mt-4">
              {progress > 0 ? `Processing... ${Math.round(progress)}%` : 'Submitting task...'}
            </p>
          </div>
        )}

        {!loading && !result && tab === 'strategic' && (
          <EmptyState icon={<Brain className="w-8 h-8" />} title="Strategic Analysis" description="AI will analyze your data based on business goals, generating insights and recommendations">
            <Button onClick={runStrategic} className="mt-4" icon={<Brain className="w-4 h-4" />}>🚀 Run Full Strategic Analysis on ALL Data</Button>
          </EmptyState>
        )}

        {!loading && !result && tab === 'auto' && (
          <EmptyState icon={<Zap className="w-8 h-8" />} title="Automatic EDA" description="Let AI automatically discover patterns, distributions, correlations, and anomalies">
            <Button onClick={runAutoEDA} className="mt-4" icon={<Zap className="w-4 h-4" />}>Run Auto EDA</Button>
          </EmptyState>
        )}

        {!loading && result && (
          <div className="space-y-6">
            {/* Strategic: analysis_markdown */}
            {result.analysis_markdown && (
              <Card>
                <h3 className="font-semibold mb-4 flex items-center gap-2"><Brain className="w-4 h-4 text-[var(--accent)]" /> Analysis Report</h3>
                <div className="flex flex-wrap items-center gap-2 mb-4 text-xs text-[var(--text-muted)]">
                  <Badge variant="default">Generated: {new Date().toLocaleString()}</Badge>
                  <Badge variant="default">Total Rows: {totalRows.toLocaleString()}</Badge>
                </div>
                <div className="prose prose-invert prose-sm max-w-none schema-markdown">
                  <ReactMarkdown>{result.analysis_markdown}</ReactMarkdown>
                </div>
              </Card>
            )}

            {Array.isArray(result.plotly_charts) && result.plotly_charts.length > 0 && (
              <Card>
                <h3 className="font-semibold mb-4 flex items-center gap-2"><BarChart3 className="w-4 h-4 text-violet-600" /> AI-Recommended Visualizations</h3>
                <p className="text-xs text-[var(--text-muted)] mb-4">Charts generated based on AI analysis of your data</p>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {result.plotly_charts.map((chart: any, index: number) => {
                    const parsed = parsePlotly(chart.plotly_json || chart);
                    if (!parsed) return null;
                    return (
                      <div key={`ai-chart-${index}`} className="rounded-lg border border-[var(--border)] overflow-hidden bg-white p-2">
                        <div className="text-xs px-2 py-1 text-[var(--text-muted)] flex items-center justify-between">
                          <span>{chart.title || `AI Chart ${index + 1}`}</span>
                          <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinChart(chart.title || `AI Chart ${index + 1}`, parsed, 'AI Recommended', chart.chart_type)}>Pin to Dashboard</Button>
                        </div>
                        <p className="text-xs px-2 pb-2 text-[var(--text-secondary)]">{buildChartInsight(chart.title || `AI Chart ${index + 1}`, chart.chart_type)}</p>
                        <div
                          ref={(el) => {
                            if (el && parsed) {
                              // @ts-ignore
                              import('plotly.js-dist-min').then((Plotly) => {
                                Plotly.newPlot(el, parsed.data || [], parsed.layout || {}, { responsive: true, displayModeBar: false });
                              }).catch(() => {});
                            }
                          }}
                          style={{ width: '100%', minHeight: 320 }}
                        />
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}

            {autoVisuals && tab === 'strategic' && (
              <Card>
                <h3 className="font-semibold mb-4 flex items-center gap-2"><BarChart3 className="w-4 h-4 text-emerald-600" /> Auto-Generated Business Visualizations</h3>
                {Array.isArray(autoVisuals.not_useful_notes) && autoVisuals.not_useful_notes.length > 0 && (
                  <div className="mb-4 space-y-1">
                    {autoVisuals.not_useful_notes.map((note: string, idx: number) => (
                      <p key={`note-${idx}`} className="text-xs text-[var(--text-muted)]">{note}</p>
                    ))}
                  </div>
                )}
                <div className="flex justify-end mb-3">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => sessionId && fetchAutoVisuals(sessionId)}
                  >
                    Regenerate Visuals
                  </Button>
                </div>

                <Tabs
                  tabs={[
                    { key: 'metrics', label: 'Key Metrics', icon: <BarChart3 className="w-4 h-4" /> },
                    { key: 'trends', label: 'Trends', icon: <BarChart3 className="w-4 h-4" /> },
                    { key: 'comparisons', label: 'Comparisons', icon: <BarChart3 className="w-4 h-4" /> },
                    { key: 'distributions', label: 'Distributions', icon: <BarChart3 className="w-4 h-4" /> },
                  ]}
                  active={autoTab}
                  onChange={setAutoTab}
                />

                <div className="mt-4">
                  {autoTab === 'metrics' && (
                    <>
                    {(autoVisuals.key_metrics || []).length === 0 && (
                      <p className="text-sm text-[var(--text-muted)]">No key metrics generated for current datasets. Try Regenerate Visuals after changing selected datasets.</p>
                    )}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {(autoVisuals.key_metrics || []).slice(0, 12).map((item: any, i: number) => (
                        <div key={`metric-${i}`} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <p className="text-xs text-[var(--text-muted)] mb-1">{item.dataset}</p>
                          <p className="text-sm font-medium">{item.metric}</p>
                          <p className="text-xs text-[var(--text-secondary)]">Total: {Number(item.total || 0).toLocaleString()}</p>
                          <p className="text-xs text-[var(--text-secondary)]">Avg: {Number(item.average || 0).toFixed(2)}</p>
                          {item.fallback && <p className="text-[10px] text-[var(--text-muted)] mt-1">Fallback metric (identifier-like column)</p>}
                          <Button
                            size="sm"
                            variant="ghost"
                            className="mt-2"
                            icon={<Pin className="w-3.5 h-3.5" />}
                            onClick={() => pinKpi(item.metric, Number(item.total || 0).toLocaleString(), undefined, item.dataset || 'Auto Metrics')}
                          >
                            Pin to Dashboard
                          </Button>
                        </div>
                      ))}
                    </div>
                    </>
                  )}

                  {autoTab === 'trends' && (
                    <>
                    {(autoVisuals.trend_charts || []).length === 0 && (
                      <p className="text-sm text-[var(--text-muted)]">No trend chart generated (date/time + numeric pair not found).</p>
                    )}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      {(autoVisuals.trend_charts || []).map((chart: any, i: number) => {
                        const parsed = parsePlotly(chart.plotly_json);
                        if (!parsed) return null;
                        return (
                          <div key={`trend-${i}`} className="rounded-lg border border-[var(--border)] overflow-hidden bg-white p-2">
                            <div className="text-xs px-2 py-1 text-[var(--text-muted)] flex items-center justify-between">
                              <span>{chart.title}</span>
                              <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinChart(chart.title, parsed, chart.dataset || 'Trends', 'line')}>Pin to Dashboard</Button>
                            </div>
                            <p className="text-xs px-2 pb-2 text-[var(--text-secondary)]">{buildChartInsight(chart.title, 'line')}</p>
                            <div
                              ref={(el) => {
                                if (el && parsed) {
                                  // @ts-ignore
                                  import('plotly.js-dist-min').then((Plotly) => {
                                    Plotly.newPlot(el, parsed.data || [], parsed.layout || {}, { responsive: true, displayModeBar: false });
                                  }).catch(() => {});
                                }
                              }}
                              style={{ width: '100%', minHeight: 320 }}
                            />
                          </div>
                        );
                      })}
                    </div>
                    </>
                  )}

                  {autoTab === 'comparisons' && (
                    <>
                    {(autoVisuals.comparison_charts || []).length === 0 && (
                      <p className="text-sm text-[var(--text-muted)]">No comparison chart generated (categorical + numeric pair not found).</p>
                    )}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      {(autoVisuals.comparison_charts || []).map((chart: any, i: number) => {
                        const parsed = parsePlotly(chart.plotly_json);
                        if (!parsed) return null;
                        return (
                          <div key={`comparison-${i}`} className="rounded-lg border border-[var(--border)] overflow-hidden bg-white p-2">
                            <div className="text-xs px-2 py-1 text-[var(--text-muted)] flex items-center justify-between">
                              <span>{chart.title}</span>
                              <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinChart(chart.title, parsed, chart.dataset || 'Comparisons', 'bar')}>Pin to Dashboard</Button>
                            </div>
                            <p className="text-xs px-2 pb-2 text-[var(--text-secondary)]">{buildChartInsight(chart.title, 'bar')}</p>
                            <div
                              ref={(el) => {
                                if (el && parsed) {
                                  // @ts-ignore
                                  import('plotly.js-dist-min').then((Plotly) => {
                                    Plotly.newPlot(el, parsed.data || [], parsed.layout || {}, { responsive: true, displayModeBar: false });
                                  }).catch(() => {});
                                }
                              }}
                              style={{ width: '100%', minHeight: 320 }}
                            />
                          </div>
                        );
                      })}
                    </div>
                    </>
                  )}

                  {autoTab === 'distributions' && (
                    <>
                    {(autoVisuals.distribution_charts || []).length === 0 && (
                      <p className="text-sm text-[var(--text-muted)]">No distribution chart generated (numeric column not found).</p>
                    )}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      {(autoVisuals.distribution_charts || []).map((chart: any, i: number) => {
                        const parsed = parsePlotly(chart.plotly_json);
                        if (!parsed) return null;
                        return (
                          <div key={`distribution-${i}`} className="rounded-lg border border-[var(--border)] overflow-hidden bg-white p-2">
                            <div className="text-xs px-2 py-1 text-[var(--text-muted)] flex items-center justify-between">
                              <span>{chart.title}</span>
                              <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinChart(chart.title, parsed, chart.dataset || 'Distributions', 'histogram')}>Pin to Dashboard</Button>
                            </div>
                            <p className="text-xs px-2 pb-2 text-[var(--text-secondary)]">{buildChartInsight(chart.title, 'histogram')}</p>
                            <div
                              ref={(el) => {
                                if (el && parsed) {
                                  // @ts-ignore
                                  import('plotly.js-dist-min').then((Plotly) => {
                                    Plotly.newPlot(el, parsed.data || [], parsed.layout || {}, { responsive: true, displayModeBar: false });
                                  }).catch(() => {});
                                }
                              }}
                              style={{ width: '100%', minHeight: 320 }}
                            />
                          </div>
                        );
                      })}
                    </div>
                    </>
                  )}
                </div>
              </Card>
            )}

            {/* Auto EDA: summary_report, column_summary, correlations, etc. */}
            {result.summary_report && (
              <Card>
                <h3 className="font-semibold mb-4 flex items-center gap-2"><BarChart3 className="w-4 h-4 text-violet-600" /> Summary Report</h3>
                <pre className="text-sm text-[var(--text-secondary)] overflow-auto max-h-[400px] whitespace-pre-wrap bg-[var(--bg-primary)] p-4 rounded-lg">
                  {typeof result.summary_report === 'string' ? result.summary_report : JSON.stringify(result.summary_report, null, 2)}
                </pre>
              </Card>
            )}

            {result.high_correlations && result.high_correlations.length > 0 && (
              <Card>
                <h3 className="font-semibold mb-4">High Correlations</h3>
                <div className="space-y-2">
                  {result.high_correlations.map((c: any, i: number) => (
                    <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-[var(--bg-secondary)]">
                      <Badge variant={c.correlation > 0.8 ? 'success' : 'warning'}>
                        {typeof c.correlation === 'number' ? c.correlation.toFixed(3) : c.correlation}
                      </Badge>
                      <span className="text-sm">{c.column_1 || c.col1} ↔ {c.column_2 || c.col2}</span>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Fallback: render any remaining result */}
            {!result.analysis_markdown && !result.summary_report && (
              <Card>
                <pre className="text-sm text-[var(--text-secondary)] overflow-auto max-h-[500px] whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
              </Card>
            )}

            {/* ── Download Report Buttons ── */}
            <Card>
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Download className="w-4 h-4 text-[var(--accent)]" /> Download Report
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Export this analysis in different formats</p>
              <div className="flex flex-wrap gap-3">
                {result.analysis_markdown && (
                  <>
                    <Button
                      variant="secondary"
                      onClick={downloadMarkdownDirect}
                      icon={<FileText className="w-4 h-4" />}
                    >
                      Markdown (.md)
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={downloadHTMLDirect}
                      icon={<FileText className="w-4 h-4" />}
                    >
                      HTML
                    </Button>
                  </>
                )}
                <Button
                  variant="secondary"
                  onClick={() => downloadReport('pdf')}
                  loading={downloading === 'pdf'}
                  icon={<FileText className="w-4 h-4" />}
                >
                  PDF
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => downloadReport('pptx')}
                  loading={downloading === 'pptx'}
                  icon={<Presentation className="w-4 h-4" />}
                >
                  PowerPoint (.pptx)
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => downloadReport('excel')}
                  loading={downloading === 'excel'}
                  icon={<FileSpreadsheet className="w-4 h-4" />}
                >
                  Excel (.xlsx)
                </Button>
              </div>
            </Card>

            {/* Trust Layer */}
            {trust && (
              <Card className="border-[var(--accent)]/30">
                <h3 className="font-semibold mb-4 flex items-center gap-2"><ShieldCheck className="w-4 h-4 text-emerald-600" /> Trust Validation</h3>
                <div className="flex items-center gap-4 mb-4">
                  <div className="text-center p-3 rounded-lg bg-[var(--bg-secondary)]">
                    <p className="text-2xl font-bold">{trust.total ?? '—'}</p>
                    <p className="text-xs text-[var(--text-muted)]">Insights Verified</p>
                  </div>
                </div>
                {trust.verified_insights?.length > 0 && (
                  <div className="space-y-3">
                    {trust.verified_insights.map((vi: any, i: number) => (
                      <div key={i} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant={vi.verification_status === 'verified' ? 'success' : vi.verification_status === 'partial' ? 'warning' : 'danger'}>
                            {vi.verification_status}
                          </Badge>
                          {vi.confidence_score != null && (
                            <span className="text-xs text-[var(--text-muted)]">{Math.round(vi.confidence_score * 100)}% confidence</span>
                          )}
                          {vi.signal_strength && <Badge variant="default">{vi.signal_strength}</Badge>}
                        </div>
                        <p className="text-sm text-[var(--text-secondary)] mt-1">{vi.insight_text}</p>
                        {vi.evidence?.length > 0 && (
                          <div className="mt-2 space-y-1">
                            {vi.evidence.map((e: any, j: number) => (
                              <p key={j} className="text-xs text-[var(--text-muted)]">📊 {e.source}: {e.description}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
