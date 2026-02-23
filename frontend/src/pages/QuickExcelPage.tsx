import { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';
import { useAuthStore } from '@/stores/auth-store';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { engine, parsePlotly, pollTask } from '@/lib/api';
import { loadPageState, savePageState } from '@/lib/pagePersistence';
import { Badge, Button, Card, DataTable, EmptyState, Input, Select, Spinner, Textarea } from '@/components/ui';
import { Download, FileSpreadsheet, MessageCircle, Sparkles, Upload } from 'lucide-react';

type QuickRole = 'Accountant' | 'Manager' | 'Finance Officer' | 'Analyst' | 'Business Owner' | 'Other';

interface DatasetMeta {
  key: string;
  sheetName: string;
  rows: number;
  columns: number;
}

interface QuickAnalysisState {
  markdown: string;
  keyFindings: string[];
  recommendations: string[];
  charts: Array<{ title: string; plotly_json: any }>;
  quality: Record<string, Record<string, any>>;
}

interface QuickQaItem {
  question: string;
  answer: string;
}

function normalizeSheetName(key: string) {
  const parts = String(key || '').split('::');
  return parts.length > 1 ? parts.slice(1).join('::') : key;
}

function extractInsights(markdown: string, max = 4) {
  if (!markdown) return [] as string[];
  const lines = markdown
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 35 && !line.startsWith('#') && !line.startsWith('|') && !line.startsWith('---'));
  return lines.slice(0, max);
}

function buildSummaryText(context: {
  filename: string;
  purpose: string;
  role: string;
  selectedSheets: string[];
  hasRelationships: boolean;
  datasets: DatasetMeta[];
  findings: string[];
  recommendations: string[];
}) {
  const selected = context.datasets.filter((item) => context.selectedSheets.includes(item.key));
  const sheetLines = selected.map((item) => `- ${item.sheetName}: ${item.rows.toLocaleString()} rows × ${item.columns} columns`);
  const findingLines = context.findings.length ? context.findings.map((f, i) => `${i + 1}. ${f}`) : ['1. No findings generated'];
  const recommendationLines = context.recommendations.length
    ? context.recommendations.map((r, i) => `${i + 1}. ${r}`)
    : ['1. Validate data quality and ask targeted follow-up questions.'];

  return [
    'QUICK EXCEL ANALYSIS SUMMARY',
    '============================',
    '',
    `File: ${context.filename || 'N/A'}`,
    `Purpose: ${context.purpose || 'N/A'}`,
    `Role: ${context.role || 'N/A'}`,
    `Multi-sheet relationships: ${context.hasRelationships ? 'Yes' : 'No'}`,
    '',
    'DATASETS',
    ...sheetLines,
    '',
    'KEY FINDINGS',
    ...findingLines,
    '',
    'RECOMMENDATIONS',
    ...recommendationLines,
  ].join('\n');
}

export default function QuickExcelPage() {
  const { user } = useAuthStore();
  const { activeWorkspace } = useWorkspaceStore();

  const [fileName, setFileName] = useState('');
  const [quickSessionId, setQuickSessionId] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [selectedSheets, setSelectedSheets] = useState<string[]>([]);
  const [purpose, setPurpose] = useState('');
  const [role, setRole] = useState<QuickRole>('Accountant');
  const [hasRelationships, setHasRelationships] = useState(false);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [customQuestionInput, setCustomQuestionInput] = useState('');
  const [customQuestions, setCustomQuestions] = useState<string[]>([]);
  const [analysis, setAnalysis] = useState<QuickAnalysisState | null>(null);
  const [qaInput, setQaInput] = useState('');
  const [qaHistory, setQaHistory] = useState<QuickQaItem[]>([]);

  const [uploading, setUploading] = useState(false);
  const [askingSuggestions, setAskingSuggestions] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [asking, setAsking] = useState(false);
  const [downloadingReport, setDownloadingReport] = useState(false);

  const cacheScope = useMemo(() => ({
    userId: user?.id,
    workspaceId: activeWorkspace?.id,
    sessionId: `quick-${activeWorkspace?.id || 'none'}`,
  }), [user?.id, activeWorkspace?.id]);

  useEffect(() => {
    const cached = loadPageState<any>('quick-excel-page', cacheScope);
    if (!cached) return;
    setFileName(cached.fileName || '');
    setQuickSessionId(cached.quickSessionId || null);
    setDatasets(Array.isArray(cached.datasets) ? cached.datasets : []);
    setSelectedSheets(Array.isArray(cached.selectedSheets) ? cached.selectedSheets : []);
    setPurpose(cached.purpose || '');
    setRole((cached.role as QuickRole) || 'Accountant');
    setHasRelationships(Boolean(cached.hasRelationships));
    setSuggestedQuestions(Array.isArray(cached.suggestedQuestions) ? cached.suggestedQuestions : []);
    setCustomQuestions(Array.isArray(cached.customQuestions) ? cached.customQuestions : []);
    setAnalysis(cached.analysis || null);
    setQaHistory(Array.isArray(cached.qaHistory) ? cached.qaHistory : []);
  }, [cacheScope]);

  useEffect(() => {
    savePageState('quick-excel-page', cacheScope, {
      fileName,
      quickSessionId,
      datasets,
      selectedSheets,
      purpose,
      role,
      hasRelationships,
      suggestedQuestions,
      customQuestions,
      analysis,
      qaHistory,
    });
  }, [
    cacheScope,
    fileName,
    quickSessionId,
    datasets,
    selectedSheets,
    purpose,
    role,
    hasRelationships,
    suggestedQuestions,
    customQuestions,
    analysis,
    qaHistory,
  ]);

  const handleUpload = async (file: File | null) => {
    if (!file) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await engine.post('/files/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const payload = data?.data || {};
      const newSessionId = payload.session_id as string;
      const sheetItems: any[] = Array.isArray(payload.sheets) ? payload.sheets : [];
      const normalized: DatasetMeta[] = sheetItems.map((sheet: any): DatasetMeta => {
        const key = String(sheet?.dataset_key || sheet?.key || '');
        return {
          key,
          sheetName: normalizeSheetName(key || String(sheet?.name || sheet?.sheet_name || 'Sheet')),
          rows: Number(sheet?.rows || sheet?.total_rows || payload?.total_rows || 0),
          columns: Number(sheet?.columns || sheet?.total_columns || payload?.total_columns || 0),
        };
      }).filter((item: DatasetMeta) => Boolean(item.key));

      setFileName(String(payload?.filename || file.name));
      setQuickSessionId(newSessionId);
      setDatasets(normalized);
      setSelectedSheets(normalized.map((item) => item.key));
      setSuggestedQuestions([]);
      setCustomQuestions([]);
      setAnalysis(null);
      setQaHistory([]);

      toast.success('File uploaded for quick analysis');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const generateSuggestions = async () => {
    if (!quickSessionId) return toast.error('Upload a file first');
    if (!purpose.trim()) return toast.error('Please provide file purpose first');

    setAskingSuggestions(true);
    try {
      const { data } = await engine.post('/strategic/suggest-questions', {
        session_id: quickSessionId,
        purpose: purpose.trim(),
        role,
        max_questions: 3,
      });
      const questions = Array.isArray(data?.data?.questions) ? data.data.questions : [];
      setSuggestedQuestions(questions.slice(0, 3));
      if (!questions.length) toast('No suggestions returned');
    } catch (err: any) {
      const fallback = [
        `What are the key metrics related to ${purpose.trim()}?`,
        'Which categories have the highest totals?',
        'What data quality issues should be fixed first?',
      ];
      setSuggestedQuestions(fallback);
      toast.error(err?.response?.data?.detail || 'Could not generate AI suggestions, using fallback');
    } finally {
      setAskingSuggestions(false);
    }
  };

  const runAnalysis = async () => {
    if (!quickSessionId) return toast.error('Upload a file first');
    if (!purpose.trim()) return toast.error('Purpose is required before analysis');
    if (!selectedSheets.length) return toast.error('Select at least one sheet');

    setAnalyzing(true);
    try {
      await engine.post('/cleaning/select-datasets', {
        session_id: quickSessionId,
        keys: selectedSheets,
      });

      const strategicTaskResp = await engine.post('/strategic/analyze', {
        session_id: quickSessionId,
        goals: {
          problem: purpose.trim(),
          objective: `Quick analysis for ${role}`,
          target: role,
        },
      });

      const taskId = strategicTaskResp?.data?.data?.task_id;
      const strategicResult = taskId ? await pollTask(taskId) : strategicTaskResp?.data?.data;

      let qualityData: Record<string, Record<string, any>> = {};
      try {
        const qualityResp = await engine.post('/analysis/data-quality', { session_id: quickSessionId });
        qualityData = (qualityResp?.data?.data?.datasets || qualityResp?.data?.data || {}) as Record<string, Record<string, any>>;
      } catch {
        qualityData = {};
      }

      let autoCharts: Array<{ title: string; plotly_json: any }> = [];
      try {
        const { data } = await engine.post('/strategic/auto-visuals', { session_id: quickSessionId });
        const payload = data?.data || {};
        const groups = [
          ...(Array.isArray(payload?.trend_charts) ? payload.trend_charts : []),
          ...(Array.isArray(payload?.comparison_charts) ? payload.comparison_charts : []),
          ...(Array.isArray(payload?.distribution_charts) ? payload.distribution_charts : []),
        ];
        autoCharts = groups.map((item: any) => ({
          title: String(item?.title || 'Auto Chart'),
          plotly_json: item?.plotly_json,
        }));
      } catch {
        autoCharts = [];
      }

      const keyFindings = extractInsights(String(strategicResult?.analysis_markdown || ''), 4);
      const recommendations = Array.isArray(strategicResult?.recommendations)
        ? strategicResult.recommendations
            .map((r: any) => (typeof r === 'string' ? r : r?.reason || r?.insight || ''))
            .filter((r: string) => Boolean(r))
            .slice(0, 4)
        : [];

      const strategicCharts = Array.isArray(strategicResult?.plotly_charts)
        ? strategicResult.plotly_charts.map((item: any) => ({
            title: String(item?.title || item?.reason || 'AI Chart'),
            plotly_json: item?.plotly_json,
          }))
        : [];

      setAnalysis({
        markdown: String(strategicResult?.analysis_markdown || ''),
        keyFindings,
        recommendations,
        charts: [...strategicCharts, ...autoCharts],
        quality: qualityData,
      });
      toast.success('Quick analysis complete');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || err?.message || 'Quick analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const askQuickQuestion = async () => {
    if (!quickSessionId) return toast.error('Upload a file first');
    const question = qaInput.trim();
    if (!question) return;

    setAsking(true);
    try {
      const { data } = await engine.post('/chat/message', {
        session_id: quickSessionId,
        message: question,
        chatbot_type: 'hybrid',
        model: 'qwen2.5:7b',
      });

      const payload = data?.data || {};
      const answer = String(
        payload?.answer
        || payload?.text
        || payload?.response
        || payload?.message
        || 'No answer returned.'
      );

      setQaHistory((prev) => [...prev, { question, answer }]);
      setQaInput('');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to answer question');
    } finally {
      setAsking(false);
    }
  };

  const exportSummary = () => {
    if (!analysis) return toast.error('Run analysis first');
    const summaryText = buildSummaryText({
      filename: fileName,
      purpose,
      role,
      selectedSheets,
      hasRelationships,
      datasets,
      findings: analysis.keyFindings,
      recommendations: analysis.recommendations,
    });
    const blob = new Blob([summaryText], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'quick_analysis_summary.txt';
    a.click();
    URL.revokeObjectURL(a.href);
    toast.success('Summary exported');
  };

  const exportPdf = async () => {
    if (!quickSessionId || !analysis) return toast.error('Run analysis first');
    setDownloadingReport(true);
    try {
      const charts = analysis.charts
        .map((item) => {
          const parsed = parsePlotly(item.plotly_json);
          if (!parsed) return null;
          return {
            title: item.title,
            plotly_json: JSON.stringify({ data: parsed.data || [], layout: parsed.layout || {} }),
            business_insight: `Quick analysis chart: ${item.title}`,
          };
        })
        .filter(Boolean);

      const { data } = await engine.post('/reports/strategic-report', {
        session_id: quickSessionId,
        report_type: 'pdf',
        analysis_markdown: analysis.markdown || analysis.keyFindings.join('\n'),
        goals: {
          problem: purpose || 'Quick Excel Analysis',
          objective: `Quick analysis for ${role}`,
          target: role,
        },
        recommendations: analysis.recommendations,
        plotly_charts: charts,
      });

      const taskId = data?.data?.task_id;
      if (!taskId) return toast.error('Failed to create export task');

      await pollTask(taskId);
      const response = await engine.get(`/reports/download/${taskId}`, { responseType: 'blob' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([response.data]));
      a.download = 'quick_excel_analysis.pdf';
      a.click();
      URL.revokeObjectURL(a.href);
      toast.success('PDF exported');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to export PDF');
    } finally {
      setDownloadingReport(false);
    }
  };

  const selectedDatasetRows = datasets.filter((item) => selectedSheets.includes(item.key));

  return (
    <div className="animate-fade-in space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Quick Excel Analysis</h1>
          <p className="text-[var(--text-secondary)] mt-1">Analyze one file instantly without running the full pipeline workflow</p>
        </div>
        {quickSessionId && <Badge variant="primary">Quick Session: {quickSessionId.slice(0, 8)}…</Badge>}
      </div>

      <Card>
        <h3 className="font-semibold mb-3 flex items-center gap-2"><Upload className="w-4 h-4 text-[var(--accent)]" /> Upload Excel File</h3>
        <div className="flex flex-wrap items-center gap-3">
          <label className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] text-white text-sm font-medium cursor-pointer hover:bg-[var(--accent-hover)] transition">
            <FileSpreadsheet className="w-4 h-4" />
            Choose One File
            <input
              type="file"
              accept=".xlsx,.xls,.csv"
              className="hidden"
              onChange={(e) => handleUpload(e.target.files?.[0] || null)}
            />
          </label>
          {uploading && <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]"><Spinner size="sm" /> Uploading…</div>}
          {fileName && !uploading && <span className="text-sm text-[var(--text-secondary)]">Loaded: <span className="font-medium text-[var(--text-primary)]">{fileName}</span></span>}
        </div>
      </Card>

      {datasets.length > 0 && (
        <Card>
          <h3 className="font-semibold mb-4">Context & Purpose (Required)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="What is the purpose of this file?"
              value={purpose}
              onChange={setPurpose}
              placeholder="e.g., Sales reports, accounting records, expense tracking"
            />
            <Select
              label="Your role"
              value={role}
              onChange={(v) => setRole(v as QuickRole)}
              options={[
                { value: 'Accountant', label: 'Accountant' },
                { value: 'Manager', label: 'Manager' },
                { value: 'Finance Officer', label: 'Finance Officer' },
                { value: 'Analyst', label: 'Analyst' },
                { value: 'Business Owner', label: 'Business Owner' },
                { value: 'Other', label: 'Other' },
              ]}
            />
          </div>

          <div className="mt-5">
            <p className="text-sm font-medium mb-2">Select sheets to analyze</p>
            <div className="flex flex-wrap gap-2">
              {datasets.map((dataset) => {
                const selected = selectedSheets.includes(dataset.key);
                return (
                  <button
                    key={dataset.key}
                    onClick={() => setSelectedSheets((prev) => selected ? prev.filter((k) => k !== dataset.key) : [...prev, dataset.key])}
                    className={`px-3 py-1.5 rounded-lg border text-sm transition cursor-pointer ${selected ? 'border-[var(--accent)] bg-[var(--accent-bg)] text-[var(--accent)]' : 'border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-secondary)]'}`}
                  >
                    {dataset.sheetName}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="mt-5">
            <p className="text-sm font-medium mb-2">Are there relationships between selected sheets?</p>
            <div className="flex gap-2">
              <Button variant={hasRelationships ? 'secondary' : 'primary'} size="sm" onClick={() => setHasRelationships(false)}>No</Button>
              <Button variant={hasRelationships ? 'primary' : 'secondary'} size="sm" onClick={() => setHasRelationships(true)}>Yes</Button>
            </div>
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            <Button
              variant="secondary"
              icon={<Sparkles className="w-4 h-4" />}
              onClick={generateSuggestions}
              loading={askingSuggestions}
            >
              Generate AI Questions
            </Button>
            <Button onClick={runAnalysis} loading={analyzing}>Analyze This File</Button>
          </div>
        </Card>
      )}

      {suggestedQuestions.length > 0 && (
        <Card>
          <h3 className="font-semibold mb-3">AI-Suggested Questions</h3>
          <div className="space-y-2">
            {suggestedQuestions.slice(0, 3).map((question, idx) => (
              <div key={`${question}-${idx}`} className="flex items-center justify-between gap-3 p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                <p className="text-sm">{question}</p>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    setCustomQuestions((prev) => prev.includes(question) ? prev : [...prev, question]);
                    toast.success('Added to custom questions');
                  }}
                >
                  Use
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      {datasets.length > 0 && (
        <Card>
          <h3 className="font-semibold mb-3">Your Custom Questions</h3>
          <div className="flex flex-col md:flex-row gap-2">
            <Input
              value={customQuestionInput}
              onChange={setCustomQuestionInput}
              placeholder="e.g., What is the average revenue per customer?"
            />
            <Button
              variant="secondary"
              onClick={() => {
                const q = customQuestionInput.trim();
                if (!q) return;
                setCustomQuestions((prev) => prev.includes(q) ? prev : [...prev, q]);
                setCustomQuestionInput('');
              }}
            >
              Add
            </Button>
            <Button variant="ghost" onClick={() => setCustomQuestions([])}>Clear All</Button>
          </div>

          {customQuestions.length > 0 && (
            <div className="mt-4 space-y-2">
              {customQuestions.map((q, idx) => (
                <div key={`${q}-${idx}`} className="flex items-center justify-between gap-3 p-2.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                  <p className="text-sm">{idx + 1}. {q}</p>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setCustomQuestions((prev) => prev.filter((item) => item !== q))}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {!datasets.length && (
        <EmptyState
          icon={<FileSpreadsheet className="w-8 h-8" />}
          title="Upload a file to begin"
          description="This page runs a fast single-file analysis outside the main pipeline steps."
        />
      )}

      {analysis && (
        <>
          <Card>
            <div className="flex items-center justify-between gap-3 mb-4">
              <h3 className="font-semibold">Data Summary</h3>
              <div className="flex gap-2">
                <Button variant="secondary" icon={<Download className="w-4 h-4" />} onClick={exportSummary}>Export Summary</Button>
                <Button variant="secondary" icon={<Download className="w-4 h-4" />} onClick={exportPdf} loading={downloadingReport}>Export PDF</Button>
              </div>
            </div>

            {selectedDatasetRows.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {selectedDatasetRows.map((item) => (
                  <div key={item.key} className="rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
                    <p className="text-xs uppercase tracking-wider text-[var(--text-muted)]">{item.sheetName}</p>
                    <p className="text-sm font-semibold mt-1">{item.rows.toLocaleString()} rows × {item.columns} cols</p>
                  </div>
                ))}
              </div>
            ) : <p className="text-sm text-[var(--text-secondary)]">No selected sheets.</p>}
          </Card>

          {analysis.keyFindings.length > 0 && (
            <Card>
              <h3 className="font-semibold mb-3">Key Findings</h3>
              <div className="space-y-2">
                {analysis.keyFindings.map((finding, idx) => (
                  <p key={`${finding}-${idx}`} className="text-sm"><span className="font-semibold">{idx + 1}.</span> {finding}</p>
                ))}
              </div>
            </Card>
          )}

          {Object.keys(analysis.quality || {}).length > 0 && (
            <Card>
              <h3 className="font-semibold mb-3">Data Quality Issues</h3>
              <div className="space-y-3">
                {Object.entries(analysis.quality).map(([dataset, details]) => (
                  <div key={dataset} className="rounded-lg border border-[var(--border)] p-3 bg-[var(--bg-secondary)]">
                    <p className="text-sm font-semibold mb-2">{normalizeSheetName(dataset)}</p>
                    <DataTable
                      columns={['Issue', 'Details']}
                      rows={Object.entries(details || {}).map(([issue, value]) => ({ Issue: issue, Details: typeof value === 'string' ? value : JSON.stringify(value) }))}
                      maxRows={10}
                    />
                  </div>
                ))}
              </div>
            </Card>
          )}

          {analysis.charts.length > 0 && (
            <Card>
              <h3 className="font-semibold mb-3">Key Visualizations</h3>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                {analysis.charts.map((chart, idx) => {
                  const parsed = parsePlotly(chart.plotly_json);
                  if (!parsed) return null;
                  return (
                    <div key={`${chart.title}-${idx}`} className="rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
                      <p className="text-sm font-medium mb-2">{chart.title}</p>
                      <Plot
                        data={parsed.data || []}
                        layout={{
                          ...(parsed.layout || {}),
                          autosize: true,
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                          font: { color: '#334155' },
                        }}
                        config={{ displaylogo: false, responsive: true }}
                        style={{ width: '100%', height: 320 }}
                        useResizeHandler
                      />
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {analysis.recommendations.length > 0 && (
            <Card>
              <h3 className="font-semibold mb-3">Recommendations</h3>
              <div className="space-y-2">
                {analysis.recommendations.map((recommendation, idx) => (
                  <p key={`${recommendation}-${idx}`} className="text-sm"><span className="font-semibold">{idx + 1}.</span> {recommendation}</p>
                ))}
              </div>
            </Card>
          )}

          <Card>
            <h3 className="font-semibold mb-3 flex items-center gap-2"><MessageCircle className="w-4 h-4 text-[var(--accent)]" /> Quick Questions</h3>
            <div className="space-y-3">
              <Textarea
                value={qaInput}
                onChange={setQaInput}
                placeholder="Ask a question about this uploaded file"
                rows={3}
              />
              <div>
                <Button onClick={askQuickQuestion} loading={asking}>Ask</Button>
              </div>
              {qaHistory.length > 0 && (
                <div className="space-y-3">
                  {qaHistory.map((item, idx) => (
                    <div key={`${item.question}-${idx}`} className="rounded-lg border border-[var(--border)] p-3 bg-[var(--bg-secondary)]">
                      <p className="text-sm font-semibold">Q: {item.question}</p>
                      <p className="text-sm mt-2 whitespace-pre-wrap">A: {item.answer}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
