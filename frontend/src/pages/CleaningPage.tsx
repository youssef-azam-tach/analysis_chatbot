import { useState, useEffect, useCallback } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { useAuthStore } from '@/stores/auth-store';
import { engine } from '@/lib/api';
import { loadPageState, savePageState } from '@/lib/pagePersistence';
import { Button, Card, Badge, Spinner, EmptyState, Input, Select } from '@/components/ui';
import {
  Eraser, AlertTriangle, AlertCircle, Info, CheckCircle2,
  Merge, PenLine, RotateCcw, Download, Wand2, Database, ShieldCheck,
  SkipForward, Layers, ArrowRight, ArrowLeft, Eye, Table2, ChevronRight
} from 'lucide-react';
import toast from 'react-hot-toast';

/* ── Types ───────────────────────────────────────────────── */
interface Issue {
  column: string;
  type: string;
  severity: string;
  count?: number;
  unique_count?: number;
  percentage?: number;
  bounds?: { lower: number; upper: number };
  col1?: string; col2?: string;
  recommendation: string;
  explanation: string;
}

interface DatasetQuality {
  quality_score: number;
  total_issues: number;
  by_severity: { critical: Issue[]; high: Issue[]; medium: Issue[]; low: Issue[] };
  all_issues: Issue[];
  summary: string;
  priority_actions: string[];
}

interface SessionInfo {
  session_id: string;
  dataframe_count: number;
  dataframe_keys: string[];
  datasets_info?: Record<string, { shape: number[]; columns: string[] }>;
  has_active_df: boolean;
  has_cleaned_df: boolean;
  has_pipeline_final: boolean;
  active_df_shape: number[] | null;
  column_names: string[];
  operations_count: number;
}

interface PreviewData {
  preview: Record<string, any>[];
  current_shape: number[];
  columns: string[];
  dtypes: Record<string, string>;
  original_shape?: number[];
  cleaned_shape?: number[];
  missing_summary?: Record<string, number>;
  fill_suggestions?: Record<string, {
    missing_count?: number;
    mean?: number | null;
    median?: number | null;
    mode?: any;
    ffill_sample?: any;
    bfill_sample?: any;
  }>;
  operations?: any[];
}

const STEPS = [
  { key: 'cleaning', label: 'Clean Data', icon: ShieldCheck, desc: 'Fix quality issues' },
  { key: 'merge', label: 'Merge', icon: Merge, desc: 'Join datasets' },
  { key: 'append', label: 'Append', icon: Layers, desc: 'Stack tables' },
  { key: 'columns', label: 'Columns', icon: PenLine, desc: 'Add / change columns' },
  { key: 'finalize', label: 'Finalize', icon: Download, desc: 'Lock & use' },
] as const;

/* ── Data Preview Component ───────────────────────────── */
function DataPreview({ data, loading: previewLoading }: { data: PreviewData | null; loading: boolean }) {
  if (previewLoading) return <div className="flex justify-center py-6"><Spinner /></div>;
  if (!data || !data.preview?.length) return null;

  const cols = data.columns || Object.keys(data.preview[0]);
  return (
    <Card className="mt-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-sm flex items-center gap-2">
          <Eye className="w-4 h-4 text-indigo-600" /> Data Preview
        </h4>
        <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
          {data.current_shape && (
            <span className="flex items-center gap-1">
              <Table2 className="w-3.5 h-3.5" />
              {data.current_shape[0].toLocaleString()} rows × {data.current_shape[1]} cols
            </span>
          )}
          {data.original_shape && data.current_shape && data.original_shape[0] !== data.current_shape[0] && (
            <span className="text-amber-600">(was {data.original_shape[0].toLocaleString()} rows)</span>
          )}
          {data.missing_summary && Object.keys(data.missing_summary).length > 0 && (
            <Badge variant="warning">{Object.keys(data.missing_summary).length} cols with missing</Badge>
          )}
        </div>
      </div>
      <div className="overflow-auto rounded-lg border border-[var(--border)]" style={{ maxHeight: 320 }}>
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-[var(--bg-tertiary)] sticky top-0">
              {cols.map((col) => (
                <th key={col} className="px-3 py-2 text-left font-semibold text-[var(--text-secondary)] whitespace-nowrap border-b border-[var(--border)]">
                  <div>{col}</div>
                  {data.dtypes?.[col] && (
                    <span className="text-[10px] font-normal text-[var(--text-muted)]">{data.dtypes[col]}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.preview.map((row, i) => (
              <tr key={i} className="border-b border-[var(--border)] hover:bg-[var(--bg-secondary)] transition">
                {cols.map((col) => {
                  const v = row[col];
                  const isEmpty = v === '' || v === null || v === undefined;
                  return (
                    <td key={col} className={`px-3 py-1.5 whitespace-nowrap ${isEmpty ? 'text-red-600/50 italic' : 'text-[var(--text-primary)]'}`}>
                      {isEmpty ? 'null' : String(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/* ── Main Page ───────────────────────────────────────────── */
export default function CleaningPage() {
  const { sessionId, activeWorkspace } = useWorkspaceStore();
  const { user } = useAuthStore();
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);

  // Session & datasets
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [activeDatasetKey, setActiveDatasetKey] = useState<string>('');

  // Preview
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Quality
  const [qualityResults, setQualityResults] = useState<Record<string, DatasetQuality>>({});
  const [qualityLoading, setQualityLoading] = useState(false);
  const [fixedIssues, setFixedIssues] = useState<Set<string>>(new Set());
  const [skippedIssues, setSkippedIssues] = useState<Set<string>>(new Set());
  const [opResult, setOpResult] = useState<any>(null);
  const [customFillByColumn, setCustomFillByColumn] = useState<Record<string, string>>({});

  // Merge
  const [mergeLeftKey, setMergeLeftKey] = useState('');
  const [mergeRightKey, setMergeRightKey] = useState('');
  const [mergeLeftOn, setMergeLeftOn] = useState('');
  const [mergeRightOn, setMergeRightOn] = useState('');
  const [mergeHow, setMergeHow] = useState('left');

  // Custom column
  const [colName, setColName] = useState('');
  const [colExpression, setColExpression] = useState('');

  // Append
  const [appendKeys, setAppendKeys] = useState<string[]>([]);

  // Change type
  const [typeColumn, setTypeColumn] = useState('');
  const [typeTarget, setTypeTarget] = useState('string');
  const [columnsDatasetKey, setColumnsDatasetKey] = useState('');

  // Inline dataset previews (for merge & append)
  const [inlinePreview, setInlinePreview] = useState<{ key: string; data: PreviewData } | null>(null);
  const [inlinePreviewLoading, setInlinePreviewLoading] = useState(false);

  // Merge/Append preview result (before committing)
  const [mergePreview, setMergePreview] = useState<any>(null);
  const [mergePreviewSignature, setMergePreviewSignature] = useState('');
  const [appendPreview, setAppendPreview] = useState<any>(null);
  const [appendPreviewSignature, setAppendPreviewSignature] = useState('');
  const [previewingOp, setPreviewingOp] = useState(false);
  const [nextStepKeys, setNextStepKeys] = useState<string[]>([]);
  const [finalizeKeys, setFinalizeKeys] = useState<string[]>([]);
  const [finalizedSummary, setFinalizedSummary] = useState<any>(null);
  const [cacheHydrated, setCacheHydrated] = useState(false);

  const dsKeys = sessionInfo?.dataframe_keys || [];

  useEffect(() => {
    const cached = loadPageState<any>('cleaning-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    });
    if (cached) {
      setStep(cached.step ?? 0);
      setActiveDatasetKey(cached.activeDatasetKey || '');
      setQualityResults(cached.qualityResults || {});
      setFixedIssues(new Set(cached.fixedIssues || []));
      setSkippedIssues(new Set(cached.skippedIssues || []));
      setOpResult(cached.opResult || null);
      setCustomFillByColumn(cached.customFillByColumn || {});
      setNextStepKeys(cached.nextStepKeys || []);
      setFinalizeKeys(cached.finalizeKeys || []);
      setFinalizedSummary(cached.finalizedSummary || null);
    }
    setCacheHydrated(true);
  }, [user?.id, activeWorkspace?.id, sessionId]);

  useEffect(() => {
    if (!cacheHydrated) return;
    savePageState('cleaning-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    }, {
      step,
      activeDatasetKey,
      qualityResults,
      fixedIssues: Array.from(fixedIssues),
      skippedIssues: Array.from(skippedIssues),
      opResult,
      customFillByColumn,
      nextStepKeys,
      finalizeKeys,
      finalizedSummary,
    });
  }, [cacheHydrated, user?.id, activeWorkspace?.id, sessionId, step, activeDatasetKey, qualityResults, fixedIssues, skippedIssues, opResult, customFillByColumn, nextStepKeys, finalizeKeys, finalizedSummary]);

  /* ── Load session info ─────────────────────────────────── */
  const loadSessionInfo = useCallback(async () => {
    if (!sessionId) return;
    try {
      const { data } = await engine.get(`/files/session-info?session_id=${sessionId}`);
      const info = data.data as SessionInfo;
      setSessionInfo(info);
      if (info.dataframe_keys.length > 0 && !activeDatasetKey) {
        setActiveDatasetKey(info.dataframe_keys[0]);
      }
    } catch { /* silent */ }
  }, [sessionId, activeDatasetKey]);

  /* ── Load preview (supports specific dataset key) ─────── */
  const loadPreview = useCallback(async (key?: string) => {
    if (!sessionId) return;
    setPreviewLoading(true);
    try {
      const params = new URLSearchParams({ session_id: sessionId });
      if (key) params.append('key', key);
      const { data } = await engine.get(`/cleaning/preview?${params}`);
      setPreviewData(data.data as PreviewData);
    } catch { /* silent */ }
    finally { setPreviewLoading(false); }
  }, [sessionId]);

  useEffect(() => { loadSessionInfo(); }, [loadSessionInfo]);
  useEffect(() => { if (activeDatasetKey) loadPreview(activeDatasetKey); else loadPreview(); }, [activeDatasetKey, step]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (step !== 3) return;
    if (columnsDatasetKey && dsKeys.includes(columnsDatasetKey)) return;
    if (activeDatasetKey && dsKeys.includes(activeDatasetKey)) {
      setColumnsDatasetKey(activeDatasetKey);
      return;
    }
    if (dsKeys.length > 0) setColumnsDatasetKey(dsKeys[0]);
  }, [step, activeDatasetKey, columnsDatasetKey, dsKeys]);

  useEffect(() => {
    setNextStepKeys((prev) => {
      if (prev.length === 0) return dsKeys;
      const filtered = prev.filter((k) => dsKeys.includes(k));
      return filtered.length > 0 ? filtered : dsKeys;
    });
  }, [dsKeys]);

  useEffect(() => {
    setFinalizeKeys((prev) => {
      if (prev.length === 0) return dsKeys;
      const filtered = prev.filter((k) => dsKeys.includes(k));
      return filtered.length > 0 ? filtered : dsKeys;
    });
  }, [dsKeys]);

  /* ── Load inline preview for a specific dataset key ───── */
  const loadInlinePreview = async (key: string) => {
    if (!sessionId) return;
    if (inlinePreview?.key === key) { setInlinePreview(null); return; } // toggle off
    setInlinePreviewLoading(true);
    try {
      const { data } = await engine.get(`/cleaning/preview?session_id=${sessionId}&key=${key}`);
      setInlinePreview({ key, data: data.data as PreviewData });
    } catch { toast.error('Failed to load preview'); }
    finally { setInlinePreviewLoading(false); }
  };

  /* ── Run quality assessment ────────────────────────────── */
  const runQuality = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setQualityLoading(true);
    try {
      const { data } = await engine.post('/analysis/data-quality', { session_id: sessionId });
      const qr = data.data as { datasets: Record<string, DatasetQuality>; dataset_count: number };
      setQualityResults(qr.datasets);
      setFixedIssues(new Set());
      setSkippedIssues(new Set());
      toast.success(`Quality assessed — ${qr.dataset_count} dataset(s)`);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || err.response?.data?.error?.message || 'Quality check failed');
    } finally { setQualityLoading(false); }
  };

  /* ── Generic API call ──────────────────────────────────── */
  const callEndpoint = async (endpoint: string, body: Record<string, any>, msg: string) => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    try {
      const { data } = await engine.post(endpoint, { session_id: sessionId, ...body });
      setOpResult(data.data);
      toast.success(msg);
      await loadSessionInfo();

      // After append/merge, auto-select the new result dataset
      if (endpoint === '/cleaning/append') {
        setActiveDatasetKey('appended');
        loadPreview('appended');
      } else if (endpoint === '/cleaning/merge') {
        setActiveDatasetKey('merged');
        loadPreview('merged');
      } else {
        loadPreview(activeDatasetKey || undefined);
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || err.response?.data?.error?.message || 'Operation failed');
    } finally { setLoading(false); }
  };

  /* ── Set active dataset on server ──────────────────────── */
  const switchActiveDataset = async (key: string) => {
    setActiveDatasetKey(key);
    if (!sessionId) return;
    try {
      await engine.post('/files/set-active', { session_id: sessionId, dataset_key: key });
    } catch { /* silent — preview still works via key param */ }
  };

  /* ── Fix actions ───────────────────────────────────────── */
  const fixMissing = (column: string, strategy: string, fillValue?: string) =>
    callEndpoint(
      '/cleaning/missing-values',
      {
        strategy,
        columns: [column],
        ...(fillValue !== undefined ? { fill_value: fillValue } : {}),
      },
      `Missing values in "${column}" handled with ${strategy}`,
    );
  const fixOutliers = (
    column: string,
    method: string,
    treatment: 'remove' | 'replace' | 'transform' = 'remove',
    replaceWith: 'median' | 'mean' | 'mode' = 'median',
    transformMethod: 'log' | 'sqrt' | 'box-cox' | 'yeo-johnson' = 'yeo-johnson',
  ) => {
    const body: Record<string, any> = { method, columns: [column], treatment };
    let msg = `Outliers in "${column}" handled (${method}, ${treatment})`;
    if (treatment === 'replace') {
      body.replace_with = replaceWith;
      msg = `Outliers in "${column}" replaced with ${replaceWith} (${method})`;
    }
    if (treatment === 'transform') {
      body.transform_method = transformMethod;
      msg = `Outlier treatment transformation applied on "${column}": ${transformMethod} (${method})`;
    }
    return callEndpoint('/cleaning/outliers', body, msg);
  };
  const fixDuplicates = () =>
    callEndpoint('/cleaning/duplicates', {}, 'Duplicate rows removed');
  const removeColumn = (column: string) =>
    callEndpoint('/cleaning/remove-column', { column }, `Column "${column}" removed`);
  const fixInconsistentValues = (
    column: string,
    replacementMode: 'nan' | 'custom' | 'impute' = 'nan',
    customValue?: string,
    imputeStrategy: 'mean' | 'median' | 'mode' = 'mean',
  ) =>
    callEndpoint(
      '/cleaning/inconsistent-values',
      {
        columns: [column],
        replacement_mode: replacementMode,
        custom_value: customValue,
        impute_strategy: imputeStrategy,
        cast_if_consistent: true,
        consistency_threshold: 0.8,
      },
      `Inconsistent values in "${column}" handled (${replacementMode})`,
    );
  const applyNextStepDatasetSelection = async () => {
    if (!sessionId) return toast.error('Upload data first');
    if (nextStepKeys.length === 0) return toast.error('Select at least one dataset');
    setLoading(true);
    try {
      await engine.post('/cleaning/select-datasets', { session_id: sessionId, keys: nextStepKeys });
      toast.success(`Continuing with ${nextStepKeys.length} dataset(s)`);
      await loadSessionInfo();
      const preferred = nextStepKeys.includes('merged') ? 'merged' : nextStepKeys.includes('appended') ? 'appended' : nextStepKeys[0];
      setActiveDatasetKey(preferred);
      loadPreview(preferred);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to apply dataset selection');
    } finally {
      setLoading(false);
    }
  };
  const finalizePipelineSelection = async () => {
    if (!sessionId) return toast.error('Upload data first');
    if (finalizeKeys.length === 0) return toast.error('Select at least one dataset');
    setLoading(true);
    try {
      const preferredActive = finalizeKeys.includes(activeDatasetKey) ? activeDatasetKey : finalizeKeys[0];
      const { data } = await engine.post('/pipeline/finalize', {
        session_id: sessionId,
        dataset_key: preferredActive,
        keys: finalizeKeys,
      });
      const result = data.data;
      setFinalizedSummary(result);
      setOpResult(result);
      toast.success(`Finalized with ${result.selected_dataset_count || finalizeKeys.length} dataset(s)`);
      await loadSessionInfo();
      setActiveDatasetKey(preferredActive);
      loadPreview(preferredActive);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Finalize failed');
    } finally {
      setLoading(false);
    }
  };
  const markFixed = (issueId: string) => setFixedIssues(prev => new Set(prev).add(issueId));
  const markSkipped = (issueId: string) => setSkippedIssues(prev => new Set(prev).add(issueId));

  /* ── Helpers ───────────────────────────────────────────── */
  const sevIcon = (sev: string) => {
    if (sev === 'critical') return <AlertCircle className="w-4 h-4 text-red-600" />;
    if (sev === 'high') return <AlertTriangle className="w-4 h-4 text-orange-600" />;
    if (sev === 'medium') return <AlertTriangle className="w-4 h-4 text-amber-600" />;
    return <Info className="w-4 h-4 text-indigo-600" />;
  };
  const sevBadge = (sev: string): 'danger' | 'warning' | 'primary' | 'default' => {
    if (sev === 'critical') return 'danger';
    if (sev === 'high') return 'warning';
    if (sev === 'medium') return 'primary';
    return 'default';
  };
  const scoreBg = (s: number) => s >= 80 ? 'text-emerald-600' : s >= 60 ? 'text-amber-600' : 'text-red-600';
  const formatSuggestion = (value: any) => {
    if (value === null || value === undefined || value === '') return 'N/A';
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) return 'N/A';
      if (Number.isInteger(value)) return value.toLocaleString();
      return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    return String(value);
  };
  const columnsDatasetColumns =
    (columnsDatasetKey && sessionInfo?.datasets_info?.[columnsDatasetKey]?.columns?.length)
      ? (sessionInfo?.datasets_info?.[columnsDatasetKey]?.columns || [])
      : (sessionInfo?.column_names || []);
  const activeQuality = activeDatasetKey && qualityResults[activeDatasetKey] ? qualityResults[activeDatasetKey] : null;

  const allIssues: (Issue & { _id: string; _idx: number })[] = [];
  if (activeQuality) {
    activeQuality.all_issues.forEach((issue, idx) => {
      const id = `${issue.type}_${issue.column || 'general'}_${idx}`;
      if (!fixedIssues.has(id) && !skippedIssues.has(id)) {
        allIssues.push({ ...issue, _id: id, _idx: idx });
      }
    });
  }

  const currentStep = STEPS[step];
  const currentMergeSignature = `${mergeLeftKey}||${mergeRightKey}||${mergeLeftOn}||${mergeRightOn}||${mergeHow}`;
  const currentAppendSignature = [...appendKeys].sort().join('||');
  const canPreviewMerge = Boolean(mergeLeftKey && mergeRightKey && mergeLeftOn && mergeRightOn);
  const canMerge = Boolean(mergePreview) && mergePreviewSignature === currentMergeSignature;
  const canPreviewAppend = appendKeys.length >= 2;
  const canAppend = Boolean(appendPreview) && appendPreviewSignature === currentAppendSignature;
  const goNext = () => { if (step < STEPS.length - 1) { setStep(step + 1); setOpResult(null); setInlinePreview(null); } };
  const goPrev = () => { if (step > 0) { setStep(step - 1); setOpResult(null); setInlinePreview(null); } };

  return (
    <div className="animate-fade-in">

      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Data Cleaning Pipeline</h1>
          <p className="text-[var(--text-secondary)] mt-1">Step-by-step wizard to clean, transform, and finalize your data</p>
        </div>
        <div className="flex items-center gap-2">
          {sessionInfo && <Badge variant="success">{sessionInfo.dataframe_count} dataset(s)</Badge>}
          {sessionInfo?.active_df_shape && (
            <Badge variant="default">{sessionInfo.active_df_shape[0].toLocaleString()} × {sessionInfo.active_df_shape[1]}</Badge>
          )}
        </div>
      </div>

      {!sessionId && (
        <EmptyState icon={<Eraser className="w-8 h-8" />} title="No data loaded" description="Upload your data first from the Upload page" />
      )}

      {sessionId && (
        <>
          {/* ── Stepper ── */}
          <div className="flex items-center gap-1 mb-8 px-2">
            {STEPS.map((s, i) => {
              const Icon = s.icon;
              const isActive = i === step;
              const isDone = i < step;
              return (
                <div key={s.key} className="flex items-center flex-1">
                  <button
                    onClick={() => { setStep(i); setOpResult(null); setInlinePreview(null); }}
                    className={`flex items-center gap-2 px-3 py-2.5 rounded-xl text-sm font-medium transition-all w-full cursor-pointer ${
                      isActive
                        ? 'bg-[var(--accent)] text-white shadow-lg shadow-[var(--accent)]/20'
                        : isDone
                          ? 'bg-emerald-50 text-emerald-600 border border-emerald-200'
                          : 'bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <div className="text-left min-w-0 hidden sm:block">
                      <div className="text-xs font-semibold truncate">{s.label}</div>
                      <div className={`text-[10px] truncate ${isActive ? 'text-white/70' : 'text-[var(--text-muted)]'}`}>{s.desc}</div>
                    </div>
                    {isDone && <CheckCircle2 className="w-3.5 h-3.5 ml-auto text-emerald-600 shrink-0" />}
                  </button>
                  {i < STEPS.length - 1 && (
                    <ChevronRight className={`w-4 h-4 mx-1 shrink-0 ${i < step ? 'text-emerald-600' : 'text-[var(--border)]'}`} />
                  )}
                </div>
              );
            })}
          </div>

          {/* ── Dataset Selector Bar ── */}
          {dsKeys.length > 0 && (
            <Card className="mb-4">
              <div className="flex items-center gap-3 flex-wrap">
                <span className="text-sm font-medium text-[var(--text-secondary)] shrink-0">
                  <Database className="w-4 h-4 inline mr-1" />Active Dataset:
                </span>
                {dsKeys.map((key) => (
                  <button key={key} onClick={() => { switchActiveDataset(key); }}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer ${
                      activeDatasetKey === key
                        ? 'bg-[var(--accent)] text-white shadow-sm'
                        : 'bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] border border-[var(--border)]'
                    }`}>
                    <Table2 className="w-3 h-3" />
                    <span className="max-w-[200px] truncate">{key.replace('::', ' → ')}</span>
                  </button>
                ))}
              </div>
            </Card>
          )}

          {/* ── Step Content ── */}
          <div className="space-y-4">

            {/* Data Preview for active dataset */}
            <DataPreview data={previewData} loading={previewLoading} />

            {/* ═══════ STEP 1: CLEAN ═══════ */}
            {currentStep.key === 'cleaning' && (
              <div className="space-y-4">
                <Card>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold flex items-center gap-2">
                        <ShieldCheck className="w-4 h-4 text-indigo-600" /> Assess & Fix Data Quality
                      </h3>
                      <p className="text-sm text-[var(--text-secondary)] mt-1">Run quality check to discover issues, then fix them one by one or in bulk</p>
                    </div>
                    <Button onClick={runQuality} loading={qualityLoading} icon={<ShieldCheck className="w-4 h-4" />}>
                      Run Quality Check
                    </Button>
                  </div>
                </Card>

                {qualityLoading && <div className="flex justify-center py-12"><Spinner size="lg" /></div>}

                {!qualityLoading && Object.keys(qualityResults).length > 0 && (
                  <>
                    {dsKeys.length > 1 && (
                      <div className="flex gap-2 flex-wrap">
                        {dsKeys.map((key) => {
                          const ds = qualityResults[key];
                          if (!ds) return null;
                          return (
                            <button key={key} onClick={() => { switchActiveDataset(key); setFixedIssues(new Set()); setSkippedIssues(new Set()); }}
                              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition cursor-pointer ${
                                activeDatasetKey === key ? 'bg-[var(--accent)] text-white' : 'bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                              }`}>
                              <Database className="w-3.5 h-3.5" />
                              <span className="max-w-[200px] truncate">{key.replace('::', ' → ')}</span>
                              <span className={`text-xs font-bold ${ds.quality_score >= 80 ? 'text-emerald-600' : ds.quality_score >= 60 ? 'text-amber-600' : 'text-red-600'}`}>
                                {Math.round(ds.quality_score)}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    )}

                    {activeQuality && (
                      <>
                        <Card>
                          <div className="flex items-center gap-6 flex-wrap">
                            <div className="flex items-center gap-2">
                              <div className={`text-2xl font-bold ${scoreBg(activeQuality.quality_score)}`}>
                                {Math.round(activeQuality.quality_score)}/100
                              </div>
                              <span className="text-sm text-[var(--text-muted)]">Quality</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-2xl font-bold">{activeQuality.total_issues}</span>
                              <span className="text-sm text-[var(--text-muted)]">Issues</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-2xl font-bold text-red-600">{activeQuality.by_severity.critical?.length || 0}</span>
                              <span className="text-sm text-[var(--text-muted)]">Critical</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-2xl font-bold text-orange-600">{activeQuality.by_severity.high?.length || 0}</span>
                              <span className="text-sm text-[var(--text-muted)]">High</span>
                            </div>
                            <div className="flex items-center gap-4 ml-auto text-sm">
                              <span className="text-emerald-600">✓ {fixedIssues.size}</span>
                              <span className="text-[var(--text-muted)]">⏭ {skippedIssues.size}</span>
                            </div>
                          </div>
                        </Card>

                        <Card>
                          <h3 className="font-semibold mb-4 flex items-center gap-2">
                            <Wand2 className="w-4 h-4 text-amber-600" /> Fix Issues ({allIssues.length} remaining)
                          </h3>

                          {allIssues.length === 0 && (
                            <div className="flex items-center gap-2 text-emerald-600 text-sm py-4">
                              <CheckCircle2 className="w-5 h-5" /> All issues handled! Dataset is clean.
                            </div>
                          )}

                          <div className="space-y-3">
                            {allIssues.map((issue) => (
                              <div key={issue._id} className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                                <div className="flex items-start gap-3">
                                  {sevIcon(issue.severity)}
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                                      <span className="font-medium text-sm">{issue.column || (issue.col1 && issue.col2 ? `${issue.col1} ↔ ${issue.col2}` : 'General')}</span>
                                      <Badge variant={sevBadge(issue.severity)}>{issue.severity}</Badge>
                                      <Badge variant="default">{issue.type.replace(/_/g, ' ')}</Badge>
                                    </div>

                                    {/* ── Affected values info ── */}
                                    <div className="flex items-center gap-3 flex-wrap my-2 text-xs">
                                      {issue.count != null && (
                                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200 font-medium">
                                          <AlertCircle className="w-3 h-3" />
                                          {issue.count.toLocaleString()} affected value{issue.count !== 1 ? 's' : ''}
                                        </span>
                                      )}
                                      {issue.percentage != null && (
                                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-medium">
                                          {issue.percentage.toFixed(1)}% of column
                                        </span>
                                      )}
                                      {issue.unique_count != null && (
                                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 font-medium">
                                          {issue.unique_count.toLocaleString()} unique values
                                        </span>
                                      )}
                                      {issue.bounds && (
                                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200 font-medium">
                                          Normal range: {issue.bounds.lower.toFixed(2)} – {issue.bounds.upper.toFixed(2)}
                                        </span>
                                      )}
                                    </div>

                                    <p className="text-sm text-[var(--text-secondary)] mb-2">{issue.explanation}</p>
                                    <p className="text-xs text-[var(--accent)] mb-3">→ {issue.recommendation}</p>
                                    <div className="flex items-center gap-2 flex-wrap">
                                      {issue.type === 'missing_values' && issue.column && (
                                        <>
                                          <Button
                                            size="sm"
                                            variant={typeof issue.percentage === 'number' && issue.percentage >= 50 ? 'danger' : 'secondary'}
                                            loading={loading}
                                            onClick={async () => { await removeColumn(issue.column); markFixed(issue._id); }}
                                          >
                                            Remove Column
                                          </Button>

                                          {['mean', 'median', 'mode', 'drop', 'ffill', 'bfill'].map((s) => {
                                            const suggestion = previewData?.fill_suggestions?.[issue.column];
                                            const suggestedValue =
                                              s === 'mean' ? suggestion?.mean
                                                : s === 'median' ? suggestion?.median
                                                  : s === 'mode' ? suggestion?.mode
                                                    : s === 'ffill' ? suggestion?.ffill_sample
                                                      : s === 'bfill' ? suggestion?.bfill_sample
                                                        : undefined;
                                            const label = s === 'drop'
                                              ? `🗑️ Drop (${issue.count ?? '?'})`
                                              : `📊 ${s}${suggestedValue !== undefined ? ` (${formatSuggestion(suggestedValue)})` : ''}`;

                                            return (
                                              <Button
                                                key={s}
                                                size="sm"
                                                variant={s === 'mean' ? 'primary' : 'secondary'}
                                                loading={loading}
                                                onClick={async () => { await fixMissing(issue.column, s); markFixed(issue._id); }}
                                              >
                                                {label}
                                              </Button>
                                            );
                                          })}

                                          <div className="flex items-center gap-2 w-full sm:w-auto">
                                            <Input
                                              value={customFillByColumn[issue.column] ?? ''}
                                              onChange={(v: string) => setCustomFillByColumn((prev) => ({ ...prev, [issue.column]: v }))}
                                              placeholder="Custom value"
                                              className="min-w-[180px]"
                                            />
                                            <Button
                                              size="sm"
                                              variant="secondary"
                                              loading={loading}
                                              onClick={async () => {
                                                const customValue = customFillByColumn[issue.column];
                                                if (customValue == null || customValue === '') {
                                                  toast.error('Enter custom value first');
                                                  return;
                                                }
                                                await fixMissing(issue.column, 'constant', customValue);
                                                markFixed(issue._id);
                                              }}
                                            >
                                              Apply custom
                                            </Button>
                                          </div>
                                        </>
                                      )}
                                      {['inconsistent_values', 'type_mismatch'].includes(issue.type) && issue.column && (
                                        <>
                                          <Button
                                            size="sm"
                                            variant="primary"
                                            loading={loading}
                                            onClick={async () => { await fixInconsistentValues(issue.column, 'nan'); markFixed(issue._id); }}
                                          >
                                            Replace invalid → NaN
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            loading={loading}
                                            onClick={async () => { await fixInconsistentValues(issue.column, 'impute', undefined, 'mean'); markFixed(issue._id); }}
                                          >
                                            Impute Mean
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            loading={loading}
                                            onClick={async () => { await fixInconsistentValues(issue.column, 'impute', undefined, 'mode'); markFixed(issue._id); }}
                                          >
                                            Impute Mode
                                          </Button>
                                          <div className="flex items-center gap-2 w-full sm:w-auto">
                                            <Input
                                              value={customFillByColumn[issue.column] ?? ''}
                                              onChange={(v: string) => setCustomFillByColumn((prev) => ({ ...prev, [issue.column]: v }))}
                                              placeholder="Custom replacement"
                                              className="min-w-[180px]"
                                            />
                                            <Button
                                              size="sm"
                                              variant="secondary"
                                              loading={loading}
                                              onClick={async () => {
                                                const customValue = customFillByColumn[issue.column];
                                                if (customValue == null || customValue === '') {
                                                  toast.error('Enter custom value first');
                                                  return;
                                                }
                                                await fixInconsistentValues(issue.column, 'custom', customValue, 'mean');
                                                markFixed(issue._id);
                                              }}
                                            >
                                              Apply custom
                                            </Button>
                                          </div>
                                        </>
                                      )}
                                      {issue.type === 'outliers' && (
                                        <>
                                          <Button size="sm" variant="primary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'remove'); markFixed(issue._id); }}>
                                            IQR Remove{issue.count != null ? ` (${issue.count})` : ''}
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'zscore', 'remove'); markFixed(issue._id); }}>
                                            Z-Score{issue.count != null ? ` (${issue.count})` : ''}
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'replace', 'median'); markFixed(issue._id); }}>
                                            Replace Median
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'replace', 'mean'); markFixed(issue._id); }}>
                                            Replace Mean
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'replace', 'mode'); markFixed(issue._id); }}>
                                            Replace Mode
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'transform', 'median', 'log'); markFixed(issue._id); }}>
                                            Transform Log
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'transform', 'median', 'sqrt'); markFixed(issue._id); }}>
                                            Transform Sqrt
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'transform', 'median', 'box-cox'); markFixed(issue._id); }}>
                                            Transform Box-Cox
                                          </Button>
                                          <Button size="sm" variant="secondary" loading={loading}
                                            onClick={async () => { await fixOutliers(issue.column, 'iqr', 'transform', 'median', 'yeo-johnson'); markFixed(issue._id); }}>
                                            Transform Yeo-Johnson
                                          </Button>
                                        </>
                                      )}
                                      {issue.type === 'duplicate_rows' && (
                                        <Button size="sm" variant="primary" loading={loading}
                                          onClick={async () => { await fixDuplicates(); markFixed(issue._id); }}>
                                          🗑️ Remove{issue.count != null ? ` ${issue.count.toLocaleString()}` : ''} Duplicate{issue.count !== 1 ? 's' : ''}
                                        </Button>
                                      )}
                                      {['high_cardinality', 'whitespace_inconsistency', 'case_inconsistency', 'high_correlation'].includes(issue.type) && (
                                        <span className="text-xs text-[var(--text-muted)] italic">Manual review recommended</span>
                                      )}
                                      <Button size="sm" variant="secondary" onClick={() => markSkipped(issue._id)}
                                        icon={<SkipForward className="w-3 h-3" />}>Skip</Button>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </Card>

                        <Card>
                          <h3 className="font-semibold mb-4 flex items-center gap-2">
                            <Wand2 className="w-4 h-4 text-violet-600" /> Quick Bulk Actions
                          </h3>
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            <Button variant="secondary" loading={loading}
                              onClick={() => callEndpoint('/cleaning/missing-values', { strategy: 'mean' }, 'All missing filled (mean)')}>Fill Missing (Mean)</Button>
                            <Button variant="secondary" loading={loading}
                              onClick={() => callEndpoint('/cleaning/outliers', { method: 'iqr' }, 'Outliers removed')}>Remove Outliers</Button>
                            <Button variant="secondary" loading={loading}
                              disabled={!activeQuality?.all_issues.some(i => i.type === 'duplicate_rows')}
                              onClick={() => callEndpoint('/cleaning/duplicates', {}, 'Duplicates removed')}>Remove Duplicates</Button>
                            <Button variant="secondary" loading={loading}
                              onClick={async () => { try { await engine.post('/cleaning/reset', null, { params: { session_id: sessionId } }); toast.success('Reset to original'); loadSessionInfo(); loadPreview(); } catch { toast.error('Reset failed'); } }}>
                              <RotateCcw className="w-4 h-4 mr-1" /> Reset All
                            </Button>
                          </div>
                        </Card>
                      </>
                    )}
                  </>
                )}

                {!qualityLoading && Object.keys(qualityResults).length === 0 && (
                  <EmptyState icon={<ShieldCheck className="w-8 h-8" />} title="Run quality check first"
                    description="Click 'Run Quality Check' to discover issues to fix">
                    <Button onClick={runQuality} className="mt-4" icon={<ShieldCheck className="w-4 h-4" />}>Run Quality Check</Button>
                  </EmptyState>
                )}
              </div>
            )}

            {/* ═══════ STEP 2: MERGE ═══════ */}
            {currentStep.key === 'merge' && (
              <div className="space-y-4">
                <Card>
                  <h3 className="font-semibold mb-4 flex items-center gap-2"><Merge className="w-4 h-4 text-indigo-600" /> Merge Datasets (VLOOKUP-style)</h3>
                  <p className="text-sm text-[var(--text-secondary)] mb-4">Combine two datasets by matching columns, like VLOOKUP in Excel.</p>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <Select label="Left Table" value={mergeLeftKey} onChange={(v) => { setMergeLeftKey(v); setMergeLeftOn(''); setMergePreview(null); setMergePreviewSignature(''); }}
                        placeholder="Select table"
                        options={dsKeys.map(k => ({ value: k, label: k.replace('::', ' → ') }))} />
                      {mergeLeftKey && (
                        <button onClick={() => loadInlinePreview(mergeLeftKey)}
                          className="mt-1.5 flex items-center gap-1 text-xs text-[var(--accent)] hover:underline cursor-pointer">
                          <Eye className="w-3 h-3" /> {inlinePreview?.key === mergeLeftKey ? 'Hide' : 'Preview'} left table
                        </button>
                      )}
                    </div>
                    <div>
                      <Select label="Right Table" value={mergeRightKey} onChange={(v) => { setMergeRightKey(v); setMergeRightOn(''); setMergePreview(null); setMergePreviewSignature(''); }}
                        placeholder="Select table"
                        options={dsKeys.filter(k => k !== mergeLeftKey).map(k => ({ value: k, label: k.replace('::', ' → ') }))} />
                      {mergeRightKey && (
                        <button onClick={() => loadInlinePreview(mergeRightKey)}
                          className="mt-1.5 flex items-center gap-1 text-xs text-[var(--accent)] hover:underline cursor-pointer">
                          <Eye className="w-3 h-3" /> {inlinePreview?.key === mergeRightKey ? 'Hide' : 'Preview'} right table
                        </button>
                      )}
                    </div>
                    <Select label="Left On (column)" value={mergeLeftOn} onChange={(v) => { setMergeLeftOn(v); setMergePreview(null); setMergePreviewSignature(''); }}
                      placeholder="Select column"
                      options={(mergeLeftKey && sessionInfo?.datasets_info?.[mergeLeftKey]?.columns || sessionInfo?.column_names || []).map((c: string) => ({ value: c, label: c }))} />
                    <Select label="Right On (column)" value={mergeRightOn} onChange={(v) => { setMergeRightOn(v); setMergePreview(null); setMergePreviewSignature(''); }}
                      placeholder="Select column"
                      options={(mergeRightKey && sessionInfo?.datasets_info?.[mergeRightKey]?.columns || sessionInfo?.column_names || []).map((c: string) => ({ value: c, label: c }))} />
                  </div>
                  <Select label="Join Type" value={mergeHow} onChange={(v) => { setMergeHow(v); setMergePreview(null); setMergePreviewSignature(''); }} options={[
                    { value: 'left', label: 'Left Join (keep all from left — like VLOOKUP)' },
                    { value: 'inner', label: 'Inner Join (only matching rows)' },
                    { value: 'outer', label: 'Outer Join (all rows from both)' },
                    { value: 'right', label: 'Right Join (keep all from right)' },
                  ]} />
                  <div className="flex items-center gap-3 mt-4">
                    <Button loading={previewingOp} variant="secondary" icon={<Eye className="w-4 h-4" />}
                      disabled={!canPreviewMerge}
                      onClick={async () => {
                        if (!canPreviewMerge) return toast.error('Fill all fields');
                        setPreviewingOp(true);
                        try {
                          const { data } = await engine.post('/cleaning/merge/preview', {
                            session_id: sessionId, left_key: mergeLeftKey, right_key: mergeRightKey,
                            left_on: mergeLeftOn, right_on: mergeRightOn, how: mergeHow,
                          });
                          setMergePreview(data.data);
                          setMergePreviewSignature(currentMergeSignature);
                          toast.success('Preview ready — review before confirming');
                        } catch (err: any) { toast.error(err.response?.data?.detail || 'Preview failed'); }
                        finally { setPreviewingOp(false); }
                      }}>
                      Preview Merge
                    </Button>
                    <Button loading={loading} icon={<Merge className="w-4 h-4" />}
                      disabled={!canMerge}
                      onClick={() => {
                        if (!canMerge) return;
                        callEndpoint('/cleaning/merge', { left_key: mergeLeftKey, right_key: mergeRightKey, left_on: mergeLeftOn, right_on: mergeRightOn, how: mergeHow }, 'Datasets merged!');
                        setMergePreview(null);
                        setMergePreviewSignature('');
                      }}>
                      Merge
                    </Button>
                  </div>
                  {!canMerge && canPreviewMerge && (
                    <p className="text-xs text-[var(--text-muted)] mt-2">Run Preview Merge first, then click Merge.</p>
                  )}
                  {dsKeys.length < 2 && (
                    <p className="text-sm text-amber-600 mt-3">⚠️ You need at least 2 datasets to merge.</p>
                  )}
                </Card>

                {/* Merge preview result */}
                {mergePreview && (
                  <Card>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-sm flex items-center gap-2">
                        <Eye className="w-4 h-4 text-indigo-600" /> Merge Preview — {mergePreview.merged_shape?.[0]?.toLocaleString()} rows × {mergePreview.merged_shape?.[1]} cols
                      </h4>
                      <button onClick={() => setMergePreview(null)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer">✕</button>
                    </div>
                    <DataPreview data={{ columns: mergePreview.columns || [], preview: mergePreview.preview || [], current_shape: mergePreview.merged_shape || [], dtypes: {} }} loading={false} />
                  </Card>
                )}

                {/* Post-merge: choose datasets for next steps */}
                {opResult?.merged_shape && (
                  <Card>
                    <h4 className="font-semibold text-sm flex items-center gap-2 mb-3">
                      <CheckCircle2 className="w-4 h-4 text-indigo-600" /> Merge Complete — Select Datasets for Next Steps
                    </h4>
                    <p className="text-sm text-[var(--text-secondary)] mb-3">
                      Keep only the datasets you need to avoid duplicates in the next stages.
                    </p>
                    <div className="space-y-2 mb-4">
                      {dsKeys.map((key) => (
                        <label key={key} className="flex items-center gap-3 p-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] cursor-pointer">
                          <input
                            type="checkbox"
                            checked={nextStepKeys.includes(key)}
                            onChange={(e) => {
                              if (e.target.checked) setNextStepKeys((prev) => [...prev, key]);
                              else setNextStepKeys((prev) => prev.filter((k) => k !== key));
                            }}
                            className="w-4 h-4 accent-[var(--accent)]"
                          />
                          <span className="text-sm">{key.replace('::', ' → ')}</span>
                        </label>
                      ))}
                    </div>
                    <Button loading={loading} onClick={applyNextStepDatasetSelection}>Continue with selected datasets</Button>
                  </Card>
                )}

                {/* Inline preview for merge tables */}
                {inlinePreviewLoading && (
                  <Card><div className="flex items-center justify-center gap-2 py-6"><Spinner size="md" /><span className="text-sm text-[var(--text-secondary)]">Loading preview…</span></div></Card>
                )}
                {!inlinePreviewLoading && inlinePreview && (
                  <Card>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-sm flex items-center gap-2">
                        <Eye className="w-4 h-4 text-indigo-600" /> Preview: {inlinePreview.key.replace('::', ' → ')}
                      </h4>
                      <button onClick={() => setInlinePreview(null)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer">✕</button>
                    </div>
                    <DataPreview data={inlinePreview.data} loading={false} />
                  </Card>
                )}
              </div>
            )}

            {/* ═══════ STEP 3: APPEND ═══════ */}
            {currentStep.key === 'append' && (
              <div className="space-y-4">
                <Card>
                  <h3 className="font-semibold mb-4 flex items-center gap-2"><Layers className="w-4 h-4 text-emerald-600" /> Append (Stack Tables)</h3>
                  <p className="text-sm text-[var(--text-secondary)] mb-4">Stack multiple tables vertically. Columns are matched by name.</p>
                  <div className="space-y-2 mb-4">
                    {dsKeys.map((key) => (
                      <div key={key} className="flex items-center gap-3 p-3 rounded-lg bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] transition">
                        <label className="flex items-center gap-3 flex-1 cursor-pointer">
                          <input type="checkbox" checked={appendKeys.includes(key)}
                            onChange={(e) => {
                              if (e.target.checked) setAppendKeys(p => [...p, key]);
                              else setAppendKeys(p => p.filter(k => k !== key));
                              setAppendPreview(null);
                              setAppendPreviewSignature('');
                            }}
                            className="w-4 h-4 accent-[var(--accent)]" />
                          <Database className="w-4 h-4 text-[var(--text-muted)]" />
                          <span className="text-sm">{key.replace('::', ' → ')}</span>
                        </label>
                        <button onClick={() => loadInlinePreview(key)}
                          className="flex items-center gap-1 text-xs text-[var(--accent)] hover:underline cursor-pointer px-2 py-1 rounded">
                          <Eye className="w-3 h-3" /> {inlinePreview?.key === key ? 'Hide' : 'Preview'}
                        </button>
                      </div>
                    ))}
                  </div>
                  <div className="flex items-center gap-3">
                    <Button loading={previewingOp} variant="secondary" icon={<Eye className="w-4 h-4" />}
                      disabled={!canPreviewAppend}
                      onClick={async () => {
                        if (!canPreviewAppend) return toast.error('Select at least 2 datasets');
                        setPreviewingOp(true);
                        try {
                          const { data } = await engine.post('/cleaning/append/preview', { session_id: sessionId, keys: appendKeys });
                          setAppendPreview(data.data);
                          setAppendPreviewSignature(currentAppendSignature);
                          toast.success('Preview ready — review before confirming');
                        } catch (err: any) {
                          if (err.response?.status === 404) {
                            toast.error('Append preview endpoint not found. Rebuild/restart AI Engine service.');
                          } else {
                            toast.error(err.response?.data?.detail || 'Preview failed');
                          }
                        }
                        finally { setPreviewingOp(false); }
                      }}>
                      Preview Append ({appendKeys.length})
                    </Button>
                    <Button loading={loading} icon={<Layers className="w-4 h-4" />} disabled={!canAppend}
                      onClick={() => {
                        if (!canAppend) return;
                        callEndpoint('/cleaning/append', { keys: appendKeys }, `Appended ${appendKeys.length} datasets!`);
                        setAppendPreview(null);
                        setAppendPreviewSignature('');
                      }}>
                      Append
                    </Button>
                  </div>
                  {!canAppend && canPreviewAppend && (
                    <p className="text-xs text-[var(--text-muted)] mt-2">Run Preview Append first, then click Append.</p>
                  )}
                  {dsKeys.length < 2 && <p className="text-sm text-amber-600 mt-3">⚠️ You need at least 2 datasets.</p>}
                </Card>

                {/* Append preview result */}
                {appendPreview && (
                  <Card>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-sm flex items-center gap-2">
                        <Eye className="w-4 h-4 text-emerald-600" /> Append Preview — {appendPreview.appended_shape?.[0]?.toLocaleString()} rows × {appendPreview.appended_shape?.[1]} cols
                      </h4>
                      <button onClick={() => setAppendPreview(null)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer">✕</button>
                    </div>
                    <DataPreview data={{ columns: appendPreview.columns || [], preview: appendPreview.preview || [], current_shape: appendPreview.appended_shape || [], dtypes: {} }} loading={false} />
                  </Card>
                )}

                {/* Post-append: choose which dataset to carry forward */}
                {opResult?.appended_shape && (
                  <Card>
                    <h4 className="font-semibold text-sm flex items-center gap-2 mb-3">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Append Complete — Select Datasets for Next Steps
                    </h4>
                    <p className="text-sm text-[var(--text-secondary)] mb-3">
                      Keep only the datasets you need (for example: appended + remaining source files) to avoid duplicates in next steps.
                    </p>
                    <div className="space-y-2 mb-4">
                      {dsKeys.map((key) => (
                        <label key={key} className="flex items-center gap-3 p-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] cursor-pointer">
                          <input
                            type="checkbox"
                            checked={nextStepKeys.includes(key)}
                            onChange={(e) => {
                              if (e.target.checked) setNextStepKeys((prev) => [...prev, key]);
                              else setNextStepKeys((prev) => prev.filter((k) => k !== key));
                            }}
                            className="w-4 h-4 accent-[var(--accent)]"
                          />
                          <Table2 className="w-3 h-3" />
                          <span className="text-sm">{key.replace('::', ' → ')}</span>
                          {key === 'appended' && <Badge variant="success">new</Badge>}
                        </label>
                      ))}
                    </div>
                    <Button loading={loading} onClick={applyNextStepDatasetSelection}>Continue with selected datasets</Button>
                  </Card>
                )}

                {/* Inline preview for append tables */}
                {inlinePreviewLoading && (
                  <Card><div className="flex items-center justify-center gap-2 py-6"><Spinner size="md" /><span className="text-sm text-[var(--text-secondary)]">Loading preview…</span></div></Card>
                )}
                {!inlinePreviewLoading && inlinePreview && (
                  <Card>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-sm flex items-center gap-2">
                        <Eye className="w-4 h-4 text-emerald-600" /> Preview: {inlinePreview.key.replace('::', ' → ')}
                      </h4>
                      <button onClick={() => setInlinePreview(null)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer">✕</button>
                    </div>
                    <DataPreview data={inlinePreview.data} loading={false} />
                  </Card>
                )}
              </div>
            )}

            {/* ═══════ STEP 4: COLUMNS ═══════ */}
            {currentStep.key === 'columns' && (
              <div className="space-y-4">
                <Card>
                  <h3 className="font-semibold mb-4 flex items-center gap-2"><Database className="w-4 h-4 text-indigo-600" /> Column Operations Dataset</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <Select
                      label="Dataset"
                      value={columnsDatasetKey}
                      onChange={async (v) => {
                        setColumnsDatasetKey(v);
                        setTypeColumn('');
                        try {
                          await switchActiveDataset(v);
                        } catch {
                          // silent
                        }
                      }}
                      placeholder="Select dataset"
                      options={dsKeys.map((k) => ({ value: k, label: k.replace('::', ' → ') }))}
                    />
                  </div>
                  <p className="text-xs text-[var(--text-muted)] mt-3">
                    For formulas between two datasets, use <strong>Merge</strong> first (on related keys), then create the custom column on the merged table.
                  </p>
                </Card>

                <Card>
                  <h3 className="font-semibold mb-4 flex items-center gap-2"><PenLine className="w-4 h-4 text-emerald-600" /> Create Calculated Column</h3>
                  <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                    <div className="xl:col-span-2">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <Input label="Column Name" value={colName} onChange={setColName} placeholder="e.g. revenue_per_unit" />
                        <Input label="Expression (simple formula)" value={colExpression} onChange={setColExpression} placeholder="e.g. [Price] * [Quantity]" />
                      </div>
                      <p className="text-xs text-[var(--text-muted)] mb-3">Use <code className="px-1 py-0.5 rounded bg-[var(--bg-tertiary)]">[ColumnName]</code> to reference columns. This supports simple expressions and helper functions, not full DAX context engine.</p>
                      {(columnsDatasetColumns.length ?? 0) > 0 && (
                        <div className="flex flex-wrap gap-1.5 mb-4">
                          {columnsDatasetColumns.map((col) => (
                            <button key={col} type="button" onClick={() => setColExpression(prev => prev ? `${prev} [${col}]` : `[${col}]`)}
                              className="px-2 py-1 text-xs rounded-md bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition cursor-pointer">
                              {col}
                            </button>
                          ))}
                        </div>
                      )}
                      <Button loading={loading} icon={<PenLine className="w-4 h-4" />}
                        onClick={() => {
                          if (!colName || !colExpression) return toast.error('Fill name and expression');
                          const pyExpr = colExpression.replace(/\[([^\]]+)\]/g, "df['$1']");
                          callEndpoint('/cleaning/custom-column', { column_name: colName, expression: pyExpr }, `Column "${colName}" created!`);
                        }}>
                        Create Column
                      </Button>
                    </div>

                    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
                      <h4 className="text-sm font-semibold mb-2">Quick Formula Help</h4>
                      <div className="space-y-1.5 text-xs text-[var(--text-secondary)]">
                        <p><strong>Math:</strong> <code>[Price] * [Qty]</code>, <code>[A] + 10</code>, <code>round([Amount], 2)</code></p>
                        <p><strong>Condition:</strong> <code>IF([Amount] &gt; 1000, 'High', 'Low')</code></p>
                        <p><strong>Date:</strong> <code>YEAR([OrderDate])</code>, <code>MONTH([OrderDate])</code>, <code>DAY([OrderDate])</code>, <code>WEEKDAY([OrderDate])</code></p>
                        <p><strong>Aggregation:</strong> <code>SUM([Sales])</code>, <code>AVG([Sales])</code>, <code>MEDIAN([Sales])</code>, <code>MIN([Sales])</code>, <code>MAX([Sales])</code>, <code>COUNT([Sales])</code>, <code>NUNIQUE([CustomerID])</code></p>
                        <p><strong>Null handling:</strong> <code>COALESCE([Discount], 0)</code></p>
                      </div>
                    </div>
                  </div>
                </Card>

                <Card>
                  <h3 className="font-semibold mb-4 flex items-center gap-2"><RotateCcw className="w-4 h-4 text-violet-600" /> Change Column Type</h3>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <Select label="Column Name" value={typeColumn} onChange={setTypeColumn}
                      placeholder="Select a column"
                      options={columnsDatasetColumns.map(c => ({ value: c, label: c }))} />
                    <Select label="Target Type" value={typeTarget} onChange={setTypeTarget} options={[
                      { value: 'string', label: 'String' },
                      { value: 'int64', label: 'Integer' },
                      { value: 'float64', label: 'Float' },
                      { value: 'datetime64[ns]', label: 'DateTime' },
                      { value: 'category', label: 'Category' },
                      { value: 'bool', label: 'Boolean' },
                    ]} />
                  </div>
                  <Button loading={loading} icon={<RotateCcw className="w-4 h-4" />}
                    onClick={() => {
                      if (!typeColumn) return toast.error('Select a column');
                      callEndpoint('/cleaning/change-type', { column: typeColumn, new_type: typeTarget }, `Column "${typeColumn}" converted to ${typeTarget}`);
                    }}>
                    Convert Type
                  </Button>
                </Card>

                <Card>
                  <h3 className="font-semibold mb-4 flex items-center gap-2"><Wand2 className="w-4 h-4 text-amber-600" /> AI Column Creator</h3>
                  <p className="text-sm text-[var(--text-secondary)] mb-4">Describe what you want in plain English and AI will create the column expression.</p>
                  <Input label="Description" value={colExpression} onChange={setColExpression}
                    placeholder="e.g. Multiply Price by Quantity to get Revenue" />
                  <div className="mt-4">
                    <Button loading={loading} icon={<Wand2 className="w-4 h-4" />}
                      onClick={() => {
                        if (!colExpression) return toast.error('Enter a description');
                        callEndpoint('/pipeline/ai-column', { description: colExpression }, 'AI column created!');
                      }}>
                      Generate with AI
                    </Button>
                  </div>
                </Card>
              </div>
            )}

            {/* ═══════ STEP 5: FINALIZE ═══════ */}
            {currentStep.key === 'finalize' && (
              <div className="space-y-4">
                <Card>
                  <h3 className="font-semibold mb-2 flex items-center gap-2"><Download className="w-4 h-4 text-emerald-600" /> Ready for Analysis</h3>
                  <p className="text-sm text-[var(--text-secondary)] mb-4">
                    Your datasets are ready! All <strong>{sessionInfo?.dataframe_count || 0} dataset(s)</strong> will be available as <strong>separate tables</strong> on KPI, Visualization, and Analysis pages — each with its own tab.
                  </p>

                  <div className="p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] mb-4">
                    <h4 className="font-medium text-sm mb-2">Select datasets for next stage</h4>
                    <p className="text-xs text-[var(--text-muted)] mb-3">Only selected datasets will continue after finalize.</p>
                    <div className="space-y-2 mb-3">
                      {dsKeys.map((key) => (
                        <label key={key} className="flex items-center gap-3 p-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border)] cursor-pointer">
                          <input
                            type="checkbox"
                            checked={finalizeKeys.includes(key)}
                            onChange={(e) => {
                              if (e.target.checked) setFinalizeKeys((prev) => [...prev, key]);
                              else setFinalizeKeys((prev) => prev.filter((k) => k !== key));
                            }}
                            className="w-4 h-4 accent-[var(--accent)]"
                          />
                          <span className="text-sm">{key.replace('::', ' → ')}</span>
                        </label>
                      ))}
                    </div>
                    <Badge variant="primary">Selected: {finalizeKeys.length} dataset(s)</Badge>
                  </div>

                  {sessionInfo && (
                    <div className="p-4 rounded-lg bg-[var(--bg-secondary)] mb-4">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                        <div><span className="text-[var(--text-muted)]">Datasets:</span> <span className="font-medium">{sessionInfo.dataframe_count}</span></div>
                        <div><span className="text-[var(--text-muted)]">Active Shape:</span> <span className="font-medium">{sessionInfo.active_df_shape?.join(' × ') || 'N/A'}</span></div>
                        <div><span className="text-[var(--text-muted)]">Cleaned:</span> <span className="font-medium">{sessionInfo.has_cleaned_df ? '✅ Yes' : '❌ No'}</span></div>
                        <div><span className="text-[var(--text-muted)]">Operations:</span> <span className="font-medium">{sessionInfo.operations_count || 0}</span></div>
                      </div>
                      {/* List all dataset keys */}
                      {(sessionInfo.dataframe_keys || []).length > 1 && (
                        <div className="mt-3 pt-3 border-t border-[var(--border)]">
                          <p className="text-xs text-[var(--text-muted)] mb-2">Datasets that will be available:</p>
                          <div className="flex flex-wrap gap-2">
                            {(sessionInfo.dataframe_keys || []).map((k: string) => (
                              <Badge key={k} variant="primary">{k.length > 30 ? k.slice(0, 30) + '...' : k}</Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Full data preview before finalizing */}
                  <DataPreview data={previewData} loading={previewLoading} />

                  <div className="flex items-center gap-3 mt-4">
                    <Button loading={loading} icon={<Download className="w-4 h-4" />}
                      disabled={finalizeKeys.length === 0}
                      onClick={finalizePipelineSelection}>
                      ✅ Mark Ready for Analysis
                    </Button>
                    <Button variant="secondary" loading={loading}
                      onClick={async () => { try { await engine.post('/cleaning/reset', null, { params: { session_id: sessionId } }); toast.success('Reset to original'); loadSessionInfo(); loadPreview(); } catch { toast.error('Reset failed'); } }}>
                      <RotateCcw className="w-4 h-4 mr-1" /> Reset to Original
                    </Button>
                  </div>
                </Card>

                {finalizedSummary && (
                  <Card>
                    <div className="flex items-center gap-3 text-emerald-600">
                      <CheckCircle2 className="w-6 h-6" />
                      <div>
                        <p className="font-semibold">Datasets are ready!</p>
                        <p className="text-sm text-[var(--text-secondary)]">{finalizedSummary.selected_dataset_count || finalizeKeys.length} dataset(s) will enter the next stage.</p>
                      </div>
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* ═══════ OPERATION RESULT ═══════ */}
            {opResult && (
              <Card className="mt-4">
                <h3 className="font-semibold flex items-center gap-2 mb-3">
                  <Badge variant="success">Operation Result</Badge>
                </h3>
                <pre className="text-sm text-[var(--text-secondary)] bg-[var(--bg-primary)] p-4 rounded-lg overflow-auto max-h-[200px] whitespace-pre-wrap">
                  {typeof opResult === 'string' ? opResult : JSON.stringify(opResult, null, 2)}
                </pre>
              </Card>
            )}

            {loading && (
              <div className="flex items-center justify-center py-8">
                <Spinner size="lg" />
              </div>
            )}
          </div>

          {/* ── Navigation Bar ── */}
          <div className="flex items-center justify-between mt-8 pt-6 border-t border-[var(--border)]">
            <Button variant="secondary" disabled={step === 0} onClick={goPrev}
              icon={<ArrowLeft className="w-4 h-4" />}>
              Previous
            </Button>

            <div className="text-sm text-[var(--text-muted)]">
              Step {step + 1} of {STEPS.length}
            </div>

            {step < STEPS.length - 1 ? (
              <Button onClick={goNext} icon={<ArrowRight className="w-4 h-4" />}>
                Next Step
              </Button>
            ) : (
              <Button disabled={finalizeKeys.length === 0} loading={loading}
                onClick={finalizePipelineSelection}
                icon={<CheckCircle2 className="w-4 h-4" />}>
                ✅ Mark Ready
              </Button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
