import { useEffect, useState } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { useAuthStore } from '@/stores/auth-store';
import { engine } from '@/lib/api';
import { loadPageState, savePageState } from '@/lib/pagePersistence';
import { Button, Card, Badge, Spinner, EmptyState } from '@/components/ui';
import { ShieldCheck, AlertTriangle, AlertCircle, Info, CheckCircle2, Lightbulb, Database } from 'lucide-react';
import toast from 'react-hot-toast';

interface Issue {
  column: string;
  type: string;
  severity: string;
  count?: number;
  unique_count?: number;
  percentage?: number;
  bounds?: { lower: number; upper: number };
  correlation?: number;
  col1?: string;
  col2?: string;
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

interface QualityResult {
  datasets: Record<string, DatasetQuality>;
  dataset_count: number;
}

export default function QualityPage() {
  const { sessionId, activeWorkspace } = useWorkspaceStore();
  const { user } = useAuthStore();
  const [result, setResult] = useState<QualityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<string>('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [cacheHydrated, setCacheHydrated] = useState(false);

  useEffect(() => {
    const cached = loadPageState<any>('quality-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    });
    if (cached) {
      setResult(cached.result || null);
      setActiveTab(cached.activeTab || '');
      setSeverityFilter(cached.severityFilter || 'all');
    }
    setCacheHydrated(true);
  }, [user?.id, activeWorkspace?.id, sessionId]);

  useEffect(() => {
    if (!cacheHydrated) return;
    savePageState('quality-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    }, {
      result,
      activeTab,
      severityFilter,
    });
  }, [cacheHydrated, user?.id, activeWorkspace?.id, sessionId, result, activeTab, severityFilter]);

  const analyze = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    try {
      const { data } = await engine.post('/analysis/data-quality', { session_id: sessionId });
      const qr = data.data as QualityResult;
      setResult(qr);
      // Default to first tab
      const keys = Object.keys(qr.datasets);
      if (keys.length > 0) setActiveTab(keys[0]);
      toast.success(`Quality analysis complete — ${qr.dataset_count} dataset(s)`);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || err.response?.data?.error?.message || 'Quality analysis failed');
    } finally { setLoading(false); }
  };

  const scoreBg = (s: number) => s >= 80 ? 'text-emerald-600' : s >= 60 ? 'text-amber-600' : 'text-red-600';
  const scoreRing = (s: number) => s >= 80 ? 'var(--success, #22c55e)' : s >= 60 ? 'var(--warning, #eab308)' : 'var(--error, #ef4444)';
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

  const dsKeys = result ? Object.keys(result.datasets) : [];
  const activeData: DatasetQuality | null = result && activeTab ? result.datasets[activeTab] : null;

  const filteredIssues = activeData
    ? severityFilter === 'all'
      ? activeData.all_issues
      : activeData.by_severity[severityFilter as keyof typeof activeData.by_severity] || []
    : [];

  const sevCounts = activeData?.by_severity
    ? { critical: activeData.by_severity.critical?.length || 0, high: activeData.by_severity.high?.length || 0, medium: activeData.by_severity.medium?.length || 0, low: activeData.by_severity.low?.length || 0 }
    : { critical: 0, high: 0, medium: 0, low: 0 };

  const issueCount = (issue: Issue) => issue.count ?? issue.unique_count ?? 0;
  const issuePct = (issue: Issue) => issue.percentage != null ? `${issue.percentage.toFixed(1)}%` : '';

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Data Quality</h1>
          <p className="text-[var(--text-secondary)] mt-1">Assess data quality for each uploaded dataset</p>
        </div>
        <Button onClick={analyze} loading={loading} icon={<ShieldCheck className="w-4 h-4" />}>Run Quality Check</Button>
      </div>

      {loading && <div className="flex items-center justify-center py-20"><Spinner size="lg" /></div>}

      {!loading && !result && (
        <EmptyState icon={<ShieldCheck className="w-8 h-8" />} title="No quality analysis yet" description="Run a quality check to assess the health of all your uploaded datasets">
          <Button onClick={analyze} className="mt-4">Analyze Quality</Button>
        </EmptyState>
      )}

      {result && (
        <div className="space-y-6">
          {/* Dataset Tabs */}
          {dsKeys.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              {dsKeys.map((key) => {
                const ds = result.datasets[key];
                return (
                  <button
                    key={key}
                    onClick={() => { setActiveTab(key); setSeverityFilter('all'); }}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition cursor-pointer ${
                      activeTab === key
                        ? 'bg-[var(--accent)] text-white'
                        : 'bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                    }`}
                  >
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

          {/* Single dataset header when only one */}
          {dsKeys.length === 1 && (
            <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
              <Database className="w-4 h-4" />
              <span>{dsKeys[0].replace('::', ' → ')}</span>
            </div>
          )}

          {activeData && (
            <>
              {/* Score + Severity Breakdown */}
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                <Card className="lg:col-span-2">
                  <div className="flex flex-col items-center py-4">
                    <div className="relative w-36 h-36 flex items-center justify-center">
                      <svg className="absolute inset-0 w-36 h-36" viewBox="0 0 120 120">
                        <circle cx="60" cy="60" r="52" fill="none" stroke="var(--border)" strokeWidth="8" />
                        <circle cx="60" cy="60" r="52" fill="none" stroke={scoreRing(activeData.quality_score)} strokeWidth="8"
                          strokeDasharray={`${(activeData.quality_score / 100) * 327} 327`}
                          strokeLinecap="round" transform="rotate(-90 60 60)" className="transition-all duration-1000" />
                      </svg>
                      <span className={`text-4xl font-bold ${scoreBg(activeData.quality_score)}`}>{Math.round(activeData.quality_score)}</span>
                    </div>
                    <p className="text-sm text-[var(--text-secondary)] mt-2">Quality Score</p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">{activeData.total_issues} total issues found</p>
                  </div>
                </Card>

                {(['critical', 'high', 'medium'] as const).map((sev) => (
                  <Card key={sev}
                    className={`cursor-pointer transition hover:border-[var(--accent)] ${severityFilter === sev ? 'border-[var(--accent)]' : ''}`}
                    onClick={() => setSeverityFilter(severityFilter === sev ? 'all' : sev)}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      {sevIcon(sev)}
                      <span className="text-sm font-medium capitalize">{sev}</span>
                    </div>
                    <p className={`text-3xl font-bold ${sev === 'critical' ? 'text-red-600' : sev === 'high' ? 'text-orange-600' : 'text-amber-600'}`}>
                      {sevCounts[sev]}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">issues</p>
                  </Card>
                ))}
              </div>

              {/* Summary */}
              {activeData.summary && (
                <Card>
                  <p className="text-sm text-[var(--text-secondary)]">{activeData.summary}</p>
                </Card>
              )}

              {/* Priority Actions */}
              {activeData.priority_actions?.length > 0 && (
                <Card>
                  <h3 className="font-semibold mb-4 flex items-center gap-2"><Lightbulb className="w-4 h-4 text-amber-600" /> Priority Actions</h3>
                  <div className="space-y-2">
                    {activeData.priority_actions.map((action, i) => (
                      <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-[var(--bg-secondary)]">
                        <span className="text-[var(--accent)] font-bold text-sm shrink-0">{i + 1}.</span>
                        <p className="text-sm">{action}</p>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* Issues */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-600" /> Issues ({filteredIssues.length})
                    {severityFilter !== 'all' && (
                      <Badge variant={sevBadge(severityFilter)}>{severityFilter}</Badge>
                    )}
                  </h3>
                  {severityFilter !== 'all' && (
                    <button onClick={() => setSeverityFilter('all')} className="text-xs text-[var(--accent)] hover:underline cursor-pointer">
                      Show all
                    </button>
                  )}
                </div>
                {filteredIssues.length === 0 ? (
                  <div className="flex items-center gap-2 text-emerald-600 text-sm"><CheckCircle2 className="w-4 h-4" /> No issues detected — data looks great!</div>
                ) : (
                  <div className="space-y-3">
                    {filteredIssues.map((issue, i) => (
                      <div key={i} className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                        <div className="flex items-start gap-3">
                          {sevIcon(issue.severity)}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                              <span className="font-medium text-sm">{issue.column || (issue.col1 && issue.col2 ? `${issue.col1} ↔ ${issue.col2}` : 'N/A')}</span>
                              <Badge variant={sevBadge(issue.severity)}>{issue.severity}</Badge>
                              <Badge variant="default">{issue.type.replace(/_/g, ' ')}</Badge>
                              {(issueCount(issue) > 0 || issuePct(issue)) && (
                                <span className="text-xs text-[var(--text-muted)]">
                                  {issueCount(issue) > 0 && `${issueCount(issue)} occurrences`}
                                  {issuePct(issue) && ` (${issuePct(issue)})`}
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-[var(--text-secondary)]">{issue.explanation}</p>
                            {issue.recommendation && <p className="text-xs text-[var(--accent)] mt-1">→ {issue.recommendation}</p>}
                            {issue.bounds && (
                              <p className="text-xs text-[var(--text-muted)] mt-1">Bounds: [{issue.bounds.lower.toFixed(2)}, {issue.bounds.upper.toFixed(2)}]</p>
                            )}
                            {issue.correlation != null && (
                              <p className="text-xs text-[var(--text-muted)] mt-1">Correlation: {issue.correlation.toFixed(3)}</p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </>
          )}
        </div>
      )}
    </div>
  );
}
