import { useState, useEffect, useCallback } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { useAuthStore } from '@/stores/auth-store';
import { engine } from '@/lib/api';
import { loadPageState, savePageState } from '@/lib/pagePersistence';
import { Button, Card, Badge, Spinner, EmptyState, Input, Select, KpiCard } from '@/components/ui';
import { TrendingUp, Plus, Wand2, CheckCircle2, AlertTriangle, Database, BarChart3, Key, Hash, FolderOpen, ChevronDown, ChevronUp, Pin } from 'lucide-react';
import DatasetSelector from '@/components/DatasetSelector';
import toast from 'react-hot-toast';

interface KPI {
  name: string;
  value: number | string;
  column?: string;
  aggregation?: string;
  function?: string;
  formatted_value?: string;
  business_definition?: string;
  column_role?: string;
  is_valid?: boolean;
  warning?: string;
  dataset?: string;
}

interface DatasetInfo {
  shape: number[];
  columns: string[];
}

interface DatasetKPIs {
  kpis: KPI[];
  column_summary: { keys: string[]; measures: string[]; categories: string[] };
  shape: number[];
  columns: string[];
}

interface AllDatasetsResult {
  datasets: Record<string, DatasetKPIs>;
  master_kpis: KPI[];
  measure_kpis: KPI[];
  count_kpis: KPI[];
  column_roles: { keys: string[]; measures: string[]; categories: string[] };
}

export default function KpisPage() {
  const { sessionId, activeWorkspace, addPinnedKpi, addPinnedCard } = useWorkspaceStore();
  const { user } = useAuthStore();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AllDatasetsResult | null>(null);
  const [datasetsInfo, setDatasetsInfo] = useState<Record<string, DatasetInfo>>({});
  const [datasetKeys, setDatasetKeys] = useState<string[]>([]);
  const [activeDatasetTab, setActiveDatasetTab] = useState('__all__');
  const [rolesExpanded, setRolesExpanded] = useState(false);

  // Custom KPI
  const [customName, setCustomName] = useState('');
  const [customColumn, setCustomColumn] = useState('');
  const [customAgg, setCustomAgg] = useState('sum');
  const [customDataset, setCustomDataset] = useState('');
  const [customBuiltKpis, setCustomBuiltKpis] = useState<KPI[]>([]);

  // Validate
  const [valColumn, setValColumn] = useState('');
  const [valAgg, setValAgg] = useState('');
  const [valResult, setValResult] = useState<any>(null);
  const [cacheHydrated, setCacheHydrated] = useState(false);

  const loadSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      const { data } = await engine.get(`/files/session-info?session_id=${sessionId}`);
      const info = data.data;
      setDatasetKeys(info?.dataframe_keys || []);
      setDatasetsInfo(info?.datasets_info || {});
      if (info?.dataframe_keys?.length > 0 && !customDataset) {
        setCustomDataset(info.dataframe_keys[0]);
      }
    } catch { /* silent */ }
  }, [sessionId]);

  useEffect(() => { loadSession(); }, [loadSession]);

  useEffect(() => {
    const cached = loadPageState<any>('kpis-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    });
    if (cached) {
      setResult(cached.result || null);
      setActiveDatasetTab(cached.activeDatasetTab || '__all__');
      setRolesExpanded(Boolean(cached.rolesExpanded));
      setCustomBuiltKpis(cached.customBuiltKpis || []);
      setValResult(cached.valResult || null);
      setCustomDataset(cached.customDataset || '');
    }
    setCacheHydrated(true);
  }, [user?.id, activeWorkspace?.id, sessionId]);

  useEffect(() => {
    if (!cacheHydrated) return;
    savePageState('kpis-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    }, {
      result,
      activeDatasetTab,
      rolesExpanded,
      customBuiltKpis,
      valResult,
      customDataset,
    });
  }, [cacheHydrated, user?.id, activeWorkspace?.id, sessionId, result, activeDatasetTab, rolesExpanded, customBuiltKpis, valResult, customDataset]);

  const analyzeAll = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    try {
      const { data } = await engine.post('/kpis/generate-all', { session_id: sessionId, max_kpis: 10 });
      const res = data.data as AllDatasetsResult;
      setResult(res);
      const dsKeys = Object.keys(res.datasets || {});
      if (dsKeys.length > 0 && !activeDatasetTab) setActiveDatasetTab(dsKeys[0]);
      const totalKpis = (res.master_kpis || []).length;
      toast.success(`${totalKpis} master KPIs generated across ${dsKeys.length} datasets`);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to generate KPIs');
    } finally { setLoading(false); }
  };

  const addCustom = async () => {
    if (!sessionId) return toast.error('Upload data first');
    if (!customColumn) return toast.error('Select a column');
    setLoading(true);
    try {
      const { data } = await engine.post('/kpis/single', {
        session_id: sessionId,
        column: customColumn,
        aggregation: customAgg || undefined,
        custom_name: customName || undefined,
        key: customDataset || undefined,
      });
      const createdKpi: KPI = { ...(data.data || {}), dataset: customDataset || undefined };
      // Add to result's datasets
      if (result && customDataset && result.datasets[customDataset]) {
        const updated = { ...result };
        updated.datasets[customDataset] = {
          ...updated.datasets[customDataset],
          kpis: [...updated.datasets[customDataset].kpis, createdKpi],
        };
        updated.master_kpis = [...updated.master_kpis, createdKpi];
        setResult(updated);
      }
      setCustomBuiltKpis((prev) => [createdKpi, ...prev]);
      setCustomName('');
      setCustomColumn('');
      toast.success('Custom KPI added');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to add KPI');
    } finally { setLoading(false); }
  };

  const validate = async () => {
    if (!sessionId) return toast.error('Upload data first');
    if (!valColumn || !valAgg) return toast.error('Enter column and aggregation');
    setLoading(true);
    try {
      const { data } = await engine.post('/kpis/validate', {
        session_id: sessionId,
        column: valColumn,
        aggregation: valAgg,
      });
      setValResult(data.data);
      if (data.data?.is_valid) toast.success('KPI is valid');
      else toast.error(data.data?.message || 'KPI not valid');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Validation failed');
    } finally { setLoading(false); }
  };

  // Compute totals
  const totalRows = Object.values(datasetsInfo).reduce((s, d) => s + (d.shape?.[0] || 0), 0);
  const totalCols = Object.values(datasetsInfo).reduce((s, d) => s + (d.shape?.[1] || 0), 0);

  // Columns for custom builder dataset
  const customColumns = customDataset && datasetsInfo[customDataset]
    ? datasetsInfo[customDataset].columns
    : Object.values(datasetsInfo).flatMap(d => d.columns).filter((v, i, a) => a.indexOf(v) === i);

  // Columns for validator (use all columns)
  const allColumns = Object.values(datasetsInfo).flatMap(d => d.columns).filter((v, i, a) => a.indexOf(v) === i);

  const shortName = (key: string) => key.length > 28 ? key.slice(0, 28) + '...' : key;

  const pinKpi = (kpi: KPI, source: string) => {
    const value = kpi.formatted_value || (typeof kpi.value === 'number' ? kpi.value.toLocaleString() : String(kpi.value ?? '—'));
    addPinnedKpi({
      id: `${sessionId || 'no-session'}::kpi::${source}::${kpi.dataset || kpi.column || 'global'}::${kpi.name}`,
      sessionId,
      label: kpi.name,
      value,
      suffix: kpi.function || kpi.aggregation,
      source,
    });
    toast.success(`Pinned KPI ${kpi.name}`);
  };

  const pinOverviewCard = (label: string, value: string) => {
    addPinnedCard({
      id: `${sessionId || 'no-session'}::kpi-overview::${label}`,
      sessionId,
      label,
      value,
      source: 'KPI Intelligence',
    });
    toast.success(`Pinned ${label}`);
  };

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">KPI Intelligence</h1>
          <p className="text-[var(--text-secondary)] mt-1">Comprehensive KPIs across all datasets — measures vs counts, per-dataset breakdown</p>
        </div>
        <Button onClick={analyzeAll} loading={loading} icon={<Wand2 className="w-4 h-4" />}>Analyze All Data</Button>
      </div>

      {/* Data Overview */}
      {datasetKeys.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div>
            <KpiCard label="Total Datasets" value={String(datasetKeys.length)} icon={<Database className="w-5 h-5" />} />
            <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinOverviewCard('Total Datasets', String(datasetKeys.length))}>Pin to Dashboard</Button>
          </div>
          <div>
            <KpiCard label="Total Rows" value={totalRows.toLocaleString()} icon={<BarChart3 className="w-5 h-5" />} />
            <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinOverviewCard('Total Rows', totalRows.toLocaleString())}>Pin to Dashboard</Button>
          </div>
          <div>
            <KpiCard label="Total Columns" value={String(totalCols)} icon={<Hash className="w-5 h-5" />} />
            <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinOverviewCard('Total Columns', String(totalCols))}>Pin to Dashboard</Button>
          </div>
          <div>
            <KpiCard label="Files Loaded" value={String(datasetKeys.length)} icon={<FolderOpen className="w-5 h-5" />} />
            <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinOverviewCard('Files Loaded', String(datasetKeys.length))}>Pin to Dashboard</Button>
          </div>
        </div>
      )}

      {/* Loading / Empty */}
      {loading && !result && <div className="flex items-center justify-center py-20"><Spinner size="lg" /></div>}

      {!loading && !result && (
        <EmptyState icon={<TrendingUp className="w-8 h-8" />} title="No KPIs yet" description="Analyze all datasets to generate master KPIs with intelligent measure/count detection">
          <Button onClick={analyzeAll} className="mt-4" icon={<Wand2 className="w-4 h-4" />}>Analyze All Data</Button>
        </EmptyState>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-8">
          {/* Column Role Detection */}
          {result.column_roles && (
            <Card>
              <button
                onClick={() => setRolesExpanded(!rolesExpanded)}
                className="flex items-center justify-between w-full text-left cursor-pointer"
              >
                <h3 className="font-semibold flex items-center gap-2">
                  <Key className="w-4 h-4 text-[var(--accent)]" /> Column Role Detection
                </h3>
                {rolesExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              {rolesExpanded && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
                  <div>
                    <p className="text-sm font-medium text-[var(--text-secondary)] mb-2">Key/ID Columns (COUNT only)</p>
                    <div className="flex flex-wrap gap-1.5">
                      {(result.column_roles.keys || []).slice(0, 12).map(c => (
                        <Badge key={c} variant="warning">{c}</Badge>
                      ))}
                      {(result.column_roles.keys || []).length === 0 && <span className="text-xs text-[var(--text-muted)]">None detected</span>}
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[var(--text-secondary)] mb-2">Measure Columns (SUM/AVG)</p>
                    <div className="flex flex-wrap gap-1.5">
                      {(result.column_roles.measures || []).slice(0, 12).map(c => (
                        <Badge key={c} variant="success">{c}</Badge>
                      ))}
                      {(result.column_roles.measures || []).length === 0 && <span className="text-xs text-[var(--text-muted)]">None detected</span>}
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[var(--text-secondary)] mb-2">Category Columns (DISTINCT)</p>
                    <div className="flex flex-wrap gap-1.5">
                      {(result.column_roles.categories || []).slice(0, 12).map(c => (
                        <Badge key={c} variant="info">{c}</Badge>
                      ))}
                      {(result.column_roles.categories || []).length === 0 && <span className="text-xs text-[var(--text-muted)]">None detected</span>}
                    </div>
                  </div>
                </div>
              )}
            </Card>
          )}

          {/* ── Master KPIs: Business Metrics (Measures) ── */}
          {(result.measure_kpis || []).length > 0 && (
            <div>
              <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-emerald-600" /> Business Metrics (Measures)
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {result.measure_kpis.slice(0, 12).map((kpi, i) => (
                  <div key={`m-${i}`}>
                    <KpiCard
                      label={kpi.name}
                      value={kpi.formatted_value || (typeof kpi.value === 'number' ? kpi.value.toLocaleString() : String(kpi.value ?? '—'))}
                      suffix={kpi.function || kpi.aggregation}
                    />
                    <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinKpi(kpi, 'KPI Intelligence: Measures')}>Pin to Dashboard</Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Master KPIs: Entity Counts (Keys & Categories) ── */}
          {(result.count_kpis || []).length > 0 && (
            <div>
              <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Hash className="w-5 h-5 text-indigo-600" /> Entity Counts (Keys & Categories)
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {result.count_kpis.slice(0, 8).map((kpi, i) => (
                  <div key={`c-${i}`}>
                    <KpiCard
                      label={kpi.name}
                      value={kpi.formatted_value || (typeof kpi.value === 'number' ? kpi.value.toLocaleString() : String(kpi.value ?? '—'))}
                      suffix={kpi.function || kpi.aggregation}
                    />
                    <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinKpi(kpi, 'KPI Intelligence: Counts')}>Pin to Dashboard</Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Per-Dataset KPIs ── */}
          {Object.keys(result.datasets || {}).length > 0 && (
            <div>
              <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                <FolderOpen className="w-5 h-5 text-blue-600" /> KPIs by Dataset
              </h2>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setActiveDatasetTab('__all__')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition cursor-pointer ${
                    activeDatasetTab === '__all__'
                      ? 'bg-[var(--accent)] text-white shadow-sm'
                      : 'bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-[var(--border)]'
                  }`}
                >
                  <FolderOpen className="w-3.5 h-3.5" /> All Datasets
                </button>
                <DatasetSelector
                  datasets={Object.keys(result.datasets)}
                  activeDataset={activeDatasetTab === '__all__' ? '' : activeDatasetTab}
                  onSelect={setActiveDatasetTab}
                  className="flex-1"
                />
              </div>

              {/* All Datasets view */}
              {activeDatasetTab === '__all__' && (
                <div className="mt-4 space-y-6">
                  {Object.entries(result.datasets).map(([dsKey, ds]) => (
                    <Card key={dsKey}>
                      <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                        <Database className="w-4 h-4 text-[var(--accent)]" /> {shortName(dsKey)}
                        <span className="text-xs text-[var(--text-muted)] ml-auto">{(ds.shape?.[0] || 0).toLocaleString()} rows × {ds.shape?.[1] || 0} cols</span>
                      </h4>
                      {(ds.kpis || []).length > 0 ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                          {ds.kpis.map((kpi, i) => (
                            <div key={`all-${dsKey}-${i}`}>
                              <KpiCard
                                label={kpi.name}
                                value={kpi.formatted_value || (typeof kpi.value === 'number' ? kpi.value.toLocaleString() : String(kpi.value ?? '—'))}
                                suffix={kpi.function || kpi.aggregation}
                              />
                              <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinKpi({ ...kpi, dataset: dsKey }, `KPI Dataset: ${dsKey}`)}>Pin to Dashboard</Button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-[var(--text-muted)]">No KPIs for this dataset.</p>
                      )}
                    </Card>
                  ))}
                </div>
              )}

              {/* Single dataset view */}
              {activeDatasetTab !== '__all__' && activeDatasetTab && result.datasets[activeDatasetTab] && (() => {
                const ds = result.datasets[activeDatasetTab];
                return (
                  <div className="mt-4 space-y-4">
                    {/* Dataset quick stats */}
                    <div className="flex flex-wrap gap-4 text-sm text-[var(--text-secondary)]">
                      <span><strong>Rows:</strong> {(ds.shape?.[0] || 0).toLocaleString()}</span>
                      <span><strong>Columns:</strong> {ds.shape?.[1] || 0}</span>
                      <span><strong>Keys:</strong> {ds.column_summary?.keys?.length || 0}</span>
                      <span><strong>Measures:</strong> {ds.column_summary?.measures?.length || 0}</span>
                    </div>

                    {/* Dataset KPIs */}
                    {(ds.kpis || []).length > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {ds.kpis.map((kpi, i) => (
                          <div key={`ds-${i}`}>
                            <KpiCard
                              label={kpi.name}
                              value={kpi.formatted_value || (typeof kpi.value === 'number' ? kpi.value.toLocaleString() : String(kpi.value ?? '—'))}
                              suffix={kpi.function || kpi.aggregation}
                            />
                            <Button size="sm" variant="ghost" className="mt-2" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinKpi({ ...kpi, dataset: activeDatasetTab }, `KPI Dataset: ${activeDatasetTab}`)}>Pin to Dashboard</Button>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-[var(--text-muted)] py-4">No KPIs generated for this dataset.</p>
                    )}

                    {/* KPI Details */}
                    {(ds.kpis || []).length > 0 && (
                      <Card>
                        <h3 className="font-semibold mb-3 text-sm">KPI Details</h3>
                        <div className="space-y-2">
                          {ds.kpis.map((kpi, i) => (
                            <div key={i} className="flex items-start gap-3 p-2.5 rounded-lg bg-[var(--bg-secondary)]">
                              <TrendingUp className="w-4 h-4 text-[var(--accent)] shrink-0 mt-0.5" />
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <p className="font-medium text-sm">{kpi.name}</p>
                                  {kpi.column && <Badge variant="default">{kpi.column}</Badge>}
                                  {(kpi.function || kpi.column_role) && <Badge variant="primary">{kpi.function || kpi.column_role}</Badge>}
                                  {kpi.is_valid === false && <Badge variant="warning">Needs review</Badge>}
                                </div>
                                {kpi.business_definition && <p className="text-xs text-[var(--text-secondary)] mt-1">{kpi.business_definition}</p>}
                                {kpi.warning && (
                                  <p className="text-xs text-amber-600 mt-1 flex items-center gap-1">
                                    <AlertTriangle className="w-3 h-3" /> {kpi.warning}
                                  </p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </Card>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* ── Custom KPI Builder ── */}
      <Card className="mt-8">
        <h3 className="font-semibold mb-4 flex items-center gap-2"><Plus className="w-4 h-4 text-emerald-600" /> Custom KPI Builder</h3>
        <p className="text-xs text-[var(--text-muted)] mb-4">Create custom KPIs with intelligent aggregation validation. Keys are protected from invalid SUM operations.</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <Select label="Dataset" value={customDataset} onChange={(v) => { setCustomDataset(v); setCustomColumn(''); }}
            placeholder="Select dataset"
            options={datasetKeys.map(k => ({ value: k, label: shortName(k) }))} />
          <Select label="Column *" value={customColumn} onChange={setCustomColumn}
            placeholder="Select column"
            options={customColumns.map(c => ({ value: c, label: c }))} />
          <Select label="Aggregation" value={customAgg} onChange={setCustomAgg} options={[
            { value: 'sum', label: 'Sum' },
            { value: 'mean', label: 'Mean' },
            { value: 'median', label: 'Median' },
            { value: 'count', label: 'Count' },
            { value: 'min', label: 'Min' },
            { value: 'max', label: 'Max' },
            { value: 'std', label: 'Std Dev' },
          ]} />
          <Input label="Custom Name (optional)" value={customName} onChange={setCustomName} placeholder="e.g. Total Revenue" />
        </div>
        <Button onClick={addCustom} loading={loading} icon={<Plus className="w-4 h-4" />}>Add KPI</Button>
      </Card>

      {customBuiltKpis.length > 0 && (
        <Card className="mt-4">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-[var(--accent)]" /> KPI Builded
          </h3>
          <p className="text-xs text-[var(--text-muted)] mb-4">Custom KPIs you created in this page. You can pin any KPI to Dashboard.</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {customBuiltKpis.map((kpi, i) => (
              <div key={`custom-built-${kpi.name}-${i}`}>
                <KpiCard
                  label={kpi.name}
                  value={kpi.formatted_value || (typeof kpi.value === 'number' ? kpi.value.toLocaleString() : String(kpi.value ?? '—'))}
                  suffix={kpi.function || kpi.aggregation}
                />
                <Button
                  size="sm"
                  variant="ghost"
                  className="mt-2"
                  icon={<Pin className="w-3.5 h-3.5" />}
                  onClick={() => pinKpi(kpi, 'KPI Builded')}
                >
                  Pin to Dashboard
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ── KPI Validator ── */}
      <Card className="mt-4">
        <h3 className="font-semibold mb-4 flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-indigo-600" /> Validate KPI</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <Select label="Column" value={valColumn} onChange={setValColumn}
            placeholder="Select column"
            options={allColumns.map(c => ({ value: c, label: c }))} />
          <Select label="Aggregation" value={valAgg} onChange={setValAgg}
            placeholder="Select aggregation"
            options={[
              { value: 'sum', label: 'Sum' },
              { value: 'mean', label: 'Mean' },
              { value: 'median', label: 'Median' },
              { value: 'count', label: 'Count' },
              { value: 'min', label: 'Min' },
              { value: 'max', label: 'Max' },
              { value: 'std', label: 'Std Dev' },
            ]} />
        </div>
        <Button onClick={validate} loading={loading} variant="secondary" icon={<CheckCircle2 className="w-4 h-4" />}>Validate</Button>
        {valResult && (
          <div className={`mt-3 p-3 rounded-lg text-sm ${valResult.is_valid ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-600'}`}>
            <p>{valResult.message}</p>
            {valResult.suggestion && <p className="mt-1 text-xs opacity-80">Suggestion: {valResult.suggestion}</p>}
          </div>
        )}
      </Card>
    </div>
  );
}
