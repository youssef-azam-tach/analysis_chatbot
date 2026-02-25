import { useState, useEffect, useCallback } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { useAuthStore } from '@/stores/auth-store';
import { engine, parsePlotly } from '@/lib/api';
import { loadPageState, savePageState } from '@/lib/pagePersistence';
import { Button, Card, Badge, Spinner, EmptyState, Select, Tabs, Input } from '@/components/ui';
import { BarChart3, Wand2, Database, FolderOpen, Link2, Pin } from 'lucide-react';
import toast from 'react-hot-toast';
import Plot from 'react-plotly.js';

interface ChartSpec {
  title: string;
  chart_type?: string;
  data: any[];
  layout: any;
  reasoning?: string;
  dataset?: string;
}

interface DatasetInfo {
  shape: number[];
  columns: string[];
}

export default function VisualizationPage() {
  const { sessionId, activeWorkspace, addPinnedChart } = useWorkspaceStore();
  const { user } = useAuthStore();
  const [tab, setTab] = useState('ai');
  const [loading, setLoading] = useState(false);
  const [charts, setCharts] = useState<ChartSpec[]>([]);
  const [manualBuiltCharts, setManualBuiltCharts] = useState<ChartSpec[]>([]);
  const [crossBuiltCharts, setCrossBuiltCharts] = useState<ChartSpec[]>([]);

  // Dataset awareness
  const [datasetKeys, setDatasetKeys] = useState<string[]>([]);
  const [datasetsInfo, setDatasetsInfo] = useState<Record<string, DatasetInfo>>({});
  const [hasPipelineFinal, setHasPipelineFinal] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('__all__');

  // Manual builder
  const [chartType, setChartType] = useState('bar');
  const [manualChartName, setManualChartName] = useState('');
  const [manualAggregation, setManualAggregation] = useState('none');
  const [xCol, setXCol] = useState('');
  const [yCol, setYCol] = useState('');

  // Cross-dataset builder
  const [crossChartType, setCrossChartType] = useState('bar');
  const [crossChartName, setCrossChartName] = useState('');
  const [crossAggregation, setCrossAggregation] = useState('none');
  const [crossDs1, setCrossDs1] = useState('');
  const [crossDs2, setCrossDs2] = useState('');
  const [crossXCol, setCrossXCol] = useState('');
  const [crossYCol, setCrossYCol] = useState('');
  const [crossJoin1, setCrossJoin1] = useState('');
  const [crossJoin2, setCrossJoin2] = useState('');
  const [crossJoinType, setCrossJoinType] = useState('inner');
  const [cacheHydrated, setCacheHydrated] = useState(false);

  const loadSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      const { data } = await engine.get(`/files/session-info?session_id=${sessionId}`);
      const info = data.data;
      const keys: string[] = info?.dataframe_keys || [];
      setDatasetKeys(keys);
      setDatasetsInfo(info?.datasets_info || {});
      const pipelineFinal = Boolean(info?.has_pipeline_final);
      setHasPipelineFinal(pipelineFinal);
      if (keys.length > 0 && selectedDataset === '') {
        setSelectedDataset(pipelineFinal ? '__working__' : keys.length > 1 ? '__all__' : keys[0]);
      }
      if (keys.length >= 2) { if (!crossDs1) setCrossDs1(keys[0]); if (!crossDs2) setCrossDs2(keys[1]); }
    } catch { /* silent */ }
  }, [sessionId]);

  useEffect(() => { loadSession(); }, [loadSession]);

  useEffect(() => {
    const cached = loadPageState<any>('visualization-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    });
    if (cached) {
      setTab(cached.tab || 'ai');
      setCharts(cached.charts || []);
      setManualBuiltCharts(cached.manualBuiltCharts || []);
      setCrossBuiltCharts(cached.crossBuiltCharts || []);
      setSelectedDataset(cached.selectedDataset || '__all__');
    }
    setCacheHydrated(true);
  }, [user?.id, activeWorkspace?.id, sessionId]);

  useEffect(() => {
    if (!hasPipelineFinal) return;
    if (selectedDataset === '__all__' && datasetKeys.length > 1) {
      setSelectedDataset('__working__');
    }
  }, [hasPipelineFinal, selectedDataset, datasetKeys.length]);

  useEffect(() => {
    if (!cacheHydrated) return;
    savePageState('visualization-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    }, {
      tab,
      charts,
      manualBuiltCharts,
      crossBuiltCharts,
      selectedDataset,
    });
  }, [cacheHydrated, user?.id, activeWorkspace?.id, sessionId, tab, charts, manualBuiltCharts, crossBuiltCharts, selectedDataset]);

  // Reset builder column picks when dataset changes
  useEffect(() => { setXCol(''); setYCol(''); }, [selectedDataset]);

  const columnNames = selectedDataset && selectedDataset !== '__all__' && selectedDataset !== '__working__' && datasetsInfo[selectedDataset]
    ? datasetsInfo[selectedDataset].columns
    : Object.values(datasetsInfo).flatMap(d => d.columns).filter((v, i, a) => a.indexOf(v) === i);

  const aiRecommendSingle = async (key: string): Promise<ChartSpec[]> => {
    const { data } = await engine.post('/strategic/viz-recommend', {
      session_id: sessionId,
      max_charts: 6,
      key: key || undefined,
    });
    const resp = data.data;
    const parsed: ChartSpec[] = [];
    if (resp?.plotly_charts) {
      for (const pc of resp.plotly_charts) {
        const fig = parsePlotly(pc.plotly_json);
        if (fig) parsed.push({ title: pc.title || 'Chart', data: fig.data, layout: fig.layout, dataset: key });
      }
    }
    if (parsed.length === 0 && resp?.recommendations) {
      for (const rec of resp.recommendations) {
        parsed.push({
          title: `${rec.chart_type}: ${rec.x_column}${rec.y_column ? ' vs ' + rec.y_column : ''}`,
          chart_type: rec.chart_type, data: [], layout: {},
          reasoning: rec.reason, dataset: key,
        });
      }
    }
    return parsed;
  };

  const aiRecommend = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    try {
      const keysToProcess = selectedDataset === '__all__' ? datasetKeys : [selectedDataset];
      const allParsed: ChartSpec[] = [];
      for (const key of keysToProcess) {
        const parsed = await aiRecommendSingle(key === '__working__' ? '' : key);
        allParsed.push(...parsed);
      }
      setCharts(allParsed);
      toast.success(`${allParsed.length} charts recommended${selectedDataset === '__all__' ? ` across ${keysToProcess.length} datasets` : ''}`);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Recommendation failed');
    } finally { setLoading(false); }
  };

  const buildChartSingle = async (key: string): Promise<ChartSpec | null> => {
    const params: Record<string, string> = { x_column: xCol };
    if (yCol) params.y_column = yCol;
    if (manualAggregation !== 'none') params.aggregation = manualAggregation;
    if (manualChartName.trim()) params.chart_name = manualChartName.trim();
    const { data } = await engine.post('/visualization/generate', {
      session_id: sessionId,
      chart_type: chartType,
      params,
      key: key || undefined,
    });
    const resp = data.data;
    const fig = parsePlotly(resp?.plotly_json);
    if (fig) {
      return {
        title: manualChartName.trim() || `${chartType}: ${xCol}${yCol ? ' vs ' + yCol : ''}`,
        chart_type: chartType,
        data: fig.data,
        layout: fig.layout,
        dataset: key,
        reasoning: manualAggregation !== 'none' ? `Aggregation applied: ${manualAggregation.toUpperCase()}` : undefined,
      };
    }
    return null;
  };

  const buildChart = async () => {
    if (!sessionId) return toast.error('Upload data first');
    if (!xCol) return toast.error('Select at least X column');
    setLoading(true);
    try {
      const keysToProcess = selectedDataset === '__all__' ? datasetKeys : [selectedDataset];
      const newCharts: ChartSpec[] = [];
      for (const key of keysToProcess) {
        const chart = await buildChartSingle(key === '__working__' ? '' : key);
        if (chart) newCharts.push(chart);
      }
      if (newCharts.length > 0) {
        setCharts((prev) => [...prev, ...newCharts]);
        setManualBuiltCharts((prev) => [...newCharts, ...prev]);
        toast.success(`${newCharts.length} chart(s) created`);
      } else {
        toast.error('Could not parse chart data');
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Chart creation failed');
    } finally { setLoading(false); }
  };

  const buildCrossChart = async () => {
    if (!sessionId) return toast.error('Upload data first');
    if (!crossDs1 || !crossDs2) return toast.error('Select both datasets');
    if (!crossXCol) return toast.error('Select X column');
    if (!crossJoin1 || !crossJoin2) return toast.error('Select join columns');
    setLoading(true);
    try {
      const { data } = await engine.post('/visualization/cross-dataset', {
        session_id: sessionId,
        chart_type: crossChartType,
        chart_name: crossChartName.trim() || undefined,
        aggregation: crossAggregation !== 'none' ? crossAggregation : undefined,
        dataset1_key: crossDs1,
        dataset2_key: crossDs2,
        x_column: crossXCol,
        y_column: crossYCol || undefined,
        join_column1: crossJoin1,
        join_column2: crossJoin2,
        join_type: crossJoinType,
      });
      const resp = data.data;
      const fig = parsePlotly(resp?.plotly_json);
      if (fig) {
        const builtChart: ChartSpec = {
          title: crossChartName.trim() || `${crossChartType}: ${crossXCol}${crossYCol ? ' vs ' + crossYCol : ''} (cross-dataset)`,
          chart_type: crossChartType,
          data: fig.data,
          layout: fig.layout,
          dataset: `${shortName(crossDs1)} + ${shortName(crossDs2)}`,
          reasoning: `Joined ${resp.merged_rows?.toLocaleString()} rows via ${crossJoinType} join${crossAggregation !== 'none' ? ` | Aggregation: ${crossAggregation.toUpperCase()}` : ''}`,
        };
        setCharts((prev) => [...prev, builtChart]);
        setCrossBuiltCharts((prev) => [builtChart, ...prev]);
        toast.success(`Cross-dataset chart created (${resp.merged_rows?.toLocaleString()} merged rows)`);
      } else {
        toast.error('Could not parse chart data');
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Cross-dataset chart failed');
    } finally { setLoading(false); }
  };

  const ds1Columns = crossDs1 && datasetsInfo[crossDs1] ? datasetsInfo[crossDs1].columns : [];
  const ds2Columns = crossDs2 && datasetsInfo[crossDs2] ? datasetsInfo[crossDs2].columns : [];

  const darkLayout = (base: any = {}) => ({
    ...base,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#6B7280', family: 'Inter, sans-serif' },
    xaxis: { ...base.xaxis, gridcolor: 'rgba(0,0,0,0.06)', zerolinecolor: 'rgba(0,0,0,0.1)' },
    yaxis: { ...base.yaxis, gridcolor: 'rgba(0,0,0,0.06)', zerolinecolor: 'rgba(0,0,0,0.1)' },
    margin: { t: 40, r: 20, b: 40, l: 50, ...base.margin },
    autosize: true,
  });

  const shortName = (key: string) => key.length > 30 ? key.slice(0, 30) + '...' : key;

  const buildBusinessInsight = (chart: ChartSpec) => {
    if (chart.reasoning && chart.reasoning.trim()) return chart.reasoning;
    const dtype = (chart.chart_type || '').toLowerCase();
    if (dtype.includes('line') || dtype.includes('trend')) {
      return 'Business Insight: This trend highlights performance changes over time and helps identify momentum shifts for planning and forecasting.';
    }
    if (dtype.includes('bar')) {
      return 'Business Insight: This comparison identifies top and low-performing segments, guiding prioritization of resources and actions.';
    }
    if (dtype.includes('pie')) {
      return 'Business Insight: This composition view shows contribution share by segment, helping focus on the largest business drivers.';
    }
    if (dtype.includes('hist') || dtype.includes('distribution')) {
      return 'Business Insight: This distribution reveals spread and concentration, useful for detecting skewness, outliers, and process stability.';
    }
    if (dtype.includes('scatter')) {
      return 'Business Insight: This relationship view helps identify correlation patterns between two metrics and supports root-cause analysis.';
    }
    return 'Business Insight: This visualization highlights actionable patterns to support strategic business decisions.';
  };

  const pinChart = (chart: ChartSpec, index: number) => {
    if (!chart.data || chart.data.length === 0) {
      toast.error('Cannot pin chart without rendered data');
      return;
    }
    addPinnedChart({
      id: `${sessionId || 'no-session'}::viz::${chart.dataset || 'global'}::${chart.title}::${index}`,
      sessionId,
      title: chart.title,
      plotly_json: JSON.stringify({ data: chart.data || [], layout: chart.layout || {} }),
      source: `Visualization${chart.dataset ? ` - ${chart.dataset}` : ''}`,
      chartType: chart.chart_type,
    });
    toast.success(`Pinned chart ${chart.title}`);
  };

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Visualization</h1>
          <p className="text-[var(--text-secondary)] mt-1">AI-recommended charts & manual chart builder — per dataset</p>
        </div>
        <Button onClick={aiRecommend} loading={loading} icon={<Wand2 className="w-4 h-4" />}>AI Recommend</Button>
      </div>

      {/* Dataset selector */}
      {(datasetKeys.length > 1 || hasPipelineFinal) && (
        <Card className="mb-6">
          <div className="flex items-center gap-4">
            <Database className="w-5 h-5 text-[var(--accent)]" />
            <Select
              label="Active Dataset"
              value={selectedDataset}
              onChange={setSelectedDataset}
              placeholder="Select dataset"
              options={[
                ...(hasPipelineFinal ? [{ value: '__working__', label: 'Working Final Dataset (joined)' }] : []),
                { value: '__all__', label: `All Datasets (${datasetKeys.length})` },
                ...datasetKeys.map(k => ({ value: k, label: shortName(k) })),
              ]}
            />
            {selectedDataset === '__working__' && (
              <div className="flex items-center gap-2 text-sm text-emerald-700 mt-5">
                <FolderOpen className="w-4 h-4" />
                <span>Using relationship-aware finalized dataset</span>
              </div>
            )}
            {selectedDataset === '__all__' && (
              <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)] mt-5">
                <FolderOpen className="w-4 h-4" />
                <span>{datasetKeys.length} datasets — charts generated for all</span>
              </div>
            )}
            {selectedDataset && selectedDataset !== '__all__' && datasetsInfo[selectedDataset] && (
              <div className="flex gap-3 text-sm text-[var(--text-secondary)] mt-5">
                <span>{(datasetsInfo[selectedDataset].shape?.[0] || 0).toLocaleString()} rows</span>
                <span>{datasetsInfo[selectedDataset].shape?.[1] || 0} columns</span>
              </div>
            )}
          </div>
        </Card>
      )}

      <Tabs
        tabs={[
          { key: 'ai', label: 'AI Charts', icon: <Wand2 className="w-4 h-4" /> },
          { key: 'builder', label: 'Chart Builder', icon: <BarChart3 className="w-4 h-4" /> },
          ...(datasetKeys.length >= 2 ? [{ key: 'cross', label: 'Cross-Dataset', icon: <Link2 className="w-4 h-4" /> }] : []),
        ]}
        active={tab}
        onChange={setTab}
      />

      <div className="mt-6">
        {loading && charts.length === 0 && <div className="flex items-center justify-center py-20"><Spinner size="lg" /></div>}

        {tab === 'ai' && !loading && charts.length === 0 && (
          <EmptyState icon={<BarChart3 className="w-8 h-8" />} title="No charts yet" description="Let AI analyze your data and recommend the best visualizations">
            <Button onClick={aiRecommend} className="mt-4" icon={<Wand2 className="w-4 h-4" />}>Get Recommendations</Button>
          </EmptyState>
        )}

        {tab === 'builder' && (
          <>
            <Card className="mb-6">
              <h3 className="font-semibold mb-4">Manual Chart Builder</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
                <Select label="Chart Type" value={chartType} onChange={setChartType} options={[
                  { value: 'bar', label: 'Bar Chart' },
                  { value: 'line', label: 'Line Chart' },
                  { value: 'scatter', label: 'Scatter Plot' },
                  { value: 'histogram', label: 'Histogram' },
                  { value: 'pie', label: 'Pie Chart' },
                  { value: 'boxplot', label: 'Box Plot' },
                  { value: 'heatmap', label: 'Heatmap' },
                  { value: 'distribution', label: 'Distribution' },
                  { value: 'trend', label: 'Trend' },
                ]} />
                <Select label="X Column" value={xCol} onChange={setXCol}
                  placeholder="Select column"
                  options={columnNames.map(c => ({ value: c, label: c }))} />
                <Select label="Y Column (optional)" value={yCol} onChange={setYCol}
                  placeholder="Select column"
                  options={[{ value: '', label: '— None —' }, ...columnNames.map(c => ({ value: c, label: c }))]} />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <Input label="Chart Name (optional)" value={manualChartName} onChange={setManualChartName} placeholder="e.g. Total Sales by Category" />
                <Select label="Aggregation (optional)" value={manualAggregation} onChange={setManualAggregation} options={[
                  { value: 'none', label: 'None' },
                  { value: 'sum', label: 'SUM' },
                  { value: 'mean', label: 'MEAN' },
                  { value: 'median', label: 'MEDIAN' },
                  { value: 'count', label: 'COUNT' },
                  { value: 'min', label: 'MIN' },
                  { value: 'max', label: 'MAX' },
                  { value: 'nunique', label: 'DISTINCT COUNT' },
                ]} />
              </div>
              <Button onClick={buildChart} loading={loading} icon={<BarChart3 className="w-4 h-4" />}>Create Chart</Button>
            </Card>

            {manualBuiltCharts.length > 0 && (
              <Card className="mb-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-[var(--accent)]" /> Chart Builded
                </h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {manualBuiltCharts.map((chart, i) => (
                    <Card key={`manual-built-${i}`}>
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h4 className="font-semibold text-sm">{chart.title}</h4>
                          {chart.dataset && datasetKeys.length > 1 && (
                            <p className="text-xs text-[var(--text-muted)] mt-0.5">{shortName(chart.dataset)}</p>
                          )}
                          <p className="text-xs text-[var(--text-muted)] mt-0.5">{buildBusinessInsight(chart)}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          {chart.chart_type && <Badge>{chart.chart_type}</Badge>}
                          <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinChart(chart, i)}>Pin to Dashboard</Button>
                        </div>
                      </div>
                      <div className="rounded-lg overflow-hidden" style={{ minHeight: 300 }}>
                        {chart.data && chart.data.length > 0 ? (
                          <Plot
                            data={chart.data}
                            layout={darkLayout(chart.layout || {})}
                            config={{ responsive: true, displayModeBar: false }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler
                          />
                        ) : (
                          <div className="flex items-center justify-center h-[300px] text-[var(--text-muted)] text-sm">
                            {chart.reasoning ? `Recommendation: ${chart.reasoning}` : 'No chart data available'}
                          </div>
                        )}
                      </div>
                    </Card>
                  ))}
                </div>
              </Card>
            )}
          </>
        )}

        {/* Cross-Dataset Builder */}
        {tab === 'cross' && datasetKeys.length >= 2 && (
          <>
            <Card className="mb-6">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <Link2 className="w-4 h-4 text-[var(--accent)]" /> Cross-Dataset Chart Builder
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">
                Join two datasets and create charts across them — pick X from one dataset, Y from another.
              </p>

            {/* Chart Type */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
              <Select label="Chart Type" value={crossChartType} onChange={setCrossChartType} options={[
                { value: 'bar', label: 'Bar Chart' },
                { value: 'line', label: 'Line Chart' },
                { value: 'scatter', label: 'Scatter Plot' },
                { value: 'histogram', label: 'Histogram' },
                { value: 'pie', label: 'Pie Chart' },
                { value: 'boxplot', label: 'Box Plot' },
                { value: 'heatmap', label: 'Heatmap' },
              ]} />
              <Input label="Chart Name (optional)" value={crossChartName} onChange={setCrossChartName} placeholder="e.g. Sales vs Discounts Across Tables" />
              <Select label="Aggregation (optional)" value={crossAggregation} onChange={setCrossAggregation} options={[
                { value: 'none', label: 'None' },
                { value: 'sum', label: 'SUM' },
                { value: 'mean', label: 'MEAN' },
                { value: 'median', label: 'MEDIAN' },
                { value: 'count', label: 'COUNT' },
                { value: 'min', label: 'MIN' },
                { value: 'max', label: 'MAX' },
                { value: 'nunique', label: 'DISTINCT COUNT' },
              ]} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
              {/* Dataset 1 */}
              <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-indigo-600" /> Dataset 1 (X Axis)
                </h4>
                <div className="space-y-3">
                  <Select label="Dataset" value={crossDs1} onChange={(v) => { setCrossDs1(v); setCrossXCol(''); setCrossJoin1(''); }}
                    placeholder="Select dataset"
                    options={datasetKeys.map(k => ({ value: k, label: shortName(k) }))} />
                  <Select label="X Column" value={crossXCol} onChange={setCrossXCol}
                    placeholder="Select column"
                    options={ds1Columns.map(c => ({ value: c, label: c }))} />
                  <Select label="Join Column" value={crossJoin1} onChange={setCrossJoin1}
                    placeholder="Key to join on"
                    options={ds1Columns.map(c => ({ value: c, label: c }))} />
                </div>
                {crossDs1 && datasetsInfo[crossDs1] && (
                  <p className="text-xs text-[var(--text-muted)] mt-2">{datasetsInfo[crossDs1].shape[0].toLocaleString()} rows × {datasetsInfo[crossDs1].shape[1]} cols</p>
                )}
              </div>

              {/* Dataset 2 */}
              <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-emerald-600" /> Dataset 2 (Y Axis)
                </h4>
                <div className="space-y-3">
                  <Select label="Dataset" value={crossDs2} onChange={(v) => { setCrossDs2(v); setCrossYCol(''); setCrossJoin2(''); }}
                    placeholder="Select dataset"
                    options={datasetKeys.map(k => ({ value: k, label: shortName(k) }))} />
                  {crossChartType !== 'histogram' && crossChartType !== 'pie' && (
                    <Select label="Y Column" value={crossYCol} onChange={setCrossYCol}
                      placeholder="Select column"
                      options={ds2Columns.map(c => ({ value: c, label: c }))} />
                  )}
                  <Select label="Join Column" value={crossJoin2} onChange={setCrossJoin2}
                    placeholder="Key to join on"
                    options={ds2Columns.map(c => ({ value: c, label: c }))} />
                </div>
                {crossDs2 && datasetsInfo[crossDs2] && (
                  <p className="text-xs text-[var(--text-muted)] mt-2">{datasetsInfo[crossDs2].shape[0].toLocaleString()} rows × {datasetsInfo[crossDs2].shape[1]} cols</p>
                )}
              </div>
            </div>

            {/* Join Settings */}
              <div className="flex items-end gap-4">
                <Select label="Join Type" value={crossJoinType} onChange={setCrossJoinType} options={[
                  { value: 'inner', label: 'Inner (matching only)' },
                  { value: 'left', label: 'Left (all from DS1)' },
                  { value: 'right', label: 'Right (all from DS2)' },
                  { value: 'outer', label: 'Outer (all rows)' },
                ]} />
                <Button onClick={buildCrossChart} loading={loading} icon={<Link2 className="w-4 h-4" />}>
                  Create Cross-Dataset Chart
                </Button>
              </div>
            </Card>

            {crossBuiltCharts.length > 0 && (
              <Card className="mb-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Link2 className="w-4 h-4 text-[var(--accent)]" /> Cross-Dataset Chart Builded
                </h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {crossBuiltCharts.map((chart, i) => (
                    <Card key={`cross-built-${i}`}>
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h4 className="font-semibold text-sm">{chart.title}</h4>
                          {chart.dataset && datasetKeys.length > 1 && (
                            <p className="text-xs text-[var(--text-muted)] mt-0.5">{shortName(chart.dataset)}</p>
                          )}
                          <p className="text-xs text-[var(--text-muted)] mt-0.5">{buildBusinessInsight(chart)}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          {chart.chart_type && <Badge>{chart.chart_type}</Badge>}
                          <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinChart(chart, i)}>Pin to Dashboard</Button>
                        </div>
                      </div>
                      <div className="rounded-lg overflow-hidden" style={{ minHeight: 300 }}>
                        {chart.data && chart.data.length > 0 ? (
                          <Plot
                            data={chart.data}
                            layout={darkLayout(chart.layout || {})}
                            config={{ responsive: true, displayModeBar: false }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler
                          />
                        ) : (
                          <div className="flex items-center justify-center h-[300px] text-[var(--text-muted)] text-sm">
                            {chart.reasoning ? `Recommendation: ${chart.reasoning}` : 'No chart data available'}
                          </div>
                        )}
                      </div>
                    </Card>
                  ))}
                </div>
              </Card>
            )}
          </>
        )}

        {/* Chart Grid */}
        {charts.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {charts.map((chart, i) => (
              <Card key={i}>
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="font-semibold text-sm">{chart.title}</h4>
                    {chart.dataset && datasetKeys.length > 1 && (
                      <p className="text-xs text-[var(--text-muted)] mt-0.5">{shortName(chart.dataset)}</p>
                    )}
                    <p className="text-xs text-[var(--text-muted)] mt-0.5">{buildBusinessInsight(chart)}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {chart.chart_type && <Badge>{chart.chart_type}</Badge>}
                    <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinChart(chart, i)}>Pin to Dashboard</Button>
                  </div>
                </div>
                <div className="rounded-lg overflow-hidden" style={{ minHeight: 300 }}>
                  {chart.data && chart.data.length > 0 ? (
                    <Plot
                      data={chart.data}
                      layout={darkLayout(chart.layout || {})}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%', height: '100%' }}
                      useResizeHandler
                    />
                  ) : (
                    <div className="flex items-center justify-center h-[300px] text-[var(--text-muted)] text-sm">
                      {chart.reasoning ? `Recommendation: ${chart.reasoning}` : 'No chart data available'}
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
