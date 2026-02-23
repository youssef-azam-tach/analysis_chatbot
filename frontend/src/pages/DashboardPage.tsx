import { useEffect, useMemo, useRef, useState } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { engine, pollTask, parsePlotly } from '@/lib/api';
import { Button, Card, Spinner, EmptyState, KpiCard, Input, Badge, Select } from '@/components/ui';
import { LayoutDashboard, Download, Wand2, Pin, Trash2, Plus, ChevronLeft, ChevronRight, Copy, Lock, Unlock, ArrowRightLeft } from 'lucide-react';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';

interface DashChart {
  title: string;
  type?: string;
  data: any[];
  layout: any;
}

type WidgetTheme = 'default' | 'ocean' | 'sunset' | 'forest';

type DashboardWidget = {
  id: string;
  kind: 'chart' | 'kpi' | 'card';
  title: string;
  source?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  locked?: boolean;
  theme?: WidgetTheme;
  chart?: DashChart;
  kpi?: { label: string; value: string; suffix?: string };
  card?: { label: string; value: string };
};

interface DashPage {
  id: string;
  title: string;
  background: string;
  widgets: DashboardWidget[];
}

interface InteractionState {
  widgetId: string;
  mode: 'drag' | 'resize';
  startClientX: number;
  startClientY: number;
  startX: number;
  startY: number;
  startW: number;
  startH: number;
}

const CANVAS_W = 1600;
const CANVAS_H = 900;
const GRID = 20;

const THEME_OPTIONS = [
  { value: 'default', label: 'Default' },
  { value: 'ocean', label: 'Ocean' },
  { value: 'sunset', label: 'Sunset' },
  { value: 'forest', label: 'Forest' },
];

function snap(value: number) {
  return Math.round(value / GRID) * GRID;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function newPage(index: number): DashPage {
  return {
    id: `page-${Date.now()}-${index}`,
    title: `Page ${index}`,
    background: '#ffffff',
    widgets: [],
  };
}

function applyThemeToChart(chart: DashChart, theme: WidgetTheme): DashChart {
  const layout = { ...(chart.layout || {}) };
  const data = Array.isArray(chart.data) ? chart.data.map((d) => ({ ...d })) : [];

  if (theme === 'ocean') {
    layout.paper_bgcolor = '#f8fdff';
    layout.plot_bgcolor = '#f2fbff';
    data.forEach((trace: any, idx: number) => {
      const colors = ['#0EA5E9', '#0284C7', '#38BDF8', '#0369A1'];
      if (trace.type === 'bar' || trace.type === 'scatter' || trace.type === 'line') trace.marker = { ...(trace.marker || {}), color: colors[idx % colors.length] };
    });
  } else if (theme === 'sunset') {
    layout.paper_bgcolor = '#fffaf5';
    layout.plot_bgcolor = '#fff7ed';
    data.forEach((trace: any, idx: number) => {
      const colors = ['#F97316', '#FB7185', '#F59E0B', '#EF4444'];
      if (trace.type === 'bar' || trace.type === 'scatter' || trace.type === 'line') trace.marker = { ...(trace.marker || {}), color: colors[idx % colors.length] };
    });
  } else if (theme === 'forest') {
    layout.paper_bgcolor = '#f7fdf9';
    layout.plot_bgcolor = '#f0fdf4';
    data.forEach((trace: any, idx: number) => {
      const colors = ['#16A34A', '#15803D', '#22C55E', '#166534'];
      if (trace.type === 'bar' || trace.type === 'scatter' || trace.type === 'line') trace.marker = { ...(trace.marker || {}), color: colors[idx % colors.length] };
    });
  }

  return { ...chart, layout, data };
}

export default function DashboardPage() {
  const {
    sessionId,
    pinnedKpis,
    pinnedCards,
    pinnedCharts,
    removePinnedKpi,
    removePinnedCard,
    removePinnedChart,
    clearPinnedForSession,
  } = useWorkspaceStore();

  const [kpis, setKpis] = useState<any[]>([]);
  const [charts, setCharts] = useState<DashChart[]>([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [exporting, setExporting] = useState<'pdf' | 'pptx' | ''>('');

  const [pages, setPages] = useState<DashPage[]>([newPage(1)]);
  const [activePageId, setActivePageId] = useState('');
  const [newPageTitle, setNewPageTitle] = useState('');
  const [selectedWidgetId, setSelectedWidgetId] = useState<string>('');
  const [interaction, setInteraction] = useState<InteractionState | null>(null);
  const [dragPageId, setDragPageId] = useState('');

  const canvasRef = useRef<HTMLDivElement | null>(null);

  const pinnedKpisForSession = pinnedKpis.filter((p) => p.sessionId === sessionId);
  const pinnedCardsForSession = pinnedCards.filter((p) => p.sessionId === sessionId);
  const pinnedChartsForSession = pinnedCharts.filter((p) => p.sessionId === sessionId);

  const activePage = useMemo(() => pages.find((p) => p.id === activePageId) || pages[0], [pages, activePageId]);
  const selectedWidget = useMemo(() => activePage?.widgets.find((w) => w.id === selectedWidgetId) || null, [activePage, selectedWidgetId]);

  useEffect(() => {
    if (!sessionId) return;
    const key = `dashboard-designer:${sessionId}`;
    const raw = localStorage.getItem(key);
    if (!raw) {
      const initial = newPage(1);
      setPages([initial]);
      setActivePageId(initial.id);
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed.pages) && parsed.pages.length > 0) {
        setPages(parsed.pages);
        setActivePageId(parsed.activePageId || parsed.pages[0].id);
      }
    } catch {
      const initial = newPage(1);
      setPages([initial]);
      setActivePageId(initial.id);
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || pages.length === 0) return;
    const key = `dashboard-designer:${sessionId}`;
    localStorage.setItem(key, JSON.stringify({ pages, activePageId }));
  }, [sessionId, pages, activePageId]);

  useEffect(() => {
    if (pages.length === 0) {
      const initial = newPage(1);
      setPages([initial]);
      setActivePageId(initial.id);
      return;
    }
    if (!activePageId || !pages.some((p) => p.id === activePageId)) {
      setActivePageId(pages[0].id);
    }
  }, [pages, activePageId]);

  useEffect(() => {
    if (!interaction) return;

    const handleMouseMove = (e: MouseEvent) => {
      const page = activePage;
      if (!page || !canvasRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      const dx = (e.clientX - interaction.startClientX) * scaleX;
      const dy = (e.clientY - interaction.startClientY) * scaleY;

      setPages((prev) => prev.map((p) => {
        if (p.id !== page.id) return p;
        return {
          ...p,
          widgets: p.widgets.map((w) => {
            if (w.id !== interaction.widgetId || w.locked) return w;

            if (interaction.mode === 'drag') {
              const nx = clamp(snap(interaction.startX + dx), 0, CANVAS_W - w.w);
              const ny = clamp(snap(interaction.startY + dy), 0, CANVAS_H - w.h);
              return { ...w, x: nx, y: ny };
            }

            const nw = clamp(snap(interaction.startW + dx), 180, CANVAS_W - w.x);
            const nh = clamp(snap(interaction.startH + dy), 120, CANVAS_H - w.y);
            return { ...w, w: nw, h: nh };
          }),
        };
      }));
    };

    const handleMouseUp = () => setInteraction(null);

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [interaction, activePage]);

  const setWidget = (widgetId: string, updater: (w: DashboardWidget) => DashboardWidget) => {
    if (!activePage) return;
    setPages((prev) => prev.map((p) => {
      if (p.id !== activePage.id) return p;
      return { ...p, widgets: p.widgets.map((w) => (w.id === widgetId ? updater(w) : w)) };
    }));
  };

  const addPage = () => {
    const title = (newPageTitle || '').trim() || `Page ${pages.length + 1}`;
    const page = { ...newPage(pages.length + 1), title };
    setPages((prev) => [...prev, page]);
    setActivePageId(page.id);
    setNewPageTitle('');
    toast.success(`Added ${title}`);
  };

  const duplicatePage = (pageId: string) => {
    const target = pages.find((p) => p.id === pageId);
    if (!target) return;
    const clone: DashPage = {
      ...target,
      id: `page-${Date.now()}-dup`,
      title: `${target.title} Copy`,
      widgets: target.widgets.map((w) => ({ ...w, id: `${w.id}-copy-${Date.now()}` })),
    };
    setPages((prev) => [...prev, clone]);
    setActivePageId(clone.id);
    toast.success('Page duplicated');
  };

  const deletePage = (pageId: string) => {
    if (pages.length <= 1) return toast.error('At least one page is required');
    const next = pages.filter((p) => p.id !== pageId);
    setPages(next);
    if (activePageId === pageId) setActivePageId(next[0].id);
    toast.success('Page deleted');
  };

  const renamePage = (pageId: string, title: string) => {
    setPages((prev) => prev.map((p) => (p.id === pageId ? { ...p, title } : p)));
  };

  const reorderPages = (fromId: string, toId: string) => {
    if (fromId === toId) return;
    setPages((prev) => {
      const fromIdx = prev.findIndex((p) => p.id === fromId);
      const toIdx = prev.findIndex((p) => p.id === toId);
      if (fromIdx < 0 || toIdx < 0) return prev;
      const copy = [...prev];
      const [moved] = copy.splice(fromIdx, 1);
      copy.splice(toIdx, 0, moved);
      return copy;
    });
  };

  const prevPage = () => {
    const idx = pages.findIndex((p) => p.id === activePageId);
    if (idx > 0) setActivePageId(pages[idx - 1].id);
  };

  const nextPage = () => {
    const idx = pages.findIndex((p) => p.id === activePageId);
    if (idx < pages.length - 1) setActivePageId(pages[idx + 1].id);
  };

  const addWidgetToActivePage = (widget: Omit<DashboardWidget, 'x' | 'y' | 'w' | 'h'>) => {
    if (!activePage) return;
    const count = activePage.widgets.length;
    const placed: DashboardWidget = {
      ...widget,
      x: snap((count % 4) * 380 + 20),
      y: snap(Math.floor(count / 4) * 220 + 20),
      w: widget.kind === 'chart' ? 360 : 320,
      h: widget.kind === 'chart' ? 220 : 150,
      theme: 'default',
    };

    setPages((prev) => prev.map((p) => (p.id === activePage.id ? { ...p, widgets: [...p.widgets, placed] } : p)));
    setSelectedWidgetId(placed.id);
  };

  const moveWidgetToPage = (widgetId: string, fromPageId: string, toPageId: string) => {
    if (fromPageId === toPageId) return;
    let moved: DashboardWidget | null = null;
    setPages((prev) => {
      const removed = prev.map((p) => {
        if (p.id !== fromPageId) return p;
        return {
          ...p,
          widgets: p.widgets.filter((w) => {
            if (w.id === widgetId) {
              moved = { ...w, x: 20, y: 20 };
              return false;
            }
            return true;
          }),
        };
      });
      if (!moved) return prev;
      return removed.map((p) => (p.id === toPageId ? { ...p, widgets: [...p.widgets, moved as DashboardWidget] } : p));
    });
    setActivePageId(toPageId);
  };

  const removeWidget = (widgetId: string, pageId: string) => {
    setPages((prev) => prev.map((p) => (p.id === pageId ? { ...p, widgets: p.widgets.filter((w) => w.id !== widgetId) } : p)));
    if (selectedWidgetId === widgetId) setSelectedWidgetId('');
  };

  const duplicateWidget = (widget: DashboardWidget) => {
    const copy: DashboardWidget = {
      ...widget,
      id: `${widget.id}-copy-${Date.now()}`,
      x: clamp(widget.x + 30, 0, CANVAS_W - widget.w),
      y: clamp(widget.y + 30, 0, CANVAS_H - widget.h),
      title: `${widget.title} Copy`,
    };
    addWidgetToActivePage(copy);
  };

  const autoGenerate = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    setProgress(0);
    try {
      const { data } = await engine.post('/dashboard/build', { session_id: sessionId });
      const taskId = data.data?.task_id;
      if (!taskId) {
        handleDashResult(data.data);
        return;
      }
      toast('Building dashboard...', { icon: '⏳' });
      const result = await pollTask(taskId, (p) => setProgress(p));
      handleDashResult(result);
      toast.success('Dashboard generated');
    } catch (err: any) {
      toast.error(err.message || err.response?.data?.detail || 'Dashboard generation failed');
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const handleDashResult = (result: any) => {
    if (!result) return;
    setKpis(result.suggested_kpis || []);
    const parsed: DashChart[] = [];
    for (const sc of result.summary_charts || []) {
      const fig = parsePlotly(sc.figure || sc.plotly_json);
      if (fig) parsed.push({ title: sc.title || 'Chart', type: sc.type, data: fig.data, layout: fig.layout });
    }
    setCharts(parsed);
  };

  const exportDashboard = async (reportType: 'pdf' | 'pptx') => {
    if (!sessionId) return toast.error('Upload data first');
    if (pages.every((p) => p.widgets.length === 0)) return toast.error('Add widgets first');
    setExporting(reportType);
    try {
      const payloadPages = pages.map((page) => ({
        id: page.id,
        name: page.title,
        background: page.background,
        widgets: page.widgets.map((widget) => ({
          id: widget.id,
          kind: widget.kind,
          title: widget.title,
          theme: widget.theme,
          x: widget.x,
          y: widget.y,
          w: widget.w,
          h: widget.h,
          chart: widget.kind === 'chart' && widget.chart
            ? {
                title: widget.chart.title,
                type: widget.chart.type,
                plotly_json: JSON.stringify({ data: widget.chart.data || [], layout: widget.chart.layout || {} }),
              }
            : undefined,
          kpi: widget.kind === 'kpi' ? widget.kpi : undefined,
          card: widget.kind === 'card' ? widget.card : undefined,
        })),
      }));

      const response = await engine.post('/dashboard/export-report', {
        session_id: sessionId,
        title: 'Dashboard Designer Report',
        report_type: reportType,
        pages: payloadPages,
      }, { responseType: 'blob' });

      const ext = reportType === 'pptx' ? 'pptx' : 'pdf';
      const url = URL.createObjectURL(new Blob([response.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = `dashboard_designer.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`${reportType.toUpperCase()} exported`);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || `Failed to export ${reportType.toUpperCase()}`);
    } finally {
      setExporting('');
    }
  };

  const darkLayout = (base: any = {}, theme: WidgetTheme = 'default') => {
    const themed = {
      default: { paper: 'transparent', plot: 'transparent' },
      ocean: { paper: '#f8fdff', plot: '#f2fbff' },
      sunset: { paper: '#fffaf5', plot: '#fff7ed' },
      forest: { paper: '#f7fdf9', plot: '#f0fdf4' },
    }[theme];

    return {
      ...base,
      paper_bgcolor: themed.paper,
      plot_bgcolor: themed.plot,
      font: { color: '#6B7280', family: 'Inter, sans-serif' },
      xaxis: { ...base.xaxis, gridcolor: 'rgba(0,0,0,0.06)' },
      yaxis: { ...base.yaxis, gridcolor: 'rgba(0,0,0,0.06)' },
      margin: { t: 40, r: 20, b: 35, l: 45, ...base.margin },
      autosize: true,
    };
  };

  const hasAnyWidgets = pages.some((p) => p.widgets.length > 0);

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Dashboard Builder</h1>
          <p className="text-[var(--text-secondary)] mt-1">Power BI–style multi-page dashboard designer</p>
        </div>
        <div className="flex items-center gap-2">
          {(pinnedCardsForSession.length > 0 || pinnedKpisForSession.length > 0 || pinnedChartsForSession.length > 0) && (
            <Button onClick={() => clearPinnedForSession(sessionId)} variant="danger" icon={<Trash2 className="w-4 h-4" />}>Clear Pinned</Button>
          )}
          {hasAnyWidgets && (
            <>
              <Button onClick={() => exportDashboard('pdf')} variant="secondary" loading={exporting === 'pdf'} icon={<Download className="w-4 h-4" />}>Export PDF</Button>
              <Button onClick={() => exportDashboard('pptx')} variant="secondary" loading={exporting === 'pptx'} icon={<Download className="w-4 h-4" />}>Export PPTX</Button>
            </>
          )}
          <Button onClick={autoGenerate} loading={loading} icon={<Wand2 className="w-4 h-4" />}>Auto Generate</Button>
        </div>
      </div>

      <Card className="mb-6">
        <h3 className="font-semibold mb-3">Page Management</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <Input label="New Page Name" value={newPageTitle} onChange={setNewPageTitle} placeholder="e.g. Executive Summary" />
          <Button className="md:mt-6" icon={<Plus className="w-4 h-4" />} onClick={addPage}>Add New Page</Button>
          <Button className="md:mt-6" icon={<Copy className="w-4 h-4" />} variant="secondary" onClick={() => activePage && duplicatePage(activePage.id)}>Duplicate Active Page</Button>
          <Button className="md:mt-6" icon={<Trash2 className="w-4 h-4" />} variant="danger" onClick={() => activePage && deletePage(activePage.id)}>Delete Active Page</Button>
        </div>
      </Card>

      {loading && (
        <div className="flex flex-col items-center justify-center py-20">
          <Spinner size="lg" />
          <p className="text-sm text-[var(--text-muted)] mt-4">Building dashboard{progress > 0 ? ` (${Math.round(progress)}%)` : '...'}</p>
        </div>
      )}

      {!loading && !hasAnyWidgets && kpis.length === 0 && charts.length === 0 && pinnedCardsForSession.length === 0 && pinnedKpisForSession.length === 0 && pinnedChartsForSession.length === 0 && (
        <EmptyState icon={<LayoutDashboard className="w-8 h-8" />} title="No dashboard yet" description="Create pages and drag widgets to design your dashboard">
          <Button onClick={autoGenerate} className="mt-4" icon={<Wand2 className="w-4 h-4" />}>Auto Generate</Button>
        </EmptyState>
      )}

      {!loading && (
        <div className="space-y-6">
          {(pinnedCardsForSession.length > 0 || pinnedKpisForSession.length > 0 || pinnedChartsForSession.length > 0 || kpis.length > 0 || charts.length > 0) && (
            <Card>
              <h3 className="font-semibold mb-4 flex items-center gap-2"><Pin className="w-4 h-4 text-[var(--accent)]" /> Widgets Library</h3>

              {pinnedCardsForSession.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                  {pinnedCardsForSession.map((card) => (
                    <div key={card.id} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                      <p className="text-xs text-[var(--text-muted)]">{card.label}</p>
                      <p className="text-lg font-semibold">{card.value}</p>
                      <div className="flex gap-2 mt-2">
                        <Button size="sm" variant="ghost" onClick={() => addWidgetToActivePage({
                          id: `${card.id}::widget::${Date.now()}`,
                          kind: 'card',
                          title: card.label,
                          card: { label: card.label, value: card.value },
                          source: card.source,
                        })}>Add</Button>
                        <Button size="sm" variant="ghost" onClick={() => removePinnedCard(card.id)}>Remove</Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {pinnedKpisForSession.length > 0 && (
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 mb-4">
                  {pinnedKpisForSession.map((kpi) => (
                    <div key={kpi.id}>
                      <KpiCard label={kpi.label} value={kpi.value} suffix={kpi.suffix} />
                      <div className="flex gap-2 mt-2">
                        <Button size="sm" variant="ghost" onClick={() => addWidgetToActivePage({
                          id: `${kpi.id}::widget::${Date.now()}`,
                          kind: 'kpi',
                          title: kpi.label,
                          kpi: { label: kpi.label, value: kpi.value, suffix: kpi.suffix },
                          source: kpi.source,
                        })}>Add</Button>
                        <Button size="sm" variant="ghost" onClick={() => removePinnedKpi(kpi.id)}>Remove</Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {pinnedChartsForSession.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                  {pinnedChartsForSession.map((chart) => {
                    const fig = parsePlotly(chart.plotly_json);
                    return (
                      <div key={chart.id} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                        <p className="font-semibold text-sm">{chart.title}</p>
                        <div className="flex gap-2 mt-2">
                          <Button size="sm" variant="ghost" onClick={() => {
                            if (!fig) return;
                            addWidgetToActivePage({
                              id: `${chart.id}::widget::${Date.now()}`,
                              kind: 'chart',
                              title: chart.title,
                              chart: { title: chart.title, type: chart.chartType, data: fig.data, layout: fig.layout },
                              source: chart.source,
                            });
                          }}>Add</Button>
                          <Button size="sm" variant="ghost" onClick={() => removePinnedChart(chart.id)}>Remove</Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {(kpis.length > 0 || charts.length > 0) && (
                <div className="border-t border-[var(--border)] pt-4">
                  <p className="text-sm font-medium mb-3">Auto Generated</p>
                  {kpis.length > 0 && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 mb-4">
                      {kpis.map((kpi: any, idx: number) => (
                        <div key={`lib-kpi-${idx}`} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <p className="text-xs text-[var(--text-muted)]">{kpi.name || kpi.label || `KPI ${idx + 1}`}</p>
                          <p className="text-sm font-semibold">{kpi.formatted_value || String(kpi.value ?? '—')}</p>
                          <Button size="sm" variant="ghost" className="mt-2" onClick={() => addWidgetToActivePage({
                            id: `gen-kpi-${idx}-${Date.now()}`,
                            kind: 'kpi',
                            title: kpi.name || kpi.label || `KPI ${idx + 1}`,
                            kpi: {
                              label: kpi.name || kpi.label || `KPI ${idx + 1}`,
                              value: kpi.formatted_value || String(kpi.value ?? '—'),
                              suffix: kpi.aggregation || kpi.unit,
                            },
                            source: 'Auto Generate',
                          })}>Add</Button>
                        </div>
                      ))}
                    </div>
                  )}

                  {charts.length > 0 && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                      {charts.map((chart, idx) => (
                        <div key={`lib-chart-${idx}`} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <p className="text-sm font-semibold">{chart.title}</p>
                          <Button size="sm" variant="ghost" className="mt-2" onClick={() => addWidgetToActivePage({
                            id: `gen-chart-${idx}-${Date.now()}`,
                            kind: 'chart',
                            title: chart.title,
                            chart,
                            source: 'Auto Generate',
                          })}>Add</Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </Card>
          )}

          <Card>
            <h3 className="font-semibold mb-3">Canvas Designer (16:9)</h3>
            <p className="text-xs text-[var(--text-muted)] mb-3">Drag widgets to move, use corner handle to resize, and select widget for customization.</p>

            <div className="flex items-center justify-between gap-3 mb-3">
              <div className="flex items-center gap-2">
                <Button size="sm" variant="secondary" onClick={prevPage} disabled={pages.findIndex((p) => p.id === activePageId) <= 0} icon={<ChevronLeft className="w-4 h-4" />}>Prev</Button>
                <Button size="sm" variant="secondary" onClick={nextPage} disabled={pages.findIndex((p) => p.id === activePageId) >= pages.length - 1} icon={<ChevronRight className="w-4 h-4" />}>Next</Button>
              </div>
              {activePage && (
                <div className="flex items-end gap-3">
                  <Input label="Active Page Name" value={activePage.title} onChange={(v: string) => renamePage(activePage.id, v)} />
                  <Input label="Background" value={activePage.background} onChange={(v: string) => {
                    setPages((prev) => prev.map((p) => p.id === activePage.id ? { ...p, background: v || '#ffffff' } : p));
                  }} />
                </div>
              )}
            </div>

            <div className="relative w-full rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
              <div
                key={activePage?.id}
                ref={canvasRef}
                className="relative w-full rounded-lg border border-[var(--border)] overflow-hidden transition-all duration-300"
                style={{ aspectRatio: '16 / 9', background: activePage?.background || '#ffffff' }}
                onClick={() => setSelectedWidgetId('')}
              >
                {(activePage?.widgets || []).map((widget) => {
                  const left = (widget.x / CANVAS_W) * 100;
                  const top = (widget.y / CANVAS_H) * 100;
                  const width = (widget.w / CANVAS_W) * 100;
                  const height = (widget.h / CANVAS_H) * 100;
                  const selected = widget.id === selectedWidgetId;

                  return (
                    <div
                      key={widget.id}
                      className={`absolute border rounded-lg overflow-hidden ${selected ? 'border-[var(--accent)] shadow-lg' : 'border-[var(--border)]'} bg-[var(--bg-card)]`}
                      style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                      onClick={(e) => { e.stopPropagation(); setSelectedWidgetId(widget.id); }}
                    >
                      <div
                        className="flex items-center justify-between px-2 py-1 text-xs border-b border-[var(--border)] bg-[var(--bg-secondary)] cursor-move"
                        onMouseDown={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          if (widget.locked) return;
                          setSelectedWidgetId(widget.id);
                          setInteraction({
                            widgetId: widget.id,
                            mode: 'drag',
                            startClientX: e.clientX,
                            startClientY: e.clientY,
                            startX: widget.x,
                            startY: widget.y,
                            startW: widget.w,
                            startH: widget.h,
                          });
                        }}
                      >
                        <span className="truncate pr-2">{widget.title}</span>
                        <Badge variant="default">{widget.kind.toUpperCase()}</Badge>
                      </div>

                      <div className="w-full h-[calc(100%-28px)]">
                        {widget.kind === 'chart' && widget.chart && (
                          <Plot
                            data={widget.chart.data}
                            layout={darkLayout(widget.chart.layout || {}, widget.theme || 'default')}
                            config={{ responsive: true, displayModeBar: false }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler
                          />
                        )}

                        {widget.kind === 'kpi' && widget.kpi && (
                          <div className="p-2 h-full">
                            <KpiCard label={widget.kpi.label} value={widget.kpi.value} suffix={widget.kpi.suffix} />
                          </div>
                        )}

                        {widget.kind === 'card' && widget.card && (
                          <div className="p-3">
                            <p className="text-xs text-[var(--text-muted)]">{widget.card.label}</p>
                            <p className="text-lg font-semibold">{widget.card.value}</p>
                          </div>
                        )}
                      </div>

                      {!widget.locked && (
                        <div
                          className="absolute w-3 h-3 right-1 bottom-1 rounded-sm bg-[var(--accent)] cursor-se-resize"
                          onMouseDown={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            setSelectedWidgetId(widget.id);
                            setInteraction({
                              widgetId: widget.id,
                              mode: 'resize',
                              startClientX: e.clientX,
                              startClientY: e.clientY,
                              startX: widget.x,
                              startY: widget.y,
                              startW: widget.w,
                              startH: widget.h,
                            });
                          }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="mt-3 flex items-center gap-2 overflow-x-auto">
              {pages.map((p) => (
                <button
                  key={p.id}
                  draggable
                  onDragStart={() => setDragPageId(p.id)}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => {
                    if (dragPageId) reorderPages(dragPageId, p.id);
                    setDragPageId('');
                  }}
                  onClick={() => setActivePageId(p.id)}
                  className={`px-3 py-1.5 rounded-lg text-sm border transition ${p.id === activePageId ? 'bg-[var(--accent)] text-white border-[var(--accent)]' : 'bg-[var(--bg-card)] border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}`}
                >
                  {p.title}
                </button>
              ))}
            </div>
          </Card>

          {selectedWidget && activePage && (
            <Card>
              <h3 className="font-semibold mb-3">Visual Customization</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <Input label="Widget Title" value={selectedWidget.title} onChange={(v: string) => setWidget(selectedWidget.id, (w) => ({ ...w, title: v }))} />
                <Select
                  label="Theme"
                  value={selectedWidget.theme || 'default'}
                  onChange={(v) => {
                    const theme = v as WidgetTheme;
                    setWidget(selectedWidget.id, (w) => {
                      if (w.kind !== 'chart' || !w.chart) return { ...w, theme };
                      return { ...w, theme, chart: applyThemeToChart(w.chart, theme) };
                    });
                  }}
                  options={THEME_OPTIONS}
                />
                {selectedWidget.kind === 'chart' && (
                  <Select
                    label="Chart Type"
                    value={selectedWidget.chart?.type || 'bar'}
                    onChange={(nextType) => {
                      setWidget(selectedWidget.id, (w) => {
                        if (w.kind !== 'chart' || !w.chart) return w;
                        const nextData = (w.chart.data || []).map((trace: any, idx: number) => ({
                          ...trace,
                          type: nextType === 'line' ? 'scatter' : nextType,
                          mode: nextType === 'line' ? 'lines+markers' : trace.mode,
                          marker: {
                            ...(trace.marker || {}),
                            color: trace?.marker?.color || ['#4F46E5', '#10B981', '#F59E0B', '#EC4899'][idx % 4],
                          },
                        }));
                        return {
                          ...w,
                          chart: {
                            ...w.chart,
                            type: nextType,
                            data: nextData,
                          },
                        };
                      });
                    }}
                    options={[
                      { value: 'bar', label: 'Bar' },
                      { value: 'line', label: 'Line' },
                      { value: 'scatter', label: 'Scatter' },
                      { value: 'pie', label: 'Pie' },
                    ]}
                  />
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" icon={<Copy className="w-4 h-4" />} onClick={() => duplicateWidget(selectedWidget)}>Duplicate</Button>
                <Button variant="secondary" icon={<ArrowRightLeft className="w-4 h-4" />} onClick={() => {
                  const nextPage = pages.find((p) => p.id !== activePage.id);
                  if (!nextPage) return toast.error('Create another page first');
                  moveWidgetToPage(selectedWidget.id, activePage.id, nextPage.id);
                }}>Move to Next Page</Button>
                <Select
                  value={activePage.id}
                  onChange={(pid) => moveWidgetToPage(selectedWidget.id, activePage.id, pid)}
                  options={pages.map((p) => ({ value: p.id, label: p.title }))}
                />
                <Button variant="secondary" icon={selectedWidget.locked ? <Unlock className="w-4 h-4" /> : <Lock className="w-4 h-4" />} onClick={() => setWidget(selectedWidget.id, (w) => ({ ...w, locked: !w.locked }))}>{selectedWidget.locked ? 'Unlock' : 'Lock'} Position</Button>
                <Button variant="danger" icon={<Trash2 className="w-4 h-4" />} onClick={() => removeWidget(selectedWidget.id, activePage.id)}>Delete</Button>
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
