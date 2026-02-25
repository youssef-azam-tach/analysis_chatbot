import { useEffect, useState } from 'react';
import { useAuthStore } from '@/stores/auth-store';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { engine, pollTask } from '@/lib/api';
import { Button, Card, Badge, EmptyState } from '@/components/ui';
import { FileText, Download, Clock, FilePieChart, Presentation } from 'lucide-react';
import toast from 'react-hot-toast';

interface Report {
  task_id: string;
  report_id?: string;
  report_type: string;
  filename?: string;
  file_size_bytes?: number;
  created_at: string;
}

export default function ReportsPage() {
  const { sessionId } = useWorkspaceStore();
  const { user } = useAuthStore();
  const [reports, setReports] = useState<Report[]>([]);
  const [reportItems, setReportItems] = useState<any[]>([]);
  const [generating, setGenerating] = useState<string | null>(null);

  useEffect(() => {
    const loadReportItems = async () => {
      if (!user) return;
      try {
        const { data } = await engine.get(`/reports/items?user_id=${user.id}`);
        setReportItems(data?.data?.items || []);
      } catch {
        setReportItems([]);
      }
    };
    loadReportItems();
  }, [user]);

  const buildPinnedExportPayload = () => {
    const sections: string[] = ['# Pinned Responses'];
    const plotlyCharts: any[] = [];

    reportItems.forEach((item: any, idx: number) => {
      const payload = item?.payload || {};
      const text = String(payload.text_markdown || payload.text || '').trim();

      sections.push(`## Item ${idx + 1}`);
      if (text) sections.push('', text);

      const charts = Array.isArray(payload.charts) ? payload.charts : [];
      charts.forEach((chart: any, cidx: number) => {
        const pj = chart?.plotly_json;
        if (!pj) return;
        plotlyCharts.push({
          title: chart?.title || `Pinned Chart ${idx + 1}.${cidx + 1}`,
          plotly_json: pj,
          business_insight: `Pinned from chat item ${idx + 1}`,
        });
      });
    });

    return {
      analysis_markdown: sections.join('\n'),
      plotly_charts: plotlyCharts,
    };
  };

  const generate = async (reportType: 'pdf' | 'pptx') => {
    if (!sessionId) return toast.error('Upload data first');
    if (!reportItems.length) return toast.error('No pinned report items found on this page');

    setGenerating(reportType);
    try {
      const payload = buildPinnedExportPayload();

      const { data } = await engine.post('/reports/strategic-report', {
        session_id: sessionId,
        report_type: reportType,
        analysis_markdown: payload.analysis_markdown,
        pinned_only: true,
        recommendations: [],
        plotly_charts: payload.plotly_charts,
      });

      const taskId = data.data?.task_id;
      if (!taskId) {
        toast.error('No task ID returned');
        return;
      }
      toast(`Generating ${reportType.toUpperCase()} report...`, { icon: '⏳' });

      const result = await pollTask(taskId);

      setReports((prev) => [{
        task_id: taskId,
        report_id: result.report_id,
        report_type: result.report_type || reportType,
        filename: result.filename,
        file_size_bytes: result.file_size_bytes,
        created_at: new Date().toISOString(),
      }, ...prev]);

      toast.success(`${reportType.toUpperCase()} report generated`);
    } catch (err: any) {
      toast.error(err.message || err.response?.data?.detail || 'Report generation failed');
    } finally { setGenerating(null); }
  };

  const download = async (report: Report) => {
    try {
      const { data } = await engine.get(`/reports/download/${report.task_id}`, { responseType: 'blob' });
      const blob = new Blob([data]);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = report.filename || `report.${report.report_type}`;
      a.click();
      URL.revokeObjectURL(a.href);
      toast.success('Downloaded');
    } catch { toast.error('Download failed'); }
  };

  const formatIcon = (fmt: string) => {
    if (fmt === 'pdf') return <FilePieChart className="w-5 h-5 text-red-600" />;
    if (fmt === 'pptx') return <Presentation className="w-5 h-5 text-orange-600" />;
    return <FileText className="w-5 h-5 text-indigo-600" />;
  };

  const formatSize = (bytes?: number) => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Reports</h1>
          <p className="text-[var(--text-secondary)] mt-1">Generate PDF or PowerPoint from all pinned items in this page</p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={() => generate('pdf')} loading={generating === 'pdf'} variant="primary" icon={<FilePieChart className="w-4 h-4" />}>PDF</Button>
          <Button onClick={() => generate('pptx')} loading={generating === 'pptx'} variant="primary" icon={<Presentation className="w-4 h-4" />}>PowerPoint</Button>
        </div>
      </div>

      {reports.length === 0 && (
        <EmptyState icon={<FileText className="w-8 h-8" />} title="No reports yet" description="Generate PDF or PowerPoint reports from pinned report items">
          <div className="flex gap-3 mt-4 flex-wrap">
            <Button onClick={() => generate('pdf')} icon={<FilePieChart className="w-4 h-4" />}>PDF Report</Button>
            <Button onClick={() => generate('pptx')} icon={<Presentation className="w-4 h-4" />}>PowerPoint</Button>
          </div>
        </EmptyState>
      )}

      {reports.length > 0 && (
        <div className="space-y-3">
          {reports.map((report, i) => (
            <Card key={report.task_id || i} hover>
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-[var(--bg-secondary)] flex items-center justify-center">
                  {formatIcon(report.report_type)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold">{report.filename || `Report ${i + 1}`}</p>
                  <div className="flex items-center gap-3 mt-1 text-xs text-[var(--text-muted)]">
                    <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{new Date(report.created_at).toLocaleDateString()}</span>
                    <Badge variant={report.report_type === 'pdf' ? 'danger' : report.report_type === 'pptx' ? 'warning' : 'primary'}>{report.report_type.toUpperCase()}</Badge>
                    {report.file_size_bytes && <span>{formatSize(report.file_size_bytes)}</span>}
                  </div>
                </div>
                <Button onClick={() => download(report)} variant="secondary" icon={<Download className="w-4 h-4" />}>Download</Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {reportItems.length > 0 && (
        <Card className="mt-6">
          <h3 className="font-semibold mb-4">Pinned Chat Report Items</h3>
          <div className="space-y-3">
            {reportItems.map((item) => {
              const payload = item.payload || {};
              const text = String(payload.text_markdown || payload.text || '');
              const charts = payload.charts || [];
              return (
                <div key={item.pin_id} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs text-[var(--text-muted)]">Chat: {item.chat_id} • Message: {item.message_id}</p>
                      <p className="text-sm mt-1 whitespace-pre-wrap">{text.length > 400 ? `${text.slice(0, 400)}...` : text}</p>
                      <p className="text-[10px] text-[var(--text-muted)] mt-1">{new Date(item.created_at).toLocaleString()} • {charts.length} chart(s)</p>
                    </div>
                    <Button
                      variant="secondary"
                      onClick={() => {
                        const blob = new Blob([text], { type: 'text/markdown' });
                        const a = document.createElement('a');
                        a.href = URL.createObjectURL(blob);
                        a.download = `pinned_report_${item.pin_id}.md`;
                        a.click();
                        URL.revokeObjectURL(a.href);
                      }}
                    >
                      Export Text
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}
