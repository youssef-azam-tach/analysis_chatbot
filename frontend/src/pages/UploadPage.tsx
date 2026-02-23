import { useState, useCallback, useEffect } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { engine } from '@/lib/api';
import { Card, Badge, DataTable, EmptyState, Spinner, Tabs, ProgressBar } from '@/components/ui';
import { Upload, FileSpreadsheet, Table2, CheckCircle2, X, Plus } from 'lucide-react';
import toast from 'react-hot-toast';

type UploadStatus = 'pending' | 'uploading' | 'done' | 'error';

interface UploadQueueItem {
  id: string;
  name: string;
  size: number;
  progress: number;
  status: UploadStatus;
  error?: string;
}

export default function UploadPage() {
  const { sessionId, setSessionId, uploadedFiles, addUploadedFiles, setUploadedFiles, clearSession } = useWorkspaceStore();
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState<any>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [tab, setTab] = useState('upload');
  const [dragOver, setDragOver] = useState(false);
  const [recovering, setRecovering] = useState(false);
  const [uploadQueue, setUploadQueue] = useState<UploadQueueItem[]>([]);

  const formatBytes = (bytes: number) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex += 1;
    }
    return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
  };

  /* ── Session recovery on mount ── */
  useEffect(() => {
    if (!sessionId || uploadedFiles.length > 0) return;
    // We have a sessionId from localStorage but no files — try to recover
    let cancelled = false;
    (async () => {
      setRecovering(true);
      try {
        const { data } = await engine.get(`/files/session-info?session_id=${sessionId}`);
        if (cancelled) return;
        const info = data.data;
        if (info && info.dataframe_keys?.length) {
          const recovered = info.dataframe_keys.map((key: string) => ({
            filename: key,
            dataset_key: key,
            total_rows: info.active_df_shape?.[0] ?? null,
            total_columns: info.active_df_shape?.[1] ?? null,
          }));
          setUploadedFiles(recovered);
        }
      } catch {
        // Session expired on server — clear local state
        clearSession();
      } finally {
        if (!cancelled) setRecovering(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const handleUpload = async (fileList: FileList) => {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    const queue = files.map((file, i) => ({
      id: `${file.name}-${file.lastModified}-${i}`,
      name: file.name,
      size: file.size,
      progress: 0,
      status: 'pending' as UploadStatus,
    }));
    setUploadQueue(queue);

    setUploading(true);
    try {
      let currentSessionId = sessionId;
      const uploaded: any[] = [];
      let failed = 0;

      for (let i = 0; i < queue.length; i += 1) {
        const item = queue[i];
        const file = files[i];

        setUploadQueue((prev) => prev.map((q) => (q.id === item.id ? { ...q, status: 'uploading', progress: 0, error: undefined } : q)));

        const formData = new FormData();
        formData.append('file', file);
        if (currentSessionId) formData.append('session_id', currentSessionId);

        try {
          const { data } = await engine.post('/files/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (evt) => {
              const total = evt.total || file.size;
              const pct = total > 0 ? Math.min(100, Math.round((evt.loaded / total) * 100)) : 0;
              setUploadQueue((prev) => prev.map((q) => (q.id === item.id ? { ...q, progress: pct } : q)));
            },
          });

          const resp = data.data;
          currentSessionId = resp.session_id || currentSessionId;
          if (currentSessionId) setSessionId(currentSessionId);

          uploaded.push({
            filename: resp.filename,
            dataset_key: resp.dataset_key,
            total_rows: resp.total_rows,
            total_columns: resp.total_columns,
            sheets: Array.isArray(resp.sheets)
              ? resp.sheets.map((s: any) => (typeof s === 'string' ? s : (s?.name || s?.sheet_name || 'Sheet')))
              : undefined,
          });

          setUploadQueue((prev) => prev.map((q) => (q.id === item.id ? { ...q, status: 'done', progress: 100 } : q)));
        } catch (err: any) {
          failed += 1;
          setUploadQueue((prev) => prev.map((q) => (q.id === item.id ? { ...q, status: 'error', error: err.response?.data?.detail || 'Upload failed' } : q)));
        }
      }

      if (uploaded.length > 0) {
        addUploadedFiles(uploaded);
        toast.success(`${uploaded.length} file(s) uploaded successfully`);
      }
      if (failed > 0) {
        toast.error(`${failed} file(s) failed to upload`);
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files);
  }, [sessionId]);

  const loadPreview = async (key: string) => {
    if (!sessionId) return;
    setSelectedKey(key);
    setLoadingPreview(true);
    try {
      const { data } = await engine.get(`/files/preview?session_id=${sessionId}&key=${key}&rows=20`);
      const payload = data.data;
      const normalizedRows = payload.rows ?? payload.preview ?? [];
      const normalizedColumns = Array.isArray(payload.columns) && payload.columns.length > 0
        ? payload.columns
        : (normalizedRows[0] ? Object.keys(normalizedRows[0]) : []);
      setPreview({
        ...payload,
        rows: normalizedRows,
        columns: normalizedColumns,
      });
    } catch { toast.error('Failed to load preview'); }
    finally { setLoadingPreview(false); }
  };

  /* ── Auto-load first file preview when switching to loaded tab ── */
  useEffect(() => {
    if (tab === 'loaded' && uploadedFiles.length > 0 && !preview && !selectedKey) {
      const firstKey = uploadedFiles[0].dataset_key || uploadedFiles[0].filename;
      loadPreview(firstKey);
    }
  }, [tab, uploadedFiles.length]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Data Upload</h1>
          <p className="text-[var(--text-secondary)] mt-1">Upload Excel or CSV files to start your analysis pipeline</p>
        </div>
        {sessionId && <Badge variant="success">Session: {sessionId.slice(0, 8)}…</Badge>}
      </div>

      <Tabs
        tabs={[
          { key: 'upload', label: 'Upload Files', icon: <Upload className="w-4 h-4" /> },
          { key: 'loaded', label: `Loaded (${uploadedFiles.length})`, icon: <FileSpreadsheet className="w-4 h-4" /> },
        ]}
        active={tab}
        onChange={setTab}
      />

      <div className="mt-6">
        {recovering && (
          <div className="flex items-center gap-2 mb-4 p-3 rounded-lg bg-[var(--accent-bg)] text-sm text-[var(--accent)]">
            <Spinner size="sm" /> Recovering session data…
          </div>
        )}
        {tab === 'upload' && (
          <div className="space-y-6">
            {/* Drop Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-200 ${
                dragOver ? 'border-[var(--accent)] bg-[var(--accent-bg)]' : 'border-[var(--border)] hover:border-[var(--border-hover)]'
              }`}
            >
              {uploading ? (
                <div className="w-full max-w-2xl mx-auto space-y-4">
                  <div className="flex items-center justify-center gap-3">
                    <Spinner size="lg" />
                    <p className="text-sm text-[var(--text-secondary)]">Uploading & processing files...</p>
                  </div>
                  <ProgressBar value={Math.round((uploadQueue.reduce((sum, f) => sum + f.progress, 0) / Math.max(1, uploadQueue.length)))} />
                  <div className="space-y-2 text-left">
                    {uploadQueue.map((f) => (
                      <div key={f.id} className="p-3 rounded-lg bg-[var(--bg-card)] border border-[var(--border)]">
                        <div className="flex items-center justify-between gap-3 mb-1.5">
                          <p className="text-sm font-medium truncate">{f.name}</p>
                          <div className="flex items-center gap-2 shrink-0">
                            <span className="text-xs text-[var(--text-muted)]">{formatBytes(f.size)}</span>
                            {f.status === 'done' && <Badge variant="success">Done</Badge>}
                            {f.status === 'uploading' && <Badge variant="primary">Uploading</Badge>}
                            {f.status === 'pending' && <Badge variant="default">Pending</Badge>}
                            {f.status === 'error' && <Badge variant="danger">Failed</Badge>}
                          </div>
                        </div>
                        <ProgressBar value={f.progress} />
                        {f.error && <p className="text-xs text-[var(--error)] mt-1">{f.error}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  <div className="w-16 h-16 rounded-2xl bg-[var(--accent-bg)] flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-8 h-8 text-[var(--accent)]" />
                  </div>
                  <p className="text-lg font-semibold mb-2">Drop files here or click to browse</p>
                  <p className="text-sm text-[var(--text-muted)] mb-6">Supports .xlsx, .xls, .csv — up to 100MB</p>
                  <label className="inline-flex items-center gap-2 px-6 py-3 rounded-lg gradient-accent text-white font-medium cursor-pointer hover:opacity-90 transition">
                    <Plus className="w-4 h-4" />
                    Choose Files
                    <input type="file" accept=".xlsx,.xls,.csv" multiple onChange={(e) => e.target.files && handleUpload(e.target.files)} className="hidden" />
                  </label>
                </>
              )}
            </div>

            {/* Recent uploads */}
            {uploadedFiles.length > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-[var(--text-secondary)]">Uploaded Files</h3>
                {uploadedFiles.map((f: any, i: number) => (
                  <Card key={i} hover onClick={() => { loadPreview(f.dataset_key || f.filename); setTab('loaded'); }}>
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-lg bg-emerald-50 flex items-center justify-center">
                        <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate">{f.filename}</p>
                        <div className="flex items-center gap-3 mt-1 text-xs text-[var(--text-muted)]">
                          <span>{f.total_rows?.toLocaleString()} rows</span>
                          <span>{f.total_columns} columns</span>
                          {f.sheets && <span>{f.sheets.length} sheet(s)</span>}
                        </div>
                      </div>
                      <Badge variant="success">Ready</Badge>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === 'loaded' && (
          <div>
            {uploadedFiles.length === 0 ? (
              <EmptyState icon={<FileSpreadsheet className="w-8 h-8" />} title="No files loaded" description="Upload files first to see them here" />
            ) : (
              <div className="space-y-4">
                {/* Dataset selector cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {uploadedFiles.map((f: any, i: number) => {
                    const key = f.dataset_key || f.filename;
                    const isSelected = selectedKey === key;
                    return (
                      <Card
                        key={i}
                        hover
                        onClick={() => loadPreview(key)}
                        className={isSelected ? 'ring-2 ring-[var(--accent)] bg-[var(--accent-bg)]' : ''}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${isSelected ? 'bg-[var(--accent)] text-white' : 'bg-[var(--surface-hover)]'}`}>
                            <Table2 className="w-4 h-4" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className={`font-medium text-sm truncate ${isSelected ? 'text-[var(--accent)]' : ''}`}>{f.filename}</p>
                            <p className="text-xs text-[var(--text-muted)]">{f.total_rows?.toLocaleString()} rows × {f.total_columns} cols</p>
                          </div>
                          {isSelected && <CheckCircle2 className="w-4 h-4 text-[var(--accent)] shrink-0" />}
                        </div>
                      </Card>
                    );
                  })}
                </div>

                {/* Data Preview */}
                {loadingPreview && (
                  <Card>
                    <div className="flex items-center justify-center gap-3 py-8">
                      <Spinner size="md" />
                      <span className="text-sm text-[var(--text-secondary)]">Loading preview…</span>
                    </div>
                  </Card>
                )}
                {preview && Array.isArray(preview.rows) && (
                  <Card>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <Table2 className="w-4 h-4 text-[var(--accent)]" />
                        <h3 className="font-semibold text-sm">{selectedKey}</h3>
                        <Badge variant="default">{preview.rows.length} rows</Badge>
                      </div>
                      <button onClick={() => { setPreview(null); setSelectedKey(null); }} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"><X className="w-4 h-4" /></button>
                    </div>
                    <DataTable columns={preview.columns} rows={preview.rows} maxRows={20} />
                  </Card>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
