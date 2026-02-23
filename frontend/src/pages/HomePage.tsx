import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWorkspaceStore, type Workspace } from '@/stores/workspace-store';
import { Button, Card, Input, Modal, EmptyState, Badge, Spinner } from '@/components/ui';
import { Plus, FolderOpen, Trash2, Calendar, Database } from 'lucide-react';
import toast from 'react-hot-toast';
import { backend } from '@/lib/api';

export default function HomePage() {
  const { workspaces, activeWorkspace, fetchWorkspaces, createWorkspace, selectWorkspace, loading } = useWorkspaceStore();
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();

  useEffect(() => { fetchWorkspaces(); }, []);

  const handleCreate = async () => {
    if (!name.trim()) return;
    setCreating(true);
    try {
      const ws = await createWorkspace(name.trim(), desc.trim() || undefined);
      selectWorkspace(ws);
      setShowCreate(false);
      setName(''); setDesc('');
      toast.success('Workspace created!');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to create workspace');
    } finally {
      setCreating(false);
    }
  };

  const handleOpen = (ws: Workspace) => {
    selectWorkspace(ws);
    navigate('/upload');
  };

  const handleDelete = async (wsId: string) => {
    if (!confirm('Delete this workspace?')) return;
    try {
      await backend.delete(`/workspaces/${wsId}`);
      await fetchWorkspaces();
      toast.success('Workspace deleted');
    } catch { toast.error('Failed to delete'); }
  };

  if (loading) return <div className="flex items-center justify-center h-96"><Spinner size="lg" /></div>;

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Welcome back! 👋</h1>
          <p className="text-[var(--text-secondary)] mt-1">Select a workspace or create a new one to start analyzing</p>
        </div>
        <Button icon={<Plus className="w-4 h-4" />} onClick={() => setShowCreate(true)}>New Workspace</Button>
      </div>

      {/* Workspaces Grid */}
      {workspaces.length === 0 ? (
        <EmptyState
          icon={<FolderOpen className="w-8 h-8" />}
          title="No workspaces yet"
          description="Create your first workspace to start uploading data and running AI analysis"
          action={<Button icon={<Plus className="w-4 h-4" />} onClick={() => setShowCreate(true)}>Create Workspace</Button>}
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {workspaces.map((ws) => (
            <Card key={ws.id} hover onClick={() => handleOpen(ws)} className={activeWorkspace?.id === ws.id ? 'ring-2 ring-[var(--accent)]' : ''}>
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-[var(--accent-bg)] flex items-center justify-center">
                    <Database className="w-5 h-5 text-[var(--accent)]" />
                  </div>
                  <div>
                    <h3 className="font-semibold">{ws.name}</h3>
                    {ws.description && <p className="text-xs text-[var(--text-muted)] mt-0.5 line-clamp-1">{ws.description}</p>}
                  </div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(ws.id); }}
                  className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--error)] hover:bg-[var(--error-bg)] transition cursor-pointer"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              <div className="flex items-center gap-3 mt-4 text-xs text-[var(--text-muted)]">
                <Badge variant="info">{ws.data_source_type || 'file'}</Badge>
                <span className="flex items-center gap-1"><Calendar className="w-3 h-3" />{new Date(ws.created_at).toLocaleDateString()}</span>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Create Modal */}
      <Modal open={showCreate} onClose={() => setShowCreate(false)} title="Create Workspace" footer={
        <>
          <Button variant="secondary" onClick={() => setShowCreate(false)}>Cancel</Button>
          <Button loading={creating} onClick={handleCreate}>Create</Button>
        </>
      }>
        <div className="space-y-4">
          <Input label="Workspace Name" placeholder="Q4 Sales Analysis" value={name} onChange={setName} />
          <Input label="Description (optional)" placeholder="Analysis of Q4 2025 sales data..." value={desc} onChange={setDesc} />
        </div>
      </Modal>
    </div>
  );
}
