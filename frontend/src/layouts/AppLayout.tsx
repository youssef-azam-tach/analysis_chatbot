import { useState } from 'react';
import { Outlet, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/auth-store';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { Button } from '@/components/ui';
import { Sidebar } from '@/components/Sidebar';
import { Toaster } from 'react-hot-toast';
import { ArrowLeft, ArrowRight } from 'lucide-react';

const FLOW_ROUTES = [
  '/upload',
  '/schema',
  '/quality',
  '/cleaning',
  '/goals',
  '/analysis',
  '/kpis',
  '/visualization',
  '/dashboard',
  '/chat',
  '/reports',
];

export default function AppLayout() {
  const { isAuthenticated } = useAuthStore();
  const { activeWorkspace, sessionId } = useWorkspaceStore();
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  if (!isAuthenticated) return <Navigate to="/login" replace />;

  const currentIndex = FLOW_ROUTES.findIndex((r) => location.pathname === r);
  const canGoBack = currentIndex > 0;
  const canGoNext = currentIndex >= 0 && currentIndex < FLOW_ROUTES.length - 1;
  const requiresSession = currentIndex >= 0;
  const nextDisabled = !canGoNext || !activeWorkspace || (requiresSession && !sessionId);
  const backDisabled = !canGoBack || !activeWorkspace;

  const goBack = () => {
    if (!canGoBack) return;
    navigate(FLOW_ROUTES[currentIndex - 1]);
  };

  const goNext = () => {
    if (nextDisabled || !canGoNext) return;
    navigate(FLOW_ROUTES[currentIndex + 1]);
  };

  return (
    <div className="min-h-screen bg-[var(--bg-primary)]">
      <Toaster position="top-right" toastOptions={{
        style: { background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border)', fontSize: '14px', boxShadow: 'var(--shadow-md)' },
      }} />
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
      <main
        className="min-h-screen transition-all duration-300"
        style={{ marginLeft: collapsed ? 68 : 260 }}
      >
        <div className="px-5 md:px-8 py-5 md:py-7 max-w-[1700px]">
          <section className="app-surface page-shell animate-fade-in">
            <Outlet />
          </section>

          {currentIndex >= 0 && (
            <div className="mt-5 sticky bottom-4 z-20">
              <div className="glass px-4 py-3 border border-[var(--border)] shadow-[var(--shadow-md)] flex items-center justify-between gap-3">
                <Button variant="secondary" onClick={goBack} disabled={backDisabled} icon={<ArrowLeft className="w-4 h-4" />}>
                  Back
                </Button>
                <Button onClick={goNext} disabled={nextDisabled} icon={<ArrowRight className="w-4 h-4" />}>
                  Go to Next Page
                </Button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
