import { useState } from 'react';
import { Outlet, Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/auth-store';
import { Sidebar } from '@/components/Sidebar';
import { Toaster } from 'react-hot-toast';

export default function AppLayout() {
  const { isAuthenticated } = useAuthStore();
  const [collapsed, setCollapsed] = useState(false);
  if (!isAuthenticated) return <Navigate to="/login" replace />;

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
        </div>
      </main>
    </div>
  );
}
