import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/stores/auth-store';
import { useWorkspaceStore } from '@/stores/workspace-store';
import {
  Home, Upload, Target, AlertTriangle, Sparkles,
  BarChart3, LayoutDashboard, MessageCircle, FileText, Brain,
  LogOut, Database,
  PanelLeftClose, PanelLeft, Layers, Check, FileSpreadsheet
} from 'lucide-react';

const navItems = [
  { to: '/', icon: Home, label: 'Home', end: true },
  { to: '/upload', icon: Upload, label: 'Data Upload' },
  { to: '/schema', icon: Database, label: 'Schema Analysis' },
  { to: '/quality', icon: AlertTriangle, label: 'Data Quality' },
  { to: '/cleaning', icon: Sparkles, label: 'Data Cleaning' },
  { to: '/goals', icon: Target, label: 'Business Goals' },
  { to: '/analysis', icon: Brain, label: 'AI Analysis' },
  { to: '/kpis', icon: BarChart3, label: 'KPIs' },
  { to: '/visualization', icon: Layers, label: 'Visualization' },
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/chat', icon: MessageCircle, label: 'AI Chat' },
  { to: '/reports', icon: FileText, label: 'Reports' },
  { to: '/quick-excel', icon: FileSpreadsheet, label: 'Quick Excel Analysis' },
];

interface SidebarProps {
  collapsed: boolean;
  setCollapsed: (v: boolean) => void;
}

export function Sidebar({ collapsed, setCollapsed }: SidebarProps) {
  const { user, logout } = useAuthStore();
  const { activeWorkspace } = useWorkspaceStore();
  const navigate = useNavigate();
  const location = useLocation();

  const flowRoutes = ['/upload', '/schema', '/quality', '/cleaning', '/goals', '/analysis', '/kpis', '/visualization', '/dashboard', '/chat', '/reports'];
  const currentFlowIdx = flowRoutes.findIndex((r) => r === location.pathname);

  const handleLogout = () => { logout(); navigate('/login'); };

  return (
    <aside className={cn(
      'fixed left-0 top-0 h-screen flex flex-col bg-[var(--sidebar-bg)] border-r border-[var(--border)] z-40 transition-all duration-300 shadow-[var(--shadow-md)]',
      collapsed ? 'w-[68px]' : 'w-[260px]'
    )}>
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-[68px] border-b border-[var(--border)] shrink-0">
        <div className="w-9 h-9 rounded-xl gradient-accent flex items-center justify-center text-white font-bold text-sm shrink-0">AI</div>
        {!collapsed && <span className="text-base font-bold gradient-text truncate">DataAnalysis</span>}
      </div>

      {/* Workspace Indicator */}
      {!collapsed && activeWorkspace && (
        <div className="mx-3 mt-3 px-3 py-2 rounded-lg bg-[var(--accent-bg)] border border-[var(--accent)]/20">
          <p className="text-[10px] uppercase tracking-wider text-[var(--accent)] font-semibold">Workspace</p>
          <p className="text-xs text-[var(--text-primary)] truncate mt-0.5">{activeWorkspace.name}</p>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => cn(
              'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150',
              isActive
                ? 'bg-[var(--sidebar-active)] text-[var(--accent)] border-l-2 border-[var(--accent)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--sidebar-hover)]',
              collapsed && 'justify-center px-2'
            )}
          >
            <item.icon className="w-[18px] h-[18px] shrink-0" />
            {!collapsed && <span className="truncate">{item.label}</span>}
          </NavLink>
        ))}

        {!collapsed && (
          <div className="mt-4 pt-4 border-t border-[var(--border)]">
            <p className="px-3 mb-2 text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">Workflow Progress</p>
            <div className="px-2">
              {flowRoutes.map((route, idx) => {
                const nav = navItems.find((n) => n.to === route);
                if (!nav) return null;
                const isCompleted = currentFlowIdx > idx;
                const isCurrent = currentFlowIdx === idx;
                const isUpcoming = currentFlowIdx < idx;

                return (
                  <div key={`flow-${route}`} className="relative pl-8 py-1.5">
                    {idx < flowRoutes.length - 1 && (
                      <span className={cn(
                        'absolute left-[15px] top-6 w-[2px] h-6 rounded-full',
                        isCompleted ? 'bg-[var(--accent)]/55' : 'bg-[var(--border)]'
                      )} />
                    )}
                    <span className={cn(
                      'absolute left-0 top-2 w-7 h-7 rounded-full border flex items-center justify-center text-[11px]',
                      isCompleted && 'bg-[var(--accent)] text-white border-[var(--accent)]',
                      isCurrent && 'bg-[var(--accent-bg)] text-[var(--accent)] border-[var(--accent)] shadow-[0_0_0_3px_rgba(79,70,229,0.14)]',
                      isUpcoming && 'bg-[var(--bg-secondary)] text-[var(--text-muted)] border-[var(--border)]'
                    )}>
                      {isCompleted ? <Check className="w-3.5 h-3.5" /> : idx + 1}
                    </span>
                    <p className={cn(
                      'text-xs leading-5',
                      isCurrent ? 'text-[var(--text-primary)] font-semibold' : isUpcoming ? 'text-[var(--text-muted)]' : 'text-[var(--text-secondary)]'
                    )}>
                      {nav.label}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </nav>

      {/* Footer */}
      <div className="border-t border-[var(--border)] p-3 space-y-2">
        {!collapsed && user && (
          <div className="flex items-center gap-3 px-2 py-2">
            <div className="w-8 h-8 rounded-full gradient-accent flex items-center justify-center text-white text-xs font-bold">
              {user.full_name?.charAt(0) || 'U'}
            </div>
            <div className="min-w-0">
              <p className="text-sm font-medium truncate">{user.full_name}</p>
              <p className="text-[11px] text-[var(--text-muted)] truncate">{user.email}</p>
            </div>
          </div>
        )}
        <div className="flex items-center gap-1">
          <button onClick={handleLogout} className={cn('flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-[var(--text-muted)] hover:text-[var(--error)] hover:bg-[var(--error-bg)] transition-all cursor-pointer', collapsed && 'justify-center w-full')}>
            <LogOut className="w-4 h-4" />
            {!collapsed && <span>Logout</span>}
          </button>
          <button onClick={() => setCollapsed(!collapsed)} className="ml-auto p-2 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--sidebar-hover)] transition cursor-pointer">
            {collapsed ? <PanelLeft className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </aside>
  );
}
