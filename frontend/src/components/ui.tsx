import type { ReactNode, ButtonHTMLAttributes, InputHTMLAttributes, TextareaHTMLAttributes } from 'react';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

/* ── Button ─────────────────────────────────────────────────── */
interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: ReactNode;
}

export function Button({ variant = 'primary', size = 'md', loading, icon, className, children, disabled, ...props }: ButtonProps) {
  const base = 'inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer';
  const variants = {
    primary: 'bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white focus:ring-[var(--accent)]',
    secondary: 'bg-[var(--bg-card)] border border-[var(--border)] hover:border-[var(--border-hover)] text-[var(--text-primary)] focus:ring-[var(--border-hover)]',
    ghost: 'hover:bg-[var(--bg-card)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]',
    danger: 'bg-[var(--error)] hover:bg-red-600 text-white focus:ring-[var(--error)]',
  };
  const sizes = { sm: 'px-3 py-1.5 text-xs', md: 'px-4 py-2 text-sm', lg: 'px-6 py-3 text-base' };

  return (
    <button className={cn(base, variants[variant], sizes[size], className)} disabled={disabled || loading} {...props}>
      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

/* ── Input ──────────────────────────────────────────────────── */
interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'onChange'> {
  label?: string;
  error?: string;
  onChange?: ((v: string) => void) | React.ChangeEventHandler<HTMLInputElement>;
}

export function Input({ label, error, className, onChange, ...props }: InputProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!onChange) return;
    if (onChange.length === 1) {
      try { (onChange as (v: string) => void)(e.target.value); } catch { (onChange as React.ChangeEventHandler<HTMLInputElement>)(e); }
    } else { (onChange as React.ChangeEventHandler<HTMLInputElement>)(e); }
  };
  return (
    <div className="space-y-1.5">
      {label && <label className="block text-sm font-medium text-[var(--text-secondary)]">{label}</label>}
      <input
        className={cn(
          'w-full px-3.5 py-2.5 rounded-lg bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all text-sm',
          error && 'border-[var(--error)] focus:ring-[var(--error)]',
          className
        )}
        onChange={handleChange}
        {...props}
      />
      {error && <p className="text-xs text-[var(--error)]">{error}</p>}
    </div>
  );
}

/* ── Textarea ───────────────────────────────────────────────── */
interface TextareaProps extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'onChange'> {
  label?: string;
  onChange?: ((v: string) => void) | React.ChangeEventHandler<HTMLTextAreaElement>;
}

export function Textarea({ label, className, onChange, ...props }: TextareaProps) {
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (!onChange) return;
    try { (onChange as (v: string) => void)(e.target.value); } catch { (onChange as React.ChangeEventHandler<HTMLTextAreaElement>)(e); }
  };
  return (
    <div className="space-y-1.5">
      {label && <label className="block text-sm font-medium text-[var(--text-secondary)]">{label}</label>}
      <textarea
        className={cn('w-full px-3.5 py-2.5 rounded-lg bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all text-sm resize-y min-h-[80px]', className)}
        onChange={handleChange}
        {...props}
      />
    </div>
  );
}

/* ── Select ─────────────────────────────────────────────────── */
interface SelectProps {
  label?: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  className?: string;
  placeholder?: string;
}

export function Select({ label, value, onChange, options, className, placeholder }: SelectProps) {
  return (
    <div className="space-y-1.5">
      {label && <label className="block text-sm font-medium text-[var(--text-secondary)]">{label}</label>}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={cn('w-full px-3.5 py-2.5 rounded-lg bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent text-sm appearance-none cursor-pointer', className)}
      >
        {placeholder && <option value="" disabled>{placeholder}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

/* ── Card ───────────────────────────────────────────────────── */
interface CardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
}

export function Card({ children, className, hover, onClick }: CardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        'rounded-xl bg-[var(--bg-card)] border border-[var(--border)] p-5 shadow-[var(--shadow-sm)]',
        hover && 'hover:border-[var(--border-hover)] hover:bg-[var(--bg-card-hover)] transition-all duration-200 cursor-pointer',
        className
      )}
    >
      {children}
    </div>
  );
}

/* ── Badge ──────────────────────────────────────────────────── */
interface BadgeProps {
  children: ReactNode;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'danger' | 'info';
  className?: string;
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  const variants: Record<string, string> = {
    default: 'bg-[var(--accent-bg)] text-[var(--accent)]',
    primary: 'bg-[var(--accent-bg)] text-[var(--accent)]',
    success: 'bg-[var(--success-bg)] text-[var(--success)]',
    warning: 'bg-[var(--warning-bg)] text-[var(--warning)]',
    error: 'bg-[var(--error-bg)] text-[var(--error)]',
    danger: 'bg-[var(--error-bg)] text-[var(--error)]',
    info: 'bg-[var(--info-bg)] text-[var(--info)]',
  };
  return <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', variants[variant], className)}>{children}</span>;
}

/* ── Spinner ────────────────────────────────────────────────── */
export function Spinner({ size = 'md', className }: { size?: 'sm' | 'md' | 'lg'; className?: string }) {
  const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-8 h-8' };
  return <Loader2 className={cn(sizes[size], 'animate-spin text-[var(--accent)]', className)} />;
}

/* ── Empty State ────────────────────────────────────────────── */
export function EmptyState({ icon, title, description, action, children }: { icon: ReactNode; title: string; description: string; action?: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center animate-fade-in">
      <div className="w-16 h-16 rounded-2xl bg-[var(--accent-bg)] flex items-center justify-center mb-4 text-[var(--accent)]">{icon}</div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm text-[var(--text-secondary)] max-w-sm mb-6">{description}</p>
      {action || children}
    </div>
  );
}

/* ── Progress Bar ───────────────────────────────────────────── */
export function ProgressBar({ value, max = 100, color = 'var(--accent)' }: { value: number; max?: number; color?: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-2 rounded-full bg-[var(--bg-input)] overflow-hidden">
      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

/* ── KPI Card ───────────────────────────────────────────────── */
/* ── KpiCard ────────────────────────────────────────────── */
const KPI_COLORS = [
  { bg: 'rgba(79, 70, 229, 0.08)', border: 'rgba(79, 70, 229, 0.18)', text: '#4F46E5', icon: '#6366F1' },
  { bg: 'rgba(16, 185, 129, 0.08)', border: 'rgba(16, 185, 129, 0.18)', text: '#059669', icon: '#10B981' },
  { bg: 'rgba(245, 158, 11, 0.08)', border: 'rgba(245, 158, 11, 0.18)', text: '#D97706', icon: '#F59E0B' },
  { bg: 'rgba(139, 92, 246, 0.08)', border: 'rgba(139, 92, 246, 0.18)', text: '#7C3AED', icon: '#8B5CF6' },
  { bg: 'rgba(236, 72, 153, 0.08)', border: 'rgba(236, 72, 153, 0.18)', text: '#DB2777', icon: '#EC4899' },
  { bg: 'rgba(14, 165, 233, 0.08)', border: 'rgba(14, 165, 233, 0.18)', text: '#0284C7', icon: '#0EA5E9' },
];
function hashColor(s: string) { let h = 0; for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0; return KPI_COLORS[Math.abs(h) % KPI_COLORS.length]; }

export function KpiCard({ label, value, icon, trend, trendUp, suffix, colorIndex }: { label: string; value: string; icon?: ReactNode; trend?: number | string; trendUp?: boolean; suffix?: string; colorIndex?: number }) {
  const trendNum = typeof trend === 'number' ? trend : parseFloat(trend || '');
  const isUp = trendUp ?? (trendNum > 0);
  const color = colorIndex != null ? KPI_COLORS[colorIndex % KPI_COLORS.length] : hashColor(label);
  return (
    <div
      className="group relative rounded-2xl p-4 transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 cursor-default"
      style={{
        background: color.bg,
        border: `1px solid ${color.border}`,
      }}
    >
      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: `linear-gradient(135deg, ${color.bg}, transparent)` }} />

      <div className="relative flex items-start gap-3">
        {icon && (
          <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0" style={{ background: color.border, color: color.icon }}>
            {icon}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium uppercase tracking-wider truncate" style={{ color: color.text, opacity: 0.7 }}>{label}</p>
          <div className="flex items-baseline gap-1.5 mt-1">
            <p className="text-xl font-bold tracking-tight" style={{ color: color.text }}>{value}</p>
            {suffix && <span className="text-xs font-medium px-1.5 py-0.5 rounded-full" style={{ background: color.border, color: color.text }}>{suffix}</span>}
          </div>
          {trend != null && !isNaN(trendNum) && (
            <div className={cn('inline-flex items-center gap-1 mt-1.5 text-xs font-semibold px-2 py-0.5 rounded-full', isUp ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700')}>
              {isUp ? '↑' : '↓'} {typeof trend === 'number' ? `${Math.abs(trend)}%` : trend}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Modal ──────────────────────────────────────────────────── */
interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
}

export function Modal({ open, onClose, title, children, footer }: ModalProps) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-lg mx-4 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] shadow-[var(--shadow-lg)] animate-fade-in">
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
          <h3 className="text-lg font-semibold">{title}</h3>
          <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition cursor-pointer">✕</button>
        </div>
        <div className="px-6 py-5">{children}</div>
        {footer && <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end gap-3">{footer}</div>}
      </div>
    </div>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
interface TabsProps {
  tabs: { key: string; label: string; icon?: ReactNode }[];
  active: string;
  onChange: (key: string) => void;
}

export function Tabs({ tabs, active, onChange }: TabsProps) {
  return (
    <div className="flex gap-1 p-1 rounded-xl bg-[var(--bg-input)] border border-[var(--border)]">
      {tabs.map((t) => (
        <button
          key={t.key}
          onClick={() => onChange(t.key)}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer',
            active === t.key
              ? 'bg-[var(--accent)] text-white shadow-lg'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
          )}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* ── DataTable ──────────────────────────────────────────────── */
interface DataTableProps {
  columns: string[];
  rows: Record<string, any>[];
  maxRows?: number;
}

export function DataTable({ columns, rows, maxRows = 50 }: DataTableProps) {
  const display = rows.slice(0, maxRows);
  return (
    <div className="overflow-auto rounded-lg border border-[var(--border)]">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-[var(--bg-input)]">
            {columns.map((c) => (
              <th key={c} className="px-4 py-3 text-left font-medium text-[var(--text-secondary)] whitespace-nowrap border-b border-[var(--border)]">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {display.map((row, i) => (
            <tr key={i} className="border-b border-[var(--border)] hover:bg-[var(--bg-card-hover)] transition-colors">
              {columns.map((c) => (
                <td key={c} className="px-4 py-2.5 whitespace-nowrap text-[var(--text-primary)]">{String(row[c] ?? '—')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > maxRows && (
        <p className="text-center py-2 text-xs text-[var(--text-muted)]">Showing {maxRows} of {rows.length} rows</p>
      )}
    </div>
  );
}
