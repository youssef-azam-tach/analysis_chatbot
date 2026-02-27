import { useState, useRef, useEffect, type ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { ChevronLeft, ChevronRight, ChevronDown, Search, Database } from 'lucide-react';

/* ── Types ──────────────────────────────────────────────── */
export interface DatasetSelectorProps {
  /** List of dataset keys */
  datasets: string[];
  /** Currently selected key */
  activeDataset: string;
  /** Callback when selection changes */
  onSelect: (key: string) => void;
  /** Optional extra content rendered beside each option (e.g. quality score) */
  renderBadge?: (key: string) => ReactNode;
  /** Optional display name formatter (default: replace '::' with ' → ') */
  formatName?: (key: string) => string;
  /** Extra wrapper className */
  className?: string;
}

const defaultFormat = (key: string) =>
  key.length > 40 ? key.slice(0, 40) + '…' : key.replace('::', ' → ');

/* ── Component ──────────────────────────────────────────── */
export default function DatasetSelector({
  datasets,
  activeDataset,
  onSelect,
  renderBadge,
  formatName = defaultFormat,
  className,
}: DatasetSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  /* close on outside click */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        setSearch('');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  /* focus search when opened */
  useEffect(() => {
    if (open) searchRef.current?.focus();
  }, [open]);

  /* keyboard nav */
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setOpen(false); setSearch(''); }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open]);

  const currentIdx = datasets.indexOf(activeDataset);
  const filtered = search.trim()
    ? datasets.filter((d) => d.toLowerCase().includes(search.toLowerCase()))
    : datasets;

  const goPrev = () => {
    if (datasets.length === 0) return;
    const idx = currentIdx <= 0 ? datasets.length - 1 : currentIdx - 1;
    onSelect(datasets[idx]);
  };

  const goNext = () => {
    if (datasets.length === 0) return;
    const idx = currentIdx >= datasets.length - 1 ? 0 : currentIdx + 1;
    onSelect(datasets[idx]);
  };

  const select = (key: string) => {
    onSelect(key);
    setOpen(false);
    setSearch('');
  };

  if (datasets.length === 0) return null;

  return (
    <div className={cn('flex items-center gap-2', className)} ref={containerRef}>
      {/* ← Prev */}
      <button
        onClick={goPrev}
        disabled={datasets.length <= 1}
        aria-label="Previous dataset"
        className={cn(
          'flex items-center justify-center w-8 h-8 rounded-lg transition cursor-pointer',
          'bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-secondary)]',
          'hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)]',
          'disabled:opacity-40 disabled:cursor-not-allowed',
        )}
      >
        <ChevronLeft className="w-4 h-4" />
      </button>

      {/* Dropdown trigger */}
      <div className="relative flex-1 min-w-0 max-w-xs">
        <button
          onClick={() => setOpen((prev) => !prev)}
          className={cn(
            'flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm font-medium transition cursor-pointer',
            'bg-[var(--bg-card)] border border-[var(--border)]',
            'hover:border-[var(--accent)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]',
            open && 'border-[var(--accent)] ring-2 ring-[var(--accent)]',
          )}
        >
          <Database className="w-4 h-4 text-[var(--accent)] shrink-0" />
          <span className="truncate flex-1 text-left text-[var(--text-primary)]">
            {activeDataset ? formatName(activeDataset) : 'Select dataset'}
          </span>
          {renderBadge && activeDataset && renderBadge(activeDataset)}
          <ChevronDown className={cn('w-4 h-4 text-[var(--text-muted)] shrink-0 transition-transform', open && 'rotate-180')} />
        </button>

        {/* Dropdown */}
        {open && (
          <div className="absolute z-50 mt-1 w-full min-w-[260px] rounded-lg bg-[var(--bg-card)] border border-[var(--border)] shadow-lg overflow-hidden animate-fade-in">
            {/* Search */}
            {datasets.length > 5 && (
              <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--border)]">
                <Search className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                <input
                  ref={searchRef}
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search datasets…"
                  className="w-full bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none"
                />
              </div>
            )}

            {/* Options */}
            <div className="max-h-60 overflow-y-auto py-1">
              {filtered.length === 0 ? (
                <div className="px-3 py-4 text-center text-xs text-[var(--text-muted)]">No datasets match "{search}"</div>
              ) : (
                filtered.map((key) => (
                  <button
                    key={key}
                    onClick={() => select(key)}
                    className={cn(
                      'flex items-center gap-2 w-full px-3 py-2 text-sm text-left transition cursor-pointer',
                      activeDataset === key
                        ? 'bg-[var(--accent)] text-white'
                        : 'text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]',
                    )}
                  >
                    <Database className={cn('w-3.5 h-3.5 shrink-0', activeDataset === key ? 'text-white/70' : 'text-[var(--text-muted)]')} />
                    <span className="truncate flex-1">{formatName(key)}</span>
                    {renderBadge && renderBadge(key)}
                  </button>
                ))
              )}
            </div>

            {/* Footer count */}
            <div className="px-3 py-1.5 border-t border-[var(--border)] text-[10px] text-[var(--text-muted)]">
              {datasets.length} dataset{datasets.length !== 1 ? 's' : ''}{' '}
              {currentIdx >= 0 && <>· #{currentIdx + 1} selected</>}
            </div>
          </div>
        )}
      </div>

      {/* → Next */}
      <button
        onClick={goNext}
        disabled={datasets.length <= 1}
        aria-label="Next dataset"
        className={cn(
          'flex items-center justify-center w-8 h-8 rounded-lg transition cursor-pointer',
          'bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-secondary)]',
          'hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)]',
          'disabled:opacity-40 disabled:cursor-not-allowed',
        )}
      >
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  );
}
