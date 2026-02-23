import { useState, useEffect, useRef, useCallback } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { useAuthStore } from '@/stores/auth-store';
import { engine } from '@/lib/api';
import { loadPageState, savePageState } from '@/lib/pagePersistence';
import { Button, Card, Badge, Spinner, EmptyState, Select } from '@/components/ui';
import {
  Database, Columns3, ArrowRight, Link as LinkIcon,
  Lightbulb, Combine, Trash2, GitFork,
  ChevronDown, ChevronUp, LayoutGrid, Rows3,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import mermaid from 'mermaid';
import toast from 'react-hot-toast';

/* ── Mermaid init ── */
mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  themeVariables: {
    darkMode: false,
    primaryColor: '#4F46E5',
    primaryTextColor: '#1F2937',
    primaryBorderColor: '#4F46E5',
    lineColor: '#9CA3AF',
    secondaryColor: '#F3F4F6',
    tertiaryColor: '#F9FAFB',
    background: '#FFFFFF',
    mainBkg: '#F3F4F6',
    nodeBorder: '#4F46E5',
    clusterBkg: '#F3F4F6',
    titleColor: '#1F2937',
    edgeLabelBackground: '#FFFFFF',
    fontSize: '13px',
  },
  er: { useMaxWidth: true, layoutDirection: 'TB' },
});

/* ── Types matching the actual API response ── */
interface DatasetInfo {
  shape: [number, number];
  columns: string[];
}

interface Relationship {
  file1: string;
  file2: string;
  column1: string;
  column2: string;
  relationship_type: string;
  confidence: string;
  is_manual?: boolean;
  source?: string;
  match_percentage?: number;
  overlap_count?: number;
  reasoning?: string;
  recommendation?: string;
  is_primary_key?: boolean;
  is_foreign_key?: boolean;
}

interface RejectedRelationship {
  file1: string;
  file2: string;
  column1: string;
  column2: string;
  overlap_count?: number;
  child_to_parent_coverage?: number;
  reason: string;
}

interface SuspiciousColumn {
  table: string;
  column: string;
  possible_reference?: string | null;
  coverage?: number;
  reason: string;
}

interface JunctionTable {
  table: string;
  fk_count: number;
  parents: string[];
  fk_columns: string[];
  extra_columns: string[];
  reason: string;
}

interface FinalErdRelationship {
  child: string;
  parent: string;
  child_fk: string;
  parent_pk: string;
  cardinality: string;
  engine_valid: boolean;
}

interface FinalErdStructure {
  entities?: { name: string; rows: number; columns: string[] }[];
  relationships?: FinalErdRelationship[];
}

interface JoinSuggestion {
  left_dataset: string;
  right_dataset: string;
  left_on: string;
  right_on: string;
  how: string;
  confidence: string;
  reasoning?: string;
}

interface SchemaResult {
  datasets: Record<string, DatasetInfo>;
  relationships: Relationship[];
  confirmed_relationships?: Relationship[];
  manual_relationships?: Relationship[];
  rejected_relationships?: RejectedRelationship[];
  suspicious_columns?: SuspiciousColumn[];
  possible_junction_tables?: JunctionTable[];
  final_erd_structure?: FinalErdStructure;
  data_modeling_observations?: string[];
  erd_summary?: string;
  unified_view: string;
  join_suggestions: JoinSuggestion[];
}

function normalizeMarkdownContent(input?: string): string {
  if (!input) return '';

  let text = String(input).trim();

  if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
    try {
      const parsed = JSON.parse(text);
      if (typeof parsed === 'string') text = parsed;
    } catch {
      // keep original text
    }
  }

  text = text.replace(/^```(?:markdown|md)?\s*/i, '').replace(/\s*```\s*$/i, '');
  text = text.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
  text = text.replace(/\\([#*_`>\-])/g, '$1');

  return text.trim();
}

/* ── ERD Mermaid Component ── */
function ErdDiagram({ datasets, relationships }: { datasets: Record<string, DatasetInfo>; relationships: Relationship[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const renderDiagram = async () => {
      try {
        // Sanitize name for Mermaid (replace dots, spaces, special chars)
        const safe = (s: string) => s.replace(/[^a-zA-Z0-9_]/g, '_');

        let code = 'erDiagram\n';

        // Add entities with their columns
        for (const [name, ds] of Object.entries(datasets)) {
          const sName = safe(name);
          code += `    ${sName} {\n`;
          for (const col of ds.columns.slice(0, 15)) {  // limit to 15 cols for readability
            const safeCol = safe(col);
            code += `        string ${safeCol}\n`;
          }
          if (ds.columns.length > 15) {
            code += `        string ___${ds.columns.length - 15}_more___\n`;
          }
          code += '    }\n';
        }

        // Add relationships
        for (const rel of relationships) {
          const from = safe(rel.file1);
          const to = safe(rel.file2);
          let arrow = '||--o{';
          if (rel.relationship_type === 'one-to-one') arrow = '||--||';
          else if (rel.relationship_type === 'many-to-many') arrow = '}o--o{';
          else if (rel.relationship_type === 'many-to-one') arrow = '}o--||';
          const label = `${rel.column1} - ${rel.column2}`;
          code += `    ${from} ${arrow} ${to} : "${label}"\n`;
        }

        const id = 'erd-' + Date.now();
        const { svg: renderedSvg } = await mermaid.render(id, code);
        setSvg(renderedSvg);
        setError('');
      } catch (e: any) {
        setError(e.message || 'Failed to render ERD');
        setSvg('');
      }
    };

    if (Object.keys(datasets).length > 0) renderDiagram();
  }, [datasets, relationships]);

  if (error) {
    return (
      <div className="p-6 rounded-xl bg-[var(--error-bg)] border border-[var(--error)]/20 text-sm text-[var(--error)]">
        ERD rendering error: {error}
      </div>
    );
  }

  if (!svg) {
    return <div className="flex items-center justify-center py-12"><Spinner size="lg" /></div>;
  }

  return (
    <div
      ref={containerRef}
      className="w-full overflow-auto rounded-xl bg-[var(--bg-card)] border border-[var(--border)] p-4 shadow-[var(--shadow-sm)]"
      style={{ maxHeight: '600px' }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

/* ── Main Page ── */
type TabKey = 'erd' | 'overview' | 'relationships' | 'joins' | 'unified';

export default function SchemaPage() {
  const { sessionId, activeWorkspace } = useWorkspaceStore();
  const { user } = useAuthStore();
  const [schema, setSchema] = useState<SchemaResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [savingManual, setSavingManual] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>('erd');
  const [expandedRel, setExpandedRel] = useState<number | null>(null);
  const [deletingIdx, setDeletingIdx] = useState<number | null>(null);
  const [manualRelation, setManualRelation] = useState({
    file1: '',
    column1: '',
    file2: '',
    column2: '',
    relationship_type: 'one-to-many',
  });
  const [cacheHydrated, setCacheHydrated] = useState(false);

  useEffect(() => {
    const cached = loadPageState<any>('schema-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    });
    if (cached) {
      setSchema(cached.schema || null);
      setActiveTab(cached.activeTab || 'erd');
      setExpandedRel(cached.expandedRel ?? null);
      setManualRelation(cached.manualRelation || {
        file1: '',
        column1: '',
        file2: '',
        column2: '',
        relationship_type: 'one-to-many',
      });
    }
    setCacheHydrated(true);
  }, [user?.id, activeWorkspace?.id, sessionId]);

  useEffect(() => {
    if (!cacheHydrated) return;
    savePageState('schema-page', {
      userId: user?.id,
      workspaceId: activeWorkspace?.id,
      sessionId,
    }, {
      schema,
      activeTab,
      expandedRel,
      manualRelation,
    });
  }, [cacheHydrated, user?.id, activeWorkspace?.id, sessionId, schema, activeTab, expandedRel, manualRelation]);

  const analyzeSchema = async () => {
    if (!sessionId) return toast.error('Upload data first');
    setLoading(true);
    try {
      const { data } = await engine.post('/analysis/schema', { session_id: sessionId });
      const raw: SchemaResult = data.data;

      const confirmedRels = raw.confirmed_relationships || raw.relationships || [];

      // Also remove join suggestions whose relationship was filtered out
      const relKey = (f1: string, c1: string, f2: string, c2: string) =>
        `${f1}|${c1}|${f2}|${c2}`;
      const kept = new Set(confirmedRels.map((r) => relKey(r.file1, r.column1, r.file2, r.column2)));
      const strongJoins = raw.join_suggestions.filter((js) =>
        kept.has(relKey(js.left_dataset, js.left_on, js.right_dataset, js.right_on)),
      );

      setSchema({
        ...raw,
        relationships: confirmedRels,
        confirmed_relationships: confirmedRels,
        join_suggestions: strongJoins,
      });
      setActiveTab('erd');
      toast.success('Schema analysis complete');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Schema analysis failed');
    } finally {
      setLoading(false);
    }
  };

  /* ── Delete a relationship (local only) ── */
  const deleteRelationship = useCallback(async (idx: number) => {
    if (!schema) return;
    const rel = schema.relationships[idx];
    if (!rel) return;

    setDeletingIdx(idx);
    try {
      if (rel.is_manual && sessionId) {
        await engine.post('/analysis/schema/relationships/remove', {
          session_id: sessionId,
          file1: rel.file1,
          column1: rel.column1,
          file2: rel.file2,
          column2: rel.column2,
        });
      }

      setSchema((prev) => {
        if (!prev) return prev;
        const updated = [...prev.relationships];
        const removed = updated[idx];
        updated.splice(idx, 1);

        const updatedJoins = prev.join_suggestions.filter(
          (js) => !(
            js.left_on === removed.column1 &&
            js.right_on === removed.column2 &&
            js.left_dataset === removed.file1 &&
            js.right_dataset === removed.file2
          ),
        );

        const confirmed = (prev.confirmed_relationships || prev.relationships).filter((_, i) => i !== idx);
        const manual = (prev.manual_relationships || []).filter(
          (m) => !(
            m.file1 === removed.file1 &&
            m.column1 === removed.column1 &&
            m.file2 === removed.file2 &&
            m.column2 === removed.column2
          ),
        );

        return {
          ...prev,
          relationships: updated,
          confirmed_relationships: confirmed,
          manual_relationships: manual,
          join_suggestions: updatedJoins,
        };
      });
      toast.success('Relationship removed');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to remove relationship');
    } finally {
      setDeletingIdx(null);
    }
  }, [schema, sessionId]);

  const addManualRelationship = useCallback(async () => {
    if (!sessionId || !schema) return;
    const { file1, column1, file2, column2, relationship_type } = manualRelation;
    if (!file1 || !column1 || !file2 || !column2) {
      toast.error('Select source/target table and column first');
      return;
    }

    setSavingManual(true);
    try {
      const { data } = await engine.post('/analysis/schema/relationships', {
        session_id: sessionId,
        file1,
        column1,
        file2,
        column2,
        relationship_type,
      });

      const added: Relationship = data?.data?.relationship;
      if (!added) {
        throw new Error('Invalid relationship response');
      }

      setSchema((prev) => {
        if (!prev) return prev;
        const same = (r: Relationship) =>
          r.file1 === added.file1 &&
          r.column1 === added.column1 &&
          r.file2 === added.file2 &&
          r.column2 === added.column2;

        const withoutDup = prev.relationships.filter((r) => !same(r));
        const updatedRelationships = [...withoutDup, added];

        const confirmedBase = prev.confirmed_relationships || prev.relationships;
        const updatedConfirmed = [...confirmedBase.filter((r) => !same(r)), added];

        const manualBase = prev.manual_relationships || [];
        const updatedManual = [...manualBase.filter((r) => !same(r)), added];

        const joinExists = prev.join_suggestions.some(
          (js) =>
            js.left_dataset === added.file1 &&
            js.left_on === added.column1 &&
            js.right_dataset === added.file2 &&
            js.right_on === added.column2,
        );

        const updatedJoins = joinExists
          ? prev.join_suggestions
          : [
              ...prev.join_suggestions,
              {
                left_dataset: added.file1,
                left_on: added.column1,
                right_dataset: added.file2,
                right_on: added.column2,
                how: 'left',
                confidence: 'high',
                reasoning: 'User-defined relationship',
              },
            ];

        return {
          ...prev,
          relationships: updatedRelationships,
          confirmed_relationships: updatedConfirmed,
          manual_relationships: updatedManual,
          join_suggestions: updatedJoins,
        };
      });

      toast.success('Manual relationship added');
      setManualRelation((prev) => ({ ...prev, column1: '', column2: '' }));
    } catch (err: any) {
      toast.error(err.response?.data?.detail || err.message || 'Failed to add relationship');
    } finally {
      setSavingManual(false);
    }
  }, [manualRelation, schema, sessionId]);

  const datasetEntries = schema ? Object.entries(schema.datasets) : [];
  const datasetOptions = datasetEntries.map(([name]) => ({ value: name, label: name }));
  const sourceColumnOptions = manualRelation.file1 && schema?.datasets[manualRelation.file1]
    ? schema.datasets[manualRelation.file1].columns.map((c) => ({ value: c, label: c }))
    : [];
  const targetColumnOptions = manualRelation.file2 && schema?.datasets[manualRelation.file2]
    ? schema.datasets[manualRelation.file2].columns.map((c) => ({ value: c, label: c }))
    : [];

  const totalRows = datasetEntries.reduce((sum, [, d]) => sum + d.shape[0], 0);
  const totalCols = datasetEntries.reduce((sum, [, d]) => sum + d.shape[1], 0);

  const confColor = (c?: string) => {
    if (c === 'high') return 'success';
    if (c === 'medium') return 'warning';
    return 'danger';
  };

  const relTypeLabel = (t: string) => {
    const map: Record<string, string> = {
      'one-to-one': '1 : 1',
      'one-to-many': '1 : N',
      'many-to-one': 'N : 1',
      'many-to-many': 'M : N',
    };
    return map[t] || t;
  };

  const tabs: { key: TabKey; label: string; icon: React.ReactNode }[] = [
    { key: 'erd', label: 'ERD Diagram', icon: <GitFork className="w-4 h-4" /> },
    { key: 'overview', label: `Datasets (${datasetEntries.length})`, icon: <LayoutGrid className="w-4 h-4" /> },
    { key: 'relationships', label: `Relations (${schema?.confirmed_relationships?.length || schema?.relationships?.length || 0})`, icon: <LinkIcon className="w-4 h-4" /> },
    { key: 'joins', label: `Joins (${schema?.join_suggestions?.length || 0})`, icon: <Combine className="w-4 h-4" /> },
    { key: 'unified', label: 'AI Insights', icon: <Lightbulb className="w-4 h-4" /> },
  ];

  return (
    <div className="animate-fade-in min-h-0 flex flex-col gap-6">
      {/* ── Header ── */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-2xl font-bold">Schema Analysis</h1>
          <p className="text-[var(--text-secondary)] mt-1 text-sm">Understand your data structure, types, and relationships</p>
        </div>
        <Button onClick={analyzeSchema} loading={loading} icon={<Database className="w-4 h-4" />}>
          {schema ? 'Re-Analyze' : 'Analyze Schema'}
        </Button>
      </div>

      {loading && <div className="flex items-center justify-center py-20"><Spinner size="lg" /></div>}

      {!loading && !schema && (
        <EmptyState icon={<Database className="w-8 h-8" />} title="No schema analysis yet" description="Click 'Analyze Schema' to inspect your data structure">
          <Button onClick={analyzeSchema} className="mt-4">Run Analysis</Button>
        </EmptyState>
      )}

      {schema && !loading && (
        <>
          {/* ── Stat Cards ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 flex-shrink-0">
            {[
              { label: 'Datasets', value: datasetEntries.length, color: 'var(--accent)', bg: 'var(--accent-bg)', Icon: Database },
              { label: 'Columns', value: totalCols, color: 'var(--success)', bg: 'var(--success-bg)', Icon: Columns3 },
              { label: 'Rows', value: totalRows.toLocaleString(), color: 'var(--info)', bg: 'var(--info-bg)', Icon: Rows3 },
              { label: 'Relationships', value: schema.confirmed_relationships?.length || schema.relationships?.length || 0, color: '#7C3AED', bg: 'rgba(124,58,237,0.08)', Icon: LinkIcon },
            ].map(({ label, value, color, bg, Icon }) => (
              <div key={label} className="flex items-center gap-3 px-4 py-3 rounded-xl bg-[var(--bg-card)] border border-[var(--border)]">
                <div className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0" style={{ background: bg }}>
                  <Icon className="w-4 h-4" style={{ color }} />
                </div>
                <div>
                  <p className="text-xl font-bold leading-tight">{value}</p>
                  <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">{label}</p>
                </div>
              </div>
            ))}
          </div>

          {/* ── Tab Bar ── */}
          <div className="flex gap-1 p-1 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border)] flex-shrink-0 overflow-x-auto scrollbar-thin">
            {tabs.map((t) => (
              <button
                key={t.key}
                onClick={() => setActiveTab(t.key)}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition whitespace-nowrap cursor-pointer ${
                  activeTab === t.key
                    ? 'bg-[var(--accent)] text-white shadow-sm'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]'
                }`}
              >
                {t.icon}{t.label}
              </button>
            ))}
          </div>

          {/* ── Tab Content ── */}
          <div className="flex-1 min-h-0 overflow-y-auto pb-6">

            {/* ERD Diagram */}
            {activeTab === 'erd' && (
              <div className="space-y-4">
                {schema.erd_summary && (
                  <Card>
                    <p className="text-sm text-[var(--text-secondary)]">{schema.erd_summary}</p>
                  </Card>
                )}
                <Card>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold flex items-center gap-2 text-sm">
                      <GitFork className="w-4 h-4 text-[var(--accent)]" /> Entity-Relationship Diagram
                    </h3>
                    <span className="text-[10px] text-[var(--text-muted)]">Scroll to pan &bull; Generated from detected relationships</span>
                  </div>
                  <ErdDiagram datasets={schema.datasets} relationships={schema.relationships} />
                </Card>
              </div>
            )}

            {/* Datasets */}
            {activeTab === 'overview' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {datasetEntries.map(([name, ds]) => (
                  <Card key={name}>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-sm truncate max-w-[75%]" title={name}>
                        <Database className="w-3.5 h-3.5 inline-block mr-1.5 text-[var(--accent)] -mt-0.5" />
                        {name}
                      </h4>
                      <span className="text-[10px] text-[var(--text-muted)] shrink-0">
                        {ds.shape[0].toLocaleString()} &times; {ds.shape[1]}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1.5 max-h-[140px] overflow-y-auto scrollbar-thin pr-1">
                      {ds.columns.map((col) => (
                        <span key={col} className="inline-block px-2 py-0.5 rounded text-[11px] bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-secondary)] font-mono">
                          {col}
                        </span>
                      ))}
                    </div>
                  </Card>
                ))}
              </div>
            )}

            {/* Relationships */}
            {activeTab === 'relationships' && (
              <div className="space-y-2">
                <Card>
                  <div className="flex items-center justify-between gap-2 mb-4">
                    <h4 className="font-semibold text-sm">Add Relationship</h4>
                    <span className="text-[11px] text-[var(--text-muted)]">This relation will be used in next analysis steps</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    <Select
                      label="Source Table"
                      value={manualRelation.file1}
                      onChange={(value) => setManualRelation((prev) => ({ ...prev, file1: value, column1: '' }))}
                      options={datasetOptions}
                      placeholder="Choose table"
                    />
                    <Select
                      label="Source Column"
                      value={manualRelation.column1}
                      onChange={(value) => setManualRelation((prev) => ({ ...prev, column1: value }))}
                      options={sourceColumnOptions}
                      placeholder="Choose column"
                    />
                    <Select
                      label="Relationship Type"
                      value={manualRelation.relationship_type}
                      onChange={(value) => setManualRelation((prev) => ({ ...prev, relationship_type: value }))}
                      options={[
                        { value: 'one-to-one', label: '1 : 1' },
                        { value: 'one-to-many', label: '1 : N' },
                        { value: 'many-to-one', label: 'N : 1' },
                        { value: 'many-to-many', label: 'M : N' },
                      ]}
                    />
                    <Select
                      label="Target Table"
                      value={manualRelation.file2}
                      onChange={(value) => setManualRelation((prev) => ({ ...prev, file2: value, column2: '' }))}
                      options={datasetOptions}
                      placeholder="Choose table"
                    />
                    <Select
                      label="Target Column"
                      value={manualRelation.column2}
                      onChange={(value) => setManualRelation((prev) => ({ ...prev, column2: value }))}
                      options={targetColumnOptions}
                      placeholder="Choose column"
                    />
                    <div className="flex items-end">
                      <Button
                        className="w-full"
                        onClick={addManualRelationship}
                        loading={savingManual}
                        icon={<LinkIcon className="w-4 h-4" />}
                      >
                        Add Relation
                      </Button>
                    </div>
                  </div>
                </Card>

                {(!schema.relationships || schema.relationships.length === 0) ? (
                  <EmptyState icon={<LinkIcon className="w-8 h-8" />} title="No relationships detected" description="Upload multiple related datasets to detect relationships" />
                ) : (
                  <>
                    {/* Header row */}
                    <div className="hidden md:grid grid-cols-[1fr_32px_1fr_100px_80px_70px_40px] gap-3 px-4 py-2 text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                      <span>Source</span><span /><span>Target</span><span>Type</span><span>Confidence</span><span>Match</span><span />
                    </div>
                    {schema.relationships.map((rel, i) => (
                      <div
                        key={`${rel.file1}-${rel.column1}-${rel.file2}-${rel.column2}-${i}`}
                        className={`rounded-xl border border-[var(--border)] bg-[var(--bg-card)] transition-all duration-300 ${
                          deletingIdx === i ? 'opacity-0 scale-95 -translate-x-4' : 'opacity-100'
                        }`}
                      >
                        {/* Main row */}
                        <div
                          className="grid grid-cols-[1fr_32px_1fr] md:grid-cols-[1fr_32px_1fr_100px_80px_70px_40px] gap-3 items-center px-4 py-3 cursor-pointer hover:bg-[var(--bg-secondary)] transition"
                          onClick={() => setExpandedRel(expandedRel === i ? null : i)}
                        >
                          {/* Source */}
                          <div className="min-w-0">
                            <p className="text-xs font-medium truncate">{rel.file1}</p>
                            <p className="text-[11px] font-mono text-[var(--accent)] truncate">{rel.column1}</p>
                          </div>

                          {/* Arrow */}
                          <div className="flex justify-center">
                            <ArrowRight className="w-4 h-4 text-[var(--text-muted)]" />
                          </div>

                          {/* Target */}
                          <div className="min-w-0">
                            <p className="text-xs font-medium truncate">{rel.file2}</p>
                            <p className="text-[11px] font-mono text-[var(--accent)] truncate">{rel.column2}</p>
                          </div>

                          {/* Type */}
                          <div className="hidden md:block">
                            <Badge variant="primary">{relTypeLabel(rel.relationship_type)}</Badge>
                            {rel.is_manual && <span className="ml-2 text-[10px] text-[var(--accent)]">Manual</span>}
                          </div>

                          {/* Confidence */}
                          <div className="hidden md:block">
                            <Badge variant={confColor(rel.confidence)}>{rel.confidence}</Badge>
                          </div>

                          {/* Match % */}
                          <div className="hidden md:block">
                            {rel.match_percentage != null ? (
                              <span className="text-xs text-[var(--text-muted)]">{rel.match_percentage}%</span>
                            ) : (
                              <span className="text-xs text-[var(--text-muted)]">—</span>
                            )}
                          </div>

                          {/* Expand icon */}
                          <div className="hidden md:flex justify-end">
                            {expandedRel === i
                              ? <ChevronUp className="w-4 h-4 text-[var(--text-muted)]" />
                              : <ChevronDown className="w-4 h-4 text-[var(--text-muted)]" />
                            }
                          </div>
                        </div>

                        {/* Mobile badges row */}
                        <div className="md:hidden flex flex-wrap gap-1.5 px-4 pb-2">
                          <Badge variant="primary">{relTypeLabel(rel.relationship_type)}</Badge>
                          <Badge variant={confColor(rel.confidence)}>{rel.confidence}</Badge>
                          {rel.is_manual && <span className="text-[11px] text-[var(--accent)]">Manual</span>}
                          {rel.match_percentage != null && (
                            <span className="text-[11px] text-[var(--text-muted)]">{rel.match_percentage}% match</span>
                          )}
                        </div>

                        {/* Expanded details */}
                        {expandedRel === i && (
                          <div className="px-4 pb-4 pt-1 border-t border-[var(--border)] space-y-3 animate-fade-in">
                            {rel.reasoning && (
                              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{rel.reasoning}</p>
                            )}
                            <div className="flex flex-wrap gap-2 text-[11px]">
                              {rel.overlap_count != null && (
                                <span className="px-2.5 py-1 rounded-md bg-[var(--bg-secondary)] text-[var(--text-muted)]">
                                  Overlap: <strong className="text-[var(--text-primary)]">{rel.overlap_count}</strong> values
                                </span>
                              )}
                              {rel.is_primary_key && (
                                <span className="px-2.5 py-1 rounded-md bg-[var(--accent-bg)] text-[var(--accent)]">PK</span>
                              )}
                              {rel.is_foreign_key && (
                                <span className="px-2.5 py-1 rounded-md bg-[var(--warning-bg)] text-[var(--warning)]">FK</span>
                              )}
                              {rel.recommendation && rel.recommendation !== 'keep' && (
                                <Badge variant="warning">{rel.recommendation}</Badge>
                              )}
                            </div>
                            <div className="flex justify-end">
                              <button
                                onClick={(e) => { e.stopPropagation(); deleteRelationship(i); }}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--error)] hover:bg-[var(--error-bg)] transition cursor-pointer"
                              >
                                <Trash2 className="w-3.5 h-3.5" /> Remove
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </>
                )}

                {schema.rejected_relationships && schema.rejected_relationships.length > 0 && (
                  <Card className="mt-4">
                    <h4 className="font-semibold text-sm mb-3">Rejected Relationships ({schema.rejected_relationships.length})</h4>
                    <div className="space-y-2">
                      {schema.rejected_relationships.map((rej, i) => (
                        <div key={`${rej.file1}-${rej.column1}-${rej.file2}-${rej.column2}-${i}`} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <div className="text-xs font-medium">{rej.file1}.{rej.column1} → {rej.file2}.{rej.column2}</div>
                          <div className="text-[11px] text-[var(--text-muted)] mt-1">{rej.reason}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {schema.suspicious_columns && schema.suspicious_columns.length > 0 && (
                  <Card className="mt-4">
                    <h4 className="font-semibold text-sm mb-3">Suspicious Columns ({schema.suspicious_columns.length})</h4>
                    <div className="space-y-2">
                      {schema.suspicious_columns.map((sus, i) => (
                        <div key={`${sus.table}-${sus.column}-${i}`} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <div className="text-xs font-medium">{sus.table}.{sus.column}</div>
                          {sus.possible_reference && (
                            <div className="text-[11px] text-[var(--accent)] mt-1">Possible reference: {sus.possible_reference}</div>
                          )}
                          <div className="text-[11px] text-[var(--text-muted)] mt-1">{sus.reason}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {schema.possible_junction_tables && schema.possible_junction_tables.length > 0 && (
                  <Card className="mt-4">
                    <h4 className="font-semibold text-sm mb-3">Possible Junction Tables ({schema.possible_junction_tables.length})</h4>
                    <div className="space-y-2">
                      {schema.possible_junction_tables.map((jt, i) => (
                        <div key={`${jt.table}-${i}`} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <div className="text-xs font-medium">{jt.table}</div>
                          <div className="text-[11px] text-[var(--text-muted)] mt-1">Parents: {jt.parents.join(', ')}</div>
                          <div className="text-[11px] text-[var(--text-muted)]">FK Columns: {jt.fk_columns.join(', ')}</div>
                          <div className="text-[11px] text-[var(--text-muted)]">{jt.reason}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {schema.final_erd_structure?.relationships && schema.final_erd_structure.relationships.length > 0 && (
                  <Card className="mt-4">
                    <h4 className="font-semibold text-sm mb-3">Final ERD Structure (Engine-valid)</h4>
                    <div className="space-y-2">
                      {schema.final_erd_structure.relationships.map((fr, i) => (
                        <div key={`${fr.child}-${fr.child_fk}-${fr.parent}-${fr.parent_pk}-${i}`} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
                          <div className="text-xs font-medium">{fr.child}.{fr.child_fk} → {fr.parent}.{fr.parent_pk}</div>
                          <div className="text-[11px] text-[var(--text-muted)] mt-1">Cardinality: {fr.cardinality}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {schema.data_modeling_observations && schema.data_modeling_observations.length > 0 && (
                  <Card className="mt-4">
                    <h4 className="font-semibold text-sm mb-3">Data Modeling Observations</h4>
                    <ul className="space-y-2 list-disc pl-5">
                      {schema.data_modeling_observations.map((obs, i) => (
                        <li key={i} className="text-sm text-[var(--text-secondary)]">{obs}</li>
                      ))}
                    </ul>
                  </Card>
                )}
              </div>
            )}

            {/* Join Suggestions */}
            {activeTab === 'joins' && (
              <div className="space-y-3">
                {(!schema.join_suggestions || schema.join_suggestions.length === 0) ? (
                  <EmptyState icon={<Combine className="w-8 h-8" />} title="No join suggestions" description="Relationships need to be detected first" />
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] border-b border-[var(--border)]">
                          <th className="text-left pb-3 pr-4">Left Dataset</th>
                          <th className="text-left pb-3 pr-4">Join</th>
                          <th className="text-left pb-3 pr-4">Right Dataset</th>
                          <th className="text-left pb-3 pr-4">ON</th>
                          <th className="text-left pb-3">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {schema.join_suggestions.map((js, i) => (
                          <tr key={i} className="border-b border-[var(--border)] hover:bg-[var(--bg-secondary)] transition">
                            <td className="py-3 pr-4">
                              <span className="text-xs font-medium">{js.left_dataset}</span>
                            </td>
                            <td className="py-3 pr-4">
                              <span className="inline-block px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[var(--accent-bg)] text-[var(--accent)]">
                                {js.how.toUpperCase()}
                              </span>
                            </td>
                            <td className="py-3 pr-4">
                              <span className="text-xs font-medium">{js.right_dataset}</span>
                            </td>
                            <td className="py-3 pr-4">
                              <code className="text-[11px] text-[var(--text-muted)]">{js.left_on} = {js.right_on}</code>
                            </td>
                            <td className="py-3">
                              <Badge variant={confColor(js.confidence)}>{js.confidence}</Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* Unified View */}
            {activeTab === 'unified' && (
              <Card>
                <h3 className="font-semibold mb-4 flex items-center gap-2 text-sm">
                  <Lightbulb className="w-4 h-4 text-[var(--warning)]" /> AI Business Insights
                </h3>
                {schema.unified_view ? (
                  <div className="schema-markdown prose prose-sm max-w-none text-[var(--text-secondary)]">
                    <ReactMarkdown>{normalizeMarkdownContent(schema.unified_view)}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="text-sm text-[var(--text-muted)]">No unified view generated.</p>
                )}
              </Card>
            )}
          </div>
        </>
      )}
    </div>
  );
}
