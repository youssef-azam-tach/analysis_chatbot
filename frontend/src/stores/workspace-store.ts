import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { backend } from '@/lib/api';

export interface Workspace {
  id: string;
  name: string;
  description?: string;
  data_source_type: string;
  enabled_pages?: string[];
  created_at: string;
}

export interface DatasetSummary {
  id: string;
  name: string;
  original_filename: string;
  file_type: string;
  row_count: number;
  column_count: number;
  quality_score?: number;
  created_at: string;
}

export interface UploadedFile {
  filename: string;
  dataset_key?: string;
  total_rows?: number;
  total_columns?: number;
  sheets?: string[];
}

export interface PinnedKpiItem {
  id: string;
  sessionId: string | null;
  label: string;
  value: string;
  suffix?: string;
  source?: string;
}

export interface PinnedCardItem {
  id: string;
  sessionId: string | null;
  label: string;
  value: string;
  source?: string;
}

export interface PinnedChartItem {
  id: string;
  sessionId: string | null;
  title: string;
  plotly_json: string;
  source?: string;
  chartType?: string;
}

interface WorkspaceState {
  workspaces: Workspace[];
  activeWorkspace: Workspace | null;
  datasets: DatasetSummary[];
  sessionId: string | null;
  uploadedFiles: UploadedFile[];
  pinnedKpis: PinnedKpiItem[];
  pinnedCards: PinnedCardItem[];
  pinnedCharts: PinnedChartItem[];
  loading: boolean;

  fetchWorkspaces: () => Promise<void>;
  createWorkspace: (name: string, description?: string) => Promise<Workspace>;
  selectWorkspace: (ws: Workspace) => void;
  fetchDatasets: (workspaceId: string) => Promise<void>;
  setSessionId: (id: string) => void;
  setUploadedFiles: (files: UploadedFile[]) => void;
  addUploadedFiles: (files: UploadedFile[]) => void;
  addPinnedKpi: (item: PinnedKpiItem) => void;
  addPinnedCard: (item: PinnedCardItem) => void;
  addPinnedChart: (item: PinnedChartItem) => void;
  removePinnedKpi: (id: string) => void;
  removePinnedCard: (id: string) => void;
  removePinnedChart: (id: string) => void;
  clearPinnedForSession: (sessionId: string | null) => void;
  clearSession: () => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      workspaces: [],
      activeWorkspace: null,
      datasets: [],
      sessionId: null,
      uploadedFiles: [],
      pinnedKpis: [],
      pinnedCards: [],
      pinnedCharts: [],
      loading: false,

      fetchWorkspaces: async () => {
        set({ loading: true });
        const { data } = await backend.get('/workspaces/');
        set({ workspaces: data.data, loading: false });
      },

      createWorkspace: async (name, description) => {
        const { data } = await backend.post('/workspaces/', { name, description });
        const ws = data.data;
        set((s) => ({ workspaces: [...s.workspaces, ws] }));
        return ws;
      },

      selectWorkspace: (ws) => {
        set({ activeWorkspace: ws, datasets: [], sessionId: null, uploadedFiles: [] });
      },

      fetchDatasets: async (workspaceId) => {
        const { data } = await backend.get(`/datasets/workspace/${workspaceId}`);
        set({ datasets: data.data });
      },

      setSessionId: (id) => set({ sessionId: id }),

      setUploadedFiles: (files) => set({ uploadedFiles: files }),

      addUploadedFiles: (files) => set((s) => ({ uploadedFiles: [...s.uploadedFiles, ...files] })),

      addPinnedKpi: (item) => set((s) => {
        if (s.pinnedKpis.some((p) => p.id === item.id)) return s;
        return { pinnedKpis: [item, ...s.pinnedKpis] };
      }),

      addPinnedCard: (item) => set((s) => {
        if (s.pinnedCards.some((p) => p.id === item.id)) return s;
        return { pinnedCards: [item, ...s.pinnedCards] };
      }),

      addPinnedChart: (item) => set((s) => {
        if (s.pinnedCharts.some((p) => p.id === item.id)) return s;
        return { pinnedCharts: [item, ...s.pinnedCharts] };
      }),

      removePinnedKpi: (id) => set((s) => ({ pinnedKpis: s.pinnedKpis.filter((p) => p.id !== id) })),
      removePinnedCard: (id) => set((s) => ({ pinnedCards: s.pinnedCards.filter((p) => p.id !== id) })),
      removePinnedChart: (id) => set((s) => ({ pinnedCharts: s.pinnedCharts.filter((p) => p.id !== id) })),

      clearPinnedForSession: (sessionId) => set((s) => ({
        pinnedKpis: s.pinnedKpis.filter((p) => p.sessionId !== sessionId),
        pinnedCards: s.pinnedCards.filter((p) => p.sessionId !== sessionId),
        pinnedCharts: s.pinnedCharts.filter((p) => p.sessionId !== sessionId),
      })),

      clearSession: () => set({ sessionId: null, uploadedFiles: [], pinnedKpis: [], pinnedCards: [], pinnedCharts: [] }),
    }),
    {
      name: 'ai-workspace',
      partialize: (state) => ({
        activeWorkspace: state.activeWorkspace,
        sessionId: state.sessionId,
        uploadedFiles: state.uploadedFiles,
        pinnedKpis: state.pinnedKpis,
        pinnedCards: state.pinnedCards,
        pinnedCharts: state.pinnedCharts,
      }),
    }
  )
);
