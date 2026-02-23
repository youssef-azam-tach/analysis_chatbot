import axios from 'axios';
import { useAuthStore } from '@/stores/auth-store';

/** Service 1 — Backend Core (Auth, Users, Workspaces, DB entities) */
export const backend = axios.create({ baseURL: '/api/backend' });

/** Service 2 — AI Engine (Files, Analysis, Cleaning, Chat, Viz) */
export const engine = axios.create({ baseURL: '/api/engine' });

// ── Request interceptor: attach JWT ──────────────────────────
function attachToken(config: any) {
  const token = useAuthStore.getState().token;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
}
backend.interceptors.request.use(attachToken);
engine.interceptors.request.use(attachToken);

// ── Response interceptor: handle 401 ─────────────────────────
function handle401(error: any) {
  if (error.response?.status === 401) {
    useAuthStore.getState().logout();
    window.location.href = '/login';
  }
  return Promise.reject(error);
}
backend.interceptors.response.use((r) => r, handle401);
engine.interceptors.response.use((r) => r, handle401);

/* ── Task polling for background AI operations ────────────── */
export interface TaskResult {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  result?: any;
  error?: string;
}

/**
 * Poll a background task until completion. Returns the task result.
 * @param taskId - from the initial TaskResponse
 * @param onProgress - optional callback for progress updates
 * @param intervalMs - poll interval (default 2s)
 * @param timeoutMs - overall timeout (default 5min)
 */
export async function pollTask(
  taskId: string,
  onProgress?: (progress: number, status: string) => void,
  intervalMs = 2000,
  timeoutMs = 300_000,
): Promise<any> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const { data } = await engine.get(`/tasks/${taskId}`);
    const task: TaskResult = data.data;
    if (onProgress) onProgress(task.progress || 0, task.status);
    if (task.status === 'completed') return task.result;
    if (task.status === 'failed') throw new Error(task.error || 'Task failed');
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error('Task timed out');
}

/**
 * Parse a Plotly JSON string or object into { data, layout }.
 */
export function parsePlotly(input: any): { data: any[]; layout: any } | null {
  if (!input) return null;
  try {
    const obj = typeof input === 'string' ? JSON.parse(input) : input;
    return { data: obj.data || [], layout: obj.layout || {} };
  } catch { return null; }
}
