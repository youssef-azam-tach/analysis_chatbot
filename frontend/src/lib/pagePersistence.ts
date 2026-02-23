interface PagePersistenceScope {
  userId?: string | number | null;
  workspaceId?: string | null;
  sessionId?: string | null;
}

function makeKey(page: string, scope: PagePersistenceScope): string | null {
  if (!page) return null;
  if (!scope?.userId || !scope?.workspaceId || !scope?.sessionId) return null;
  return `page-cache:${String(scope.userId)}:${scope.workspaceId}:${scope.sessionId}:${page}`;
}

export function loadPageState<T>(page: string, scope: PagePersistenceScope): T | null {
  const key = makeKey(page, scope);
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function savePageState<T>(page: string, scope: PagePersistenceScope, state: T): void {
  const key = makeKey(page, scope);
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(state));
  } catch {
    // ignore storage failures
  }
}

export function clearPageState(page: string, scope: PagePersistenceScope): void {
  const key = makeKey(page, scope);
  if (!key) return;
  try {
    localStorage.removeItem(key);
  } catch {
    // ignore storage failures
  }
}
