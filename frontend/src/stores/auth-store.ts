import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { backend } from '@/lib/api';

interface User {
  id: string;
  email: string;
  full_name: string;
  avatar_url?: string;
  enabled_pages?: string[];
}

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, fullName: string, password: string) => Promise<void>;
  fetchMe: () => Promise<void>;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      refreshToken: null,
      user: null,
      isAuthenticated: false,

      login: async (email, password) => {
        const { data } = await backend.post('/auth/login', { email, password });
        const t = data.data;
        set({ token: t.access_token, refreshToken: t.refresh_token, isAuthenticated: true });
        await get().fetchMe();
      },

      register: async (email, fullName, password) => {
        await backend.post('/auth/register', { email, full_name: fullName, password });
      },

      fetchMe: async () => {
        const { data } = await backend.get('/auth/me');
        set({ user: data.data });
      },

      logout: () => {
        set({ token: null, refreshToken: null, user: null, isAuthenticated: false });
      },
    }),
    { name: 'ai-auth' }
  )
);
