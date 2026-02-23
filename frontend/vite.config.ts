import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/api/backend': {
        target: 'http://10.100.102.6:9867',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/backend/, '/api/v1'),
      },
      '/api/engine': {
        target: 'http://10.100.102.6:3490',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/engine/, '/api/v1'),
      },
      '/ws': {
        target: 'ws://10.100.102.6:3490',
        ws: true,
        rewrite: (p) => p.replace(/^\/ws/, '/api/v1/ws'),
      },
    },
  },
})
