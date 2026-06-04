import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

function normalizeBasePath(value: string | undefined): string {
  const trimmed = (value || '/chatbot').trim();
  if (!trimmed || trimmed === '/') return '';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  const webBasePath = normalizeBasePath(env.VITE_WEB_BASE_PATH);
  const apiBasePath = normalizeBasePath(env.VITE_API_BASE || '/api');
  const apiTarget = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8088';

  const proxiedRoots = ['/ask', '/sessions', '/auth', '/users', '/health', '/ingest', '/rag', '/tasks'];

  const proxy = Object.fromEntries(
    [...proxiedRoots, ...proxiedRoots.map((path) => `${apiBasePath}${path}`)].map((path) => [
      path,
      apiTarget,
    ]),
  );

  return {
    base: `${webBasePath || ''}/static/`,

    plugins: [react()],

    build: {
      outDir: 'dist',
      emptyOutDir: true,
    },

    server: {
      host: '0.0.0.0',
      port: 8001,
      allowedHosts: [
        'www.pader.top',
        'pader.top',
      ],
      proxy,
    },
  };
});
