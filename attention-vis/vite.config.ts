import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
    base: './',
    plugins: [react()],
    resolve: {
        alias: {
            '@/src': path.resolve(__dirname, 'src'),
            '@/styles': path.resolve(__dirname, 'styles'),
        },
    },
    css: {
        preprocessorOptions: {
            scss: { api: 'legacy', silenceDeprecations: ['legacy-js-api'] },
        },
    },
});
