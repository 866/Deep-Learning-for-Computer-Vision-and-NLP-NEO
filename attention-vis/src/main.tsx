import React from 'react';
import { createRoot } from 'react-dom/client';
import '@/styles/main.css';
import '@fortawesome/fontawesome-svg-core/styles.css';
import { config } from '@fortawesome/fontawesome-svg-core';
import { App } from './App';
config.autoAddCss = false;

createRoot(document.getElementById('root')!).render(<App />);
