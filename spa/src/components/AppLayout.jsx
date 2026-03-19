// AppLayout — main layout shell with sidebar navigation + top bar + content area.
// What it shows: Application chrome — nav sidebar, system title bar, main content region.
// Decision it drives: Navigate between dashboard sections.

import { signal } from '@preact/signals';
import Topbar from './Topbar.jsx';

// Route signal — controls which page is displayed
export const currentRoute = signal('dashboard');

const NAV_ITEMS = [
  { id: 'dashboard', label: 'DASHBOARD', icon: '>' },
  { id: 'lessons', label: 'LESSONS', icon: '#' },
  { id: 'triage', label: 'TRIAGE', icon: '!' },
  { id: 'eval', label: 'EVAL', icon: '%' },
  { id: 'admin', label: 'ADMIN', icon: '*' },
];

export default function AppLayout({ children }) {
  return (
    <div id="app-root">
      <Topbar />
      <div class="layout-root">
        <nav class="layout-sidebar">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.id}
              class={`nav-item ${currentRoute.value === item.id ? 'active' : ''}`}
              onClick={() => { currentRoute.value = item.id; }}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </a>
          ))}
        </nav>
        <main class="layout-main sh-stagger-children">
          {children}
        </main>
      </div>
    </div>
  );
}
