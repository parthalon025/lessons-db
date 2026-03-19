// App root — route dispatcher + boot sequence + polling orchestration.
// What it shows: The active page based on sidebar navigation selection.
// Decision it drives: Which section of the dashboard to display.

import { useEffect } from 'preact/hooks';
import AppLayout, { currentRoute } from './components/AppLayout.jsx';
import { connectionStatus } from './components/Topbar.jsx';
import { refreshStats } from './stores/stats.js';
import { refreshHealth } from './stores/health.js';
import Dashboard from './pages/Dashboard.jsx';
import Lessons from './pages/Lessons.jsx';
import Triage from './pages/Triage.jsx';
import Eval from './pages/Eval.jsx';
import Admin from './pages/Admin.jsx';

const ROUTES = {
  dashboard: Dashboard,
  lessons: Lessons,
  triage: Triage,
  eval: Eval,
  admin: Admin,
};

export default function App() {
  // Boot sequence — initial data load + set connection status
  useEffect(() => {
    async function boot() {
      try {
        await refreshStats();
        connectionStatus.value = 'CONNECTED';
      } catch (err) {
        connectionStatus.value = 'OFFLINE';
      }
    }
    boot();
  }, []);

  const PageComponent = ROUTES[currentRoute.value] || Dashboard;

  return (
    <AppLayout>
      <PageComponent />
    </AppLayout>
  );
}
