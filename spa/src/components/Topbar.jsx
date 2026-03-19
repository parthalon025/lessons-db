// Topbar — system title bar with connection status indicator.
// What it shows: Application name + live connection status (last poll timestamp).
// Decision it drives: "Is the dashboard connected to the API? Is data fresh?"

import { signal } from '@preact/signals';

export const connectionStatus = signal('CONNECTING...');

export default function Topbar() {
  return (
    <header class="topbar">
      <span class="topbar-title">LESSONS-DB</span>
      <span class="topbar-status">{connectionStatus.value}</span>
    </header>
  );
}
