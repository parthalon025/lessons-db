import { render } from 'preact';

// Minimal entry point — full app wired in Task 3.5
function App() {
  return (
    <div class="loading-container">
      LESSONS DB LOADING...
    </div>
  );
}

render(<App />, document.getElementById('app'));
