// LoadingState — terminal-voice loading indicator.
// What it shows: A loading message while data is being fetched.
// Decision it drives: None — tells the user to wait.

export default function LoadingState({ message }) {
  return (
    <div class="loading-container">
      {message || 'LOADING...'}
    </div>
  );
}
