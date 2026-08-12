import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

// Error Boundary: catches render errors and prevents white-screen crashes.
// In a voice AI app, a crash during a live call is unacceptable.
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[ErrorBoundary] Uncaught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', minHeight: '100vh', padding: '2rem',
          background: 'hsl(225, 25%, 8%)', color: 'hsl(220, 20%, 92%)',
          fontFamily: 'Inter, sans-serif', textAlign: 'center',
        }}>
          <h1 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Something went wrong</h1>
          <p style={{ color: 'hsl(220, 15%, 65%)', maxWidth: '400px', marginBottom: '1.5rem' }}>
            The application encountered an unexpected error. Please refresh the page to continue.
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 24px', borderRadius: '8px', border: 'none',
              background: 'hsl(217, 91%, 60%)', color: '#fff', fontSize: '0.9rem',
              cursor: 'pointer', fontWeight: 600,
            }}
          >
            Refresh Page
          </button>
          <pre style={{
            marginTop: '2rem', padding: '1rem', borderRadius: '8px',
            background: 'hsl(225, 20%, 12%)', color: 'hsl(0, 72%, 51%)',
            fontSize: '0.75rem', maxWidth: '600px', overflow: 'auto',
            textAlign: 'left',
          }}>
            {this.state.error?.toString()}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
