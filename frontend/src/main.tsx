import { Component, StrictMode, type ErrorInfo, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

class AppErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, message: '' };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, message: error?.message || 'Error inesperado en la interfaz' };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('FriendFinder UI Error:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="pin-gate">
          <div className="pin-card" style={{ maxWidth: 460 }}>
            <h1 className="section-title" style={{ margin: 0 }}>
              No se pudo cargar FriendFinder
            </h1>
            <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--ink-secondary)' }}>
              {this.state.message}
            </p>
            <button
              className="btn btn-primary"
              onClick={() => window.location.reload()}
            >
              Recargar página
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </StrictMode>
);
