import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';

// Simple working App component
const App = () => {
  return (
    <div style={{
      fontFamily: 'Arial, sans-serif',
      background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
      minHeight: '100vh',
      color: 'white',
      padding: '2rem'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto'
      }}>
        <header style={{
          textAlign: 'center',
          marginBottom: '3rem',
          padding: '2rem',
          background: 'rgba(255,255,255,0.1)',
          borderRadius: '15px',
          backdropFilter: 'blur(10px)'
        }}>
          <h1 style={{
            fontSize: '3rem',
            marginBottom: '1rem',
            textShadow: '2px 2px 4px rgba(0,0,0,0.5)'
          }}>
            🛡️ Project Argus Dashboard
          </h1>
          <p style={{ fontSize: '1.2rem', opacity: '0.9' }}>
            Advanced AI-Powered Border Detection and Monitoring System
          </p>
        </header>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '2rem'
        }}>
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '2rem',
            borderRadius: '15px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            <h3 style={{ color: '#4CAF50', marginBottom: '1rem' }}>📹 Live Camera Feeds</h3>
            <div style={{
              background: 'rgba(0,0,0,0.3)',
              height: '200px',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem'
            }}>
              <div style={{ textAlign: 'center' }}>
                🎥 Camera Feed Placeholder<br/>
                <small>4 Active Cameras</small>
              </div>
            </div>
            <button style={{
              background: 'linear-gradient(45deg, #4CAF50, #45a049)',
              color: 'white',
              border: 'none',
              padding: '0.8rem 1.5rem',
              borderRadius: '25px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}>
              View All Cameras
            </button>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '2rem',
            borderRadius: '15px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            <h3 style={{ color: '#FF5722', marginBottom: '1rem' }}>🚨 Active Alerts</h3>
            <div style={{
              background: 'rgba(255,87,34,0.2)',
              padding: '1rem',
              borderRadius: '10px',
              marginBottom: '1rem',
              borderLeft: '4px solid #FF5722'
            }}>
              <strong>VIRTUAL LINE CROSSING</strong><br/>
              <small>Camera: Border Alpha Main</small><br/>
              <span style={{
                background: '#FF5722',
                padding: '0.3rem 0.8rem',
                borderRadius: '15px',
                fontSize: '0.8rem',
                fontWeight: 'bold'
              }}>HIGH</span>
            </div>
            <button style={{
              background: 'linear-gradient(45deg, #FF5722, #E64A19)',
              color: 'white',
              border: 'none',
              padding: '0.8rem 1.5rem',
              borderRadius: '25px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}>
              View All Alerts
            </button>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '2rem',
            borderRadius: '15px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            <h3 style={{ color: '#2196F3', marginBottom: '1rem' }}>📋 Recent Incidents</h3>
            <div style={{
              background: 'rgba(33,150,243,0.2)',
              padding: '1rem',
              borderRadius: '10px',
              marginBottom: '1rem',
              borderLeft: '4px solid #2196F3'
            }}>
              <strong>Incident #INC-2024-001</strong><br/>
              <small>Unauthorized crossing detected</small><br/>
              <span style={{
                background: '#2196F3',
                padding: '0.3rem 0.8rem',
                borderRadius: '15px',
                fontSize: '0.8rem',
                fontWeight: 'bold'
              }}>OPEN</span>
            </div>
            <button style={{
              background: 'linear-gradient(45deg, #2196F3, #1976D2)',
              color: 'white',
              border: 'none',
              padding: '0.8rem 1.5rem',
              borderRadius: '25px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}>
              View All Incidents
            </button>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '2rem',
            borderRadius: '15px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            <h3 style={{ color: '#4CAF50', marginBottom: '1rem' }}>📊 System Status</h3>
            <div style={{ marginBottom: '1rem' }}>
              <p><span style={{ color: '#4CAF50' }}>●</span> API Gateway: Online</p>
              <p><span style={{ color: '#4CAF50' }}>●</span> Alert Service: Online</p>
              <p><span style={{ color: '#4CAF50' }}>●</span> Tracking Service: Online</p>
              <p><span style={{ color: '#4CAF50' }}>●</span> Edge Node: Active</p>
            </div>
            <button style={{
              background: 'linear-gradient(45deg, #4CAF50, #45a049)',
              color: 'white',
              border: 'none',
              padding: '0.8rem 1.5rem',
              borderRadius: '25px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }} onClick={() => window.open('http://localhost:8000/docs', '_blank')}>
              Open API Docs
            </button>
          </div>
        </div>

        <div style={{
          marginTop: '3rem',
          textAlign: 'center',
          background: 'rgba(255,255,255,0.05)',
          padding: '2rem',
          borderRadius: '15px'
        }}>
          <h2 style={{ marginBottom: '1rem' }}>🔗 Quick Access</h2>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button style={{
              background: 'linear-gradient(45deg, #4CAF50, #45a049)',
              color: 'white',
              border: 'none',
              padding: '1rem 2rem',
              borderRadius: '25px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }} onClick={() => window.open('http://localhost:8000', '_blank')}>
              🌐 API Gateway
            </button>
            <button style={{
              background: 'linear-gradient(45deg, #2196F3, #1976D2)',
              color: 'white',
              border: 'none',
              padding: '1rem 2rem',
              borderRadius: '25px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }} onClick={() => window.open('http://localhost:8000/docs', '_blank')}>
              📖 API Documentation
            </button>
            <button style={{
              background: 'linear-gradient(45deg, #FF9800, #F57C00)',
              color: 'white',
              border: 'none',
              padding: '1rem 2rem',
              borderRadius: '25px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }} onClick={() => window.open('http://localhost:8000/health', '_blank')}>
              ❤️ System Health
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);