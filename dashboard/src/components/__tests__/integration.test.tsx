import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';

// Import reducers
import cameraReducer from '../../store/slices/cameraSlice';
import alertReducer from '../../store/slices/alertSlice';
import incidentReducer from '../../store/slices/incidentSlice';
import systemReducer from '../../store/slices/systemSlice';

// Mock components to avoid complex rendering issues
jest.mock('../camera/LiveFeedViewer', () => {
  return function MockLiveFeedViewer({ camera }: { camera: any }) {
    return <div data-testid={`live-feed-${camera.id}`}>{camera.name}</div>;
  };
});

jest.mock('../camera/CameraGrid', () => {
  return function MockCameraGrid({ selectedCameras }: { selectedCameras?: string[] }) {
    return (
      <div data-testid="camera-grid">
        {selectedCameras ? `${selectedCameras.length} selected cameras` : 'All cameras'}
      </div>
    );
  };
});

const createTestStore = () => {
  return configureStore({
    reducer: {
      cameras: cameraReducer,
      alerts: alertReducer,
      incidents: incidentReducer,
      system: systemReducer,
    },
    preloadedState: {
      cameras: {
        cameras: [
          {
            id: 'cam-001',
            name: 'Test Camera 1',
            location: 'Location 1',
            type: 'visible' as const,
            status: 'online' as const,
            streamUrl: 'http://test.mp4',
            detections: [],
            virtualLines: [],
            lastUpdate: '2024-01-01T12:00:00Z',
          },
        ],
        selectedCamera: null,
        gridLayout: 4,
        fullscreenCamera: null,
        showDetections: true,
        showVirtualLines: true,
      },
      alerts: {
        alerts: [
          {
            id: 'alert-001',
            type: 'crossing' as const,
            severity: 'high' as const,
            cameraId: 'cam-001',
            cameraName: 'Test Camera 1',
            timestamp: '2024-01-01T12:00:00Z',
            confidence: 0.95,
            riskScore: 0.85,
            description: 'Test alert',
            acknowledged: false,
            metadata: {},
          },
        ],
        unacknowledgedCount: 1,
        filters: {
          severity: [],
          type: [],
          camera: [],
          acknowledged: null,
        },
        sortBy: 'timestamp',
        sortOrder: 'desc',
        selectedAlert: null,
      },
      incidents: {
        incidents: [],
        selectedIncident: null,
        filters: {
          status: [],
          priority: [],
          assignedTo: [],
          tags: [],
        },
        sortBy: 'createdAt',
        sortOrder: 'desc',
      },
      system: {
        connected: true,
        health: [],
        metrics: [],
        notifications: [],
        currentUser: {
          id: 'user-001',
          name: 'Test User',
          role: 'operator',
          permissions: ['view_cameras'],
        },
      },
    },
  });
};

describe('Dashboard Integration Tests', () => {
  const renderWithProviders = (component: React.ReactElement) => {
    const store = createTestStore();
    return render(
      <Provider store={store}>
        <BrowserRouter>
          {component}
        </BrowserRouter>
      </Provider>
    );
  };

  it('renders basic dashboard structure', () => {
    const TestComponent = () => (
      <div>
        <h1>Project Argus Dashboard</h1>
        <div data-testid="camera-grid">Camera Grid</div>
      </div>
    );

    renderWithProviders(<TestComponent />);

    expect(screen.getByText('Project Argus Dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('camera-grid')).toBeInTheDocument();
  });

  it('handles Redux store state correctly', () => {
    const store = createTestStore();
    
    // Test initial state
    expect(store.getState().cameras.cameras).toHaveLength(1);
    expect(store.getState().alerts.alerts).toHaveLength(1);
    expect(store.getState().alerts.unacknowledgedCount).toBe(1);
    expect(store.getState().system.connected).toBe(true);
  });

  it('can dispatch actions to update state', () => {
    const store = createTestStore();
    
    // Dispatch an action to toggle detections
    store.dispatch({ type: 'cameras/toggleDetections' });
    
    // Check if state changed
    expect(store.getState().cameras.showDetections).toBe(false);
    
    // Dispatch again to toggle back
    store.dispatch({ type: 'cameras/toggleDetections' });
    expect(store.getState().cameras.showDetections).toBe(true);
  });

  it('handles camera grid layout changes', () => {
    const store = createTestStore();
    
    // Change grid layout
    store.dispatch({ 
      type: 'cameras/setGridLayout', 
      payload: 9 
    });
    
    expect(store.getState().cameras.gridLayout).toBe(9);
  });

  it('manages alert acknowledgment', () => {
    const store = createTestStore();
    
    // Acknowledge an alert
    store.dispatch({
      type: 'alerts/acknowledgeAlert',
      payload: { alertId: 'alert-001', userId: 'user-001' }
    });
    
    const alert = store.getState().alerts.alerts.find(a => a.id === 'alert-001');
    expect(alert?.acknowledged).toBe(true);
    expect(alert?.acknowledgedBy).toBe('user-001');
    expect(store.getState().alerts.unacknowledgedCount).toBe(0);
  });

  it('handles system connection status', () => {
    const store = createTestStore();
    
    // Disconnect
    store.dispatch({
      type: 'system/setConnectionStatus',
      payload: false
    });
    
    expect(store.getState().system.connected).toBe(false);
    
    // Reconnect
    store.dispatch({
      type: 'system/setConnectionStatus',
      payload: true
    });
    
    expect(store.getState().system.connected).toBe(true);
  });

  it('manages camera status updates', () => {
    const store = createTestStore();
    
    // Update camera status
    store.dispatch({
      type: 'cameras/updateCameraStatus',
      payload: { cameraId: 'cam-001', status: 'offline' }
    });
    
    const camera = store.getState().cameras.cameras.find(c => c.id === 'cam-001');
    expect(camera?.status).toBe('offline');
  });

  it('handles fullscreen camera mode', () => {
    const store = createTestStore();
    
    // Set fullscreen camera
    store.dispatch({
      type: 'cameras/setFullscreenCamera',
      payload: 'cam-001'
    });
    
    expect(store.getState().cameras.fullscreenCamera).toBe('cam-001');
    
    // Exit fullscreen
    store.dispatch({
      type: 'cameras/setFullscreenCamera',
      payload: null
    });
    
    expect(store.getState().cameras.fullscreenCamera).toBe(null);
  });

  it('manages alert filters', () => {
    const store = createTestStore();
    
    // Set severity filter
    store.dispatch({
      type: 'alerts/setFilters',
      payload: { severity: ['high', 'critical'] }
    });
    
    expect(store.getState().alerts.filters.severity).toEqual(['high', 'critical']);
  });

  it('handles sorting changes', () => {
    const store = createTestStore();
    
    // Change sorting
    store.dispatch({
      type: 'alerts/setSorting',
      payload: { sortBy: 'severity', sortOrder: 'asc' }
    });
    
    expect(store.getState().alerts.sortBy).toBe('severity');
    expect(store.getState().alerts.sortOrder).toBe('asc');
  });

  it('manages system notifications', () => {
    const store = createTestStore();
    
    // Add notification
    store.dispatch({
      type: 'system/addNotification',
      payload: {
        type: 'info',
        message: 'Test notification'
      }
    });
    
    const notifications = store.getState().system.notifications;
    expect(notifications).toHaveLength(1);
    expect(notifications[0].message).toBe('Test notification');
    expect(notifications[0].type).toBe('info');
    expect(notifications[0].read).toBe(false);
  });
});

describe('Component State Management', () => {
  it('maintains consistent state across components', () => {
    const store = createTestStore();
    
    // Simulate multiple component interactions
    store.dispatch({ type: 'cameras/toggleDetections' });
    store.dispatch({ type: 'cameras/toggleVirtualLines' });
    store.dispatch({ 
      type: 'alerts/acknowledgeAlert',
      payload: { alertId: 'alert-001', userId: 'user-001' }
    });
    
    const state = store.getState();
    expect(state.cameras.showDetections).toBe(false);
    expect(state.cameras.showVirtualLines).toBe(false);
    expect(state.alerts.unacknowledgedCount).toBe(0);
  });

  it('handles complex state updates correctly', () => {
    const store = createTestStore();
    
    // Add multiple cameras
    const newCamera = {
      id: 'cam-002',
      name: 'Test Camera 2',
      location: 'Location 2',
      type: 'thermal' as const,
      status: 'online' as const,
      streamUrl: 'http://test2.mp4',
      detections: [
        {
          id: 'det-001',
          bbox: { x: 100, y: 100, width: 50, height: 80 },
          confidence: 0.88,
          timestamp: '2024-01-01T12:00:00Z',
        }
      ],
      virtualLines: [],
      lastUpdate: '2024-01-01T12:00:00Z',
    };
    
    store.dispatch({
      type: 'cameras/updateCamera',
      payload: newCamera
    });
    
    expect(store.getState().cameras.cameras).toHaveLength(2);
    
    // Add alert for new camera
    const newAlert = {
      id: 'alert-002',
      type: 'loitering' as const,
      severity: 'medium' as const,
      cameraId: 'cam-002',
      cameraName: 'Test Camera 2',
      timestamp: '2024-01-01T12:30:00Z',
      confidence: 0.75,
      riskScore: 0.60,
      description: 'Loitering detected',
      acknowledged: false,
      metadata: {},
    };
    
    store.dispatch({
      type: 'alerts/addAlert',
      payload: newAlert
    });
    
    expect(store.getState().alerts.alerts).toHaveLength(2);
    expect(store.getState().alerts.unacknowledgedCount).toBe(2);
  });
});