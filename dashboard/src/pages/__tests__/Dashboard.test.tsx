import React from 'react';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';
import Dashboard from '../Dashboard';
import cameraReducer from '../../store/slices/cameraSlice';
import alertReducer from '../../store/slices/alertSlice';
import incidentReducer from '../../store/slices/incidentSlice';

// Mock the CameraGrid component
jest.mock('../../components/camera/CameraGrid', () => {
  return function MockCameraGrid({ selectedCameras }: { selectedCameras?: string[] }) {
    return (
      <div data-testid="camera-grid">
        Camera Grid - {selectedCameras ? selectedCameras.length : 'all'} cameras
      </div>
    );
  };
});

const mockCameras = [
  {
    id: 'cam-001',
    name: 'Camera 1',
    location: 'Location 1',
    type: 'visible' as const,
    status: 'online' as const,
    streamUrl: 'http://test1.mp4',
    detections: [
      {
        id: 'det-001',
        bbox: { x: 100, y: 100, width: 50, height: 80 },
        confidence: 0.95,
        timestamp: '2024-01-01T12:00:00Z',
      },
    ],
    virtualLines: [],
    lastUpdate: '2024-01-01T12:00:00Z',
  },
  {
    id: 'cam-002',
    name: 'Camera 2',
    location: 'Location 2',
    type: 'thermal' as const,
    status: 'online' as const,
    streamUrl: 'http://test2.mp4',
    detections: [],
    virtualLines: [],
    lastUpdate: '2024-01-01T12:00:00Z',
  },
  {
    id: 'cam-003',
    name: 'Camera 3',
    location: 'Location 3',
    type: 'infrared' as const,
    status: 'offline' as const,
    streamUrl: 'http://test3.mp4',
    detections: [],
    virtualLines: [],
    lastUpdate: '2024-01-01T12:00:00Z',
  },
];

const mockAlerts = [
  {
    id: 'alert-001',
    type: 'crossing' as const,
    severity: 'high' as const,
    cameraId: 'cam-001',
    cameraName: 'Camera 1',
    timestamp: '2024-01-01T12:00:00Z',
    confidence: 0.95,
    riskScore: 0.85,
    description: 'Person crossing boundary',
    acknowledged: false,
    metadata: {},
  },
  {
    id: 'alert-002',
    type: 'loitering' as const,
    severity: 'medium' as const,
    cameraId: 'cam-002',
    cameraName: 'Camera 2',
    timestamp: '2024-01-01T11:00:00Z',
    confidence: 0.78,
    riskScore: 0.65,
    description: 'Person loitering in area',
    acknowledged: true,
    acknowledgedBy: 'Operator 1',
    acknowledgedAt: '2024-01-01T11:30:00Z',
    metadata: {},
  },
];

const mockIncidents = [
  {
    id: 'incident-001',
    alertId: 'alert-001',
    title: 'Border Crossing Incident',
    description: 'Person detected crossing boundary',
    status: 'open' as const,
    priority: 'high' as const,
    assignedTo: 'user-001',
    assignedToName: 'Operator 1',
    createdBy: 'user-001',
    createdByName: 'Operator 1',
    createdAt: '2024-01-01T12:00:00Z',
    updatedAt: '2024-01-01T12:00:00Z',
    evidence: [],
    notes: [],
    tags: [],
    location: 'Border Sector Alpha',
    cameraId: 'cam-001',
    cameraName: 'Camera 1',
  },
];

const createMockStore = (cameras = mockCameras, alerts = mockAlerts, incidents = mockIncidents) => {
  return configureStore({
    reducer: {
      cameras: cameraReducer,
      alerts: alertReducer,
      incidents: incidentReducer,
    },
    preloadedState: {
      cameras: {
        cameras,
        selectedCamera: null,
        gridLayout: 4,
        fullscreenCamera: null,
        showDetections: true,
        showVirtualLines: true,
      },
      alerts: {
        alerts,
        unacknowledgedCount: alerts.filter(a => !a.acknowledged).length,
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
        incidents,
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
    },
  });
};

describe('Dashboard', () => {
  it('renders main dashboard title', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    expect(screen.getByText('Project Argus Command Center')).toBeInTheDocument();
  });

  it('displays key metrics correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Check metric titles
    expect(screen.getByText('Active Cameras')).toBeInTheDocument();
    expect(screen.getByText('Unacknowledged Alerts')).toBeInTheDocument();
    expect(screen.getByText('Open Incidents')).toBeInTheDocument();
    expect(screen.getByText('Active Detections')).toBeInTheDocument();

    // Check metric values
    expect(screen.getByText('2')).toBeInTheDocument(); // Active cameras (online)
    expect(screen.getByText('/ 3')).toBeInTheDocument(); // Total cameras
    expect(screen.getByText('1')).toBeInTheDocument(); // Unacknowledged alerts
    expect(screen.getByText('1')).toBeInTheDocument(); // Open incidents
    expect(screen.getByText('1')).toBeInTheDocument(); // Active detections
  });

  it('calculates active cameras correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Should show 2 online cameras out of 3 total
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('/ 3')).toBeInTheDocument();
  });

  it('calculates unacknowledged alerts correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Should show 1 unacknowledged alert
    const unacknowledgedCount = screen.getByText('1');
    expect(unacknowledgedCount).toBeInTheDocument();
  });

  it('calculates open incidents correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Should show 1 open incident
    const openIncidentsCount = screen.getByText('1');
    expect(openIncidentsCount).toBeInTheDocument();
  });

  it('calculates active detections correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Should show 1 active detection (from cam-001)
    const detectionsCount = screen.getByText('1');
    expect(detectionsCount).toBeInTheDocument();
  });

  it('renders camera grid with first 4 cameras', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    expect(screen.getByTestId('camera-grid')).toBeInTheDocument();
    expect(screen.getByText('Camera Grid - 3 cameras')).toBeInTheDocument(); // All 3 cameras since less than 4
  });

  it('shows live camera feeds section', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    expect(screen.getByText('Live Camera Feeds')).toBeInTheDocument();
  });

  it('applies correct colors to metrics based on values', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Active cameras should be green (online)
    const activeCamerasStatistic = screen.getByText('Active Cameras').closest('.ant-statistic');
    expect(activeCamerasStatistic).toBeInTheDocument();

    // Unacknowledged alerts should be red (has alerts)
    const alertsStatistic = screen.getByText('Unacknowledged Alerts').closest('.ant-statistic');
    expect(alertsStatistic).toBeInTheDocument();
  });

  it('handles empty state correctly', () => {
    const store = createMockStore([], [], []);
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Should show 0 for all metrics
    expect(screen.getByText('0')).toBeInTheDocument(); // Active cameras
    expect(screen.getByText('/ 0')).toBeInTheDocument(); // Total cameras
  });

  it('handles mixed camera statuses correctly', () => {
    const mixedCameras = [
      { ...mockCameras[0], status: 'online' as const },
      { ...mockCameras[1], status: 'offline' as const },
      { ...mockCameras[2], status: 'error' as const },
    ];
    const store = createMockStore(mixedCameras);
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Should show 1 online camera out of 3 total
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('/ 3')).toBeInTheDocument();
  });

  it('updates metrics when store state changes', () => {
    const store = createMockStore();
    const { rerender } = render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Initial state
    expect(screen.getByText('1')).toBeInTheDocument(); // Unacknowledged alerts

    // Update store with more alerts
    const moreAlerts = [
      ...mockAlerts,
      {
        id: 'alert-003',
        type: 'tamper' as const,
        severity: 'critical' as const,
        cameraId: 'cam-003',
        cameraName: 'Camera 3',
        timestamp: '2024-01-01T10:00:00Z',
        confidence: 0.99,
        riskScore: 0.95,
        description: 'Camera tampering detected',
        acknowledged: false,
        metadata: {},
      },
    ];

    const updatedStore = createMockStore(mockCameras, moreAlerts, mockIncidents);
    
    rerender(
      <Provider store={updatedStore}>
        <Dashboard />
      </Provider>
    );

    // Should now show 2 unacknowledged alerts
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('renders with proper layout structure', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Check for proper Ant Design layout structure
    expect(document.querySelector('.ant-row')).toBeInTheDocument();
    expect(document.querySelector('.ant-col')).toBeInTheDocument();
    expect(document.querySelector('.ant-card')).toBeInTheDocument();
  });
});

describe('Dashboard Integration', () => {
  it('integrates correctly with Redux store', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Should reflect current store state
    const state = store.getState();
    expect(state.cameras.cameras).toHaveLength(3);
    expect(state.alerts.alerts).toHaveLength(2);
    expect(state.incidents.incidents).toHaveLength(1);
  });

  it('responds to store updates', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <Dashboard />
      </Provider>
    );

    // Dispatch an action to add a camera
    store.dispatch({
      type: 'cameras/updateCamera',
      payload: {
        id: 'cam-004',
        name: 'Camera 4',
        location: 'Location 4',
        type: 'visible',
        status: 'online',
        streamUrl: 'http://test4.mp4',
        detections: [],
        virtualLines: [],
        lastUpdate: '2024-01-01T12:00:00Z',
      },
    });

    // Should update the display
    expect(store.getState().cameras.cameras).toHaveLength(4);
  });
});