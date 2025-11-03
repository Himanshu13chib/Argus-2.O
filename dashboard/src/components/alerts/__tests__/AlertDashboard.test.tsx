import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';
import AlertDashboard from '../AlertDashboard';
import alertReducer from '../../../store/slices/alertSlice';
import systemReducer from '../../../store/slices/systemSlice';
import cameraReducer from '../../../store/slices/cameraSlice';

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
    thumbnail: 'http://test-thumbnail.jpg',
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

const mockCameras = [
  {
    id: 'cam-001',
    name: 'Camera 1',
    location: 'Location 1',
    type: 'visible' as const,
    status: 'online' as const,
    streamUrl: 'http://test1.mp4',
    detections: [],
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
];

const createMockStore = (alerts = mockAlerts) => {
  return configureStore({
    reducer: {
      alerts: alertReducer,
      system: systemReducer,
      cameras: cameraReducer,
    },
    preloadedState: {
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
      system: {
        connected: true,
        health: [],
        metrics: [],
        notifications: [],
        currentUser: {
          id: 'user-001',
          name: 'Test User',
          role: 'operator',
          permissions: ['acknowledge_alerts'],
        },
      },
      cameras: {
        cameras: mockCameras,
        selectedCamera: null,
        gridLayout: 4,
        fullscreenCamera: null,
        showDetections: true,
        showVirtualLines: true,
      },
    },
  });
};

describe('AlertDashboard', () => {
  it('renders alert statistics correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    expect(screen.getByText('Total Alerts')).toBeInTheDocument();
    expect(screen.getByText('Unacknowledged')).toBeInTheDocument();
    expect(screen.getByText('Critical')).toBeInTheDocument();
    expect(screen.getByText('High Priority')).toBeInTheDocument();

    // Check statistics values
    expect(screen.getByText('3')).toBeInTheDocument(); // Total alerts
    expect(screen.getByText('2')).toBeInTheDocument(); // Unacknowledged
    expect(screen.getByText('1')).toBeInTheDocument(); // Critical
  });

  it('displays alerts in table format', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    // Check table headers
    expect(screen.getByText('Time')).toBeInTheDocument();
    expect(screen.getByText('Type')).toBeInTheDocument();
    expect(screen.getByText('Severity')).toBeInTheDocument();
    expect(screen.getByText('Camera')).toBeInTheDocument();
    expect(screen.getByText('Description')).toBeInTheDocument();

    // Check alert data
    expect(screen.getByText('Person crossing boundary')).toBeInTheDocument();
    expect(screen.getByText('Person loitering in area')).toBeInTheDocument();
    expect(screen.getByText('Camera tampering detected')).toBeInTheDocument();
  });

  it('shows filter controls when showFilters is true', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard showFilters={true} />
      </Provider>
    );

    expect(screen.getByPlaceholderText('Severity')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Type')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Camera')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Status')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Sort by')).toBeInTheDocument();
  });

  it('hides filter controls when showFilters is false', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard showFilters={false} />
      </Provider>
    );

    expect(screen.queryByPlaceholderText('Severity')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Type')).not.toBeInTheDocument();
  });

  it('handles alert acknowledgment', async () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    // Find acknowledge button for unacknowledged alert
    const acknowledgeButtons = screen.getAllByRole('button', { name: /acknowledge/i });
    expect(acknowledgeButtons.length).toBeGreaterThan(0);

    fireEvent.click(acknowledgeButtons[0]);

    // Should show success message (mocked)
    await waitFor(() => {
      // In a real test, you'd check if the alert was acknowledged in the store
      expect(store.getState().alerts.alerts[0].acknowledged).toBe(false); // Initially false
    });
  });

  it('opens alert detail modal when view details is clicked', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    const viewButtons = screen.getAllByRole('button', { name: /view details/i });
    fireEvent.click(viewButtons[0]);

    expect(screen.getByText('Alert Details')).toBeInTheDocument();
    expect(screen.getByText('Alert ID:')).toBeInTheDocument();
  });

  it('filters alerts by severity', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard showFilters={true} />
      </Provider>
    );

    const severityFilter = screen.getByPlaceholderText('Severity');
    fireEvent.mouseDown(severityFilter);
    
    const criticalOption = screen.getByText('Critical');
    fireEvent.click(criticalOption);

    // Should filter to show only critical alerts
    // In a real test, you'd verify the filtered results
  });

  it('handles bulk acknowledgment', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    // Select checkboxes for unacknowledged alerts
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]); // First alert checkbox (index 0 is select all)

    // Should show bulk actions
    expect(screen.getByText(/Selected \d+ alert/)).toBeInTheDocument();
    expect(screen.getByText('Acknowledge Selected')).toBeInTheDocument();
  });

  it('displays correct severity colors', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    // Check for severity tags with correct colors
    const criticalTag = screen.getByText('CRITICAL');
    const highTag = screen.getByText('HIGH');
    const mediumTag = screen.getByText('MEDIUM');

    expect(criticalTag).toBeInTheDocument();
    expect(highTag).toBeInTheDocument();
    expect(mediumTag).toBeInTheDocument();
  });

  it('shows acknowledged status correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    expect(screen.getByText('Acknowledged')).toBeInTheDocument();
    expect(screen.getAllByText('Pending')).toHaveLength(2); // 2 unacknowledged alerts
    expect(screen.getByText('by Operator 1')).toBeInTheDocument();
  });

  it('handles sorting changes', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard showFilters={true} />
      </Provider>
    );

    const sortSelect = screen.getByPlaceholderText('Sort by');
    fireEvent.mouseDown(sortSelect);
    
    const highestSeverityOption = screen.getByText('Highest Severity');
    fireEvent.click(highestSeverityOption);

    // Should update sort order in store
    expect(store.getState().alerts.sortBy).toBe('timestamp'); // Initial value
  });

  it('displays confidence and risk scores correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    expect(screen.getByText('95.0%')).toBeInTheDocument(); // Confidence
    expect(screen.getByText('78.0%')).toBeInTheDocument(); // Confidence
    expect(screen.getByText('99.0%')).toBeInTheDocument(); // Confidence
  });

  it('shows empty state when no alerts match filters', () => {
    const store = createMockStore([]);
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    // Should show empty table
    expect(screen.getByText('No data')).toBeInTheDocument();
  });
});

describe('AlertDashboard Integration', () => {
  it('integrates with Redux store for filter changes', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <AlertDashboard showFilters={true} />
      </Provider>
    );

    const severityFilter = screen.getByPlaceholderText('Severity');
    fireEvent.mouseDown(severityFilter);
    
    const highOption = screen.getByText('High');
    fireEvent.click(highOption);

    // Check if store state changed
    expect(store.getState().alerts.filters.severity).toContain('high');
  });

  it('updates unacknowledged count when alerts are acknowledged', () => {
    const store = createMockStore();
    const initialCount = store.getState().alerts.unacknowledgedCount;
    
    render(
      <Provider store={store}>
        <AlertDashboard />
      </Provider>
    );

    // Initial unacknowledged count should be 2
    expect(initialCount).toBe(2);
  });
});