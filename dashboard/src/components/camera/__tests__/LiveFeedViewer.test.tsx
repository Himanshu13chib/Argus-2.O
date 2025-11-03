import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';
import LiveFeedViewer from '../LiveFeedViewer';
import cameraReducer from '../../../store/slices/cameraSlice';
import systemReducer from '../../../store/slices/systemSlice';

// Mock video element
Object.defineProperty(HTMLVideoElement.prototype, 'load', {
  writable: true,
  value: jest.fn(),
});

Object.defineProperty(HTMLVideoElement.prototype, 'play', {
  writable: true,
  value: jest.fn().mockResolvedValue(undefined),
});

// Mock canvas context
const mockGetContext = jest.fn(() => ({
  clearRect: jest.fn(),
  strokeRect: jest.fn(),
  fillText: jest.fn(),
  beginPath: jest.fn(),
  moveTo: jest.fn(),
  lineTo: jest.fn(),
  stroke: jest.fn(),
  setLineDash: jest.fn(),
}));

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  writable: true,
  value: mockGetContext,
});

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

const mockStore = configureStore({
  reducer: {
    cameras: cameraReducer,
    system: systemReducer,
  },
  preloadedState: {
    cameras: {
      cameras: [],
      selectedCamera: null,
      gridLayout: 4,
      fullscreenCamera: null,
      showDetections: true,
      showVirtualLines: true,
    },
    system: {
      connected: true,
      health: [],
      metrics: [],
      notifications: [],
      currentUser: null,
    },
  },
});

const mockCamera = {
  id: 'cam-001',
  name: 'Test Camera',
  location: 'Test Location',
  type: 'visible' as const,
  status: 'online' as const,
  streamUrl: 'http://test-stream.mp4',
  detections: [
    {
      id: 'det-001',
      bbox: { x: 100, y: 100, width: 50, height: 80 },
      confidence: 0.95,
      timestamp: '2024-01-01T12:00:00Z',
    },
  ],
  virtualLines: [
    {
      id: 'vl-001',
      points: [
        { x: 0, y: 200 },
        { x: 400, y: 200 },
      ],
      direction: 'both' as const,
      active: true,
    },
  ],
  lastUpdate: '2024-01-01T12:00:00Z',
};

describe('LiveFeedViewer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders camera feed with basic information', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} />
      </Provider>
    );

    expect(screen.getByText('Test Camera')).toBeInTheDocument();
    expect(screen.getByText('(visible)')).toBeInTheDocument();
  });

  it('displays detection count badge when detections are present', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} />
      </Provider>
    );

    // Check for alert badge indicating active detections
    const alertIcon = screen.getByRole('img', { name: /alert/i });
    expect(alertIcon).toBeInTheDocument();
  });

  it('shows control buttons when showControls is true', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} showControls={true} />
      </Provider>
    );

    // Check for control buttons
    expect(screen.getByRole('button', { name: /show detections|hide detections/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /show virtual lines|hide virtual lines/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /fullscreen/i })).toBeInTheDocument();
  });

  it('hides control buttons when showControls is false', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} showControls={false} />
      </Provider>
    );

    // Control buttons should not be present
    expect(screen.queryByRole('button', { name: /show detections|hide detections/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /fullscreen/i })).not.toBeInTheDocument();
  });

  it('displays loading spinner initially', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} />
      </Provider>
    );

    expect(screen.getByRole('img', { name: /loading/i })).toBeInTheDocument();
  });

  it('shows error state when camera is offline', () => {
    const offlineCamera = { ...mockCamera, status: 'offline' as const };
    
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={offlineCamera} />
      </Provider>
    );

    // Should show offline status
    expect(screen.getByText('Camera Offline')).toBeInTheDocument();
    expect(screen.getByText('Unable to connect to video stream')).toBeInTheDocument();
  });

  it('handles fullscreen toggle', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} showControls={true} />
      </Provider>
    );

    const fullscreenButton = screen.getByRole('button', { name: /fullscreen/i });
    fireEvent.click(fullscreenButton);

    // Should dispatch fullscreen action (tested via Redux store state)
    // In a real test, you'd check if the action was dispatched
  });

  it('renders video element with correct attributes', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} />
      </Provider>
    );

    const video = screen.getByRole('application'); // video elements have application role
    expect(video).toHaveAttribute('autoplay');
    expect(video).toHaveAttribute('muted');
    expect(video).toHaveAttribute('playsinline');
  });

  it('renders canvas overlay for detections and virtual lines', () => {
    render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} />
      </Provider>
    );

    // Canvas should be present for overlays
    const canvas = document.querySelector('canvas');
    expect(canvas).toBeInTheDocument();
    expect(canvas).toHaveClass('video-overlay');
  });

  it('applies correct status color based on camera status', () => {
    const { rerender } = render(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={mockCamera} />
      </Provider>
    );

    // Test online status (success badge)
    expect(screen.getByRole('img', { name: /success/i })).toBeInTheDocument();

    // Test error status
    const errorCamera = { ...mockCamera, status: 'error' as const };
    rerender(
      <Provider store={mockStore}>
        <LiveFeedViewer camera={errorCamera} />
      </Provider>
    );

    expect(screen.getByRole('img', { name: /error/i })).toBeInTheDocument();
  });
});

describe('LiveFeedViewer Integration', () => {
  it('integrates with Redux store for detection visibility toggle', () => {
    const store = configureStore({
      reducer: {
        cameras: cameraReducer,
        system: systemReducer,
      },
      preloadedState: {
        cameras: {
          cameras: [mockCamera],
          selectedCamera: null,
          gridLayout: 4,
          fullscreenCamera: null,
          showDetections: false, // Initially hidden
          showVirtualLines: true,
        },
        system: {
          connected: true,
          health: [],
          metrics: [],
          notifications: [],
          currentUser: null,
        },
      },
    });

    render(
      <Provider store={store}>
        <LiveFeedViewer camera={mockCamera} showControls={true} />
      </Provider>
    );

    const toggleButton = screen.getByRole('button', { name: /show detections/i });
    fireEvent.click(toggleButton);

    // Check if store state changed
    expect(store.getState().cameras.showDetections).toBe(true);
  });
});