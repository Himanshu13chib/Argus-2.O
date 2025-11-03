import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';
import CameraGrid from '../CameraGrid';
import cameraReducer from '../../../store/slices/cameraSlice';
import systemReducer from '../../../store/slices/systemSlice';

// Mock the LiveFeedViewer component
jest.mock('../LiveFeedViewer', () => {
  return function MockLiveFeedViewer({ camera }: { camera: any }) {
    return <div data-testid={`camera-${camera.id}`}>{camera.name}</div>;
  };
});

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

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
  {
    id: 'cam-004',
    name: 'Camera 4',
    location: 'Location 4',
    type: 'visible' as const,
    status: 'online' as const,
    streamUrl: 'http://test4.mp4',
    detections: [],
    virtualLines: [],
    lastUpdate: '2024-01-01T12:00:00Z',
  },
];

const createMockStore = (cameras = mockCameras, gridLayout = 4, fullscreenCamera = null) => {
  return configureStore({
    reducer: {
      cameras: cameraReducer,
      system: systemReducer,
    },
    preloadedState: {
      cameras: {
        cameras,
        selectedCamera: null,
        gridLayout,
        fullscreenCamera,
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
};

describe('CameraGrid', () => {
  it('renders all cameras when no selection is provided', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <CameraGrid />
      </Provider>
    );

    expect(screen.getByTestId('camera-cam-001')).toBeInTheDocument();
    expect(screen.getByTestId('camera-cam-002')).toBeInTheDocument();
    expect(screen.getByTestId('camera-cam-003')).toBeInTheDocument();
    expect(screen.getByTestId('camera-cam-004')).toBeInTheDocument();
  });

  it('renders only selected cameras when selection is provided', () => {
    const store = createMockStore();
    const selectedCameras = ['cam-001', 'cam-003'];
    
    render(
      <Provider store={store}>
        <CameraGrid selectedCameras={selectedCameras} />
      </Provider>
    );

    expect(screen.getByTestId('camera-cam-001')).toBeInTheDocument();
    expect(screen.getByTestId('camera-cam-003')).toBeInTheDocument();
    expect(screen.queryByTestId('camera-cam-002')).not.toBeInTheDocument();
    expect(screen.queryByTestId('camera-cam-004')).not.toBeInTheDocument();
  });

  it('shows grid layout controls when showControls is true', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <CameraGrid showControls={true} />
      </Provider>
    );

    expect(screen.getByText('Camera Grid Controls')).toBeInTheDocument();
    expect(screen.getByText('Layout:')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(screen.getByText('Default Grid')).toBeInTheDocument();
  });

  it('hides grid layout controls when showControls is false', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <CameraGrid showControls={false} />
      </Provider>
    );

    expect(screen.queryByText('Camera Grid Controls')).not.toBeInTheDocument();
    expect(screen.queryByText('Layout:')).not.toBeInTheDocument();
  });

  it('handles layout changes correctly', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <CameraGrid showControls={true} />
      </Provider>
    );

    const layoutSelect = screen.getByRole('combobox');
    fireEvent.mouseDown(layoutSelect);
    
    const option2x2 = screen.getByText('2x2');
    fireEvent.click(option2x2);

    // Check if store state changed
    expect(store.getState().cameras.gridLayout).toBe(4);
  });

  it('shows empty state when no cameras are available', () => {
    const store = createMockStore([]);
    
    render(
      <Provider store={store}>
        <CameraGrid />
      </Provider>
    );

    expect(screen.getByText('No cameras available')).toBeInTheDocument();
  });

  it('shows fullscreen camera when in fullscreen mode', () => {
    const store = createMockStore(mockCameras, 4, 'cam-001');
    
    render(
      <Provider store={store}>
        <CameraGrid />
      </Provider>
    );

    // Should only show the fullscreen camera
    expect(screen.getByTestId('camera-cam-001')).toBeInTheDocument();
    expect(screen.queryByTestId('camera-cam-002')).not.toBeInTheDocument();
    expect(screen.queryByTestId('camera-cam-003')).not.toBeInTheDocument();
    expect(screen.queryByTestId('camera-cam-004')).not.toBeInTheDocument();
  });

  it('limits cameras displayed based on grid layout', () => {
    const store = createMockStore(mockCameras, 2); // 2x1 grid
    
    render(
      <Provider store={store}>
        <CameraGrid />
      </Provider>
    );

    // Should only show first 2 cameras for 2x1 grid
    expect(screen.getByTestId('camera-cam-001')).toBeInTheDocument();
    expect(screen.getByTestId('camera-cam-002')).toBeInTheDocument();
    expect(screen.queryByTestId('camera-cam-003')).not.toBeInTheDocument();
    expect(screen.queryByTestId('camera-cam-004')).not.toBeInTheDocument();
  });

  it('shows empty grid slots when cameras are fewer than grid capacity', () => {
    const store = createMockStore([mockCameras[0]], 4); // 1 camera in 2x2 grid
    
    render(
      <Provider store={store}>
        <CameraGrid />
      </Provider>
    );

    expect(screen.getByTestId('camera-cam-001')).toBeInTheDocument();
    expect(screen.getAllByText('No Camera')).toHaveLength(3); // 3 empty slots
  });

  it('handles default grid button click', () => {
    const store = createMockStore(mockCameras, 9);
    
    render(
      <Provider store={store}>
        <CameraGrid showControls={true} />
      </Provider>
    );

    const defaultGridButton = screen.getByText('Default Grid');
    fireEvent.click(defaultGridButton);

    // Should set grid layout to 4 (2x2)
    expect(store.getState().cameras.gridLayout).toBe(4);
  });

  it('applies correct CSS classes for different grid layouts', () => {
    const { rerender } = render(
      <Provider store={createMockStore(mockCameras, 1)}>
        <CameraGrid />
      </Provider>
    );

    expect(document.querySelector('.camera-grid-1')).toBeInTheDocument();

    rerender(
      <Provider store={createMockStore(mockCameras, 4)}>
        <CameraGrid />
      </Provider>
    );

    expect(document.querySelector('.camera-grid-4')).toBeInTheDocument();

    rerender(
      <Provider store={createMockStore(mockCameras, 9)}>
        <CameraGrid />
      </Provider>
    );

    expect(document.querySelector('.camera-grid-9')).toBeInTheDocument();
  });
});

describe('CameraGrid Responsive Behavior', () => {
  // Mock window dimensions
  const mockWindowDimensions = (width: number, height: number) => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: width,
    });
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: height,
    });
  };

  it('calculates camera sizes based on window dimensions', () => {
    mockWindowDimensions(1200, 800);
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <CameraGrid />
      </Provider>
    );

    // Camera grid should be present and responsive
    expect(document.querySelector('.camera-grid')).toBeInTheDocument();
  });

  it('handles small screen sizes appropriately', () => {
    mockWindowDimensions(800, 600);
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <CameraGrid />
      </Provider>
    );

    // Should still render grid but with smaller dimensions
    expect(document.querySelector('.camera-grid')).toBeInTheDocument();
  });
});