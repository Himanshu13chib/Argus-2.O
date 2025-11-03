import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Detection {
  id: string;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  timestamp: string;
}

export interface VirtualLine {
  id: string;
  points: Array<{ x: number; y: number }>;
  direction: 'in' | 'out' | 'both';
  active: boolean;
}

export interface Camera {
  id: string;
  name: string;
  location: string;
  type: 'visible' | 'thermal' | 'infrared';
  status: 'online' | 'offline' | 'error';
  streamUrl: string;
  detections: Detection[];
  virtualLines: VirtualLine[];
  lastUpdate: string;
}

interface CameraState {
  cameras: Camera[];
  selectedCamera: string | null;
  gridLayout: 1 | 2 | 4 | 6 | 9;
  fullscreenCamera: string | null;
  showDetections: boolean;
  showVirtualLines: boolean;
}

const initialState: CameraState = {
  cameras: [],
  selectedCamera: null,
  gridLayout: 4,
  fullscreenCamera: null,
  showDetections: true,
  showVirtualLines: true,
};

const cameraSlice = createSlice({
  name: 'cameras',
  initialState,
  reducers: {
    setCameras: (state, action: PayloadAction<Camera[]>) => {
      state.cameras = action.payload;
    },
    updateCamera: (state, action: PayloadAction<Camera>) => {
      const index = state.cameras.findIndex(cam => cam.id === action.payload.id);
      if (index !== -1) {
        state.cameras[index] = action.payload;
      } else {
        state.cameras.push(action.payload);
      }
    },
    updateDetections: (state, action: PayloadAction<{ cameraId: string; detections: Detection[] }>) => {
      const camera = state.cameras.find(cam => cam.id === action.payload.cameraId);
      if (camera) {
        camera.detections = action.payload.detections;
        camera.lastUpdate = new Date().toISOString();
      }
    },
    updateVirtualLines: (state, action: PayloadAction<{ cameraId: string; virtualLines: VirtualLine[] }>) => {
      const camera = state.cameras.find(cam => cam.id === action.payload.cameraId);
      if (camera) {
        camera.virtualLines = action.payload.virtualLines;
      }
    },
    setSelectedCamera: (state, action: PayloadAction<string | null>) => {
      state.selectedCamera = action.payload;
    },
    setGridLayout: (state, action: PayloadAction<1 | 2 | 4 | 6 | 9>) => {
      state.gridLayout = action.payload;
    },
    setFullscreenCamera: (state, action: PayloadAction<string | null>) => {
      state.fullscreenCamera = action.payload;
    },
    toggleDetections: (state) => {
      state.showDetections = !state.showDetections;
    },
    toggleVirtualLines: (state) => {
      state.showVirtualLines = !state.showVirtualLines;
    },
    updateCameraStatus: (state, action: PayloadAction<{ cameraId: string; status: Camera['status'] }>) => {
      const camera = state.cameras.find(cam => cam.id === action.payload.cameraId);
      if (camera) {
        camera.status = action.payload.status;
      }
    },
  },
});

export const {
  setCameras,
  updateCamera,
  updateDetections,
  updateVirtualLines,
  setSelectedCamera,
  setGridLayout,
  setFullscreenCamera,
  toggleDetections,
  toggleVirtualLines,
  updateCameraStatus,
} = cameraSlice.actions;

export default cameraSlice.reducer;