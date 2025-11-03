import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface SystemHealth {
  component: string;
  status: 'healthy' | 'warning' | 'error';
  message: string;
  lastCheck: string;
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
  activeConnections: number;
  timestamp: string;
}

interface SystemState {
  connected: boolean;
  health: SystemHealth[];
  metrics: SystemMetrics[];
  notifications: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: string;
    read: boolean;
  }>;
  currentUser: {
    id: string;
    name: string;
    role: 'operator' | 'auditor' | 'administrator';
    permissions: string[];
  } | null;
}

const initialState: SystemState = {
  connected: false,
  health: [],
  metrics: [],
  notifications: [],
  currentUser: null,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.connected = action.payload;
    },
    setSystemHealth: (state, action: PayloadAction<SystemHealth[]>) => {
      state.health = action.payload;
    },
    updateSystemMetrics: (state, action: PayloadAction<SystemMetrics>) => {
      state.metrics.push(action.payload);
      // Keep only last 100 metrics
      if (state.metrics.length > 100) {
        state.metrics = state.metrics.slice(-100);
      }
    },
    addNotification: (state, action: PayloadAction<Omit<SystemState['notifications'][0], 'id' | 'timestamp' | 'read'>>) => {
      const notification = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        read: false,
      };
      state.notifications.unshift(notification);
    },
    markNotificationRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setCurrentUser: (state, action: PayloadAction<SystemState['currentUser']>) => {
      state.currentUser = action.payload;
    },
  },
});

export const {
  setConnectionStatus,
  setSystemHealth,
  updateSystemMetrics,
  addNotification,
  markNotificationRead,
  clearNotifications,
  setCurrentUser,
} = systemSlice.actions;

export default systemSlice.reducer;