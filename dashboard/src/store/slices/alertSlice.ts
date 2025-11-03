import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Alert {
  id: string;
  type: 'crossing' | 'loitering' | 'tamper' | 'system';
  severity: 'low' | 'medium' | 'high' | 'critical';
  cameraId: string;
  cameraName: string;
  timestamp: string;
  confidence: number;
  riskScore: number;
  thumbnail?: string;
  description: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  metadata: Record<string, any>;
}

interface AlertState {
  alerts: Alert[];
  unacknowledgedCount: number;
  filters: {
    severity: string[];
    type: string[];
    camera: string[];
    acknowledged: boolean | null;
  };
  sortBy: 'timestamp' | 'severity' | 'confidence' | 'riskScore';
  sortOrder: 'asc' | 'desc';
  selectedAlert: string | null;
}

const initialState: AlertState = {
  alerts: [],
  unacknowledgedCount: 0,
  filters: {
    severity: [],
    type: [],
    camera: [],
    acknowledged: null,
  },
  sortBy: 'timestamp',
  sortOrder: 'desc',
  selectedAlert: null,
};

const alertSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    setAlerts: (state, action: PayloadAction<Alert[]>) => {
      state.alerts = action.payload;
      state.unacknowledgedCount = action.payload.filter(alert => !alert.acknowledged).length;
    },
    addAlert: (state, action: PayloadAction<Alert>) => {
      state.alerts.unshift(action.payload);
      if (!action.payload.acknowledged) {
        state.unacknowledgedCount += 1;
      }
    },
    updateAlert: (state, action: PayloadAction<Alert>) => {
      const index = state.alerts.findIndex(alert => alert.id === action.payload.id);
      if (index !== -1) {
        const wasAcknowledged = state.alerts[index].acknowledged;
        state.alerts[index] = action.payload;
        
        // Update unacknowledged count
        if (!wasAcknowledged && action.payload.acknowledged) {
          state.unacknowledgedCount -= 1;
        } else if (wasAcknowledged && !action.payload.acknowledged) {
          state.unacknowledgedCount += 1;
        }
      }
    },
    acknowledgeAlert: (state, action: PayloadAction<{ alertId: string; userId: string }>) => {
      const alert = state.alerts.find(alert => alert.id === action.payload.alertId);
      if (alert && !alert.acknowledged) {
        alert.acknowledged = true;
        alert.acknowledgedBy = action.payload.userId;
        alert.acknowledgedAt = new Date().toISOString();
        state.unacknowledgedCount -= 1;
      }
    },
    setFilters: (state, action: PayloadAction<Partial<AlertState['filters']>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    setSorting: (state, action: PayloadAction<{ sortBy: AlertState['sortBy']; sortOrder: AlertState['sortOrder'] }>) => {
      state.sortBy = action.payload.sortBy;
      state.sortOrder = action.payload.sortOrder;
    },
    setSelectedAlert: (state, action: PayloadAction<string | null>) => {
      state.selectedAlert = action.payload;
    },
    clearAlerts: (state) => {
      state.alerts = [];
      state.unacknowledgedCount = 0;
    },
  },
});

export const {
  setAlerts,
  addAlert,
  updateAlert,
  acknowledgeAlert,
  setFilters,
  setSorting,
  setSelectedAlert,
  clearAlerts,
} = alertSlice.actions;

export default alertSlice.reducer;