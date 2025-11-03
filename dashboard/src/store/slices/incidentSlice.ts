import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Evidence {
  id: string;
  type: 'image' | 'video' | 'metadata';
  filePath: string;
  thumbnail?: string;
  timestamp: string;
  description: string;
}

export interface Note {
  id: string;
  content: string;
  authorId: string;
  authorName: string;
  timestamp: string;
}

export interface Incident {
  id: string;
  alertId: string;
  title: string;
  description: string;
  status: 'open' | 'investigating' | 'resolved' | 'closed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignedTo?: string;
  assignedToName?: string;
  createdBy: string;
  createdByName: string;
  createdAt: string;
  updatedAt: string;
  closedAt?: string;
  evidence: Evidence[];
  notes: Note[];
  tags: string[];
  location: string;
  cameraId: string;
  cameraName: string;
}

interface IncidentState {
  incidents: Incident[];
  selectedIncident: string | null;
  filters: {
    status: string[];
    priority: string[];
    assignedTo: string[];
    tags: string[];
  };
  sortBy: 'createdAt' | 'updatedAt' | 'priority' | 'status';
  sortOrder: 'asc' | 'desc';
}

const initialState: IncidentState = {
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
};

const incidentSlice = createSlice({
  name: 'incidents',
  initialState,
  reducers: {
    setIncidents: (state, action: PayloadAction<Incident[]>) => {
      state.incidents = action.payload;
    },
    addIncident: (state, action: PayloadAction<Incident>) => {
      state.incidents.unshift(action.payload);
    },
    updateIncident: (state, action: PayloadAction<Incident>) => {
      const index = state.incidents.findIndex(incident => incident.id === action.payload.id);
      if (index !== -1) {
        state.incidents[index] = action.payload;
      }
    },
    addNote: (state, action: PayloadAction<{ incidentId: string; note: Note }>) => {
      const incident = state.incidents.find(inc => inc.id === action.payload.incidentId);
      if (incident) {
        incident.notes.push(action.payload.note);
        incident.updatedAt = new Date().toISOString();
      }
    },
    addEvidence: (state, action: PayloadAction<{ incidentId: string; evidence: Evidence }>) => {
      const incident = state.incidents.find(inc => inc.id === action.payload.incidentId);
      if (incident) {
        incident.evidence.push(action.payload.evidence);
        incident.updatedAt = new Date().toISOString();
      }
    },
    updateIncidentStatus: (state, action: PayloadAction<{ incidentId: string; status: Incident['status'] }>) => {
      const incident = state.incidents.find(inc => inc.id === action.payload.incidentId);
      if (incident) {
        incident.status = action.payload.status;
        incident.updatedAt = new Date().toISOString();
        if (action.payload.status === 'closed') {
          incident.closedAt = new Date().toISOString();
        }
      }
    },
    assignIncident: (state, action: PayloadAction<{ incidentId: string; assignedTo: string; assignedToName: string }>) => {
      const incident = state.incidents.find(inc => inc.id === action.payload.incidentId);
      if (incident) {
        incident.assignedTo = action.payload.assignedTo;
        incident.assignedToName = action.payload.assignedToName;
        incident.updatedAt = new Date().toISOString();
      }
    },
    setSelectedIncident: (state, action: PayloadAction<string | null>) => {
      state.selectedIncident = action.payload;
    },
    setFilters: (state, action: PayloadAction<Partial<IncidentState['filters']>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    setSorting: (state, action: PayloadAction<{ sortBy: IncidentState['sortBy']; sortOrder: IncidentState['sortOrder'] }>) => {
      state.sortBy = action.payload.sortBy;
      state.sortOrder = action.payload.sortOrder;
    },
  },
});

export const {
  setIncidents,
  addIncident,
  updateIncident,
  addNote,
  addEvidence,
  updateIncidentStatus,
  assignIncident,
  setSelectedIncident,
  setFilters,
  setSorting,
} = incidentSlice.actions;

export default incidentSlice.reducer;