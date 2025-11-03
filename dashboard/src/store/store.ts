import { configureStore } from '@reduxjs/toolkit';
import cameraReducer from './slices/cameraSlice';
import alertReducer from './slices/alertSlice';
import incidentReducer from './slices/incidentSlice';
import systemReducer from './slices/systemSlice';

export const store = configureStore({
  reducer: {
    cameras: cameraReducer,
    alerts: alertReducer,
    incidents: incidentReducer,
    system: systemReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;