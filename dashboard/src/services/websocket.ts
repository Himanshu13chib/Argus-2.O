import { io, Socket } from 'socket.io-client';
import { store } from '../store/store';
import { updateDetections, updateCameraStatus } from '../store/slices/cameraSlice';
import { addAlert } from '../store/slices/alertSlice';
import { setConnectionStatus, addNotification } from '../store/slices/systemSlice';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect() {
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    
    this.socket = io(wsUrl, {
      transports: ['websocket'],
      autoConnect: true,
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      store.dispatch(setConnectionStatus(true));
      store.dispatch(addNotification({
        type: 'success',
        message: 'Connected to Project Argus system',
      }));
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      store.dispatch(setConnectionStatus(false));
      store.dispatch(addNotification({
        type: 'warning',
        message: 'Disconnected from Project Argus system',
      }));
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.handleReconnect();
    });

    // Detection updates
    this.socket.on('detection_update', (data: {
      cameraId: string;
      detections: any[];
    }) => {
      store.dispatch(updateDetections(data));
    });

    // Camera status updates
    this.socket.on('camera_status', (data: {
      cameraId: string;
      status: 'online' | 'offline' | 'error';
    }) => {
      store.dispatch(updateCameraStatus(data));
    });

    // New alerts
    this.socket.on('new_alert', (alert: any) => {
      store.dispatch(addAlert(alert));
      store.dispatch(addNotification({
        type: 'error',
        message: `New ${alert.severity} alert: ${alert.description}`,
      }));
    });

    // System notifications
    this.socket.on('system_notification', (notification: {
      type: 'info' | 'warning' | 'error' | 'success';
      message: string;
    }) => {
      store.dispatch(addNotification(notification));
    });
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
      
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, delay);
    } else {
      store.dispatch(addNotification({
        type: 'error',
        message: 'Failed to reconnect to Project Argus system',
      }));
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // Send commands to backend
  acknowledgeAlert(alertId: string) {
    if (this.socket) {
      this.socket.emit('acknowledge_alert', { alertId });
    }
  }

  updateVirtualLines(cameraId: string, virtualLines: any[]) {
    if (this.socket) {
      this.socket.emit('update_virtual_lines', { cameraId, virtualLines });
    }
  }

  requestCameraStatus() {
    if (this.socket) {
      this.socket.emit('get_camera_status');
    }
  }
}

export const websocketService = new WebSocketService();