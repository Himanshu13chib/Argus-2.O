import React, { useEffect } from 'react';
import { Typography } from 'antd';
import { useDispatch } from 'react-redux';
import { setAlerts } from '../store/slices/alertSlice';
import AlertDashboard from '../components/alerts/AlertDashboard';

const { Title } = Typography;

const Alerts: React.FC = () => {
  const dispatch = useDispatch();

  // Mock data for development
  useEffect(() => {
    const mockAlerts = [
      {
        id: 'alert-001',
        type: 'crossing' as const,
        severity: 'high' as const,
        cameraId: 'cam-001',
        cameraName: 'Border Sector Alpha',
        timestamp: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
        confidence: 0.92,
        riskScore: 0.85,
        thumbnail: 'https://via.placeholder.com/300x200/ff4d4f/ffffff?text=Person+Detected',
        description: 'Person detected crossing virtual line boundary',
        acknowledged: false,
        metadata: {
          detectionId: 'det-001',
          crossingDirection: 'inbound',
          virtualLineId: 'vl-001',
        },
      },
      {
        id: 'alert-002',
        type: 'loitering' as const,
        severity: 'medium' as const,
        cameraId: 'cam-002',
        cameraName: 'Border Sector Beta',
        timestamp: new Date(Date.now() - 900000).toISOString(), // 15 minutes ago
        confidence: 0.78,
        riskScore: 0.65,
        description: 'Person detected loitering in restricted area for 10+ minutes',
        acknowledged: true,
        acknowledgedBy: 'Operator Smith',
        acknowledgedAt: new Date(Date.now() - 600000).toISOString(), // 10 minutes ago
        metadata: {
          loiteringDuration: 600, // 10 minutes
          area: 'restricted-zone-2',
        },
      },
      {
        id: 'alert-003',
        type: 'tamper' as const,
        severity: 'critical' as const,
        cameraId: 'cam-003',
        cameraName: 'Border Sector Gamma',
        timestamp: new Date(Date.now() - 1800000).toISOString(), // 30 minutes ago
        confidence: 0.95,
        riskScore: 0.95,
        description: 'Camera lens obstruction detected - possible tampering attempt',
        acknowledged: false,
        metadata: {
          tamperType: 'lens_obstruction',
          obstructionPercentage: 85,
        },
      },
      {
        id: 'alert-004',
        type: 'system' as const,
        severity: 'low' as const,
        cameraId: 'cam-004',
        cameraName: 'Border Sector Delta',
        timestamp: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
        confidence: 1.0,
        riskScore: 0.3,
        description: 'Network connectivity restored after brief interruption',
        acknowledged: true,
        acknowledgedBy: 'System Admin',
        acknowledgedAt: new Date(Date.now() - 3300000).toISOString(), // 55 minutes ago
        metadata: {
          downtime: 180, // 3 minutes
          cause: 'network_interruption',
        },
      },
      {
        id: 'alert-005',
        type: 'crossing' as const,
        severity: 'critical' as const,
        cameraId: 'cam-001',
        cameraName: 'Border Sector Alpha',
        timestamp: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
        confidence: 0.88,
        riskScore: 0.92,
        thumbnail: 'https://via.placeholder.com/300x200/ff4d4f/ffffff?text=Multiple+Persons',
        description: 'Multiple persons detected crossing boundary simultaneously',
        acknowledged: true,
        acknowledgedBy: 'Border Operator',
        acknowledgedAt: new Date(Date.now() - 6900000).toISOString(), // 1h 55m ago
        metadata: {
          personCount: 3,
          crossingDirection: 'inbound',
          groupMovement: true,
        },
      },
    ];

    dispatch(setAlerts(mockAlerts));
  }, [dispatch]);

  return (
    <div style={{ padding: 16 }}>
      <Title level={2} style={{ marginBottom: 24 }}>
        Alert Management
      </Title>
      <AlertDashboard showFilters={true} />
    </div>
  );
};

export default Alerts;