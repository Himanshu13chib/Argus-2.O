import React, { useEffect } from 'react';
import { Typography } from 'antd';
import { useDispatch } from 'react-redux';
import { setIncidents } from '../store/slices/incidentSlice';
import IncidentWorkflow from '../components/incidents/IncidentWorkflow';

const { Title } = Typography;

const Incidents: React.FC = () => {
  const dispatch = useDispatch();

  // Mock data for development
  useEffect(() => {
    const mockIncidents = [
      {
        id: 'incident-001',
        alertId: 'alert-001',
        title: 'Unauthorized Border Crossing - Sector Alpha',
        description: 'Person detected crossing virtual line boundary with high confidence. Requires immediate investigation and response.',
        status: 'investigating' as const,
        priority: 'high' as const,
        assignedTo: 'user-001',
        assignedToName: 'Border Operator',
        createdBy: 'user-001',
        createdByName: 'Border Operator',
        createdAt: new Date(Date.now() - 1800000).toISOString(), // 30 minutes ago
        updatedAt: new Date(Date.now() - 900000).toISOString(), // 15 minutes ago
        evidence: [
          {
            id: 'evidence-001',
            type: 'image' as const,
            filePath: '/evidence/crossing-001.jpg',
            thumbnail: 'https://via.placeholder.com/150x100/ff4d4f/ffffff?text=Crossing+Evidence',
            timestamp: new Date(Date.now() - 1700000).toISOString(),
            description: 'High-resolution image of person crossing boundary',
          },
          {
            id: 'evidence-002',
            type: 'video' as const,
            filePath: '/evidence/crossing-001.mp4',
            timestamp: new Date(Date.now() - 1600000).toISOString(),
            description: '30-second video clip of crossing event',
          },
        ],
        notes: [
          {
            id: 'note-001',
            content: 'Initial assessment completed. Person appears to be carrying a backpack. Notified patrol unit for field response.',
            authorId: 'user-001',
            authorName: 'Border Operator',
            timestamp: new Date(Date.now() - 1500000).toISOString(),
          },
          {
            id: 'note-002',
            content: 'Patrol unit dispatched to coordinates. ETA 15 minutes.',
            authorId: 'user-001',
            authorName: 'Border Operator',
            timestamp: new Date(Date.now() - 900000).toISOString(),
          },
        ],
        tags: ['border-crossing', 'high-priority', 'patrol-dispatched'],
        location: 'Border Sector Alpha - Grid Reference: 34.123, -118.456',
        cameraId: 'cam-001',
        cameraName: 'Border Sector Alpha',
      },
      {
        id: 'incident-002',
        alertId: 'alert-003',
        title: 'Camera Tampering Detected - Sector Gamma',
        description: 'Critical tamper alert indicating possible lens obstruction or camera movement. Immediate technical response required.',
        status: 'open' as const,
        priority: 'critical' as const,
        assignedTo: 'user-002',
        assignedToName: 'Technical Operator',
        createdBy: 'user-001',
        createdByName: 'Border Operator',
        createdAt: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
        updatedAt: new Date(Date.now() - 3600000).toISOString(),
        evidence: [
          {
            id: 'evidence-003',
            type: 'metadata' as const,
            filePath: '/evidence/tamper-log-003.json',
            timestamp: new Date(Date.now() - 3500000).toISOString(),
            description: 'System logs showing camera status changes',
          },
        ],
        notes: [],
        tags: ['camera-tamper', 'critical', 'technical-issue'],
        location: 'Border Sector Gamma - Camera Mount Position',
        cameraId: 'cam-003',
        cameraName: 'Border Sector Gamma',
      },
      {
        id: 'incident-003',
        alertId: 'alert-005',
        title: 'Multiple Person Crossing Event',
        description: 'Group of three individuals detected crossing boundary simultaneously. Coordinated response initiated.',
        status: 'resolved' as const,
        priority: 'high' as const,
        assignedTo: 'user-001',
        assignedToName: 'Border Operator',
        createdBy: 'user-001',
        createdByName: 'Border Operator',
        createdAt: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
        updatedAt: new Date(Date.now() - 5400000).toISOString(), // 1.5 hours ago
        closedAt: new Date(Date.now() - 5400000).toISOString(),
        evidence: [
          {
            id: 'evidence-004',
            type: 'image' as const,
            filePath: '/evidence/group-crossing-001.jpg',
            thumbnail: 'https://via.placeholder.com/150x100/faad14/ffffff?text=Group+Crossing',
            timestamp: new Date(Date.now() - 7100000).toISOString(),
            description: 'Image showing three individuals crossing together',
          },
          {
            id: 'evidence-005',
            type: 'video' as const,
            filePath: '/evidence/group-crossing-001.mp4',
            timestamp: new Date(Date.now() - 7000000).toISOString(),
            description: 'Full video sequence of group crossing event',
          },
        ],
        notes: [
          {
            id: 'note-003',
            content: 'Group of 3 individuals detected. Patrol units Alpha and Beta dispatched.',
            authorId: 'user-001',
            authorName: 'Border Operator',
            timestamp: new Date(Date.now() - 7000000).toISOString(),
          },
          {
            id: 'note-004',
            content: 'Individuals apprehended by patrol units. Processing at checkpoint facility.',
            authorId: 'user-001',
            authorName: 'Border Operator',
            timestamp: new Date(Date.now() - 6000000).toISOString(),
          },
          {
            id: 'note-005',
            content: 'Incident resolved. All individuals processed according to protocol.',
            authorId: 'user-001',
            authorName: 'Border Operator',
            timestamp: new Date(Date.now() - 5400000).toISOString(),
          },
        ],
        tags: ['group-crossing', 'resolved', 'apprehended'],
        location: 'Border Sector Alpha - Multiple crossing points',
        cameraId: 'cam-001',
        cameraName: 'Border Sector Alpha',
      },
    ];

    dispatch(setIncidents(mockIncidents));
  }, [dispatch]);

  return (
    <div style={{ padding: 16 }}>
      <Title level={2} style={{ marginBottom: 24 }}>
        Incident Management
      </Title>
      <IncidentWorkflow showFilters={true} />
    </div>
  );
};

export default Incidents;