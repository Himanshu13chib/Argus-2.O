import React, { useEffect } from 'react';
import { Card, Row, Col, Statistic, Space, Tag, Button } from 'antd';
import { 
  VideoCameraOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { setCameras, updateCameraStatus } from '../store/slices/cameraSlice';
import CameraGrid from '../components/camera/CameraGrid';

const LiveFeeds: React.FC = () => {
  const dispatch = useDispatch();
  const { cameras } = useSelector((state: RootState) => state.cameras);

  // Mock data for development - in production this would come from API
  useEffect(() => {
    const mockCameras = [
      {
        id: 'cam-001',
        name: 'Border Sector Alpha',
        location: 'North Perimeter',
        type: 'visible' as const,
        status: 'online' as const,
        streamUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
        detections: [
          {
            id: 'det-001',
            bbox: { x: 100, y: 150, width: 80, height: 120 },
            confidence: 0.92,
            timestamp: new Date().toISOString(),
          }
        ],
        virtualLines: [
          {
            id: 'vl-001',
            points: [
              { x: 50, y: 200 },
              { x: 350, y: 180 },
              { x: 400, y: 250 }
            ],
            direction: 'both' as const,
            active: true,
          }
        ],
        lastUpdate: new Date().toISOString(),
      },
      {
        id: 'cam-002',
        name: 'Border Sector Beta',
        location: 'East Perimeter',
        type: 'thermal' as const,
        status: 'online' as const,
        streamUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
        detections: [],
        virtualLines: [
          {
            id: 'vl-002',
            points: [
              { x: 0, y: 300 },
              { x: 400, y: 280 }
            ],
            direction: 'in' as const,
            active: true,
          }
        ],
        lastUpdate: new Date().toISOString(),
      },
      {
        id: 'cam-003',
        name: 'Border Sector Gamma',
        location: 'South Perimeter',
        type: 'infrared' as const,
        status: 'offline' as const,
        streamUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
        detections: [],
        virtualLines: [],
        lastUpdate: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
      },
      {
        id: 'cam-004',
        name: 'Border Sector Delta',
        location: 'West Perimeter',
        type: 'visible' as const,
        status: 'error' as const,
        streamUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4',
        detections: [],
        virtualLines: [],
        lastUpdate: new Date(Date.now() - 600000).toISOString(), // 10 minutes ago
      },
    ];

    dispatch(setCameras(mockCameras));
  }, [dispatch]);

  const onlineCameras = cameras.filter(cam => cam.status === 'online').length;
  const offlineCameras = cameras.filter(cam => cam.status === 'offline').length;
  const errorCameras = cameras.filter(cam => cam.status === 'error').length;
  const totalDetections = cameras.reduce((sum, cam) => sum + cam.detections.length, 0);

  const handleRefresh = () => {
    // In production, this would refresh camera data from API
    cameras.forEach(camera => {
      const randomStatus = Math.random() > 0.8 ? 'offline' : 'online';
      dispatch(updateCameraStatus({ cameraId: camera.id, status: randomStatus }));
    });
  };

  return (
    <div style={{ padding: 16 }}>
      {/* Statistics Cards */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Cameras"
              value={cameras.length}
              prefix={<VideoCameraOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Online"
              value={onlineCameras}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Offline"
              value={offlineCameras}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Detections"
              value={totalDetections}
              prefix={<CloseCircleOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Camera Status Overview */}
      <Card 
        title="Camera Status Overview" 
        size="small" 
        style={{ marginBottom: 16 }}
        extra={
          <Button 
            icon={<ReloadOutlined />} 
            onClick={handleRefresh}
            size="small"
          >
            Refresh
          </Button>
        }
      >
        <Space wrap>
          {cameras.map(camera => (
            <Tag
              key={camera.id}
              color={
                camera.status === 'online' ? 'green' :
                camera.status === 'offline' ? 'orange' : 'red'
              }
              style={{ margin: '2px' }}
            >
              {camera.name} ({camera.type})
            </Tag>
          ))}
        </Space>
      </Card>

      {/* Camera Grid */}
      <CameraGrid showControls={true} />
    </div>
  );
};

export default LiveFeeds;