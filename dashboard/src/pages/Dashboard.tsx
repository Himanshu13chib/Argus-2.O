import React from 'react';
import { Card, Row, Col, Statistic, Space, Typography } from 'antd';
import { 
  VideoCameraOutlined, 
  AlertOutlined, 
  FileTextOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';
import CameraGrid from '../components/camera/CameraGrid';

const { Title } = Typography;

const Dashboard: React.FC = () => {
  const { cameras } = useSelector((state: RootState) => state.cameras);
  const { alerts, unacknowledgedCount } = useSelector((state: RootState) => state.alerts);
  const { incidents } = useSelector((state: RootState) => state.incidents);

  const onlineCameras = cameras.filter(cam => cam.status === 'online').length;
  const totalDetections = cameras.reduce((sum, cam) => sum + cam.detections.length, 0);
  const openIncidents = incidents.filter(inc => inc.status === 'open').length;

  return (
    <div style={{ padding: 16 }}>
      <Title level={2} style={{ marginBottom: 24 }}>
        Project Argus Command Center
      </Title>

      {/* Key Metrics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Cameras"
              value={onlineCameras}
              suffix={`/ ${cameras.length}`}
              prefix={<VideoCameraOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Unacknowledged Alerts"
              value={unacknowledgedCount}
              prefix={<AlertOutlined />}
              valueStyle={{ color: unacknowledgedCount > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Open Incidents"
              value={openIncidents}
              prefix={<FileTextOutlined />}
              valueStyle={{ color: openIncidents > 0 ? '#faad14' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Detections"
              value={totalDetections}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: totalDetections > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Live Camera Grid - Show first 4 cameras */}
      <Card title="Live Camera Feeds" style={{ marginBottom: 16 }}>
        <CameraGrid 
          selectedCameras={cameras.slice(0, 4).map(cam => cam.id)}
          showControls={false}
        />
      </Card>
    </div>
  );
};

export default Dashboard;