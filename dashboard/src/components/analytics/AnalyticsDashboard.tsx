import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Select,
  DatePicker,
  Button,
  Space,
  Typography,
  Table,
  Progress,
  Tag,
  Divider,
} from 'antd';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  TrendingUpOutlined,
  TrendingDownOutlined,
  AlertOutlined,
  VideoCameraOutlined,
  ClockCircleOutlined,
  DownloadOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useSelector } from 'react-redux';
import dayjs from 'dayjs';
import { RootState } from '../../store/store';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface AnalyticsDashboardProps {
  timeRange?: [string, string];
}

const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ timeRange }) => {
  const { cameras } = useSelector((state: RootState) => state.cameras);
  const { alerts } = useSelector((state: RootState) => state.alerts);
  const { incidents } = useSelector((state: RootState) => state.incidents);

  const [selectedTimeRange, setSelectedTimeRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(7, 'days'),
    dayjs(),
  ]);
  const [selectedMetric, setSelectedMetric] = useState<string>('detections');

  // Mock analytics data - in production this would come from API
  const [analyticsData, setAnalyticsData] = useState({
    detectionTrends: [
      { date: '2024-01-01', detections: 45, alerts: 12, incidents: 3 },
      { date: '2024-01-02', detections: 52, alerts: 15, incidents: 4 },
      { date: '2024-01-03', detections: 38, alerts: 8, incidents: 2 },
      { date: '2024-01-04', detections: 61, alerts: 18, incidents: 5 },
      { date: '2024-01-05', detections: 43, alerts: 11, incidents: 3 },
      { date: '2024-01-06', detections: 55, alerts: 14, incidents: 4 },
      { date: '2024-01-07', detections: 49, alerts: 13, incidents: 3 },
    ],
    hourlyActivity: [
      { hour: '00:00', activity: 5 },
      { hour: '02:00', activity: 3 },
      { hour: '04:00', activity: 8 },
      { hour: '06:00', activity: 15 },
      { hour: '08:00', activity: 12 },
      { hour: '10:00', activity: 18 },
      { hour: '12:00', activity: 22 },
      { hour: '14:00', activity: 25 },
      { hour: '16:00', activity: 20 },
      { hour: '18:00', activity: 28 },
      { hour: '20:00', activity: 32 },
      { hour: '22:00', activity: 18 },
    ],
    cameraPerformance: [
      { camera: 'Sector Alpha', uptime: 99.2, detections: 156, alerts: 42 },
      { camera: 'Sector Beta', uptime: 98.8, detections: 134, alerts: 38 },
      { camera: 'Sector Gamma', uptime: 95.5, detections: 89, alerts: 25 },
      { camera: 'Sector Delta', uptime: 97.3, detections: 112, alerts: 31 },
    ],
    alertTypes: [
      { name: 'Crossing', value: 65, color: '#ff4d4f' },
      { name: 'Loitering', value: 20, color: '#faad14' },
      { name: 'Tamper', value: 10, color: '#ff7a45' },
      { name: 'System', value: 5, color: '#1890ff' },
    ],
    systemHealth: [
      { component: 'Edge Nodes', status: 98.5, trend: 'up' },
      { component: 'Network', status: 99.1, trend: 'stable' },
      { component: 'Storage', status: 96.8, trend: 'down' },
      { component: 'Processing', status: 97.9, trend: 'up' },
    ],
  });

  const totalDetections = analyticsData.detectionTrends.reduce((sum, day) => sum + day.detections, 0);
  const totalAlerts = analyticsData.detectionTrends.reduce((sum, day) => sum + day.alerts, 0);
  const totalIncidents = analyticsData.detectionTrends.reduce((sum, day) => sum + day.incidents, 0);
  const avgDetectionsPerDay = Math.round(totalDetections / analyticsData.detectionTrends.length);

  const handleExportReport = () => {
    // Mock export functionality
    console.log('Exporting report for range:', selectedTimeRange);
  };

  const handleRefreshData = () => {
    // Mock refresh functionality
    console.log('Refreshing analytics data...');
  };

  const cameraPerformanceColumns = [
    {
      title: 'Camera',
      dataIndex: 'camera',
      key: 'camera',
    },
    {
      title: 'Uptime',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => (
        <Space>
          <Progress
            percent={uptime}
            size="small"
            status={uptime > 98 ? 'success' : uptime > 95 ? 'normal' : 'exception'}
            showInfo={false}
            style={{ width: 60 }}
          />
          <Text>{uptime}%</Text>
        </Space>
      ),
    },
    {
      title: 'Detections',
      dataIndex: 'detections',
      key: 'detections',
      render: (detections: number) => (
        <Statistic value={detections} valueStyle={{ fontSize: 14 }} />
      ),
    },
    {
      title: 'Alerts',
      dataIndex: 'alerts',
      key: 'alerts',
      render: (alerts: number) => (
        <Statistic value={alerts} valueStyle={{ fontSize: 14, color: '#ff4d4f' }} />
      ),
    },
  ];

  const systemHealthColumns = [
    {
      title: 'Component',
      dataIndex: 'component',
      key: 'component',
    },
    {
      title: 'Health Score',
      dataIndex: 'status',
      key: 'status',
      render: (status: number) => (
        <Space>
          <Progress
            percent={status}
            size="small"
            status={status > 98 ? 'success' : status > 95 ? 'normal' : 'exception'}
            showInfo={false}
            style={{ width: 80 }}
          />
          <Text>{status}%</Text>
        </Space>
      ),
    },
    {
      title: 'Trend',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend: string) => {
        const icon = trend === 'up' ? <TrendingUpOutlined style={{ color: '#52c41a' }} /> :
                    trend === 'down' ? <TrendingDownOutlined style={{ color: '#ff4d4f' }} /> :
                    <ClockCircleOutlined style={{ color: '#faad14' }} />;
        return (
          <Space>
            {icon}
            <Text style={{ textTransform: 'capitalize' }}>{trend}</Text>
          </Space>
        );
      },
    },
  ];

  return (
    <div>
      {/* Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={8}>
            <Space>
              <Text strong>Time Range:</Text>
              <RangePicker
                value={selectedTimeRange}
                onChange={(dates) => dates && setSelectedTimeRange(dates)}
                format="YYYY-MM-DD"
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space>
              <Text strong>Metric:</Text>
              <Select
                value={selectedMetric}
                onChange={setSelectedMetric}
                style={{ width: 120 }}
              >
                <Option value="detections">Detections</Option>
                <Option value="alerts">Alerts</Option>
                <Option value="incidents">Incidents</Option>
              </Select>
            </Space>
          </Col>
          <Col span={10} style={{ textAlign: 'right' }}>
            <Space>
              <Button icon={<ReloadOutlined />} onClick={handleRefreshData}>
                Refresh
              </Button>
              <Button type="primary" icon={<DownloadOutlined />} onClick={handleExportReport}>
                Export Report
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Key Metrics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Detections"
              value={totalDetections}
              prefix={<VideoCameraOutlined />}
              valueStyle={{ color: '#1890ff' }}
              suffix={
                <Text type="secondary" style={{ fontSize: 12 }}>
                  (7 days)
                </Text>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Alerts"
              value={totalAlerts}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
              suffix={
                <Text type="secondary" style={{ fontSize: 12 }}>
                  (7 days)
                </Text>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Incidents"
              value={totalIncidents}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
              suffix={
                <Text type="secondary" style={{ fontSize: 12 }}>
                  (7 days)
                </Text>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Avg. Detections/Day"
              value={avgDetectionsPerDay}
              valueStyle={{ color: '#52c41a' }}
              prefix={
                <TrendingUpOutlined style={{ color: '#52c41a' }} />
              }
            />
          </Card>
        </Col>
      </Row>

      {/* Charts Row 1 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={16}>
          <Card title="Detection Trends (7 Days)" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={analyticsData.detectionTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => dayjs(value).format('MM/DD')}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => dayjs(value).format('YYYY-MM-DD')}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="detections" 
                  stroke="#1890ff" 
                  strokeWidth={2}
                  name="Detections"
                />
                <Line 
                  type="monotone" 
                  dataKey="alerts" 
                  stroke="#ff4d4f" 
                  strokeWidth={2}
                  name="Alerts"
                />
                <Line 
                  type="monotone" 
                  dataKey="incidents" 
                  stroke="#faad14" 
                  strokeWidth={2}
                  name="Incidents"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="Alert Types Distribution" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={analyticsData.alertTypes}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {analyticsData.alertTypes.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Charts Row 2 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={16}>
          <Card title="Hourly Activity Pattern" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={analyticsData.hourlyActivity}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="activity" 
                  stroke="#52c41a" 
                  fill="#52c41a" 
                  fillOpacity={0.3}
                  name="Activity Level"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="System Health Overview" size="small">
            <Table
              dataSource={analyticsData.systemHealth}
              columns={systemHealthColumns}
              pagination={false}
              size="small"
              rowKey="component"
            />
          </Card>
        </Col>
      </Row>

      {/* Performance Tables */}
      <Row gutter={16}>
        <Col span={24}>
          <Card title="Camera Performance Summary" size="small">
            <Table
              dataSource={analyticsData.cameraPerformance}
              columns={cameraPerformanceColumns}
              pagination={false}
              size="small"
              rowKey="camera"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default AnalyticsDashboard;