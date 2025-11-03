import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Table, 
  Tag, 
  Button, 
  Space, 
  Badge, 
  Modal, 
  Image, 
  Typography, 
  Row, 
  Col,
  Statistic,
  Select,
  Input,
  DatePicker,
  Tooltip,
  message
} from 'antd';
import { 
  AlertOutlined, 
  CheckOutlined, 
  EyeOutlined, 
  FilterOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { useSelector, useDispatch } from 'react-redux';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import { RootState } from '../../store/store';
import { Alert, acknowledgeAlert, setFilters, setSorting } from '../../store/slices/alertSlice';
import { websocketService } from '../../services/websocket';

dayjs.extend(relativeTime);

const { Title, Text } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface AlertDashboardProps {
  showFilters?: boolean;
  maxHeight?: number;
}

const AlertDashboard: React.FC<AlertDashboardProps> = ({ 
  showFilters = true,
  maxHeight 
}) => {
  const dispatch = useDispatch();
  const { alerts, filters, sortBy, sortOrder, selectedAlert } = useSelector(
    (state: RootState) => state.alerts
  );
  const { currentUser } = useSelector((state: RootState) => state.system);
  const { cameras } = useSelector((state: RootState) => state.cameras);
  
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedAlertDetail, setSelectedAlertDetail] = useState<Alert | null>(null);

  // Filter and sort alerts
  const filteredAlerts = alerts.filter(alert => {
    if (filters.severity.length > 0 && !filters.severity.includes(alert.severity)) {
      return false;
    }
    if (filters.type.length > 0 && !filters.type.includes(alert.type)) {
      return false;
    }
    if (filters.camera.length > 0 && !filters.camera.includes(alert.cameraId)) {
      return false;
    }
    if (filters.acknowledged !== null && alert.acknowledged !== filters.acknowledged) {
      return false;
    }
    return true;
  }).sort((a, b) => {
    let comparison = 0;
    
    switch (sortBy) {
      case 'timestamp':
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
        break;
      case 'severity':
        const severityOrder = { low: 1, medium: 2, high: 3, critical: 4 };
        comparison = severityOrder[a.severity] - severityOrder[b.severity];
        break;
      case 'confidence':
        comparison = a.confidence - b.confidence;
        break;
      case 'riskScore':
        comparison = a.riskScore - b.riskScore;
        break;
    }
    
    return sortOrder === 'asc' ? comparison : -comparison;
  });

  const unacknowledgedAlerts = filteredAlerts.filter(alert => !alert.acknowledged);
  const criticalAlerts = filteredAlerts.filter(alert => alert.severity === 'critical');
  const highAlerts = filteredAlerts.filter(alert => alert.severity === 'high');

  const handleAcknowledge = async (alertIds: string[]) => {
    if (!currentUser) return;

    try {
      for (const alertId of alertIds) {
        dispatch(acknowledgeAlert({ alertId, userId: currentUser.id }));
        websocketService.acknowledgeAlert(alertId);
      }
      message.success(`Acknowledged ${alertIds.length} alert(s)`);
      setSelectedRowKeys([]);
    } catch (error) {
      message.error('Failed to acknowledge alerts');
    }
  };

  const handleViewDetails = (alert: Alert) => {
    setSelectedAlertDetail(alert);
    setDetailModalVisible(true);
  };

  const getSeverityColor = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'yellow';
      case 'low': return 'blue';
      default: return 'default';
    }
  };

  const getTypeIcon = (type: Alert['type']) => {
    switch (type) {
      case 'crossing': return <AlertOutlined />;
      case 'loitering': return <ClockCircleOutlined />;
      case 'tamper': return <ExclamationCircleOutlined />;
      case 'system': return <ExclamationCircleOutlined />;
      default: return <AlertOutlined />;
    }
  };

  const columns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 120,
      render: (timestamp: string) => (
        <Tooltip title={dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss')}>
          <Text style={{ fontSize: 12 }}>
            {dayjs(timestamp).fromNow()}
          </Text>
        </Tooltip>
      ),
      sorter: true,
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: 100,
      render: (type: Alert['type']) => (
        <Space>
          {getTypeIcon(type)}
          <Text style={{ textTransform: 'capitalize' }}>{type}</Text>
        </Space>
      ),
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: Alert['severity']) => (
        <Tag color={getSeverityColor(severity)} style={{ textTransform: 'uppercase' }}>
          {severity}
        </Tag>
      ),
      sorter: true,
    },
    {
      title: 'Camera',
      dataIndex: 'cameraName',
      key: 'cameraName',
      width: 150,
      render: (cameraName: string) => (
        <Text style={{ fontSize: 12 }}>{cameraName}</Text>
      ),
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      render: (description: string) => (
        <Text style={{ fontSize: 12 }}>{description}</Text>
      ),
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => (
        <Text style={{ fontSize: 12 }}>
          {(confidence * 100).toFixed(1)}%
        </Text>
      ),
      sorter: true,
    },
    {
      title: 'Risk Score',
      dataIndex: 'riskScore',
      key: 'riskScore',
      width: 100,
      render: (riskScore: number) => (
        <Badge 
          count={riskScore.toFixed(1)} 
          style={{ 
            backgroundColor: riskScore > 0.8 ? '#ff4d4f' : 
                           riskScore > 0.6 ? '#faad14' : '#52c41a'
          }} 
        />
      ),
      sorter: true,
    },
    {
      title: 'Status',
      dataIndex: 'acknowledged',
      key: 'acknowledged',
      width: 100,
      render: (acknowledged: boolean, record: Alert) => (
        <Space direction="vertical" size={0}>
          <Tag color={acknowledged ? 'green' : 'red'}>
            {acknowledged ? 'Acknowledged' : 'Pending'}
          </Tag>
          {acknowledged && record.acknowledgedBy && (
            <Text style={{ fontSize: 10, color: '#666' }}>
              by {record.acknowledgedBy}
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record: Alert) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetails(record)}
            />
          </Tooltip>
          {!record.acknowledged && (
            <Tooltip title="Acknowledge">
              <Button
                type="text"
                size="small"
                icon={<CheckOutlined />}
                onClick={() => handleAcknowledge([record.id])}
              />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ];

  const rowSelection = {
    selectedRowKeys,
    onChange: (keys: React.Key[]) => setSelectedRowKeys(keys as string[]),
    getCheckboxProps: (record: Alert) => ({
      disabled: record.acknowledged,
    }),
  };

  return (
    <div>
      {/* Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Total Alerts"
              value={filteredAlerts.length}
              prefix={<AlertOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Unacknowledged"
              value={unacknowledgedAlerts.length}
              valueStyle={{ color: unacknowledgedAlerts.length > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Critical"
              value={criticalAlerts.length}
              valueStyle={{ color: criticalAlerts.length > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="High Priority"
              value={highAlerts.length}
              valueStyle={{ color: highAlerts.length > 0 ? '#faad14' : '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      {showFilters && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Row gutter={16} align="middle">
            <Col span={4}>
              <Select
                placeholder="Severity"
                mode="multiple"
                allowClear
                style={{ width: '100%' }}
                value={filters.severity}
                onChange={(value) => dispatch(setFilters({ severity: value }))}
              >
                <Option value="critical">Critical</Option>
                <Option value="high">High</Option>
                <Option value="medium">Medium</Option>
                <Option value="low">Low</Option>
              </Select>
            </Col>
            <Col span={4}>
              <Select
                placeholder="Type"
                mode="multiple"
                allowClear
                style={{ width: '100%' }}
                value={filters.type}
                onChange={(value) => dispatch(setFilters({ type: value }))}
              >
                <Option value="crossing">Crossing</Option>
                <Option value="loitering">Loitering</Option>
                <Option value="tamper">Tamper</Option>
                <Option value="system">System</Option>
              </Select>
            </Col>
            <Col span={4}>
              <Select
                placeholder="Camera"
                mode="multiple"
                allowClear
                style={{ width: '100%' }}
                value={filters.camera}
                onChange={(value) => dispatch(setFilters({ camera: value }))}
              >
                {cameras.map(camera => (
                  <Option key={camera.id} value={camera.id}>
                    {camera.name}
                  </Option>
                ))}
              </Select>
            </Col>
            <Col span={4}>
              <Select
                placeholder="Status"
                allowClear
                style={{ width: '100%' }}
                value={filters.acknowledged}
                onChange={(value) => dispatch(setFilters({ acknowledged: value }))}
              >
                <Option value={false}>Unacknowledged</Option>
                <Option value={true}>Acknowledged</Option>
              </Select>
            </Col>
            <Col span={4}>
              <Select
                placeholder="Sort by"
                style={{ width: '100%' }}
                value={`${sortBy}-${sortOrder}`}
                onChange={(value) => {
                  const [field, order] = value.split('-');
                  dispatch(setSorting({ 
                    sortBy: field as any, 
                    sortOrder: order as 'asc' | 'desc' 
                  }));
                }}
              >
                <Option value="timestamp-desc">Newest First</Option>
                <Option value="timestamp-asc">Oldest First</Option>
                <Option value="severity-desc">Highest Severity</Option>
                <Option value="confidence-desc">Highest Confidence</Option>
                <Option value="riskScore-desc">Highest Risk</Option>
              </Select>
            </Col>
            <Col span={4}>
              <Button icon={<ReloadOutlined />} onClick={() => window.location.reload()}>
                Refresh
              </Button>
            </Col>
          </Row>
        </Card>
      )}

      {/* Bulk Actions */}
      {selectedRowKeys.length > 0 && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Space>
            <Text>Selected {selectedRowKeys.length} alert(s)</Text>
            <Button
              type="primary"
              icon={<CheckOutlined />}
              onClick={() => handleAcknowledge(selectedRowKeys)}
            >
              Acknowledge Selected
            </Button>
          </Space>
        </Card>
      )}

      {/* Alerts Table */}
      <Card>
        <Table
          rowSelection={rowSelection}
          columns={columns}
          dataSource={filteredAlerts}
          rowKey="id"
          size="small"
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} alerts`,
          }}
          scroll={{ y: maxHeight }}
          rowClassName={(record) => 
            record.acknowledged ? 'acknowledged-row' : 'unacknowledged-row'
          }
        />
      </Card>

      {/* Alert Detail Modal */}
      <Modal
        title="Alert Details"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            Close
          </Button>,
          selectedAlertDetail && !selectedAlertDetail.acknowledged && (
            <Button
              key="acknowledge"
              type="primary"
              icon={<CheckOutlined />}
              onClick={() => {
                if (selectedAlertDetail) {
                  handleAcknowledge([selectedAlertDetail.id]);
                  setDetailModalVisible(false);
                }
              }}
            >
              Acknowledge
            </Button>
          ),
        ]}
        width={800}
      >
        {selectedAlertDetail && (
          <div>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={12}>
                <Space direction="vertical" size={0}>
                  <Text strong>Alert ID:</Text>
                  <Text code>{selectedAlertDetail.id}</Text>
                </Space>
              </Col>
              <Col span={12}>
                <Space direction="vertical" size={0}>
                  <Text strong>Timestamp:</Text>
                  <Text>{dayjs(selectedAlertDetail.timestamp).format('YYYY-MM-DD HH:mm:ss')}</Text>
                </Space>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={8}>
                <Space direction="vertical" size={0}>
                  <Text strong>Type:</Text>
                  <Space>
                    {getTypeIcon(selectedAlertDetail.type)}
                    <Text style={{ textTransform: 'capitalize' }}>
                      {selectedAlertDetail.type}
                    </Text>
                  </Space>
                </Space>
              </Col>
              <Col span={8}>
                <Space direction="vertical" size={0}>
                  <Text strong>Severity:</Text>
                  <Tag color={getSeverityColor(selectedAlertDetail.severity)}>
                    {selectedAlertDetail.severity.toUpperCase()}
                  </Tag>
                </Space>
              </Col>
              <Col span={8}>
                <Space direction="vertical" size={0}>
                  <Text strong>Status:</Text>
                  <Tag color={selectedAlertDetail.acknowledged ? 'green' : 'red'}>
                    {selectedAlertDetail.acknowledged ? 'Acknowledged' : 'Pending'}
                  </Tag>
                </Space>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={12}>
                <Space direction="vertical" size={0}>
                  <Text strong>Camera:</Text>
                  <Text>{selectedAlertDetail.cameraName}</Text>
                </Space>
              </Col>
              <Col span={12}>
                <Space direction="vertical" size={0}>
                  <Text strong>Confidence:</Text>
                  <Text>{(selectedAlertDetail.confidence * 100).toFixed(1)}%</Text>
                </Space>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={24}>
                <Space direction="vertical" size={0}>
                  <Text strong>Description:</Text>
                  <Text>{selectedAlertDetail.description}</Text>
                </Space>
              </Col>
            </Row>

            {selectedAlertDetail.thumbnail && (
              <Row gutter={16} style={{ marginBottom: 16 }}>
                <Col span={24}>
                  <Space direction="vertical" size={0}>
                    <Text strong>Thumbnail:</Text>
                    <Image
                      src={selectedAlertDetail.thumbnail}
                      alt="Alert thumbnail"
                      style={{ maxWidth: '100%', maxHeight: 300 }}
                    />
                  </Space>
                </Col>
              </Row>
            )}

            {selectedAlertDetail.acknowledged && (
              <Row gutter={16}>
                <Col span={12}>
                  <Space direction="vertical" size={0}>
                    <Text strong>Acknowledged By:</Text>
                    <Text>{selectedAlertDetail.acknowledgedBy}</Text>
                  </Space>
                </Col>
                <Col span={12}>
                  <Space direction="vertical" size={0}>
                    <Text strong>Acknowledged At:</Text>
                    <Text>
                      {selectedAlertDetail.acknowledgedAt && 
                        dayjs(selectedAlertDetail.acknowledgedAt).format('YYYY-MM-DD HH:mm:ss')
                      }
                    </Text>
                  </Space>
                </Col>
              </Row>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AlertDashboard;