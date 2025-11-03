import React, { useState } from 'react';
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Upload,
  Tag,
  Space,
  Timeline,
  Typography,
  Row,
  Col,
  Statistic,
  Badge,
  Avatar,
  Tooltip,
  message,
  Divider,
  Image,
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  EyeOutlined,
  FileTextOutlined,
  UserOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  UploadOutlined,
  PaperClipOutlined,
  MessageOutlined,
} from '@ant-design/icons';
import { useSelector, useDispatch } from 'react-redux';
import dayjs from 'dayjs';
import { RootState } from '../../store/store';
import {
  Incident,
  Evidence,
  Note,
  addIncident,
  updateIncident,
  addNote,
  addEvidence,
  updateIncidentStatus,
  assignIncident,
  setSelectedIncident,
} from '../../store/slices/incidentSlice';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface IncidentWorkflowProps {
  showFilters?: boolean;
  maxHeight?: number;
}

const IncidentWorkflow: React.FC<IncidentWorkflowProps> = ({
  showFilters = true,
  maxHeight,
}) => {
  const dispatch = useDispatch();
  const { incidents, selectedIncident } = useSelector((state: RootState) => state.incidents);
  const { alerts } = useSelector((state: RootState) => state.alerts);
  const { currentUser } = useSelector((state: RootState) => state.system);

  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedIncidentDetail, setSelectedIncidentDetail] = useState<Incident | null>(null);
  const [noteModalVisible, setNoteModalVisible] = useState(false);
  const [evidenceModalVisible, setEvidenceModalVisible] = useState(false);
  const [createForm] = Form.useForm();
  const [noteForm] = Form.useForm();
  const [evidenceForm] = Form.useForm();

  const openIncidents = incidents.filter(inc => inc.status === 'open').length;
  const investigatingIncidents = incidents.filter(inc => inc.status === 'investigating').length;
  const resolvedIncidents = incidents.filter(inc => inc.status === 'resolved').length;
  const myIncidents = incidents.filter(inc => inc.assignedTo === currentUser?.id).length;

  const getStatusColor = (status: Incident['status']) => {
    switch (status) {
      case 'open': return 'red';
      case 'investigating': return 'orange';
      case 'resolved': return 'blue';
      case 'closed': return 'green';
      default: return 'default';
    }
  };

  const getPriorityColor = (priority: Incident['priority']) => {
    switch (priority) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'yellow';
      case 'low': return 'blue';
      default: return 'default';
    }
  };

  const handleCreateIncident = async (values: any) => {
    if (!currentUser) return;

    const newIncident: Incident = {
      id: `incident-${Date.now()}`,
      alertId: values.alertId,
      title: values.title,
      description: values.description,
      status: 'open',
      priority: values.priority,
      assignedTo: values.assignedTo,
      assignedToName: values.assignedTo === currentUser.id ? currentUser.name : 'Other Operator',
      createdBy: currentUser.id,
      createdByName: currentUser.name,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      evidence: [],
      notes: [],
      tags: values.tags || [],
      location: values.location,
      cameraId: values.cameraId,
      cameraName: values.cameraName,
    };

    dispatch(addIncident(newIncident));
    message.success('Incident created successfully');
    setCreateModalVisible(false);
    createForm.resetFields();
  };

  const handleViewDetails = (incident: Incident) => {
    setSelectedIncidentDetail(incident);
    setDetailModalVisible(true);
  };

  const handleStatusChange = (incidentId: string, status: Incident['status']) => {
    dispatch(updateIncidentStatus({ incidentId, status }));
    message.success(`Incident status updated to ${status}`);
  };

  const handleAddNote = async (values: any) => {
    if (!currentUser || !selectedIncidentDetail) return;

    const newNote: Note = {
      id: `note-${Date.now()}`,
      content: values.content,
      authorId: currentUser.id,
      authorName: currentUser.name,
      timestamp: new Date().toISOString(),
    };

    dispatch(addNote({ incidentId: selectedIncidentDetail.id, note: newNote }));
    message.success('Note added successfully');
    setNoteModalVisible(false);
    noteForm.resetFields();
  };

  const handleAddEvidence = async (values: any) => {
    if (!selectedIncidentDetail) return;

    const newEvidence: Evidence = {
      id: `evidence-${Date.now()}`,
      type: values.type,
      filePath: values.filePath || 'mock-file-path.jpg',
      thumbnail: values.type === 'image' ? 'https://via.placeholder.com/150x100' : undefined,
      timestamp: new Date().toISOString(),
      description: values.description,
    };

    dispatch(addEvidence({ incidentId: selectedIncidentDetail.id, evidence: newEvidence }));
    message.success('Evidence added successfully');
    setEvidenceModalVisible(false);
    evidenceForm.resetFields();
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (id: string) => (
        <Text code style={{ fontSize: 11 }}>
          {id.split('-')[1]}
        </Text>
      ),
    },
    {
      title: 'Title',
      dataIndex: 'title',
      key: 'title',
      ellipsis: true,
      render: (title: string) => (
        <Text strong style={{ fontSize: 12 }}>
          {title}
        </Text>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: Incident['status']) => (
        <Tag color={getStatusColor(status)} style={{ textTransform: 'capitalize' }}>
          {status}
        </Tag>
      ),
    },
    {
      title: 'Priority',
      dataIndex: 'priority',
      key: 'priority',
      width: 100,
      render: (priority: Incident['priority']) => (
        <Tag color={getPriorityColor(priority)} style={{ textTransform: 'capitalize' }}>
          {priority}
        </Tag>
      ),
    },
    {
      title: 'Assigned To',
      dataIndex: 'assignedToName',
      key: 'assignedToName',
      width: 120,
      render: (name: string) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text style={{ fontSize: 11 }}>{name}</Text>
        </Space>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 100,
      render: (createdAt: string) => (
        <Text style={{ fontSize: 11 }}>
          {dayjs(createdAt).format('MM/DD HH:mm')}
        </Text>
      ),
    },
    {
      title: 'Evidence',
      dataIndex: 'evidence',
      key: 'evidence',
      width: 80,
      render: (evidence: Evidence[]) => (
        <Badge count={evidence.length} size="small">
          <PaperClipOutlined />
        </Badge>
      ),
    },
    {
      title: 'Notes',
      dataIndex: 'notes',
      key: 'notes',
      width: 80,
      render: (notes: Note[]) => (
        <Badge count={notes.length} size="small">
          <MessageOutlined />
        </Badge>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record: Incident) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetails(record)}
            />
          </Tooltip>
          <Tooltip title="Edit">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => {
                // Handle edit
                message.info('Edit functionality would be implemented here');
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div>
      {/* Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Total Incidents"
              value={incidents.length}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Open"
              value={openIncidents}
              valueStyle={{ color: openIncidents > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Investigating"
              value={investigatingIncidents}
              valueStyle={{ color: investigatingIncidents > 0 ? '#faad14' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Assigned to Me"
              value={myIncidents}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Actions */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setCreateModalVisible(true)}
          >
            Create Incident
          </Button>
          <Button icon={<FileTextOutlined />}>
            Export Report
          </Button>
        </Space>
      </Card>

      {/* Incidents Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={incidents}
          rowKey="id"
          size="small"
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showTotal: (total, range) =>
              `${range[0]}-${range[1]} of ${total} incidents`,
          }}
          scroll={{ y: maxHeight }}
        />
      </Card>

      {/* Create Incident Modal */}
      <Modal
        title="Create New Incident"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={createForm}
          layout="vertical"
          onFinish={handleCreateIncident}
        >
          <Form.Item
            name="alertId"
            label="Related Alert"
            rules={[{ required: true, message: 'Please select an alert' }]}
          >
            <Select placeholder="Select related alert">
              {alerts.map(alert => (
                <Option key={alert.id} value={alert.id}>
                  {alert.description} - {alert.cameraName}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="title"
            label="Title"
            rules={[{ required: true, message: 'Please enter incident title' }]}
          >
            <Input placeholder="Enter incident title" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
            rules={[{ required: true, message: 'Please enter description' }]}
          >
            <TextArea rows={4} placeholder="Describe the incident..." />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="priority"
                label="Priority"
                rules={[{ required: true, message: 'Please select priority' }]}
              >
                <Select placeholder="Select priority">
                  <Option value="low">Low</Option>
                  <Option value="medium">Medium</Option>
                  <Option value="high">High</Option>
                  <Option value="critical">Critical</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="assignedTo"
                label="Assign To"
                initialValue={currentUser?.id}
              >
                <Select placeholder="Select assignee">
                  <Option value={currentUser?.id}>{currentUser?.name}</Option>
                  <Option value="other-operator">Other Operator</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="location" label="Location">
            <Input placeholder="Enter location details" />
          </Form.Item>

          <Form.Item name="tags" label="Tags">
            <Select mode="tags" placeholder="Add tags">
              <Option value="border-crossing">Border Crossing</Option>
              <Option value="suspicious-activity">Suspicious Activity</Option>
              <Option value="equipment-issue">Equipment Issue</Option>
            </Select>
          </Form.Item>

          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setCreateModalVisible(false)}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit">
                Create Incident
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Incident Detail Modal */}
      <Modal
        title="Incident Details"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={1000}
      >
        {selectedIncidentDetail && (
          <div>
            {/* Header */}
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={16}>
                <Title level={4}>{selectedIncidentDetail.title}</Title>
                <Paragraph>{selectedIncidentDetail.description}</Paragraph>
              </Col>
              <Col span={8}>
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <div>
                    <Text strong>Status: </Text>
                    <Select
                      value={selectedIncidentDetail.status}
                      style={{ width: 120 }}
                      onChange={(value) => handleStatusChange(selectedIncidentDetail.id, value)}
                    >
                      <Option value="open">Open</Option>
                      <Option value="investigating">Investigating</Option>
                      <Option value="resolved">Resolved</Option>
                      <Option value="closed">Closed</Option>
                    </Select>
                  </div>
                  <div>
                    <Text strong>Priority: </Text>
                    <Tag color={getPriorityColor(selectedIncidentDetail.priority)}>
                      {selectedIncidentDetail.priority.toUpperCase()}
                    </Tag>
                  </div>
                  <div>
                    <Text strong>Assigned to: </Text>
                    <Text>{selectedIncidentDetail.assignedToName}</Text>
                  </div>
                </Space>
              </Col>
            </Row>

            <Divider />

            {/* Timeline and Actions */}
            <Row gutter={16}>
              <Col span={16}>
                <Title level={5}>Timeline</Title>
                <Timeline>
                  <Timeline.Item
                    dot={<ClockCircleOutlined />}
                    color="blue"
                  >
                    <Text strong>Incident Created</Text>
                    <br />
                    <Text type="secondary">
                      {dayjs(selectedIncidentDetail.createdAt).format('YYYY-MM-DD HH:mm:ss')}
                    </Text>
                    <br />
                    <Text>Created by {selectedIncidentDetail.createdByName}</Text>
                  </Timeline.Item>

                  {selectedIncidentDetail.notes.map((note) => (
                    <Timeline.Item
                      key={note.id}
                      dot={<MessageOutlined />}
                      color="green"
                    >
                      <Text strong>Note Added</Text>
                      <br />
                      <Text type="secondary">
                        {dayjs(note.timestamp).format('YYYY-MM-DD HH:mm:ss')}
                      </Text>
                      <br />
                      <Text>{note.content}</Text>
                      <br />
                      <Text type="secondary">by {note.authorName}</Text>
                    </Timeline.Item>
                  ))}

                  {selectedIncidentDetail.evidence.map((evidence) => (
                    <Timeline.Item
                      key={evidence.id}
                      dot={<PaperClipOutlined />}
                      color="orange"
                    >
                      <Text strong>Evidence Added</Text>
                      <br />
                      <Text type="secondary">
                        {dayjs(evidence.timestamp).format('YYYY-MM-DD HH:mm:ss')}
                      </Text>
                      <br />
                      <Text>{evidence.description}</Text>
                      {evidence.thumbnail && (
                        <div style={{ marginTop: 8 }}>
                          <Image
                            src={evidence.thumbnail}
                            alt="Evidence"
                            width={100}
                            height={60}
                            style={{ objectFit: 'cover' }}
                          />
                        </div>
                      )}
                    </Timeline.Item>
                  ))}
                </Timeline>
              </Col>

              <Col span={8}>
                <Title level={5}>Actions</Title>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button
                    type="primary"
                    icon={<MessageOutlined />}
                    onClick={() => setNoteModalVisible(true)}
                    block
                  >
                    Add Note
                  </Button>
                  <Button
                    icon={<PaperClipOutlined />}
                    onClick={() => setEvidenceModalVisible(true)}
                    block
                  >
                    Add Evidence
                  </Button>
                  <Button
                    icon={<FileTextOutlined />}
                    block
                  >
                    Generate Report
                  </Button>
                </Space>

                <Divider />

                <Title level={5}>Evidence ({selectedIncidentDetail.evidence.length})</Title>
                <Space direction="vertical" style={{ width: '100%' }}>
                  {selectedIncidentDetail.evidence.map((evidence) => (
                    <Card key={evidence.id} size="small">
                      <Space>
                        <PaperClipOutlined />
                        <div>
                          <Text strong>{evidence.type}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {evidence.description}
                          </Text>
                        </div>
                      </Space>
                    </Card>
                  ))}
                </Space>
              </Col>
            </Row>
          </div>
        )}
      </Modal>

      {/* Add Note Modal */}
      <Modal
        title="Add Note"
        open={noteModalVisible}
        onCancel={() => setNoteModalVisible(false)}
        footer={null}
      >
        <Form form={noteForm} layout="vertical" onFinish={handleAddNote}>
          <Form.Item
            name="content"
            label="Note Content"
            rules={[{ required: true, message: 'Please enter note content' }]}
          >
            <TextArea rows={4} placeholder="Enter your note..." />
          </Form.Item>
          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setNoteModalVisible(false)}>Cancel</Button>
              <Button type="primary" htmlType="submit">Add Note</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Add Evidence Modal */}
      <Modal
        title="Add Evidence"
        open={evidenceModalVisible}
        onCancel={() => setEvidenceModalVisible(false)}
        footer={null}
      >
        <Form form={evidenceForm} layout="vertical" onFinish={handleAddEvidence}>
          <Form.Item
            name="type"
            label="Evidence Type"
            rules={[{ required: true, message: 'Please select evidence type' }]}
          >
            <Select placeholder="Select evidence type">
              <Option value="image">Image</Option>
              <Option value="video">Video</Option>
              <Option value="metadata">Metadata</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="description"
            label="Description"
            rules={[{ required: true, message: 'Please enter description' }]}
          >
            <TextArea rows={3} placeholder="Describe the evidence..." />
          </Form.Item>
          <Form.Item name="file" label="Upload File">
            <Upload>
              <Button icon={<UploadOutlined />}>Select File</Button>
            </Upload>
          </Form.Item>
          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setEvidenceModalVisible(false)}>Cancel</Button>
              <Button type="primary" htmlType="submit">Add Evidence</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default IncidentWorkflow;