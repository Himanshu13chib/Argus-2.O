import React from 'react';
import { Card, Typography } from 'antd';

const { Title } = Typography;

const Settings: React.FC = () => {
  return (
    <div style={{ padding: 16 }}>
      <Title level={2}>System Settings</Title>
      <Card>
        <p>System settings and configuration interface</p>
      </Card>
    </div>
  );
};

export default Settings;