import React from 'react';
import { Typography } from 'antd';
import AnalyticsDashboard from '../components/analytics/AnalyticsDashboard';

const { Title } = Typography;

const Analytics: React.FC = () => {
  return (
    <div style={{ padding: 16 }}>
      <Title level={2} style={{ marginBottom: 24 }}>
        Analytics & Reporting
      </Title>
      <AnalyticsDashboard />
    </div>
  );
};

export default Analytics;