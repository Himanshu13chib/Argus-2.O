import React, { useState } from 'react';
import { Layout, Menu, Badge, Avatar, Dropdown, Space, Typography } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSelector } from 'react-redux';
import {
  DashboardOutlined,
  VideoCameraOutlined,
  AlertOutlined,
  FileTextOutlined,
  BarChartOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  BellOutlined,
} from '@ant-design/icons';
import { RootState } from '../../store/store';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  
  const { unacknowledgedCount } = useSelector((state: RootState) => state.alerts);
  const { connected, notifications, currentUser } = useSelector((state: RootState) => state.system);
  
  const unreadNotifications = notifications.filter(n => !n.read).length;

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/live-feeds',
      icon: <VideoCameraOutlined />,
      label: 'Live Feeds',
    },
    {
      key: '/alerts',
      icon: <AlertOutlined />,
      label: (
        <Space>
          Alerts
          {unacknowledgedCount > 0 && (
            <Badge count={unacknowledgedCount} size="small" />
          )}
        </Space>
      ),
    },
    {
      key: '/incidents',
      icon: <FileTextOutlined />,
      label: 'Incidents',
    },
    {
      key: '/analytics',
      icon: <BarChartOutlined />,
      label: 'Analytics',
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
  ];

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
    },
  ];

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key);
  };

  const handleUserMenuClick = ({ key }: { key: string }) => {
    if (key === 'logout') {
      // Handle logout
      console.log('Logout clicked');
    } else if (key === 'profile') {
      // Handle profile
      console.log('Profile clicked');
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="dark"
        width={250}
      >
        <div style={{ 
          height: 64, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          borderBottom: '1px solid #303030'
        }}>
          <Text style={{ color: '#fff', fontSize: collapsed ? 14 : 18, fontWeight: 'bold' }}>
            {collapsed ? 'PA' : 'Project Argus'}
          </Text>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
        />
      </Sider>
      
      <Layout>
        <Header style={{ 
          padding: '0 24px', 
          background: '#fff', 
          display: 'flex', 
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid #f0f0f0'
        }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Badge 
              status={connected ? 'success' : 'error'} 
              text={connected ? 'Connected' : 'Disconnected'}
            />
          </div>
          
          <Space size="large">
            <Badge count={unreadNotifications} size="small">
              <BellOutlined style={{ fontSize: 18, cursor: 'pointer' }} />
            </Badge>
            
            <Dropdown
              menu={{
                items: userMenuItems,
                onClick: handleUserMenuClick,
              }}
              placement="bottomRight"
            >
              <Space style={{ cursor: 'pointer' }}>
                <Avatar icon={<UserOutlined />} />
                <Text>{currentUser?.name || 'Operator'}</Text>
              </Space>
            </Dropdown>
          </Space>
        </Header>
        
        <Content>
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout;