import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';
import MainLayout from '../MainLayout';
import alertReducer from '../../../store/slices/alertSlice';
import systemReducer from '../../../store/slices/systemSlice';

const createMockStore = (connected = true, unacknowledgedCount = 0, currentUser = null) => {
  return configureStore({
    reducer: {
      alerts: alertReducer,
      system: systemReducer,
    },
    preloadedState: {
      alerts: {
        alerts: [],
        unacknowledgedCount,
        filters: {
          severity: [],
          type: [],
          camera: [],
          acknowledged: null,
        },
        sortBy: 'timestamp',
        sortOrder: 'desc',
        selectedAlert: null,
      },
      system: {
        connected,
        health: [],
        metrics: [],
        notifications: [],
        currentUser,
      },
    },
  });
};

const renderWithProviders = (component: React.ReactElement, store: any) => {
  return render(
    <Provider store={store}>
      <BrowserRouter>
        {component}
      </BrowserRouter>
    </Provider>
  );
};

describe('MainLayout', () => {
  it('renders main navigation menu', () => {
    const store = createMockStore();
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    expect(screen.getByText('Project Argus')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Live Feeds')).toBeInTheDocument();
    expect(screen.getByText('Alerts')).toBeInTheDocument();
    expect(screen.getByText('Incidents')).toBeInTheDocument();
    expect(screen.getByText('Analytics')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('displays connection status correctly', () => {
    const connectedStore = createMockStore(true);
    const { rerender } = renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      connectedStore
    );

    expect(screen.getByText('Connected')).toBeInTheDocument();

    const disconnectedStore = createMockStore(false);
    rerender(
      <Provider store={disconnectedStore}>
        <BrowserRouter>
          <MainLayout>
            <div>Test Content</div>
          </MainLayout>
        </BrowserRouter>
      </Provider>
    );

    expect(screen.getByText('Disconnected')).toBeInTheDocument();
  });

  it('shows alert badge when there are unacknowledged alerts', () => {
    const store = createMockStore(true, 5);
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Should show badge with count
    const alertsMenuItem = screen.getByText('Alerts').closest('li');
    expect(alertsMenuItem).toBeInTheDocument();
    
    // Check for badge (Ant Design renders badges as spans)
    const badge = screen.getByText('5');
    expect(badge).toBeInTheDocument();
  });

  it('hides alert badge when no unacknowledged alerts', () => {
    const store = createMockStore(true, 0);
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Should not show badge
    expect(screen.queryByText('0')).not.toBeInTheDocument();
  });

  it('displays current user information', () => {
    const mockUser = {
      id: 'user-001',
      name: 'John Operator',
      role: 'operator' as const,
      permissions: ['view_cameras'],
    };
    const store = createMockStore(true, 0, mockUser);
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    expect(screen.getByText('John Operator')).toBeInTheDocument();
  });

  it('shows default user when no current user', () => {
    const store = createMockStore(true, 0, null);
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    expect(screen.getByText('Operator')).toBeInTheDocument();
  });

  it('handles sidebar collapse/expand', () => {
    const store = createMockStore();
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Find the collapse trigger (usually an icon button)
    const collapseButton = document.querySelector('.ant-layout-sider-trigger');
    expect(collapseButton).toBeInTheDocument();

    if (collapseButton) {
      fireEvent.click(collapseButton);
      // After collapse, should show abbreviated title
      expect(screen.getByText('PA')).toBeInTheDocument();
    }
  });

  it('renders notification bell with count', () => {
    const store = createMockStore();
    // Add some notifications to the store
    store.dispatch({
      type: 'system/addNotification',
      payload: {
        type: 'info',
        message: 'Test notification',
      },
    });
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Should show notification bell
    const bellIcon = screen.getByRole('img', { name: /bell/i });
    expect(bellIcon).toBeInTheDocument();
  });

  it('handles user menu interactions', () => {
    const mockUser = {
      id: 'user-001',
      name: 'John Operator',
      role: 'operator' as const,
      permissions: ['view_cameras'],
    };
    const store = createMockStore(true, 0, mockUser);
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Click on user avatar/name to open dropdown
    const userArea = screen.getByText('John Operator');
    fireEvent.click(userArea);

    // Should show dropdown menu items
    expect(screen.getByText('Profile')).toBeInTheDocument();
    expect(screen.getByText('Logout')).toBeInTheDocument();
  });

  it('renders children content correctly', () => {
    const store = createMockStore();
    
    renderWithProviders(
      <MainLayout>
        <div data-testid="child-content">Test Child Content</div>
      </MainLayout>,
      store
    );

    expect(screen.getByTestId('child-content')).toBeInTheDocument();
    expect(screen.getByText('Test Child Content')).toBeInTheDocument();
  });

  it('applies correct layout structure', () => {
    const store = createMockStore();
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Check for Ant Design layout structure
    expect(document.querySelector('.ant-layout')).toBeInTheDocument();
    expect(document.querySelector('.ant-layout-sider')).toBeInTheDocument();
    expect(document.querySelector('.ant-layout-header')).toBeInTheDocument();
    expect(document.querySelector('.ant-layout-content')).toBeInTheDocument();
  });

  it('shows correct menu item as selected based on current route', () => {
    const store = createMockStore();
    
    // Mock location pathname
    delete (window as any).location;
    (window as any).location = { pathname: '/alerts' };
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // The alerts menu item should be selected
    const alertsMenuItem = screen.getByText('Alerts').closest('li');
    expect(alertsMenuItem).toHaveClass('ant-menu-item-selected');
  });
});

describe('MainLayout Responsive Behavior', () => {
  it('handles mobile viewport correctly', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    });

    const store = createMockStore();
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Layout should still render correctly on mobile
    expect(document.querySelector('.ant-layout')).toBeInTheDocument();
  });

  it('adjusts sidebar behavior on small screens', () => {
    // Mock small screen
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 600,
    });

    const store = createMockStore();
    
    renderWithProviders(
      <MainLayout>
        <div>Test Content</div>
      </MainLayout>,
      store
    );

    // Sidebar should be present but may be collapsed by default
    expect(document.querySelector('.ant-layout-sider')).toBeInTheDocument();
  });
});