import React from 'react';
import { Row, Col, Button, Space, Select, Card, Empty } from 'antd';
import { AppstoreOutlined, BorderOutlined } from '@ant-design/icons';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store/store';
import { setGridLayout } from '../../store/slices/cameraSlice';
import LiveFeedViewer from './LiveFeedViewer';

const { Option } = Select;

interface CameraGridProps {
  selectedCameras?: string[];
  showControls?: boolean;
}

const CameraGrid: React.FC<CameraGridProps> = ({ 
  selectedCameras,
  showControls = true 
}) => {
  const dispatch = useDispatch();
  const { cameras, gridLayout, fullscreenCamera } = useSelector(
    (state: RootState) => state.cameras
  );

  // Filter cameras based on selection or show all
  const displayCameras = selectedCameras 
    ? cameras.filter(camera => selectedCameras.includes(camera.id))
    : cameras;

  // If in fullscreen mode, show only that camera
  if (fullscreenCamera) {
    const fullscreenCam = cameras.find(cam => cam.id === fullscreenCamera);
    if (fullscreenCam) {
      return <LiveFeedViewer camera={fullscreenCam} showControls={showControls} />;
    }
  }

  const handleLayoutChange = (layout: 1 | 2 | 4 | 6 | 9) => {
    dispatch(setGridLayout(layout));
  };

  const getGridColumns = () => {
    switch (gridLayout) {
      case 1: return 1;
      case 2: return 2;
      case 4: return 2;
      case 6: return 3;
      case 9: return 3;
      default: return 2;
    }
  };

  const getGridRows = () => {
    switch (gridLayout) {
      case 1: return 1;
      case 2: return 1;
      case 4: return 2;
      case 6: return 2;
      case 9: return 3;
      default: return 2;
    }
  };

  const getCameraSize = () => {
    const cols = getGridColumns();
    const rows = getGridRows();
    const containerWidth = window.innerWidth - 300; // Account for sidebar
    const containerHeight = window.innerHeight - 150; // Account for header and padding
    
    return {
      width: Math.floor(containerWidth / cols) - 20,
      height: Math.floor(containerHeight / rows) - 80,
    };
  };

  const cameraSize = getCameraSize();

  if (displayCameras.length === 0) {
    return (
      <Card style={{ margin: 16 }}>
        <Empty
          image={<BorderOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
          description="No cameras available"
        />
      </Card>
    );
  }

  return (
    <div style={{ padding: 16 }}>
      {showControls && (
        <Card 
          size="small" 
          style={{ marginBottom: 16 }}
          title="Camera Grid Controls"
        >
          <Space>
            <span>Layout:</span>
            <Select
              value={gridLayout}
              onChange={handleLayoutChange}
              style={{ width: 120 }}
            >
              <Option value={1}>1x1</Option>
              <Option value={2}>2x1</Option>
              <Option value={4}>2x2</Option>
              <Option value={6}>3x2</Option>
              <Option value={9}>3x3</Option>
            </Select>
            
            <Button
              icon={<AppstoreOutlined />}
              onClick={() => handleLayoutChange(4)}
            >
              Default Grid
            </Button>
          </Space>
        </Card>
      )}

      <div 
        className={`camera-grid camera-grid-${gridLayout}`}
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${getGridColumns()}, 1fr)`,
          gridTemplateRows: `repeat(${getGridRows()}, 1fr)`,
          gap: 16,
          minHeight: 'calc(100vh - 200px)',
        }}
      >
        {displayCameras.slice(0, gridLayout).map((camera) => (
          <div key={camera.id}>
            <LiveFeedViewer
              camera={camera}
              width={cameraSize.width}
              height={cameraSize.height}
              showControls={showControls}
            />
          </div>
        ))}
        
        {/* Fill empty grid slots if needed */}
        {Array.from({ length: Math.max(0, gridLayout - displayCameras.length) }).map((_, index) => (
          <Card
            key={`empty-${index}`}
            style={{
              width: cameraSize.width,
              height: cameraSize.height + 60,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            bodyStyle={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              height: '100%'
            }}
          >
            <Empty
              image={<BorderOutlined style={{ fontSize: 32, color: '#d9d9d9' }} />}
              description="No Camera"
              imageStyle={{ height: 40 }}
            />
          </Card>
        ))}
      </div>
    </div>
  );
};

export default CameraGrid;