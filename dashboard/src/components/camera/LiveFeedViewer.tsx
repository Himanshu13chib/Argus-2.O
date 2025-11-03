import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Card, Badge, Button, Space, Tooltip, Spin } from 'antd';
import { 
  FullscreenOutlined, 
  FullscreenExitOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
  SettingOutlined,
  AlertOutlined
} from '@ant-design/icons';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store/store';
import { Camera, Detection, VirtualLine } from '../../store/slices/cameraSlice';
import { setFullscreenCamera, toggleDetections, toggleVirtualLines } from '../../store/slices/cameraSlice';

interface LiveFeedViewerProps {
  camera: Camera;
  width?: number;
  height?: number;
  showControls?: boolean;
  className?: string;
}

const LiveFeedViewer: React.FC<LiveFeedViewerProps> = ({
  camera,
  width = 400,
  height = 300,
  showControls = true,
  className = '',
}) => {
  const dispatch = useDispatch();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  
  const { showDetections, showVirtualLines, fullscreenCamera } = useSelector(
    (state: RootState) => state.cameras
  );
  
  const isFullscreen = fullscreenCamera === camera.id;
  const hasActiveAlerts = camera.detections.length > 0;

  // Initialize video stream
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadStart = () => setIsLoading(true);
    const handleCanPlay = () => {
      setIsLoading(false);
      setHasError(false);
    };
    const handleError = () => {
      setIsLoading(false);
      setHasError(true);
    };

    video.addEventListener('loadstart', handleLoadStart);
    video.addEventListener('canplay', handleCanPlay);
    video.addEventListener('error', handleError);

    // Set video source
    video.src = camera.streamUrl;
    video.load();

    return () => {
      video.removeEventListener('loadstart', handleLoadStart);
      video.removeEventListener('canplay', handleCanPlay);
      video.removeEventListener('error', handleError);
    };
  }, [camera.streamUrl]);

  // Draw overlays on canvas
  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw detection boxes
    if (showDetections && camera.detections.length > 0) {
      camera.detections.forEach((detection: Detection) => {
        const { x, y, width: boxWidth, height: boxHeight } = detection.bbox;
        
        // Scale coordinates to canvas size
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;
        
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = boxWidth * scaleX;
        const scaledHeight = boxHeight * scaleY;

        // Draw bounding box
        ctx.strokeStyle = '#ff4d4f';
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

        // Draw confidence score
        ctx.fillStyle = '#ff4d4f';
        ctx.font = '12px Arial';
        ctx.fillText(
          `${(detection.confidence * 100).toFixed(1)}%`,
          scaledX,
          scaledY - 5
        );
      });
    }

    // Draw virtual lines
    if (showVirtualLines && camera.virtualLines.length > 0) {
      camera.virtualLines.forEach((line: VirtualLine) => {
        if (!line.active) return;

        ctx.strokeStyle = '#52c41a';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        
        ctx.beginPath();
        line.points.forEach((point, index) => {
          const scaleX = canvas.width / video.videoWidth;
          const scaleY = canvas.height / video.videoHeight;
          
          const scaledX = point.x * scaleX;
          const scaledY = point.y * scaleY;
          
          if (index === 0) {
            ctx.moveTo(scaledX, scaledY);
          } else {
            ctx.lineTo(scaledX, scaledY);
          }
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }
  }, [camera.detections, camera.virtualLines, showDetections, showVirtualLines]);

  // Update overlays when detections or virtual lines change
  useEffect(() => {
    const interval = setInterval(drawOverlays, 100); // Update 10 times per second
    return () => clearInterval(interval);
  }, [drawOverlays]);

  // Handle canvas resize
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (canvas && video) {
      const resizeCanvas = () => {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        drawOverlays();
      };

      const resizeObserver = new ResizeObserver(resizeCanvas);
      resizeObserver.observe(video);
      
      return () => resizeObserver.disconnect();
    }
  }, [drawOverlays]);

  const handleFullscreen = () => {
    dispatch(setFullscreenCamera(isFullscreen ? null : camera.id));
  };

  const handleToggleDetections = () => {
    dispatch(toggleDetections());
  };

  const handleToggleVirtualLines = () => {
    dispatch(toggleVirtualLines());
  };

  const getStatusColor = () => {
    switch (camera.status) {
      case 'online': return 'success';
      case 'offline': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const cardStyle = isFullscreen ? {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    zIndex: 1000,
    margin: 0,
  } : {
    width: width,
    height: height + 60, // Account for header
  };

  return (
    <Card
      className={`camera-feed-card ${className}`}
      style={cardStyle}
      bodyStyle={{ padding: 0 }}
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Badge status={getStatusColor()} />
            <span>{camera.name}</span>
            <span style={{ fontSize: '12px', color: '#666' }}>
              ({camera.type})
            </span>
          </Space>
          
          {hasActiveAlerts && (
            <Badge count={camera.detections.length} size="small">
              <AlertOutlined style={{ color: '#ff4d4f' }} />
            </Badge>
          )}
        </div>
      }
      extra={
        showControls && (
          <Space>
            <Tooltip title={showDetections ? 'Hide Detections' : 'Show Detections'}>
              <Button
                type="text"
                size="small"
                icon={showDetections ? <EyeOutlined /> : <EyeInvisibleOutlined />}
                onClick={handleToggleDetections}
              />
            </Tooltip>
            
            <Tooltip title={showVirtualLines ? 'Hide Virtual Lines' : 'Show Virtual Lines'}>
              <Button
                type="text"
                size="small"
                icon={showVirtualLines ? <EyeOutlined /> : <EyeInvisibleOutlined />}
                onClick={handleToggleVirtualLines}
                style={{ color: showVirtualLines ? '#52c41a' : undefined }}
              />
            </Tooltip>
            
            <Tooltip title="Settings">
              <Button
                type="text"
                size="small"
                icon={<SettingOutlined />}
              />
            </Tooltip>
            
            <Tooltip title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
              <Button
                type="text"
                size="small"
                icon={isFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
                onClick={handleFullscreen}
              />
            </Tooltip>
          </Space>
        )
      }
    >
      <div 
        className="video-container"
        style={{ 
          position: 'relative',
          width: '100%',
          height: isFullscreen ? 'calc(100vh - 64px)' : height,
          background: '#000',
        }}
      >
        {isLoading && (
          <div className="loading-spinner">
            <Spin size="large" />
          </div>
        )}
        
        {hasError && (
          <div className="loading-spinner">
            <div style={{ textAlign: 'center', color: '#ff4d4f' }}>
              <AlertOutlined style={{ fontSize: 24, marginBottom: 8 }} />
              <div>Camera Offline</div>
              <div style={{ fontSize: 12, color: '#666' }}>
                Unable to connect to video stream
              </div>
            </div>
          </div>
        )}
        
        <video
          ref={videoRef}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: hasError ? 'none' : 'block',
          }}
          autoPlay
          muted
          playsInline
        />
        
        <canvas
          ref={canvasRef}
          className="video-overlay"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            display: hasError ? 'none' : 'block',
          }}
        />
      </div>
    </Card>
  );
};

export default LiveFeedViewer;