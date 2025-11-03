# Project Argus Demo Environment

This demo environment showcases the capabilities of Project Argus border security system.

## Demo Components

### 1. Camera Feeds
- **demo_camera_001**: Main Gate - Primary crossing point monitoring
- **demo_camera_002**: Perimeter East - Thermal night vision coverage  
- **demo_camera_003**: Patrol Route - Mobile patrol monitoring
- **demo_camera_004**: Remote Outpost - Radar-integrated monitoring

### 2. Video Scenarios
- **authorized_crossing_daytime**: Normal authorized personnel crossing
- **unauthorized_crossing_night**: Security incident with night vision
- **multiple_crossings_simultaneous**: High activity scenario
- **false_positive_scenario**: Environmental false positive handling
- **patrol_verification**: Security patrol verification

### 3. Sample Data
- **50 sample detections** across 24-hour period
- **20 sample alerts** with various severity levels
- **10 sample incidents** showing complete workflow
- **30 evidence items** with chain of custody

### 4. Interactive Features
- Real-time alert dashboard
- Incident management workflow
- Evidence review and export
- System health monitoring
- Performance analytics

## Running the Demo

1. **Start Demo Environment**:
   ```bash
   python tests/integration/run_demo.py
   ```

2. **Access Dashboard**:
   - Open browser to http://localhost:3000
   - Login with demo credentials (operator/demo123)

3. **Demo Scenarios**:
   - Use scenario selector to run different demonstrations
   - Monitor alerts and incidents in real-time
   - Review evidence and generate reports

## Demo Credentials

- **Operator**: username=`demo_operator`, password=`demo123`
- **Supervisor**: username=`demo_supervisor`, password=`demo456`
- **Auditor**: username=`demo_auditor`, password=`demo789`

## Key Demonstrations

### Detection and Tracking
- Person detection with confidence scoring
- Multi-camera tracking and re-identification
- Virtual line crossing detection

### Alert Management
- Real-time alert generation
- Risk scoring and prioritization
- Escalation workflows

### Incident Response
- Incident creation and assignment
- Evidence collection and sealing
- Resolution and reporting

### Security Features
- Encrypted evidence storage
- Audit trail maintenance
- Role-based access control

### System Monitoring
- Camera health monitoring
- Performance metrics
- Tamper detection

## Technical Specifications

- **Detection Latency**: < 300ms average
- **False Positive Rate**: < 1 alert/camera/day
- **System Uptime**: 99.9% target
- **Concurrent Cameras**: 100+ supported
- **Evidence Encryption**: AES-256
- **Data Integrity**: HMAC-SHA256

## Support

For technical support or questions about the demo:
- Email: support@projectargus.demo
- Documentation: /docs/
- API Reference: /api/docs/

---
*Project Argus Demo Environment v1.0*
*Generated: 2025-11-02 21:31:03*
