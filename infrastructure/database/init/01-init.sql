-- Project Argus Database Initialization
-- This script sets up the initial database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create enum types
CREATE TYPE camera_type AS ENUM ('visible_light', 'thermal', 'infrared', 'ptz', 'fixed', 'dome');
CREATE TYPE camera_status AS ENUM ('active', 'inactive', 'maintenance', 'error', 'tampered', 'offline');
CREATE TYPE detection_class AS ENUM ('person', 'vehicle', 'animal', 'unknown');
CREATE TYPE track_status AS ENUM ('active', 'lost', 'completed', 'merged');
CREATE TYPE alert_type AS ENUM ('virtual_line_crossing', 'loitering_detected', 'suspicious_behavior', 'tamper_detected', 'system_health', 'multiple_crossings', 'high_risk_crossing');
CREATE TYPE alert_severity AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE incident_status AS ENUM ('open', 'in_progress', 'pending_review', 'resolved', 'closed', 'escalated');
CREATE TYPE incident_priority AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE evidence_type AS ENUM ('image', 'video', 'metadata', 'audio', 'sensor_data', 'log_file', 'report');
CREATE TYPE evidence_status AS ENUM ('pending', 'verified', 'sealed', 'archived', 'purged');
CREATE TYPE user_role AS ENUM ('operator', 'auditor', 'administrator', 'supervisor', 'viewer');
CREATE TYPE component_status AS ENUM ('healthy', 'warning', 'critical', 'offline', 'maintenance');
CREATE TYPE component_type AS ENUM ('camera', 'edge_node', 'api_service', 'database', 'message_broker', 'storage', 'network', 'detection_engine', 'tracking_service');

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(255) NOT NULL,
    role user_role NOT NULL DEFAULT 'viewer',
    active BOOLEAN DEFAULT TRUE,
    locked BOOLEAN DEFAULT FALSE,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    phone_number VARCHAR(50),
    department VARCHAR(255),
    badge_number VARCHAR(100),
    supervisor_id UUID REFERENCES users(id),
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    password_expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    last_password_change TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Cameras table
CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type camera_type NOT NULL,
    status camera_status DEFAULT 'active',
    location VARCHAR(500),
    gps_coordinates POINT,
    elevation FLOAT,
    orientation FLOAT,
    ip_address INET,
    port INTEGER DEFAULT 554,
    rtsp_url VARCHAR(500),
    username VARCHAR(255),
    password VARCHAR(255),
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    firmware_version VARCHAR(100),
    serial_number VARCHAR(255),
    capabilities JSONB DEFAULT '{}',
    detection_enabled BOOLEAN DEFAULT TRUE,
    recording_enabled BOOLEAN DEFAULT TRUE,
    motion_detection BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP,
    uptime_hours FLOAT DEFAULT 0,
    temperature FLOAT,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_usage FLOAT,
    tags TEXT[],
    metadata JSONB DEFAULT '{}'
);

-- Virtual lines table
CREATE TABLE virtual_lines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL DEFAULT 'line',
    points JSONB NOT NULL,
    direction VARCHAR(50) DEFAULT 'bidirectional',
    sensitivity FLOAT DEFAULT 0.8,
    active BOOLEAN DEFAULT TRUE,
    color VARCHAR(7) DEFAULT '#FF0000',
    thickness INTEGER DEFAULT 2,
    opacity FLOAT DEFAULT 0.7,
    active_hours INTEGER[],
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    tags TEXT[],
    metadata JSONB DEFAULT '{}'
);

-- Detections table
CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id),
    timestamp TIMESTAMP NOT NULL,
    bbox JSONB NOT NULL,
    confidence FLOAT NOT NULL,
    detection_class detection_class NOT NULL,
    features BYTEA,
    image_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Tracks table
CREATE TABLE tracks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    global_id UUID,
    camera_id UUID REFERENCES cameras(id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status track_status DEFAULT 'active',
    trajectory JSONB,
    confidence_history FLOAT[],
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Track detections junction table
CREATE TABLE track_detections (
    track_id UUID REFERENCES tracks(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    PRIMARY KEY (track_id, detection_id)
);

-- Alerts table
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type alert_type NOT NULL,
    severity alert_severity NOT NULL,
    camera_id UUID REFERENCES cameras(id),
    detection_id UUID REFERENCES detections(id),
    timestamp TIMESTAMP NOT NULL,
    confidence FLOAT NOT NULL,
    risk_score FLOAT NOT NULL,
    gps_coordinates POINT,
    location_accuracy FLOAT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by UUID REFERENCES users(id),
    acknowledged_at TIMESTAMP,
    escalated BOOLEAN DEFAULT FALSE,
    escalated_by UUID REFERENCES users(id),
    escalated_at TIMESTAMP,
    image_path VARCHAR(500),
    video_snippet_path VARCHAR(500),
    crossing_event JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Incidents table
CREATE TABLE incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id UUID REFERENCES alerts(id),
    operator_id UUID REFERENCES users(id),
    status incident_status DEFAULT 'open',
    priority incident_priority DEFAULT 'medium',
    title VARCHAR(500),
    description TEXT,
    location VARCHAR(500),
    gps_coordinates POINT,
    assigned_to UUID REFERENCES users(id),
    supervisor_notified BOOLEAN DEFAULT FALSE,
    supervisor_id UUID REFERENCES users(id),
    escalation_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    tags TEXT[],
    metadata JSONB DEFAULT '{}'
);

-- Incident notes table
CREATE TABLE incident_notes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID REFERENCES incidents(id) ON DELETE CASCADE,
    author_id UUID REFERENCES users(id),
    content TEXT NOT NULL,
    note_type VARCHAR(50) DEFAULT 'general',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Evidence table
CREATE TABLE evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID REFERENCES incidents(id),
    type evidence_type NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    original_filename VARCHAR(500),
    file_size BIGINT,
    mime_type VARCHAR(100),
    hash_sha256 VARCHAR(64) NOT NULL,
    hmac_signature VARCHAR(256) NOT NULL,
    encryption_key_id VARCHAR(255),
    status evidence_status DEFAULT 'pending',
    retention_until TIMESTAMP,
    auto_purge BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    camera_id UUID REFERENCES cameras(id),
    detection_id UUID REFERENCES detections(id),
    tags TEXT[],
    metadata JSONB DEFAULT '{}'
);

-- Chain of custody table
CREATE TABLE chain_of_custody (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id UUID REFERENCES evidence(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT NOW(),
    action VARCHAR(100) NOT NULL,
    operator_id UUID REFERENCES users(id),
    details TEXT,
    entry_id UUID DEFAULT uuid_generate_v4()
);

-- System health table
CREATE TABLE system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_id VARCHAR(255) NOT NULL,
    component_type component_type NOT NULL,
    status component_status DEFAULT 'healthy',
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_usage FLOAT,
    network_latency FLOAT,
    temperature FLOAT,
    uptime_seconds FLOAT,
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    last_check TIMESTAMP DEFAULT NOW(),
    last_error TIMESTAMP,
    last_warning TIMESTAMP,
    status_message TEXT,
    error_messages TEXT[],
    warning_messages TEXT[],
    metadata JSONB DEFAULT '{}'
);

-- Audit logs table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP DEFAULT NOW(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    source_ip INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    details JSONB DEFAULT '{}',
    session_id VARCHAR(255)
);

-- User sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    active BOOLEAN DEFAULT TRUE
);

-- Create indexes for performance
CREATE INDEX idx_cameras_status ON cameras(status);
CREATE INDEX idx_cameras_location ON cameras USING GIST(gps_coordinates);
CREATE INDEX idx_detections_camera_timestamp ON detections(camera_id, timestamp);
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
CREATE INDEX idx_tracks_camera_time ON tracks(camera_id, start_time);
CREATE INDEX idx_tracks_global_id ON tracks(global_id);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX idx_alerts_camera ON alerts(camera_id);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_incidents_status ON incidents(status);
CREATE INDEX idx_incidents_operator ON incidents(operator_id);
CREATE INDEX idx_incidents_created ON incidents(created_at);
CREATE INDEX idx_evidence_incident ON evidence(incident_id);
CREATE INDEX idx_evidence_hash ON evidence(hash_sha256);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_system_health_component ON system_health(component_id);
CREATE INDEX idx_system_health_timestamp ON system_health(last_check);

-- Create GIN indexes for JSONB columns
CREATE INDEX idx_cameras_metadata ON cameras USING GIN(metadata);
CREATE INDEX idx_detections_metadata ON detections USING GIN(metadata);
CREATE INDEX idx_alerts_metadata ON alerts USING GIN(metadata);
CREATE INDEX idx_incidents_metadata ON incidents USING GIN(metadata);
CREATE INDEX idx_evidence_metadata ON evidence USING GIN(metadata);
CREATE INDEX idx_audit_logs_details ON audit_logs USING GIN(details);

-- Create text search indexes
CREATE INDEX idx_incidents_title_search ON incidents USING GIN(to_tsvector('english', title));
CREATE INDEX idx_incidents_description_search ON incidents USING GIN(to_tsvector('english', description));

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_incidents_updated_at BEFORE UPDATE ON incidents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user (password: admin123 - CHANGE IN PRODUCTION!)
INSERT INTO users (username, email, full_name, password_hash, salt, role) VALUES 
('admin', 'admin@projectargus.local', 'System Administrator', 
 '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 
 'randomsalt123', 'administrator');

-- Insert sample camera for testing
INSERT INTO cameras (name, type, location, ip_address, rtsp_url) VALUES 
('Test Camera 1', 'fixed', 'Border Checkpoint Alpha', '192.168.1.100', 'rtsp://192.168.1.100:554/stream');

COMMIT;