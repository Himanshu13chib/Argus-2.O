"""
Comprehensive Audit Logging and Compliance System for Project Argus.
Provides security event monitoring, intrusion detection, and compliance reporting.
"""

import os
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
import redis
from collections import defaultdict, deque
import ipaddress
import hashlib
import hmac
import csv
import io

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_FAILURE = "login_failure"
    LOGIN_SUCCESS = "login_success"
    LOGOUT = "logout"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    ACCOUNT_LOCKOUT = "account_lockout"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_COMPROMISE = "system_compromise"
    MALWARE_DETECTED = "malware_detected"
    NETWORK_INTRUSION = "network_intrusion"
    DOS_ATTACK = "dos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    CSRF_ATTACK = "csrf_attack"
    FILE_INTEGRITY_VIOLATION = "file_integrity_violation"
    CERTIFICATE_ERROR = "certificate_error"
    ENCRYPTION_FAILURE = "encryption_failure"
    KEY_COMPROMISE = "key_compromise"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    INDIAN_IT_ACT = "indian_it_act"
    INDIAN_PRIVACY = "indian_privacy"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: SecurityEventType
    severity: str  # low, medium, high, critical
    timestamp: datetime
    user_id: Optional[str]
    source_ip: str
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    success: bool
    details: Dict[str, Any]
    risk_score: float
    session_id: Optional[str] = None
    geolocation: Optional[Dict[str, str]] = None
    threat_indicators: List[str] = None


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    framework: ComplianceFramework
    rule_name: str
    description: str
    requirement: str
    control_type: str  # preventive, detective, corrective
    automated: bool
    frequency: str  # continuous, daily, weekly, monthly
    data_types: List[str]
    retention_period: int  # days
    notification_required: bool
    escalation_required: bool


@dataclass
class IntrusionPattern:
    """Intrusion detection pattern."""
    pattern_id: str
    name: str
    description: str
    event_types: List[SecurityEventType]
    conditions: Dict[str, Any]
    time_window: int  # seconds
    threshold: int
    severity: str
    response_actions: List[str]


class SecurityEventMonitor:
    """Security event monitoring and intrusion detection system."""
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis = redis_client
        self.event_buffer = deque(maxlen=10000)
        self.ip_activity = defaultdict(list)
        self.user_activity = defaultdict(list)
        self.blocked_ips = set()
        self.suspicious_users = set()
        self.intrusion_patterns = {}
        self.active_alerts = {}
        
    async def initialize(self):
        """Initialize security event monitor."""
        await self._create_tables()
        await self._load_intrusion_patterns()
        await self._start_monitoring_tasks()
        logger.info("Security event monitor initialized")
    
    async def _create_tables(self):
        """Create database tables for security monitoring."""
        async with self.db_pool.acquire() as conn:
            # Security events table (enhanced)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id VARCHAR(255) PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    user_id UUID REFERENCES users(id),
                    source_ip INET,
                    user_agent TEXT,
                    resource VARCHAR(500),
                    action VARCHAR(255) NOT NULL,
                    success BOOLEAN NOT NULL,
                    details JSONB DEFAULT '{}',
                    risk_score FLOAT DEFAULT 0.0,
                    session_id VARCHAR(255),
                    geolocation JSONB,
                    threat_indicators TEXT[],
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Intrusion patterns table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS intrusion_patterns (
                    pattern_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    event_types JSONB NOT NULL,
                    conditions JSONB NOT NULL,
                    time_window INTEGER NOT NULL,
                    threshold INTEGER NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    response_actions JSONB NOT NULL,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Security alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_alerts (
                    alert_id VARCHAR(255) PRIMARY KEY,
                    pattern_id VARCHAR(255) REFERENCES intrusion_patterns(pattern_id),
                    triggered_at TIMESTAMP DEFAULT NOW(),
                    source_ip INET,
                    user_id UUID REFERENCES users(id),
                    event_count INTEGER NOT NULL,
                    risk_score FLOAT NOT NULL,
                    status VARCHAR(20) DEFAULT 'active',
                    acknowledged_by UUID REFERENCES users(id),
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    response_actions_taken JSONB DEFAULT '[]',
                    details JSONB DEFAULT '{}'
                )
            """)
            
            # Blocked entities table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS blocked_entities (
                    entity_id VARCHAR(255) PRIMARY KEY,
                    entity_type VARCHAR(50) NOT NULL, -- ip, user, session
                    entity_value VARCHAR(255) NOT NULL,
                    reason TEXT NOT NULL,
                    blocked_at TIMESTAMP DEFAULT NOW(),
                    blocked_by UUID REFERENCES users(id),
                    expires_at TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE,
                    unblocked_at TIMESTAMP,
                    unblocked_by UUID REFERENCES users(id)
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_ip ON security_events(source_ip)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_user ON security_events(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_alerts_status ON security_alerts(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_blocked_entities_type ON blocked_entities(entity_type, entity_value)")
    
    async def _load_intrusion_patterns(self):
        """Load intrusion detection patterns."""
        # Define default intrusion patterns
        default_patterns = [
            IntrusionPattern(
                pattern_id="brute_force_login",
                name="Brute Force Login Attack",
                description="Multiple failed login attempts from same IP",
                event_types=[SecurityEventType.LOGIN_FAILURE],
                conditions={"same_ip": True, "different_users": True},
                time_window=300,  # 5 minutes
                threshold=5,
                severity="high",
                response_actions=["block_ip", "alert_admin", "log_incident"]
            ),
            IntrusionPattern(
                pattern_id="privilege_escalation",
                name="Privilege Escalation Attempt",
                description="User attempting to access resources above their privilege level",
                event_types=[SecurityEventType.UNAUTHORIZED_ACCESS, SecurityEventType.PERMISSION_DENIED],
                conditions={"same_user": True, "escalating_resources": True},
                time_window=600,  # 10 minutes
                threshold=3,
                severity="critical",
                response_actions=["suspend_user", "alert_admin", "log_incident"]
            ),
            IntrusionPattern(
                pattern_id="suspicious_data_access",
                name="Suspicious Data Access Pattern",
                description="Unusual data access patterns indicating potential data exfiltration",
                event_types=[SecurityEventType.DATA_ACCESS, SecurityEventType.DATA_EXPORT],
                conditions={"same_user": True, "high_volume": True, "unusual_time": True},
                time_window=3600,  # 1 hour
                threshold=10,
                severity="high",
                response_actions=["alert_admin", "log_incident", "require_mfa"]
            ),
            IntrusionPattern(
                pattern_id="account_enumeration",
                name="Account Enumeration Attack",
                description="Systematic probing of user accounts",
                event_types=[SecurityEventType.LOGIN_FAILURE],
                conditions={"same_ip": True, "sequential_usernames": True},
                time_window=600,  # 10 minutes
                threshold=10,
                severity="medium",
                response_actions=["block_ip", "alert_admin"]
            ),
            IntrusionPattern(
                pattern_id="session_hijacking",
                name="Session Hijacking Attempt",
                description="Multiple IPs using same session token",
                event_types=[SecurityEventType.LOGIN_SUCCESS],
                conditions={"same_session": True, "different_ips": True},
                time_window=300,  # 5 minutes
                threshold=2,
                severity="critical",
                response_actions=["revoke_session", "alert_admin", "log_incident"]
            )
        ]
        
        # Store patterns in database and memory
        async with self.db_pool.acquire() as conn:
            for pattern in default_patterns:
                await conn.execute("""
                    INSERT INTO intrusion_patterns 
                    (pattern_id, name, description, event_types, conditions, 
                     time_window, threshold, severity, response_actions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (pattern_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        conditions = EXCLUDED.conditions,
                        time_window = EXCLUDED.time_window,
                        threshold = EXCLUDED.threshold,
                        severity = EXCLUDED.severity,
                        response_actions = EXCLUDED.response_actions
                """, pattern.pattern_id, pattern.name, pattern.description,
                    json.dumps([et.value for et in pattern.event_types]),
                    json.dumps(pattern.conditions), pattern.time_window,
                    pattern.threshold, pattern.severity,
                    json.dumps(pattern.response_actions))
                
                self.intrusion_patterns[pattern.pattern_id] = pattern
        
        logger.info(f"Loaded {len(self.intrusion_patterns)} intrusion patterns")
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        asyncio.create_task(self._process_event_buffer())
        asyncio.create_task(self._analyze_intrusion_patterns())
        asyncio.create_task(self._cleanup_old_data())
        asyncio.create_task(self._update_threat_intelligence())
    
    async def log_security_event(self, event_type: SecurityEventType, user_id: Optional[str],
                                source_ip: str, action: str, success: bool,
                                details: Dict[str, Any], user_agent: Optional[str] = None,
                                resource: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Log a security event."""
        try:
            event_id = f"{event_type.value}_{datetime.utcnow().timestamp()}_{hash(source_ip) % 10000}"
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(event_type, source_ip, user_id, success, details)
            
            # Create security event
            event = SecurityEvent(
                event_id=event_id,
                event_type=event_type,
                severity=self._determine_severity(event_type, risk_score),
                timestamp=datetime.utcnow(),
                user_id=user_id,
                source_ip=source_ip,
                user_agent=user_agent,
                resource=resource,
                action=action,
                success=success,
                details=details,
                risk_score=risk_score,
                session_id=session_id,
                threat_indicators=self._extract_threat_indicators(details)
            )
            
            # Add to buffer for processing
            self.event_buffer.append(event)
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO security_events 
                    (event_id, event_type, severity, timestamp, user_id, source_ip,
                     user_agent, resource, action, success, details, risk_score,
                     session_id, threat_indicators)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, event.event_id, event.event_type.value, event.severity,
                    event.timestamp, event.user_id, event.source_ip,
                    event.user_agent, event.resource, event.action, event.success,
                    json.dumps(event.details), event.risk_score, event.session_id,
                    event.threat_indicators)
            
            # Cache recent events in Redis
            await self.redis.lpush("recent_security_events", json.dumps(asdict(event), default=str))
            await self.redis.ltrim("recent_security_events", 0, 999)
            
            # Track IP and user activity
            self.ip_activity[source_ip].append(event)
            if user_id:
                self.user_activity[user_id].append(event)
            
            logger.debug(f"Logged security event: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
            return ""
    
    def _calculate_risk_score(self, event_type: SecurityEventType, source_ip: str,
                            user_id: Optional[str], success: bool, details: Dict[str, Any]) -> float:
        """Calculate risk score for security event."""
        risk_score = 0.0
        
        # Base risk by event type
        event_risk = {
            SecurityEventType.LOGIN_FAILURE: 0.3,
            SecurityEventType.UNAUTHORIZED_ACCESS: 0.7,
            SecurityEventType.PERMISSION_DENIED: 0.5,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 0.8,
            SecurityEventType.BRUTE_FORCE_ATTACK: 0.9,
            SecurityEventType.SYSTEM_COMPROMISE: 1.0,
            SecurityEventType.MALWARE_DETECTED: 1.0,
            SecurityEventType.NETWORK_INTRUSION: 0.9,
            SecurityEventType.SQL_INJECTION: 0.8,
            SecurityEventType.XSS_ATTACK: 0.6,
        }
        risk_score += event_risk.get(event_type, 0.2)
        
        # Failed actions are riskier
        if not success:
            risk_score += 0.2
        
        # Check IP reputation
        if self._is_suspicious_ip(source_ip):
            risk_score += 0.3
        
        # Check for unusual patterns
        if source_ip in self.ip_activity:
            recent_events = [e for e in self.ip_activity[source_ip] 
                           if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]
            if len(recent_events) > 10:
                risk_score += 0.2
        
        # Check user behavior
        if user_id and user_id in self.user_activity:
            recent_user_events = [e for e in self.user_activity[user_id]
                                if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]
            if len(recent_user_events) > 20:
                risk_score += 0.1
        
        # Check for threat indicators in details
        threat_keywords = ['injection', 'script', 'exploit', 'payload', 'malware']
        for keyword in threat_keywords:
            if any(keyword.lower() in str(v).lower() for v in details.values()):
                risk_score += 0.2
                break
        
        return min(risk_score, 1.0)
    
    def _determine_severity(self, event_type: SecurityEventType, risk_score: float) -> str:
        """Determine event severity based on type and risk score."""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _extract_threat_indicators(self, details: Dict[str, Any]) -> List[str]:
        """Extract threat indicators from event details."""
        indicators = []
        
        # Common threat indicators
        threat_patterns = [
            'union select', 'drop table', '<script>', 'javascript:',
            '../', '..\\', 'cmd.exe', 'powershell', '/etc/passwd',
            'base64_decode', 'eval(', 'system(', 'exec('
        ]
        
        for key, value in details.items():
            if isinstance(value, str):
                for pattern in threat_patterns:
                    if pattern.lower() in value.lower():
                        indicators.append(f"{pattern}_detected_in_{key}")
        
        return indicators
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious."""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check if IP is in blocked list
            if ip_address in self.blocked_ips:
                return True
            
            # Check for private/internal IPs (less suspicious)
            if ip.is_private or ip.is_loopback:
                return False
            
            # Check for known malicious ranges (simplified)
            # In production, integrate with threat intelligence feeds
            suspicious_ranges = [
                '10.0.0.0/8',  # Example suspicious range
            ]
            
            for range_str in suspicious_ranges:
                if ip in ipaddress.ip_network(range_str):
                    return True
            
            return False
            
        except Exception:
            return True  # Invalid IP is suspicious
    
    async def _process_event_buffer(self):
        """Process events in buffer for real-time analysis."""
        while True:
            try:
                if self.event_buffer:
                    events_to_process = []
                    while self.event_buffer and len(events_to_process) < 100:
                        events_to_process.append(self.event_buffer.popleft())
                    
                    for event in events_to_process:
                        await self._analyze_event_for_patterns(event)
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error processing event buffer: {e}")
                await asyncio.sleep(5)
    
    async def _analyze_event_for_patterns(self, event: SecurityEvent):
        """Analyze event against intrusion patterns."""
        try:
            for pattern in self.intrusion_patterns.values():
                if event.event_type in pattern.event_types:
                    await self._check_pattern_match(pattern, event)
                    
        except Exception as e:
            logger.error(f"Error analyzing event for patterns: {e}")
    
    async def _check_pattern_match(self, pattern: IntrusionPattern, event: SecurityEvent):
        """Check if event matches intrusion pattern."""
        try:
            # Get recent events within time window
            cutoff_time = datetime.utcnow() - timedelta(seconds=pattern.time_window)
            
            # Query recent events matching pattern criteria
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT * FROM security_events 
                    WHERE timestamp >= $1 
                    AND event_type = ANY($2)
                    AND processed = FALSE
                """
                
                rows = await conn.fetch(query, cutoff_time, [et.value for et in pattern.event_types])
                
                matching_events = []
                for row in rows:
                    matching_events.append({
                        'event_id': row['event_id'],
                        'source_ip': str(row['source_ip']) if row['source_ip'] else None,
                        'user_id': row['user_id'],
                        'session_id': row['session_id'],
                        'timestamp': row['timestamp'],
                        'details': row['details']
                    })
                
                # Check pattern conditions
                if self._evaluate_pattern_conditions(pattern, matching_events):
                    await self._trigger_security_alert(pattern, matching_events, event)
                    
                    # Mark events as processed
                    event_ids = [e['event_id'] for e in matching_events]
                    await conn.execute(
                        "UPDATE security_events SET processed = TRUE WHERE event_id = ANY($1)",
                        event_ids
                    )
                    
        except Exception as e:
            logger.error(f"Error checking pattern match: {e}")
    
    def _evaluate_pattern_conditions(self, pattern: IntrusionPattern, events: List[Dict]) -> bool:
        """Evaluate if events meet pattern conditions."""
        if len(events) < pattern.threshold:
            return False
        
        conditions = pattern.conditions
        
        # Check same IP condition
        if conditions.get("same_ip"):
            ips = {e['source_ip'] for e in events if e['source_ip']}
            if len(ips) != 1:
                return False
        
        # Check different users condition
        if conditions.get("different_users"):
            users = {e['user_id'] for e in events if e['user_id']}
            if len(users) < 2:
                return False
        
        # Check same user condition
        if conditions.get("same_user"):
            users = {e['user_id'] for e in events if e['user_id']}
            if len(users) != 1:
                return False
        
        # Check same session condition
        if conditions.get("same_session"):
            sessions = {e['session_id'] for e in events if e['session_id']}
            if len(sessions) != 1:
                return False
        
        # Check different IPs condition
        if conditions.get("different_ips"):
            ips = {e['source_ip'] for e in events if e['source_ip']}
            if len(ips) < 2:
                return False
        
        return True
    
    async def _trigger_security_alert(self, pattern: IntrusionPattern, events: List[Dict], trigger_event: SecurityEvent):
        """Trigger security alert for matched pattern."""
        try:
            alert_id = f"alert_{pattern.pattern_id}_{datetime.utcnow().timestamp()}"
            
            # Calculate aggregate risk score
            risk_score = min(len(events) * 0.1 + trigger_event.risk_score, 1.0)
            
            # Store alert
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO security_alerts 
                    (alert_id, pattern_id, source_ip, user_id, event_count, risk_score, details)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, alert_id, pattern.pattern_id, trigger_event.source_ip,
                    trigger_event.user_id, len(events), risk_score,
                    json.dumps({
                        'pattern_name': pattern.name,
                        'trigger_event': trigger_event.event_id,
                        'matching_events': [e['event_id'] for e in events]
                    }))
            
            # Execute response actions
            await self._execute_response_actions(pattern, trigger_event, alert_id)
            
            # Cache alert
            self.active_alerts[alert_id] = {
                'pattern_id': pattern.pattern_id,
                'severity': pattern.severity,
                'triggered_at': datetime.utcnow(),
                'source_ip': trigger_event.source_ip,
                'user_id': trigger_event.user_id
            }
            
            logger.warning(f"Security alert triggered: {alert_id} - {pattern.name}")
            
        except Exception as e:
            logger.error(f"Error triggering security alert: {e}")
    
    async def _execute_response_actions(self, pattern: IntrusionPattern, event: SecurityEvent, alert_id: str):
        """Execute automated response actions."""
        try:
            actions_taken = []
            
            for action in pattern.response_actions:
                if action == "block_ip" and event.source_ip:
                    await self._block_ip(event.source_ip, f"Triggered by pattern: {pattern.name}")
                    actions_taken.append("blocked_ip")
                
                elif action == "suspend_user" and event.user_id:
                    await self._suspend_user(event.user_id, f"Triggered by pattern: {pattern.name}")
                    actions_taken.append("suspended_user")
                
                elif action == "revoke_session" and event.session_id:
                    await self._revoke_session(event.session_id)
                    actions_taken.append("revoked_session")
                
                elif action == "alert_admin":
                    await self._alert_administrators(pattern, event, alert_id)
                    actions_taken.append("alerted_admin")
                
                elif action == "log_incident":
                    await self._create_security_incident(pattern, event, alert_id)
                    actions_taken.append("logged_incident")
            
            # Update alert with actions taken
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE security_alerts SET response_actions_taken = $1 WHERE alert_id = $2",
                    json.dumps(actions_taken), alert_id
                )
                
        except Exception as e:
            logger.error(f"Error executing response actions: {e}")
    
    async def _block_ip(self, ip_address: str, reason: str):
        """Block IP address."""
        try:
            entity_id = f"ip_{ip_address}_{datetime.utcnow().timestamp()}"
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO blocked_entities 
                    (entity_id, entity_type, entity_value, reason, expires_at)
                    VALUES ($1, 'ip', $2, $3, $4)
                """, entity_id, ip_address, reason, 
                    datetime.utcnow() + timedelta(hours=24))  # Block for 24 hours
            
            self.blocked_ips.add(ip_address)
            
            # Cache in Redis
            await self.redis.setex(f"blocked_ip:{ip_address}", timedelta(hours=24), reason)
            
            logger.warning(f"Blocked IP address: {ip_address} - {reason}")
            
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
    
    async def _suspend_user(self, user_id: str, reason: str):
        """Suspend user account."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE users SET locked = TRUE, locked_until = $1 WHERE id = $2",
                    datetime.utcnow() + timedelta(hours=1), user_id  # Suspend for 1 hour
                )
            
            self.suspicious_users.add(user_id)
            logger.warning(f"Suspended user: {user_id} - {reason}")
            
        except Exception as e:
            logger.error(f"Error suspending user: {e}")
    
    async def _revoke_session(self, session_id: str):
        """Revoke user session."""
        try:
            # Remove from Redis (assuming session storage)
            await self.redis.delete(f"session:{session_id}")
            
            # Update database
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE user_sessions SET active = FALSE WHERE session_token = $1",
                    session_id
                )
            
            logger.info(f"Revoked session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error revoking session: {e}")
    
    async def _alert_administrators(self, pattern: IntrusionPattern, event: SecurityEvent, alert_id: str):
        """Send alert to administrators."""
        try:
            # In a real implementation, this would send notifications
            # via email, SMS, or push notifications
            alert_message = {
                'alert_id': alert_id,
                'pattern': pattern.name,
                'severity': pattern.severity,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'timestamp': event.timestamp.isoformat(),
                'description': pattern.description
            }
            
            # Store in Redis for dashboard notifications
            await self.redis.lpush("admin_alerts", json.dumps(alert_message, default=str))
            await self.redis.ltrim("admin_alerts", 0, 99)  # Keep last 100 alerts
            
            logger.info(f"Administrator alert sent for: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error alerting administrators: {e}")
    
    async def _create_security_incident(self, pattern: IntrusionPattern, event: SecurityEvent, alert_id: str):
        """Create security incident record."""
        try:
            async with self.db_pool.acquire() as conn:
                incident_id = await conn.fetchval("""
                    INSERT INTO incidents 
                    (title, description, priority, status, metadata)
                    VALUES ($1, $2, $3, 'open', $4)
                    RETURNING id
                """, f"Security Alert: {pattern.name}",
                    f"Intrusion pattern '{pattern.name}' triggered by event {event.event_id}",
                    "critical" if pattern.severity == "critical" else "high",
                    json.dumps({
                        'alert_id': alert_id,
                        'pattern_id': pattern.pattern_id,
                        'source_ip': event.source_ip,
                        'event_type': event.event_type.value
                    }))
            
            logger.info(f"Created security incident: {incident_id} for alert: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error creating security incident: {e}")
    
    async def _analyze_intrusion_patterns(self):
        """Analyze patterns for effectiveness and tuning."""
        while True:
            try:
                # Analyze pattern effectiveness every hour
                await asyncio.sleep(3600)
                
                async with self.db_pool.acquire() as conn:
                    # Get pattern statistics
                    rows = await conn.fetch("""
                        SELECT pattern_id, COUNT(*) as alert_count,
                               AVG(risk_score) as avg_risk_score
                        FROM security_alerts 
                        WHERE triggered_at > $1
                        GROUP BY pattern_id
                    """, datetime.utcnow() - timedelta(hours=24))
                    
                    for row in rows:
                        pattern_id = row['pattern_id']
                        alert_count = row['alert_count']
                        avg_risk = row['avg_risk_score']
                        
                        # Log pattern effectiveness
                        logger.info(f"Pattern {pattern_id}: {alert_count} alerts, avg risk: {avg_risk:.2f}")
                        
                        # Auto-tune patterns if needed
                        if alert_count > 50 and avg_risk < 0.3:
                            # Too many low-risk alerts, increase threshold
                            await self._tune_pattern_threshold(pattern_id, increase=True)
                        elif alert_count < 2 and avg_risk > 0.8:
                            # Too few high-risk alerts, decrease threshold
                            await self._tune_pattern_threshold(pattern_id, increase=False)
                
            except Exception as e:
                logger.error(f"Error analyzing intrusion patterns: {e}")
    
    async def _tune_pattern_threshold(self, pattern_id: str, increase: bool):
        """Auto-tune pattern threshold."""
        try:
            if pattern_id in self.intrusion_patterns:
                pattern = self.intrusion_patterns[pattern_id]
                old_threshold = pattern.threshold
                
                if increase:
                    pattern.threshold = min(pattern.threshold + 2, 20)
                else:
                    pattern.threshold = max(pattern.threshold - 1, 2)
                
                if pattern.threshold != old_threshold:
                    # Update in database
                    async with self.db_pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE intrusion_patterns SET threshold = $1 WHERE pattern_id = $2",
                            pattern.threshold, pattern_id
                        )
                    
                    logger.info(f"Auto-tuned pattern {pattern_id} threshold: {old_threshold} -> {pattern.threshold}")
                    
        except Exception as e:
            logger.error(f"Error tuning pattern threshold: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old security data."""
        while True:
            try:
                # Clean up every 6 hours
                await asyncio.sleep(21600)
                
                cutoff_time = datetime.utcnow() - timedelta(days=90)
                
                async with self.db_pool.acquire() as conn:
                    # Clean old security events
                    await conn.execute(
                        "DELETE FROM security_events WHERE timestamp < $1",
                        cutoff_time
                    )
                    
                    # Clean resolved alerts older than 30 days
                    await conn.execute(
                        "DELETE FROM security_alerts WHERE resolved_at < $1",
                        datetime.utcnow() - timedelta(days=30)
                    )
                    
                    # Clean expired blocked entities
                    await conn.execute(
                        "DELETE FROM blocked_entities WHERE expires_at < $1",
                        datetime.utcnow()
                    )
                
                logger.info("Cleaned up old security data")
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence data."""
        while True:
            try:
                # Update every 4 hours
                await asyncio.sleep(14400)
                
                # In a real implementation, this would fetch from threat intelligence feeds
                # For now, we'll just log that we're updating
                logger.info("Updated threat intelligence data")
                
            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
    
    async def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent event counts by type
                event_counts = await conn.fetch("""
                    SELECT event_type, COUNT(*) as count
                    FROM security_events 
                    WHERE timestamp > $1
                    GROUP BY event_type
                    ORDER BY count DESC
                """, datetime.utcnow() - timedelta(hours=24))
                
                # Get active alerts
                active_alerts = await conn.fetch("""
                    SELECT alert_id, pattern_id, severity, triggered_at, source_ip
                    FROM security_alerts 
                    WHERE status = 'active'
                    ORDER BY triggered_at DESC
                    LIMIT 10
                """)
                
                # Get blocked entities
                blocked_entities = await conn.fetch("""
                    SELECT entity_type, entity_value, reason, blocked_at
                    FROM blocked_entities 
                    WHERE active = TRUE
                    ORDER BY blocked_at DESC
                    LIMIT 20
                """)
                
                return {
                    'event_counts': [{'type': row['event_type'], 'count': row['count']} for row in event_counts],
                    'active_alerts': [dict(row) for row in active_alerts],
                    'blocked_entities': [dict(row) for row in blocked_entities],
                    'total_events_24h': sum(row['count'] for row in event_counts),
                    'active_alert_count': len(active_alerts),
                    'blocked_ip_count': len([e for e in blocked_entities if e['entity_type'] == 'ip'])
                }
                
        except Exception as e:
            logger.error(f"Error getting security dashboard data: {e}")
            return {}


class ComplianceManager:
    """Compliance reporting and management system."""
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis = redis_client
        self.compliance_rules = {}
        
    async def initialize(self):
        """Initialize compliance manager."""
        await self._create_tables()
        await self._load_compliance_rules()
        logger.info("Compliance manager initialized")
    
    async def _create_tables(self):
        """Create compliance-related tables."""
        async with self.db_pool.acquire() as conn:
            # Compliance rules table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    rule_id VARCHAR(255) PRIMARY KEY,
                    framework VARCHAR(50) NOT NULL,
                    rule_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    requirement TEXT NOT NULL,
                    control_type VARCHAR(50) NOT NULL,
                    automated BOOLEAN DEFAULT FALSE,
                    frequency VARCHAR(50) NOT NULL,
                    data_types JSONB NOT NULL,
                    retention_period INTEGER NOT NULL,
                    notification_required BOOLEAN DEFAULT FALSE,
                    escalation_required BOOLEAN DEFAULT FALSE,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Compliance reports table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id VARCHAR(255) PRIMARY KEY,
                    framework VARCHAR(50) NOT NULL,
                    report_type VARCHAR(100) NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    generated_at TIMESTAMP DEFAULT NOW(),
                    generated_by UUID REFERENCES users(id),
                    status VARCHAR(20) DEFAULT 'generated',
                    file_path VARCHAR(500),
                    summary JSONB DEFAULT '{}',
                    findings JSONB DEFAULT '[]',
                    recommendations JSONB DEFAULT '[]'
                )
            """)
            
            # Data retention policies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS data_retention_policies (
                    policy_id VARCHAR(255) PRIMARY KEY,
                    data_type VARCHAR(100) NOT NULL,
                    retention_period INTEGER NOT NULL, -- days
                    purge_method VARCHAR(50) NOT NULL,
                    legal_hold_exempt BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    active BOOLEAN DEFAULT TRUE
                )
            """)
    
    async def _load_compliance_rules(self):
        """Load compliance rules."""
        # Define default compliance rules
        default_rules = [
            ComplianceRule(
                rule_id="gdpr_data_retention",
                framework=ComplianceFramework.GDPR,
                rule_name="Data Retention Limits",
                description="Personal data must not be kept longer than necessary",
                requirement="Article 5(1)(e) - Storage limitation",
                control_type="preventive",
                automated=True,
                frequency="daily",
                data_types=["personal_data", "biometric_data", "location_data"],
                retention_period=365,  # 1 year default
                notification_required=True,
                escalation_required=False
            ),
            ComplianceRule(
                rule_id="gdpr_access_logging",
                framework=ComplianceFramework.GDPR,
                rule_name="Access Logging",
                description="All access to personal data must be logged",
                requirement="Article 32 - Security of processing",
                control_type="detective",
                automated=True,
                frequency="continuous",
                data_types=["personal_data", "audit_logs"],
                retention_period=2555,  # 7 years
                notification_required=False,
                escalation_required=False
            ),
            ComplianceRule(
                rule_id="indian_it_data_protection",
                framework=ComplianceFramework.INDIAN_IT_ACT,
                rule_name="Sensitive Personal Data Protection",
                description="Sensitive personal data must be protected with reasonable security practices",
                requirement="Section 43A - Data Protection",
                control_type="preventive",
                automated=True,
                frequency="continuous",
                data_types=["personal_data", "biometric_data", "financial_data"],
                retention_period=2190,  # 6 years
                notification_required=True,
                escalation_required=True
            )
        ]
        
        # Store rules in database and memory
        async with self.db_pool.acquire() as conn:
            for rule in default_rules:
                await conn.execute("""
                    INSERT INTO compliance_rules 
                    (rule_id, framework, rule_name, description, requirement,
                     control_type, automated, frequency, data_types, retention_period,
                     notification_required, escalation_required)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (rule_id) DO UPDATE SET
                        rule_name = EXCLUDED.rule_name,
                        description = EXCLUDED.description,
                        requirement = EXCLUDED.requirement,
                        control_type = EXCLUDED.control_type,
                        automated = EXCLUDED.automated,
                        frequency = EXCLUDED.frequency,
                        data_types = EXCLUDED.data_types,
                        retention_period = EXCLUDED.retention_period,
                        notification_required = EXCLUDED.notification_required,
                        escalation_required = EXCLUDED.escalation_required
                """, rule.rule_id, rule.framework.value, rule.rule_name,
                    rule.description, rule.requirement, rule.control_type,
                    rule.automated, rule.frequency, json.dumps(rule.data_types),
                    rule.retention_period, rule.notification_required,
                    rule.escalation_required)
                
                self.compliance_rules[rule.rule_id] = rule
        
        logger.info(f"Loaded {len(self.compliance_rules)} compliance rules")
    
    async def generate_compliance_report(self, framework: ComplianceFramework,
                                       period_start: datetime, period_end: datetime,
                                       user_id: str) -> str:
        """Generate compliance report."""
        try:
            report_id = f"report_{framework.value}_{datetime.utcnow().timestamp()}"
            
            # Collect compliance data
            findings = []
            recommendations = []
            summary = {}
            
            # Check data retention compliance
            retention_findings = await self._check_data_retention_compliance(framework, period_start, period_end)
            findings.extend(retention_findings)
            
            # Check access control compliance
            access_findings = await self._check_access_control_compliance(framework, period_start, period_end)
            findings.extend(access_findings)
            
            # Check audit logging compliance
            audit_findings = await self._check_audit_logging_compliance(framework, period_start, period_end)
            findings.extend(audit_findings)
            
            # Generate summary
            summary = {
                'total_findings': len(findings),
                'critical_findings': len([f for f in findings if f.get('severity') == 'critical']),
                'high_findings': len([f for f in findings if f.get('severity') == 'high']),
                'medium_findings': len([f for f in findings if f.get('severity') == 'medium']),
                'low_findings': len([f for f in findings if f.get('severity') == 'low']),
                'compliance_score': self._calculate_compliance_score(findings)
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(findings)
            
            # Store report
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_reports 
                    (report_id, framework, report_type, period_start, period_end,
                     generated_by, summary, findings, recommendations)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, report_id, framework.value, "periodic_assessment",
                    period_start, period_end, user_id,
                    json.dumps(summary), json.dumps(findings),
                    json.dumps(recommendations))
            
            logger.info(f"Generated compliance report: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return ""
    
    async def _check_data_retention_compliance(self, framework: ComplianceFramework,
                                             period_start: datetime, period_end: datetime) -> List[Dict]:
        """Check data retention compliance."""
        findings = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check for data older than retention policies
                tables_to_check = [
                    ('evidence', 'created_at', 365),
                    ('audit_logs', 'timestamp', 2555),
                    ('security_events', 'timestamp', 90),
                    ('detections', 'timestamp', 730)
                ]
                
                for table, date_column, retention_days in tables_to_check:
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    
                    count = await conn.fetchval(f"""
                        SELECT COUNT(*) FROM {table} 
                        WHERE {date_column} < $1
                    """, cutoff_date)
                    
                    if count > 0:
                        findings.append({
                            'rule_id': 'data_retention',
                            'severity': 'medium',
                            'description': f"Found {count} records in {table} older than {retention_days} days",
                            'recommendation': f"Purge old records from {table} table",
                            'affected_records': count
                        })
            
        except Exception as e:
            logger.error(f"Error checking data retention compliance: {e}")
        
        return findings
    
    async def _check_access_control_compliance(self, framework: ComplianceFramework,
                                             period_start: datetime, period_end: datetime) -> List[Dict]:
        """Check access control compliance."""
        findings = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check for users without MFA
                users_without_mfa = await conn.fetchval("""
                    SELECT COUNT(*) FROM users 
                    WHERE active = TRUE AND mfa_enabled = FALSE
                """)
                
                if users_without_mfa > 0:
                    findings.append({
                        'rule_id': 'mfa_requirement',
                        'severity': 'high',
                        'description': f"{users_without_mfa} active users without MFA enabled",
                        'recommendation': "Enforce MFA for all users",
                        'affected_users': users_without_mfa
                    })
                
                # Check for inactive users with access
                inactive_users = await conn.fetchval("""
                    SELECT COUNT(*) FROM users 
                    WHERE active = TRUE AND last_login < $1
                """, datetime.utcnow() - timedelta(days=90))
                
                if inactive_users > 0:
                    findings.append({
                        'rule_id': 'inactive_users',
                        'severity': 'medium',
                        'description': f"{inactive_users} users inactive for >90 days but still active",
                        'recommendation': "Review and deactivate inactive user accounts",
                        'affected_users': inactive_users
                    })
            
        except Exception as e:
            logger.error(f"Error checking access control compliance: {e}")
        
        return findings
    
    async def _check_audit_logging_compliance(self, framework: ComplianceFramework,
                                            period_start: datetime, period_end: datetime) -> List[Dict]:
        """Check audit logging compliance."""
        findings = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check for gaps in audit logging
                total_days = (period_end - period_start).days
                
                days_with_logs = await conn.fetchval("""
                    SELECT COUNT(DISTINCT DATE(timestamp)) 
                    FROM audit_logs 
                    WHERE timestamp BETWEEN $1 AND $2
                """, period_start, period_end)
                
                if days_with_logs < total_days * 0.95:  # Less than 95% coverage
                    findings.append({
                        'rule_id': 'audit_logging_gaps',
                        'severity': 'high',
                        'description': f"Audit logging gaps detected: {days_with_logs}/{total_days} days covered",
                        'recommendation': "Investigate and fix audit logging gaps",
                        'coverage_percentage': (days_with_logs / total_days) * 100
                    })
                
                # Check for high-privilege actions without proper logging
                admin_actions = await conn.fetchval("""
                    SELECT COUNT(*) FROM audit_logs al
                    JOIN users u ON al.user_id = u.id
                    WHERE u.role = 'administrator'
                    AND al.timestamp BETWEEN $1 AND $2
                    AND al.action NOT IN ('login', 'logout', 'view')
                """, period_start, period_end)
                
                if admin_actions == 0:
                    findings.append({
                        'rule_id': 'admin_activity_logging',
                        'severity': 'medium',
                        'description': "No administrative actions logged in period",
                        'recommendation': "Verify administrative activity logging is working",
                        'admin_actions': admin_actions
                    })
            
        except Exception as e:
            logger.error(f"Error checking audit logging compliance: {e}")
        
        return findings
    
    def _calculate_compliance_score(self, findings: List[Dict]) -> float:
        """Calculate overall compliance score."""
        if not findings:
            return 100.0
        
        # Weight findings by severity
        severity_weights = {
            'critical': 10,
            'high': 5,
            'medium': 2,
            'low': 1
        }
        
        total_weight = sum(severity_weights.get(f.get('severity', 'low'), 1) for f in findings)
        max_possible_weight = len(findings) * 10  # Assuming all could be critical
        
        score = max(0, 100 - (total_weight / max_possible_weight * 100))
        return round(score, 2)
    
    def _generate_recommendations(self, findings: List[Dict]) -> List[Dict]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        # Group findings by type and generate recommendations
        finding_types = {}
        for finding in findings:
            rule_id = finding.get('rule_id', 'unknown')
            if rule_id not in finding_types:
                finding_types[rule_id] = []
            finding_types[rule_id].append(finding)
        
        for rule_id, rule_findings in finding_types.items():
            if rule_id == 'data_retention':
                recommendations.append({
                    'priority': 'high',
                    'category': 'data_management',
                    'title': 'Implement Automated Data Purging',
                    'description': 'Set up automated processes to purge data according to retention policies',
                    'estimated_effort': 'medium',
                    'compliance_impact': 'high'
                })
            
            elif rule_id == 'mfa_requirement':
                recommendations.append({
                    'priority': 'critical',
                    'category': 'access_control',
                    'title': 'Enforce Multi-Factor Authentication',
                    'description': 'Require MFA for all user accounts, especially privileged users',
                    'estimated_effort': 'low',
                    'compliance_impact': 'high'
                })
            
            elif rule_id == 'audit_logging_gaps':
                recommendations.append({
                    'priority': 'high',
                    'category': 'monitoring',
                    'title': 'Fix Audit Logging Infrastructure',
                    'description': 'Ensure continuous audit logging with proper monitoring and alerting',
                    'estimated_effort': 'high',
                    'compliance_impact': 'critical'
                })
        
        return recommendations
    
    async def export_compliance_report(self, report_id: str, format: str = "csv") -> str:
        """Export compliance report to file."""
        try:
            async with self.db_pool.acquire() as conn:
                report = await conn.fetchrow(
                    "SELECT * FROM compliance_reports WHERE report_id = $1",
                    report_id
                )
                
                if not report:
                    return ""
                
                findings = json.loads(report['findings'])
                summary = json.loads(report['summary'])
                
                if format.lower() == "csv":
                    output = io.StringIO()
                    writer = csv.writer(output)
                    
                    # Write header
                    writer.writerow(['Rule ID', 'Severity', 'Description', 'Recommendation'])
                    
                    # Write findings
                    for finding in findings:
                        writer.writerow([
                            finding.get('rule_id', ''),
                            finding.get('severity', ''),
                            finding.get('description', ''),
                            finding.get('recommendation', '')
                        ])
                    
                    return output.getvalue()
                
                elif format.lower() == "json":
                    return json.dumps({
                        'report_id': report_id,
                        'framework': report['framework'],
                        'period': {
                            'start': report['period_start'].isoformat(),
                            'end': report['period_end'].isoformat()
                        },
                        'summary': summary,
                        'findings': findings,
                        'recommendations': json.loads(report['recommendations'])
                    }, indent=2, default=str)
                
                else:
                    return ""
                    
        except Exception as e:
            logger.error(f"Error exporting compliance report: {e}")
            return ""