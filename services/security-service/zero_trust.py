"""
Zero-Trust Network Security Implementation for Project Argus.
Implements micro-segmentation, network policies, and secure service-to-service communication.
"""

import os
import logging
import asyncio
import json
import ipaddress
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import asyncpg
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID
import ssl
import socket

logger = logging.getLogger(__name__)


class NetworkZone(Enum):
    """Network security zones."""
    DMZ = "dmz"
    INTERNAL = "internal"
    SECURE = "secure"
    EDGE = "edge"
    MANAGEMENT = "management"


class ServiceType(Enum):
    """Service types for micro-segmentation."""
    API_GATEWAY = "api-gateway"
    AUTH_SERVICE = "auth-service"
    SECURITY_SERVICE = "security-service"
    ALERT_SERVICE = "alert-service"
    EVIDENCE_SERVICE = "evidence-service"
    TRACKING_SERVICE = "tracking-service"
    DASHBOARD = "dashboard"
    EDGE_NODE = "edge-node"
    DATABASE = "database"
    CACHE = "cache"
    STORAGE = "storage"


@dataclass
class NetworkPolicy:
    """Network access policy definition."""
    id: str
    name: str
    source_zone: NetworkZone
    destination_zone: NetworkZone
    source_services: List[ServiceType]
    destination_services: List[ServiceType]
    allowed_ports: List[int]
    protocol: str  # tcp, udp, icmp
    action: str  # allow, deny, log
    priority: int
    conditions: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    active: bool = True


@dataclass
class ServiceIdentity:
    """Service identity for zero-trust authentication."""
    service_id: str
    service_type: ServiceType
    zone: NetworkZone
    certificate_fingerprint: str
    ip_addresses: List[str]
    allowed_endpoints: List[str]
    security_level: str
    last_verified: datetime
    expires_at: datetime


@dataclass
class NetworkConnection:
    """Network connection tracking."""
    connection_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    source_service: Optional[ServiceType]
    destination_service: Optional[ServiceType]
    established_at: datetime
    last_activity: datetime
    bytes_sent: int
    bytes_received: int
    status: str  # established, closed, suspicious
    risk_score: float


class ZeroTrustManager:
    """Zero-trust network security manager."""
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis = redis_client
        self.network_policies: Dict[str, NetworkPolicy] = {}
        self.service_identities: Dict[str, ServiceIdentity] = {}
        self.active_connections: Dict[str, NetworkConnection] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_activities: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize zero-trust manager."""
        await self._create_tables()
        await self._load_network_policies()
        await self._load_service_identities()
        await self._start_monitoring()
        logger.info("Zero-trust manager initialized")
    
    async def _create_tables(self):
        """Create database tables for zero-trust data."""
        async with self.db_pool.acquire() as conn:
            # Network policies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS network_policies (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    source_zone VARCHAR(50) NOT NULL,
                    destination_zone VARCHAR(50) NOT NULL,
                    source_services JSONB NOT NULL,
                    destination_services JSONB NOT NULL,
                    allowed_ports JSONB NOT NULL,
                    protocol VARCHAR(10) NOT NULL,
                    action VARCHAR(10) NOT NULL,
                    priority INTEGER NOT NULL,
                    conditions JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Service identities table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS service_identities (
                    service_id VARCHAR(255) PRIMARY KEY,
                    service_type VARCHAR(50) NOT NULL,
                    zone VARCHAR(50) NOT NULL,
                    certificate_fingerprint VARCHAR(255) NOT NULL,
                    ip_addresses JSONB NOT NULL,
                    allowed_endpoints JSONB NOT NULL,
                    security_level VARCHAR(20) NOT NULL,
                    last_verified TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Network connections table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS network_connections (
                    connection_id VARCHAR(255) PRIMARY KEY,
                    source_ip INET NOT NULL,
                    destination_ip INET NOT NULL,
                    source_port INTEGER NOT NULL,
                    destination_port INTEGER NOT NULL,
                    protocol VARCHAR(10) NOT NULL,
                    source_service VARCHAR(50),
                    destination_service VARCHAR(50),
                    established_at TIMESTAMP DEFAULT NOW(),
                    last_activity TIMESTAMP DEFAULT NOW(),
                    bytes_sent BIGINT DEFAULT 0,
                    bytes_received BIGINT DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'established',
                    risk_score FLOAT DEFAULT 0.0
                )
            """)
            
            # Security events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id VARCHAR(255) PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    source_ip INET,
                    destination_ip INET,
                    service_id VARCHAR(255),
                    description TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_network_policies_zones ON network_policies(source_zone, destination_zone)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_service_identities_type ON service_identities(service_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_network_connections_ips ON network_connections(source_ip, destination_ip)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type, created_at)")
    
    async def _load_network_policies(self):
        """Load network policies from database."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM network_policies WHERE active = TRUE")
            
            for row in rows:
                policy = NetworkPolicy(
                    id=row['id'],
                    name=row['name'],
                    source_zone=NetworkZone(row['source_zone']),
                    destination_zone=NetworkZone(row['destination_zone']),
                    source_services=[ServiceType(s) for s in row['source_services']],
                    destination_services=[ServiceType(s) for s in row['destination_services']],
                    allowed_ports=row['allowed_ports'],
                    protocol=row['protocol'],
                    action=row['action'],
                    priority=row['priority'],
                    conditions=row['conditions'] or {},
                    created_at=row['created_at'],
                    expires_at=row['expires_at'],
                    active=row['active']
                )
                self.network_policies[policy.id] = policy
        
        logger.info(f"Loaded {len(self.network_policies)} network policies")
    
    async def _load_service_identities(self):
        """Load service identities from database."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM service_identities WHERE expires_at > NOW()")
            
            for row in rows:
                identity = ServiceIdentity(
                    service_id=row['service_id'],
                    service_type=ServiceType(row['service_type']),
                    zone=NetworkZone(row['zone']),
                    certificate_fingerprint=row['certificate_fingerprint'],
                    ip_addresses=row['ip_addresses'],
                    allowed_endpoints=row['allowed_endpoints'],
                    security_level=row['security_level'],
                    last_verified=row['last_verified'],
                    expires_at=row['expires_at']
                )
                self.service_identities[identity.service_id] = identity
        
        logger.info(f"Loaded {len(self.service_identities)} service identities")
    
    async def _start_monitoring(self):
        """Start network monitoring tasks."""
        asyncio.create_task(self._monitor_connections())
        asyncio.create_task(self._cleanup_expired_data())
        asyncio.create_task(self._analyze_security_events())
    
    async def register_service(self, service_id: str, service_type: ServiceType, 
                             zone: NetworkZone, certificate_data: bytes,
                             ip_addresses: List[str], allowed_endpoints: List[str],
                             security_level: str = "standard") -> bool:
        """Register a service identity."""
        try:
            # Extract certificate fingerprint
            cert = x509.load_pem_x509_certificate(certificate_data, default_backend())
            fingerprint = cert.fingerprint(cert.signature_hash_algorithm).hex()
            
            # Validate IP addresses
            for ip in ip_addresses:
                ipaddress.ip_address(ip)
            
            identity = ServiceIdentity(
                service_id=service_id,
                service_type=service_type,
                zone=zone,
                certificate_fingerprint=fingerprint,
                ip_addresses=ip_addresses,
                allowed_endpoints=allowed_endpoints,
                security_level=security_level,
                last_verified=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO service_identities 
                    (service_id, service_type, zone, certificate_fingerprint, 
                     ip_addresses, allowed_endpoints, security_level, last_verified, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (service_id) DO UPDATE SET
                        certificate_fingerprint = EXCLUDED.certificate_fingerprint,
                        ip_addresses = EXCLUDED.ip_addresses,
                        allowed_endpoints = EXCLUDED.allowed_endpoints,
                        last_verified = EXCLUDED.last_verified,
                        expires_at = EXCLUDED.expires_at
                """, service_id, service_type.value, zone.value, fingerprint,
                    json.dumps(ip_addresses), json.dumps(allowed_endpoints),
                    security_level, identity.last_verified, identity.expires_at)
            
            self.service_identities[service_id] = identity
            
            # Cache in Redis
            await self.redis.setex(
                f"service_identity:{service_id}",
                timedelta(hours=24),
                json.dumps(asdict(identity), default=str)
            )
            
            logger.info(f"Registered service identity: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering service identity: {e}")
            return False
    
    async def verify_service_identity(self, service_id: str, 
                                    certificate_data: bytes,
                                    source_ip: str) -> Tuple[bool, Optional[ServiceIdentity]]:
        """Verify service identity using certificate and IP."""
        try:
            # Get service identity
            identity = self.service_identities.get(service_id)
            if not identity:
                # Try loading from cache
                cached = await self.redis.get(f"service_identity:{service_id}")
                if cached:
                    identity_data = json.loads(cached)
                    identity = ServiceIdentity(**identity_data)
                    self.service_identities[service_id] = identity
                else:
                    return False, None
            
            # Check expiration
            if identity.expires_at < datetime.utcnow():
                logger.warning(f"Service identity expired: {service_id}")
                return False, None
            
            # Verify certificate fingerprint
            cert = x509.load_pem_x509_certificate(certificate_data, default_backend())
            fingerprint = cert.fingerprint(cert.signature_hash_algorithm).hex()
            
            if fingerprint != identity.certificate_fingerprint:
                logger.warning(f"Certificate fingerprint mismatch for service: {service_id}")
                await self._log_security_event("certificate_mismatch", "high", source_ip, 
                                              service_id, "Certificate fingerprint mismatch")
                return False, None
            
            # Verify IP address
            if source_ip not in identity.ip_addresses:
                logger.warning(f"IP address not allowed for service {service_id}: {source_ip}")
                await self._log_security_event("unauthorized_ip", "medium", source_ip,
                                              service_id, f"Unauthorized IP: {source_ip}")
                return False, None
            
            # Update last verified
            identity.last_verified = datetime.utcnow()
            
            return True, identity
            
        except Exception as e:
            logger.error(f"Error verifying service identity: {e}")
            return False, None
    
    async def check_network_policy(self, source_ip: str, destination_ip: str,
                                 destination_port: int, protocol: str,
                                 source_service: Optional[ServiceType] = None,
                                 destination_service: Optional[ServiceType] = None) -> Tuple[bool, str]:
        """Check if network connection is allowed by policy."""
        try:
            # Determine zones based on IP addresses
            source_zone = self._get_zone_for_ip(source_ip)
            dest_zone = self._get_zone_for_ip(destination_ip)
            
            # Find applicable policies
            applicable_policies = []
            for policy in self.network_policies.values():
                if (policy.source_zone == source_zone and 
                    policy.destination_zone == dest_zone and
                    protocol.lower() == policy.protocol.lower()):
                    
                    # Check service types
                    if source_service and source_service not in policy.source_services:
                        continue
                    if destination_service and destination_service not in policy.destination_services:
                        continue
                    
                    # Check port
                    if destination_port not in policy.allowed_ports:
                        continue
                    
                    applicable_policies.append(policy)
            
            if not applicable_policies:
                # Default deny
                await self._log_security_event("policy_violation", "medium", source_ip,
                                              None, f"No policy allows {source_ip}:{destination_port}")
                return False, "No applicable policy found"
            
            # Sort by priority and apply first matching policy
            applicable_policies.sort(key=lambda p: p.priority)
            policy = applicable_policies[0]
            
            if policy.action == "allow":
                return True, f"Allowed by policy: {policy.name}"
            elif policy.action == "deny":
                await self._log_security_event("policy_deny", "low", source_ip,
                                              None, f"Denied by policy: {policy.name}")
                return False, f"Denied by policy: {policy.name}"
            else:
                # Log action
                await self._log_security_event("policy_log", "info", source_ip,
                                              None, f"Logged by policy: {policy.name}")
                return True, f"Logged by policy: {policy.name}"
                
        except Exception as e:
            logger.error(f"Error checking network policy: {e}")
            return False, "Policy check failed"
    
    def _get_zone_for_ip(self, ip_address: str) -> NetworkZone:
        """Determine network zone for IP address."""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Define zone IP ranges
            if ip.is_private:
                if ip in ipaddress.ip_network("10.0.0.0/24"):  # Management
                    return NetworkZone.MANAGEMENT
                elif ip in ipaddress.ip_network("10.0.1.0/24"):  # Secure services
                    return NetworkZone.SECURE
                elif ip in ipaddress.ip_network("10.0.2.0/24"):  # Internal services
                    return NetworkZone.INTERNAL
                elif ip in ipaddress.ip_network("10.0.3.0/24"):  # Edge nodes
                    return NetworkZone.EDGE
                else:
                    return NetworkZone.INTERNAL
            else:
                return NetworkZone.DMZ
                
        except Exception:
            return NetworkZone.DMZ
    
    async def track_connection(self, source_ip: str, destination_ip: str,
                             source_port: int, destination_port: int,
                             protocol: str, source_service: Optional[ServiceType] = None,
                             destination_service: Optional[ServiceType] = None) -> str:
        """Track network connection."""
        connection_id = f"{source_ip}:{source_port}-{destination_ip}:{destination_port}-{protocol}"
        
        connection = NetworkConnection(
            connection_id=connection_id,
            source_ip=source_ip,
            destination_ip=destination_ip,
            source_port=source_port,
            destination_port=destination_port,
            protocol=protocol,
            source_service=source_service,
            destination_service=destination_service,
            established_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            bytes_sent=0,
            bytes_received=0,
            status="established",
            risk_score=0.0
        )
        
        self.active_connections[connection_id] = connection
        
        # Store in database
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO network_connections 
                    (connection_id, source_ip, destination_ip, source_port, destination_port,
                     protocol, source_service, destination_service, established_at, last_activity,
                     bytes_sent, bytes_received, status, risk_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, connection_id, source_ip, destination_ip, source_port, destination_port,
                    protocol, source_service.value if source_service else None,
                    destination_service.value if destination_service else None,
                    connection.established_at, connection.last_activity,
                    connection.bytes_sent, connection.bytes_received,
                    connection.status, connection.risk_score)
        except Exception as e:
            logger.error(f"Error storing connection: {e}")
        
        return connection_id
    
    async def update_connection_stats(self, connection_id: str, 
                                    bytes_sent: int, bytes_received: int):
        """Update connection statistics."""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            connection.bytes_sent += bytes_sent
            connection.bytes_received += bytes_received
            connection.last_activity = datetime.utcnow()
            
            # Calculate risk score based on traffic patterns
            connection.risk_score = self._calculate_risk_score(connection)
    
    def _calculate_risk_score(self, connection: NetworkConnection) -> float:
        """Calculate risk score for connection."""
        risk_score = 0.0
        
        # High data volume
        total_bytes = connection.bytes_sent + connection.bytes_received
        if total_bytes > 100 * 1024 * 1024:  # 100MB
            risk_score += 0.3
        
        # Long duration
        duration = (datetime.utcnow() - connection.established_at).total_seconds()
        if duration > 3600:  # 1 hour
            risk_score += 0.2
        
        # Unusual ports
        if connection.destination_port not in [80, 443, 8000, 8001, 8002, 8003, 8004, 8005]:
            risk_score += 0.1
        
        # Cross-zone communication
        source_zone = self._get_zone_for_ip(connection.source_ip)
        dest_zone = self._get_zone_for_ip(connection.destination_ip)
        if source_zone != dest_zone:
            risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    async def _monitor_connections(self):
        """Monitor active connections for suspicious activity."""
        while True:
            try:
                current_time = datetime.utcnow()
                suspicious_connections = []
                
                for connection_id, connection in self.active_connections.items():
                    # Check for stale connections
                    if (current_time - connection.last_activity).total_seconds() > 300:  # 5 minutes
                        connection.status = "stale"
                    
                    # Check for high-risk connections
                    if connection.risk_score > 0.7:
                        suspicious_connections.append(connection)
                
                # Handle suspicious connections
                for connection in suspicious_connections:
                    await self._log_security_event("suspicious_connection", "medium",
                                                  connection.source_ip, None,
                                                  f"High-risk connection: {connection.connection_id}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring connections: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_data(self):
        """Clean up expired data."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Remove expired service identities
                expired_services = [
                    service_id for service_id, identity in self.service_identities.items()
                    if identity.expires_at < current_time
                ]
                
                for service_id in expired_services:
                    del self.service_identities[service_id]
                    await self.redis.delete(f"service_identity:{service_id}")
                
                # Remove old connections
                old_connections = [
                    conn_id for conn_id, connection in self.active_connections.items()
                    if (current_time - connection.last_activity).total_seconds() > 3600  # 1 hour
                ]
                
                for conn_id in old_connections:
                    self.active_connections[conn_id].status = "closed"
                    del self.active_connections[conn_id]
                
                # Clean up database
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM network_connections WHERE last_activity < $1",
                        current_time - timedelta(hours=24)
                    )
                    await conn.execute(
                        "DELETE FROM security_events WHERE created_at < $1 AND resolved = TRUE",
                        current_time - timedelta(days=30)
                    )
                
                logger.info(f"Cleaned up {len(expired_services)} expired services and {len(old_connections)} old connections")
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up expired data: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_security_events(self):
        """Analyze security events for patterns."""
        while True:
            try:
                # Analyze recent security events
                async with self.db_pool.acquire() as conn:
                    # Find IPs with multiple violations
                    rows = await conn.fetch("""
                        SELECT source_ip, COUNT(*) as violation_count
                        FROM security_events 
                        WHERE created_at > $1 AND severity IN ('high', 'medium')
                        GROUP BY source_ip
                        HAVING COUNT(*) > 5
                    """, datetime.utcnow() - timedelta(hours=1))
                    
                    for row in rows:
                        ip = str(row['source_ip'])
                        count = row['violation_count']
                        
                        if ip not in self.blocked_ips:
                            self.blocked_ips.add(ip)
                            await self._log_security_event("ip_blocked", "high", ip, None,
                                                          f"IP blocked due to {count} violations")
                            logger.warning(f"Blocked IP {ip} due to {count} security violations")
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error analyzing security events: {e}")
                await asyncio.sleep(300)
    
    async def _log_security_event(self, event_type: str, severity: str,
                                source_ip: str, service_id: Optional[str],
                                description: str, metadata: Optional[Dict[str, Any]] = None):
        """Log security event."""
        try:
            event_id = f"{event_type}_{datetime.utcnow().timestamp()}"
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO security_events 
                    (event_id, event_type, severity, source_ip, service_id, description, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, event_id, event_type, severity, source_ip, service_id, description,
                    json.dumps(metadata) if metadata else None)
            
            # Cache recent events in Redis
            await self.redis.lpush("recent_security_events", json.dumps({
                "event_id": event_id,
                "event_type": event_type,
                "severity": severity,
                "source_ip": source_ip,
                "service_id": service_id,
                "description": description,
                "timestamp": datetime.utcnow().isoformat()
            }))
            await self.redis.ltrim("recent_security_events", 0, 999)  # Keep last 1000 events
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
    
    async def unblock_ip(self, ip_address: str) -> bool:
        """Unblock IP address."""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            await self._log_security_event("ip_unblocked", "info", ip_address, None,
                                          "IP address manually unblocked")
            return True
        return False
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "active_policies": len(self.network_policies),
            "registered_services": len(self.service_identities),
            "active_connections": len(self.active_connections),
            "blocked_ips": len(self.blocked_ips),
            "high_risk_connections": len([
                c for c in self.active_connections.values() if c.risk_score > 0.7
            ]),
            "zones": {
                zone.value: len([
                    s for s in self.service_identities.values() if s.zone == zone
                ]) for zone in NetworkZone
            }
        }