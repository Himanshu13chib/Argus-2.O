"""
Core security system tests focusing on individual components.
Tests authentication flows, access control enforcement, encryption, and audit logging.
"""

import pytest
import asyncio
import json
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

from audit_compliance import (
    SecurityEventMonitor, ComplianceManager, SecurityEventType, 
    ComplianceFramework, SecurityEvent
)


@pytest.fixture
def mock_db_pool():
    """Mock database pool for testing."""
    pool = AsyncMock()
    conn = AsyncMock()
    
    # Properly mock the async context manager
    async def mock_acquire():
        return conn
    
    pool.acquire = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    
    return pool, conn


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    return AsyncMock()


@pytest.fixture
def security_monitor(mock_db_pool, mock_redis):
    """Create security event monitor instance."""
    db_pool, _ = mock_db_pool
    return SecurityEventMonitor(db_pool, mock_redis)


@pytest.fixture
def compliance_manager(mock_db_pool, mock_redis):
    """Create compliance manager instance."""
    db_pool, _ = mock_db_pool
    return ComplianceManager(db_pool, mock_redis)


class TestCryptographicOperations:
    """Test core cryptographic operations."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        salt = "random_salt_value"
        
        # Simple PBKDF2 implementation for testing
        import hashlib
        
        # Hash password
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        password_hash_hex = password_hash.hex()
        
        # Verify password
        verify_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        verify_hash_hex = verify_hash.hex()
        
        assert password_hash_hex == verify_hash_hex
        
        # Test wrong password
        wrong_hash = hashlib.pbkdf2_hmac('sha256', "wrong_password".encode(), salt.encode(), 100000)
        assert wrong_hash.hex() != password_hash_hex
    
    def test_symmetric_encryption(self):
        """Test symmetric encryption and decryption."""
        # Generate key
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        # Test data
        plaintext = b"sensitive data to encrypt"
        
        # Encrypt
        ciphertext = fernet.encrypt(plaintext)
        assert ciphertext != plaintext
        
        # Decrypt
        decrypted = fernet.decrypt(ciphertext)
        assert decrypted == plaintext
    
    def test_digital_signatures(self):
        """Test RSA digital signatures."""
        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Test data
        message = b"message to sign"
        
        # Sign message
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Verify signature
        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            signature_valid = True
        except:
            signature_valid = False
        
        assert signature_valid == True
        
        # Test invalid signature
        try:
            public_key.verify(
                signature,
                b"different message",
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            invalid_signature_valid = True
        except:
            invalid_signature_valid = False
        
        assert invalid_signature_valid == False
    
    def test_hmac_operations(self):
        """Test HMAC generation and verification."""
        key = b"secret_key_for_hmac_testing_32b"
        message = b"message to authenticate"
        
        # Generate HMAC
        mac = hmac.new(key, message, hashlib.sha256)
        hmac_hex = mac.hexdigest()
        
        # Verify HMAC
        verify_mac = hmac.new(key, message, hashlib.sha256)
        verify_hmac_hex = verify_mac.hexdigest()
        
        # Use constant-time comparison
        is_valid = hmac.compare_digest(hmac_hex, verify_hmac_hex)
        assert is_valid == True
        
        # Test invalid HMAC
        wrong_mac = hmac.new(key, b"different message", hashlib.sha256)
        wrong_hmac_hex = wrong_mac.hexdigest()
        
        is_invalid = hmac.compare_digest(hmac_hex, wrong_hmac_hex)
        assert is_invalid == False


class TestSecurityEventMonitoring:
    """Test security event monitoring and intrusion detection."""
    
    @pytest.mark.asyncio
    async def test_security_event_logging(self, security_monitor, mock_db_pool, mock_redis):
        """Test security event logging functionality."""
        db_pool, conn = mock_db_pool
        
        # Mock database operations
        conn.execute.return_value = None
        mock_redis.lpush.return_value = None
        mock_redis.ltrim.return_value = None
        
        # Log security event
        event_id = await security_monitor.log_security_event(
            event_type=SecurityEventType.LOGIN_FAILURE,
            user_id="test-user-123",
            source_ip="192.168.1.100",
            action="login_attempt",
            success=False,
            details={"username": "testuser", "reason": "invalid_password"}
        )
        
        # Verify event was logged
        assert event_id != ""
        assert event_id.startswith("login_failure_")
        
        # Verify database call
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0]
        assert "INSERT INTO security_events" in call_args[0]
        
        # Verify Redis caching
        mock_redis.lpush.assert_called_once()
        mock_redis.ltrim.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_risk_score_calculation(self, security_monitor):
        """Test security event risk score calculation."""
        # Test high-risk event
        high_risk_score = security_monitor._calculate_risk_score(
            SecurityEventType.SYSTEM_COMPROMISE,
            "192.168.1.100",
            "test-user",
            False,  # Failed action
            {"malware": "detected", "exploit": "buffer_overflow"}
        )
        
        assert high_risk_score >= 0.8
        
        # Test low-risk event
        low_risk_score = security_monitor._calculate_risk_score(
            SecurityEventType.LOGIN_SUCCESS,
            "10.0.0.1",  # Internal IP
            "test-user",
            True,  # Successful action
            {"method": "password"}
        )
        
        assert low_risk_score <= 0.5
        
        # Test medium-risk event
        medium_risk_score = security_monitor._calculate_risk_score(
            SecurityEventType.UNAUTHORIZED_ACCESS,
            "203.0.113.1",  # External IP
            "test-user",
            False,  # Failed action
            {"resource": "admin_panel"}
        )
        
        assert 0.3 <= medium_risk_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_threat_indicator_extraction(self, security_monitor):
        """Test extraction of threat indicators from event details."""
        # Test SQL injection indicators
        sql_injection_details = {
            "query": "SELECT * FROM users WHERE id = 1 UNION SELECT password FROM admin",
            "user_input": "'; DROP TABLE users; --",
            "normal_field": "normal_value"
        }
        
        indicators = security_monitor._extract_threat_indicators(sql_injection_details)
        
        assert len(indicators) >= 1
        assert any("union select" in indicator.lower() for indicator in indicators)
        
        # Test XSS indicators
        xss_details = {
            "user_agent": "Mozilla/5.0 <script>alert('xss')</script>",
            "input_field": "javascript:void(0)",
            "comment": "Normal comment text"
        }
        
        xss_indicators = security_monitor._extract_threat_indicators(xss_details)
        
        assert len(xss_indicators) >= 1
        assert any("script" in indicator.lower() for indicator in xss_indicators)
    
    @pytest.mark.asyncio
    async def test_intrusion_pattern_evaluation(self, security_monitor):
        """Test intrusion detection pattern evaluation."""
        from audit_compliance import IntrusionPattern
        
        # Create brute force pattern
        brute_force_pattern = IntrusionPattern(
            pattern_id="brute_force_test",
            name="Brute Force Attack",
            description="Multiple failed login attempts",
            event_types=[SecurityEventType.LOGIN_FAILURE],
            conditions={"same_ip": True, "different_users": True},
            time_window=300,
            threshold=3,
            severity="high",
            response_actions=["block_ip", "alert_admin"]
        )
        
        # Test events that match pattern
        matching_events = [
            {"source_ip": "192.168.1.100", "user_id": "user1", "session_id": None},
            {"source_ip": "192.168.1.100", "user_id": "user2", "session_id": None},
            {"source_ip": "192.168.1.100", "user_id": "user3", "session_id": None}
        ]
        
        result = security_monitor._evaluate_pattern_conditions(brute_force_pattern, matching_events)
        assert result == True
        
        # Test events that don't match pattern (same user)
        non_matching_events = [
            {"source_ip": "192.168.1.100", "user_id": "user1", "session_id": None},
            {"source_ip": "192.168.1.100", "user_id": "user1", "session_id": None},
            {"source_ip": "192.168.1.100", "user_id": "user1", "session_id": None}
        ]
        
        result = security_monitor._evaluate_pattern_conditions(brute_force_pattern, non_matching_events)
        assert result == False
        
        # Test session hijacking pattern
        session_hijack_pattern = IntrusionPattern(
            pattern_id="session_hijack_test",
            name="Session Hijacking",
            description="Same session from different IPs",
            event_types=[SecurityEventType.LOGIN_SUCCESS],
            conditions={"same_session": True, "different_ips": True},
            time_window=300,
            threshold=2,
            severity="critical",
            response_actions=["revoke_session", "alert_admin"]
        )
        
        # Test session hijacking events
        hijack_events = [
            {"source_ip": "192.168.1.100", "user_id": "user1", "session_id": "session123"},
            {"source_ip": "203.0.113.1", "user_id": "user1", "session_id": "session123"}
        ]
        
        hijack_result = security_monitor._evaluate_pattern_conditions(session_hijack_pattern, hijack_events)
        assert hijack_result == True
    
    @pytest.mark.asyncio
    async def test_ip_reputation_checking(self, security_monitor):
        """Test IP address reputation and blocking."""
        # Test private IP (less suspicious)
        assert security_monitor._is_suspicious_ip("10.0.0.1") == False
        assert security_monitor._is_suspicious_ip("192.168.1.100") == False
        assert security_monitor._is_suspicious_ip("127.0.0.1") == False
        
        # Test blocked IP
        security_monitor.blocked_ips.add("203.0.113.100")
        assert security_monitor._is_suspicious_ip("203.0.113.100") == True
        
        # Test invalid IP format
        assert security_monitor._is_suspicious_ip("invalid-ip-address") == True
        assert security_monitor._is_suspicious_ip("999.999.999.999") == True


class TestComplianceManagement:
    """Test compliance management and reporting."""
    
    @pytest.mark.asyncio
    async def test_compliance_rule_loading(self, compliance_manager, mock_db_pool):
        """Test loading of compliance rules."""
        db_pool, conn = mock_db_pool
        
        # Mock database operations
        conn.execute.return_value = None
        
        # Load compliance rules
        await compliance_manager._load_compliance_rules()
        
        # Verify rules were loaded
        assert len(compliance_manager.compliance_rules) > 0
        
        # Check specific rules exist
        assert "gdpr_data_retention" in compliance_manager.compliance_rules
        assert "gdpr_access_logging" in compliance_manager.compliance_rules
        assert "indian_it_data_protection" in compliance_manager.compliance_rules
        
        # Verify database operations
        assert conn.execute.call_count >= len(compliance_manager.compliance_rules)
    
    @pytest.mark.asyncio
    async def test_data_retention_compliance_check(self, compliance_manager, mock_db_pool):
        """Test data retention compliance checking."""
        db_pool, conn = mock_db_pool
        
        # Mock database responses - simulate old data found
        conn.fetchval.side_effect = [
            150,  # evidence table - old records
            75,   # audit_logs table - old records
            25,   # security_events table - old records
            10    # detections table - old records
        ]
        
        # Check data retention compliance
        findings = await compliance_manager._check_data_retention_compliance(
            ComplianceFramework.GDPR,
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        # Verify findings
        assert len(findings) == 4  # One finding per table with old data
        
        for finding in findings:
            assert finding["rule_id"] == "data_retention"
            assert finding["severity"] == "medium"
            assert "records" in finding["description"]
            assert "Purge old records" in finding["recommendation"]
            assert finding["affected_records"] > 0
    
    @pytest.mark.asyncio
    async def test_access_control_compliance_check(self, compliance_manager, mock_db_pool):
        """Test access control compliance checking."""
        db_pool, conn = mock_db_pool
        
        # Mock database responses
        conn.fetchval.side_effect = [
            5,  # Users without MFA
            3   # Inactive users
        ]
        
        # Check access control compliance
        findings = await compliance_manager._check_access_control_compliance(
            ComplianceFramework.GDPR,
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        # Verify findings
        assert len(findings) == 2
        
        # Check MFA finding
        mfa_finding = next(f for f in findings if f["rule_id"] == "mfa_requirement")
        assert mfa_finding["severity"] == "high"
        assert mfa_finding["affected_users"] == 5
        
        # Check inactive users finding
        inactive_finding = next(f for f in findings if f["rule_id"] == "inactive_users")
        assert inactive_finding["severity"] == "medium"
        assert inactive_finding["affected_users"] == 3
    
    @pytest.mark.asyncio
    async def test_audit_logging_compliance_check(self, compliance_manager, mock_db_pool):
        """Test audit logging compliance checking."""
        db_pool, conn = mock_db_pool
        
        # Mock database responses
        conn.fetchval.side_effect = [
            25,  # Days with logs (out of 30) - gap detected
            0    # Admin actions - suspicious
        ]
        
        # Check audit logging compliance
        findings = await compliance_manager._check_audit_logging_compliance(
            ComplianceFramework.GDPR,
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        # Verify findings
        assert len(findings) == 2
        
        # Check logging gaps finding
        gaps_finding = next(f for f in findings if f["rule_id"] == "audit_logging_gaps")
        assert gaps_finding["severity"] == "high"
        assert gaps_finding["coverage_percentage"] < 95
        
        # Check admin activity finding
        admin_finding = next(f for f in findings if f["rule_id"] == "admin_activity_logging")
        assert admin_finding["severity"] == "medium"
        assert admin_finding["admin_actions"] == 0
    
    @pytest.mark.asyncio
    async def test_compliance_score_calculation(self, compliance_manager):
        """Test compliance score calculation."""
        # Test perfect compliance (no findings)
        perfect_score = compliance_manager._calculate_compliance_score([])
        assert perfect_score == 100.0
        
        # Test with various severity findings
        findings = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "low"}
        ]
        
        score = compliance_manager._calculate_compliance_score(findings)
        assert 0 <= score <= 100
        assert score < 100  # Should be less than perfect
        
        # Test with only low severity findings
        low_findings = [{"severity": "low"} for _ in range(5)]
        low_score = compliance_manager._calculate_compliance_score(low_findings)
        
        # Test with only critical findings
        critical_findings = [{"severity": "critical"} for _ in range(2)]
        critical_score = compliance_manager._calculate_compliance_score(critical_findings)
        
        # Critical findings should result in lower score
        assert critical_score < low_score
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, compliance_manager):
        """Test generation of compliance recommendations."""
        # Test findings that should generate recommendations
        findings = [
            {"rule_id": "data_retention", "severity": "medium"},
            {"rule_id": "mfa_requirement", "severity": "high"},
            {"rule_id": "audit_logging_gaps", "severity": "high"},
            {"rule_id": "inactive_users", "severity": "medium"}
        ]
        
        recommendations = compliance_manager._generate_recommendations(findings)
        
        # Verify recommendations were generated
        assert len(recommendations) >= 3
        
        # Check for specific recommendation categories
        categories = [rec["category"] for rec in recommendations]
        assert "data_management" in categories
        assert "access_control" in categories
        assert "monitoring" in categories
        
        # Check recommendation structure
        for rec in recommendations:
            assert "priority" in rec
            assert "category" in rec
            assert "title" in rec
            assert "description" in rec
            assert "estimated_effort" in rec
            assert "compliance_impact" in rec
            
            # Verify priority levels are valid
            assert rec["priority"] in ["low", "medium", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_compliance_report_export(self, compliance_manager, mock_db_pool):
        """Test compliance report export functionality."""
        db_pool, conn = mock_db_pool
        
        # Mock report data
        mock_report = {
            'report_id': 'test_report_123',
            'framework': 'gdpr',
            'period_start': datetime.now() - timedelta(days=30),
            'period_end': datetime.now(),
            'findings': json.dumps([
                {
                    "rule_id": "test_rule",
                    "severity": "high",
                    "description": "Test compliance finding",
                    "recommendation": "Fix the issue"
                }
            ]),
            'summary': json.dumps({"total_findings": 1, "compliance_score": 85.5}),
            'recommendations': json.dumps([
                {
                    "priority": "high",
                    "category": "access_control",
                    "title": "Implement MFA",
                    "description": "Enable multi-factor authentication"
                }
            ])
        }
        
        conn.fetchrow.return_value = mock_report
        
        # Test CSV export
        csv_data = await compliance_manager.export_compliance_report("test_report_123", "csv")
        
        assert csv_data != ""
        assert "Rule ID,Severity,Description,Recommendation" in csv_data
        assert "test_rule,high,Test compliance finding,Fix the issue" in csv_data
        
        # Test JSON export
        json_data = await compliance_manager.export_compliance_report("test_report_123", "json")
        
        assert json_data != ""
        parsed_json = json.loads(json_data)
        assert parsed_json["report_id"] == "test_report_123"
        assert parsed_json["framework"] == "gdpr"
        assert "findings" in parsed_json
        assert "recommendations" in parsed_json
        assert parsed_json["summary"]["total_findings"] == 1
        assert parsed_json["summary"]["compliance_score"] == 85.5


class TestSecurityIntegration:
    """Test integration between security components."""
    
    @pytest.mark.asyncio
    async def test_event_to_compliance_integration(self, security_monitor, compliance_manager, 
                                                  mock_db_pool, mock_redis):
        """Test integration between security events and compliance reporting."""
        db_pool, conn = mock_db_pool
        
        # Mock database operations
        conn.execute.return_value = None
        conn.fetchval.side_effect = ["report_123"] + [0] * 10  # Report ID + compliance checks
        mock_redis.lpush.return_value = None
        mock_redis.ltrim.return_value = None
        
        # Log security events that affect compliance
        events = [
            (SecurityEventType.UNAUTHORIZED_ACCESS, "admin_panel"),
            (SecurityEventType.DATA_ACCESS, "personal_data"),
            (SecurityEventType.DATA_EXPORT, "user_records"),
            (SecurityEventType.PERMISSION_DENIED, "sensitive_config")
        ]
        
        event_ids = []
        for event_type, resource in events:
            event_id = await security_monitor.log_security_event(
                event_type=event_type,
                user_id="test-user",
                source_ip="192.168.1.100",
                action="access_attempt",
                success=False,
                details={"resource": resource}
            )
            event_ids.append(event_id)
        
        # Verify events were logged
        assert all(event_id != "" for event_id in event_ids)
        assert conn.execute.call_count == len(events)
        
        # Generate compliance report
        report_id = await compliance_manager.generate_compliance_report(
            framework=ComplianceFramework.GDPR,
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            user_id="admin_user"
        )
        
        assert report_id == "report_123"
        
        # Verify both systems interacted with database
        assert conn.execute.call_count > len(events)  # Events + report generation
    
    @pytest.mark.asyncio
    async def test_automated_security_response(self, security_monitor, mock_db_pool, mock_redis):
        """Test automated security response mechanisms."""
        db_pool, conn = mock_db_pool
        
        # Mock database operations
        conn.execute.return_value = None
        mock_redis.setex.return_value = None
        mock_redis.delete.return_value = None
        
        # Test IP blocking response
        await security_monitor._block_ip("192.168.1.100", "Brute force attack detected")
        
        # Verify IP was blocked
        assert "192.168.1.100" in security_monitor.blocked_ips
        
        # Verify database and Redis operations
        conn.execute.assert_called()
        mock_redis.setex.assert_called()
        
        # Test user suspension (mock)
        await security_monitor._suspend_user("suspicious-user-123", "Multiple violations")
        
        # Verify database operation for user suspension
        assert conn.execute.call_count >= 2  # Block IP + suspend user
    
    @pytest.mark.asyncio
    async def test_security_dashboard_data_aggregation(self, security_monitor, mock_db_pool):
        """Test security dashboard data aggregation."""
        db_pool, conn = mock_db_pool
        
        # Mock database responses for dashboard queries
        conn.fetch.side_effect = [
            # Event counts by type
            [
                {"event_type": "login_failure", "count": 25},
                {"event_type": "unauthorized_access", "count": 10},
                {"event_type": "suspicious_activity", "count": 5}
            ],
            # Active alerts
            [
                {
                    "alert_id": "alert_001",
                    "pattern_id": "brute_force_login",
                    "severity": "high",
                    "triggered_at": datetime.now(),
                    "source_ip": "192.168.1.100"
                }
            ],
            # Blocked entities
            [
                {
                    "entity_type": "ip",
                    "entity_value": "192.168.1.100",
                    "reason": "Brute force attack",
                    "blocked_at": datetime.now()
                },
                {
                    "entity_type": "user",
                    "entity_value": "suspicious_user",
                    "reason": "Multiple violations",
                    "blocked_at": datetime.now()
                }
            ]
        ]
        
        # Get dashboard data
        dashboard_data = await security_monitor.get_security_dashboard_data()
        
        # Verify dashboard data structure
        assert "event_counts" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "blocked_entities" in dashboard_data
        assert "total_events_24h" in dashboard_data
        assert "active_alert_count" in dashboard_data
        assert "blocked_ip_count" in dashboard_data
        
        # Verify data content
        assert dashboard_data["total_events_24h"] == 40  # 25 + 10 + 5
        assert dashboard_data["active_alert_count"] == 1
        assert dashboard_data["blocked_ip_count"] == 1  # Only IP entities
        
        # Verify event counts structure
        event_counts = dashboard_data["event_counts"]
        assert len(event_counts) == 3
        assert any(ec["type"] == "login_failure" and ec["count"] == 25 for ec in event_counts)


if __name__ == "__main__":
    pytest.main([__file__])