"""
Tests for audit logging and compliance system.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncpg

from audit_compliance import (
    SecurityEventMonitor, ComplianceManager, SecurityEventType, 
    ComplianceFramework, SecurityEvent, ComplianceRule
)


@pytest.fixture
def mock_db_pool():
    """Mock database pool."""
    pool = AsyncMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    return pool, conn


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    return redis


@pytest.fixture
def security_monitor(mock_db_pool, mock_redis):
    """Create security event monitor instance."""
    db_pool, _ = mock_db_pool
    monitor = SecurityEventMonitor(db_pool, mock_redis)
    return monitor


@pytest.fixture
def compliance_manager(mock_db_pool, mock_redis):
    """Create compliance manager instance."""
    db_pool, _ = mock_db_pool
    manager = ComplianceManager(db_pool, mock_redis)
    return manager


class TestSecurityEventMonitor:
    """Test security event monitoring."""
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, security_monitor, mock_db_pool, mock_redis):
        """Test logging security events."""
        db_pool, conn = mock_db_pool
        
        # Mock database operations
        conn.execute.return_value = None
        
        # Log a security event
        event_id = await security_monitor.log_security_event(
            event_type=SecurityEventType.LOGIN_FAILURE,
            user_id="test-user-id",
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
        
        # Verify Redis call
        mock_redis.lpush.assert_called_once()
        mock_redis.ltrim.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_risk_score(self, security_monitor):
        """Test risk score calculation."""
        # Test high-risk event
        risk_score = security_monitor._calculate_risk_score(
            SecurityEventType.SYSTEM_COMPROMISE,
            "192.168.1.100",
            "test-user",
            False,
            {"malware": "detected"}
        )
        assert risk_score >= 0.8
        
        # Test low-risk event
        risk_score = security_monitor._calculate_risk_score(
            SecurityEventType.LOGIN_SUCCESS,
            "192.168.1.100",
            "test-user",
            True,
            {}
        )
        assert risk_score <= 0.5
    
    @pytest.mark.asyncio
    async def test_determine_severity(self, security_monitor):
        """Test severity determination."""
        # Test critical severity
        severity = security_monitor._determine_severity(
            SecurityEventType.SYSTEM_COMPROMISE, 0.9
        )
        assert severity == "critical"
        
        # Test low severity
        severity = security_monitor._determine_severity(
            SecurityEventType.LOGIN_SUCCESS, 0.1
        )
        assert severity == "low"
    
    @pytest.mark.asyncio
    async def test_extract_threat_indicators(self, security_monitor):
        """Test threat indicator extraction."""
        details = {
            "query": "SELECT * FROM users WHERE id = 1 UNION SELECT password FROM admin",
            "user_agent": "Mozilla/5.0 <script>alert('xss')</script>",
            "normal_field": "normal_value"
        }
        
        indicators = security_monitor._extract_threat_indicators(details)
        
        assert len(indicators) >= 2
        assert any("union select" in indicator.lower() for indicator in indicators)
        assert any("script" in indicator.lower() for indicator in indicators)
    
    @pytest.mark.asyncio
    async def test_is_suspicious_ip(self, security_monitor):
        """Test suspicious IP detection."""
        # Test blocked IP
        security_monitor.blocked_ips.add("192.168.1.100")
        assert security_monitor._is_suspicious_ip("192.168.1.100") == True
        
        # Test private IP (less suspicious)
        assert security_monitor._is_suspicious_ip("10.0.0.1") == False
        
        # Test invalid IP (suspicious)
        assert security_monitor._is_suspicious_ip("invalid-ip") == True
    
    @pytest.mark.asyncio
    async def test_evaluate_pattern_conditions(self, security_monitor):
        """Test pattern condition evaluation."""
        # Create test pattern
        from audit_compliance import IntrusionPattern
        pattern = IntrusionPattern(
            pattern_id="test_pattern",
            name="Test Pattern",
            description="Test",
            event_types=[SecurityEventType.LOGIN_FAILURE],
            conditions={"same_ip": True, "different_users": True},
            time_window=300,
            threshold=3,
            severity="high",
            response_actions=["block_ip"]
        )
        
        # Test events that meet conditions
        events = [
            {"source_ip": "192.168.1.100", "user_id": "user1"},
            {"source_ip": "192.168.1.100", "user_id": "user2"},
            {"source_ip": "192.168.1.100", "user_id": "user3"}
        ]
        
        result = security_monitor._evaluate_pattern_conditions(pattern, events)
        assert result == True
        
        # Test events that don't meet conditions (same user)
        events_same_user = [
            {"source_ip": "192.168.1.100", "user_id": "user1"},
            {"source_ip": "192.168.1.100", "user_id": "user1"},
            {"source_ip": "192.168.1.100", "user_id": "user1"}
        ]
        
        result = security_monitor._evaluate_pattern_conditions(pattern, events_same_user)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_get_security_dashboard_data(self, security_monitor, mock_db_pool):
        """Test security dashboard data retrieval."""
        db_pool, conn = mock_db_pool
        
        # Mock database responses
        conn.fetch.side_effect = [
            [{"event_type": "login_failure", "count": 10}],  # event_counts
            [{"alert_id": "alert1", "pattern_id": "pattern1", "severity": "high", 
              "triggered_at": datetime.utcnow(), "source_ip": "192.168.1.100"}],  # active_alerts
            [{"entity_type": "ip", "entity_value": "192.168.1.100", 
              "reason": "brute_force", "blocked_at": datetime.utcnow()}]  # blocked_entities
        ]
        
        dashboard_data = await security_monitor.get_security_dashboard_data()
        
        assert "event_counts" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "blocked_entities" in dashboard_data
        assert dashboard_data["total_events_24h"] == 10
        assert dashboard_data["active_alert_count"] == 1
        assert dashboard_data["blocked_ip_count"] == 1


class TestComplianceManager:
    """Test compliance management."""
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, compliance_manager, mock_db_pool):
        """Test compliance report generation."""
        db_pool, conn = mock_db_pool
        
        # Mock database operations
        conn.fetchval.side_effect = [
            "report_id_123",  # INSERT RETURNING
            5,  # data retention check
            2,  # users without MFA
            1,  # inactive users
            30,  # days with logs
            0   # admin actions
        ]
        
        report_id = await compliance_manager.generate_compliance_report(
            framework=ComplianceFramework.GDPR,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            user_id="test-user"
        )
        
        assert report_id != ""
        
        # Verify database calls
        assert conn.execute.call_count >= 1
        assert conn.fetchval.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_check_data_retention_compliance(self, compliance_manager, mock_db_pool):
        """Test data retention compliance checking."""
        db_pool, conn = mock_db_pool
        
        # Mock old data found
        conn.fetchval.side_effect = [100, 50, 25, 10]  # Old records in different tables
        
        findings = await compliance_manager._check_data_retention_compliance(
            ComplianceFramework.GDPR,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        
        assert len(findings) == 4  # One finding per table with old data
        assert all(finding["rule_id"] == "data_retention" for finding in findings)
        assert all(finding["severity"] == "medium" for finding in findings)
    
    @pytest.mark.asyncio
    async def test_check_access_control_compliance(self, compliance_manager, mock_db_pool):
        """Test access control compliance checking."""
        db_pool, conn = mock_db_pool
        
        # Mock compliance issues
        conn.fetchval.side_effect = [5, 3]  # Users without MFA, inactive users
        
        findings = await compliance_manager._check_access_control_compliance(
            ComplianceFramework.GDPR,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        
        assert len(findings) == 2
        assert findings[0]["rule_id"] == "mfa_requirement"
        assert findings[0]["severity"] == "high"
        assert findings[1]["rule_id"] == "inactive_users"
        assert findings[1]["severity"] == "medium"
    
    @pytest.mark.asyncio
    async def test_check_audit_logging_compliance(self, compliance_manager, mock_db_pool):
        """Test audit logging compliance checking."""
        db_pool, conn = mock_db_pool
        
        # Mock audit logging issues
        conn.fetchval.side_effect = [25, 0]  # Days with logs (out of 30), admin actions
        
        findings = await compliance_manager._check_audit_logging_compliance(
            ComplianceFramework.GDPR,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        
        assert len(findings) == 2
        assert findings[0]["rule_id"] == "audit_logging_gaps"
        assert findings[0]["severity"] == "high"
        assert findings[1]["rule_id"] == "admin_activity_logging"
        assert findings[1]["severity"] == "medium"
    
    @pytest.mark.asyncio
    async def test_calculate_compliance_score(self, compliance_manager):
        """Test compliance score calculation."""
        # Test perfect compliance (no findings)
        score = compliance_manager._calculate_compliance_score([])
        assert score == 100.0
        
        # Test with findings
        findings = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "low"}
        ]
        
        score = compliance_manager._calculate_compliance_score(findings)
        assert 0 <= score <= 100
        assert score < 100  # Should be less than perfect
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, compliance_manager):
        """Test recommendation generation."""
        findings = [
            {"rule_id": "data_retention", "severity": "medium"},
            {"rule_id": "mfa_requirement", "severity": "high"},
            {"rule_id": "audit_logging_gaps", "severity": "high"}
        ]
        
        recommendations = compliance_manager._generate_recommendations(findings)
        
        assert len(recommendations) == 3
        assert any(rec["category"] == "data_management" for rec in recommendations)
        assert any(rec["category"] == "access_control" for rec in recommendations)
        assert any(rec["category"] == "monitoring" for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_export_compliance_report(self, compliance_manager, mock_db_pool):
        """Test compliance report export."""
        db_pool, conn = mock_db_pool
        
        # Mock report data
        mock_report = {
            'report_id': 'test_report',
            'framework': 'gdpr',
            'period_start': datetime.utcnow() - timedelta(days=30),
            'period_end': datetime.utcnow(),
            'findings': json.dumps([
                {"rule_id": "test_rule", "severity": "high", 
                 "description": "Test finding", "recommendation": "Fix it"}
            ]),
            'summary': json.dumps({"total_findings": 1}),
            'recommendations': json.dumps([])
        }
        
        conn.fetchrow.return_value = mock_report
        
        # Test CSV export
        csv_data = await compliance_manager.export_compliance_report("test_report", "csv")
        assert csv_data != ""
        assert "Rule ID,Severity,Description,Recommendation" in csv_data
        assert "test_rule,high,Test finding,Fix it" in csv_data
        
        # Test JSON export
        json_data = await compliance_manager.export_compliance_report("test_report", "json")
        assert json_data != ""
        parsed_json = json.loads(json_data)
        assert parsed_json["report_id"] == "test_report"
        assert parsed_json["framework"] == "gdpr"


class TestIntegration:
    """Integration tests for audit and compliance system."""
    
    @pytest.mark.asyncio
    async def test_security_event_to_compliance_report_flow(self, security_monitor, 
                                                           compliance_manager, mock_db_pool):
        """Test end-to-end flow from security event to compliance report."""
        db_pool, conn = mock_db_pool
        
        # Mock database operations for security event
        conn.execute.return_value = None
        
        # Log multiple security events
        for i in range(5):
            await security_monitor.log_security_event(
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                user_id=f"user_{i}",
                source_ip="192.168.1.100",
                action="access_denied",
                success=False,
                details={"resource": "sensitive_data"}
            )
        
        # Mock compliance report generation
        conn.fetchval.side_effect = ["report_123"] + [0] * 10  # Report ID + compliance checks
        
        # Generate compliance report
        report_id = await compliance_manager.generate_compliance_report(
            framework=ComplianceFramework.GDPR,
            period_start=datetime.utcnow() - timedelta(days=1),
            period_end=datetime.utcnow(),
            user_id="admin_user"
        )
        
        assert report_id == "report_123"
        
        # Verify both systems were called
        assert conn.execute.call_count >= 5  # At least 5 security events + report
    
    @pytest.mark.asyncio
    async def test_intrusion_detection_and_response(self, security_monitor, mock_db_pool):
        """Test intrusion detection and automated response."""
        db_pool, conn = mock_db_pool
        
        # Mock pattern loading
        security_monitor.intrusion_patterns = {
            "brute_force_login": MagicMock(
                pattern_id="brute_force_login",
                event_types=[SecurityEventType.LOGIN_FAILURE],
                conditions={"same_ip": True},
                threshold=3,
                response_actions=["block_ip"]
            )
        }
        
        # Mock database operations
        conn.execute.return_value = None
        conn.fetch.return_value = [
            {"event_id": "event1", "source_ip": "192.168.1.100", "user_id": "user1"},
            {"event_id": "event2", "source_ip": "192.168.1.100", "user_id": "user2"},
            {"event_id": "event3", "source_ip": "192.168.1.100", "user_id": "user3"}
        ]
        
        # Create test event
        test_event = SecurityEvent(
            event_id="test_event",
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity="medium",
            timestamp=datetime.utcnow(),
            user_id="test_user",
            source_ip="192.168.1.100",
            user_agent=None,
            resource=None,
            action="login_attempt",
            success=False,
            details={},
            risk_score=0.5
        )
        
        # Test pattern analysis
        await security_monitor._analyze_event_for_patterns(test_event)
        
        # Verify database calls were made
        assert conn.fetch.called
        assert conn.execute.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__])