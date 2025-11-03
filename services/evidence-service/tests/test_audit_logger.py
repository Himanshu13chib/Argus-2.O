"""
Tests for Audit Logger Implementation.
Tests comprehensive audit logging functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from audit_logger import AuditLogger


@pytest.fixture
def audit_logger():
    """Create audit logger instance for testing."""
    database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
    logger = AuditLogger(database_url=database_url)
    
    # Mock the database session
    logger.async_session = AsyncMock()
    
    return logger


class TestAuditLogger:
    """Test cases for Audit Logger functionality."""
    
    @pytest.mark.asyncio
    async def test_log_user_action(self, audit_logger):
        """Test user action logging."""
        # Mock database operations
        mock_session = AsyncMock()
        audit_logger.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        audit_logger.async_session.return_value.__aexit__ = AsyncMock()
        
        # Log user action
        await audit_logger.log_user_action(
            user_id="user-123",
            action="evidence_viewed",
            resource_type="evidence",
            resource_id="evidence-456",
            details={"access_type": "metadata"}
        )
        
        # Verify database operations
        assert mock_session.execute.called
        assert mock_session.commit.called
        
        # Verify the SQL query parameters
        call_args = mock_session.execute.call_args
        assert "user_id" in str(call_args)
        assert "evidence_viewed" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_log_system_event(self, audit_logger):
        """Test system event logging."""
        # Mock database operations
        mock_session = AsyncMock()
        audit_logger.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        audit_logger.async_session.return_value.__aexit__ = AsyncMock()
        
        # Log system event
        await audit_logger.log_system_event(
            event_type="service_started",
            component="evidence_service",
            details={"version": "1.0.0", "startup_time": "2024-01-01T00:00:00"}
        )
        
        # Verify database operations
        assert mock_session.execute.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_logger):
        """Test security event logging."""
        # Mock database operations
        mock_session = AsyncMock()
        audit_logger.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        audit_logger.async_session.return_value.__aexit__ = AsyncMock()
        
        # Log security event
        await audit_logger.log_security_event(
            event_type="failed_login",
            user_id="user-123",
            source_ip="192.168.1.100",
            details={"reason": "invalid_password", "attempts": 3}
        )
        
        # Verify database operations
        assert mock_session.execute.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_log_evidence_access(self, audit_logger):
        """Test evidence access logging."""
        # Mock database operations
        mock_session = AsyncMock()
        audit_logger.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        audit_logger.async_session.return_value.__aexit__ = AsyncMock()
        
        # Log evidence access
        await audit_logger.log_evidence_access(
            evidence_id="evidence-123",
            user_id="user-456",
            action="downloaded",
            details={"filename": "evidence.jpg", "file_size": 1024}
        )
        
        # Verify database operations - should be called twice (audit log + chain of custody)
        assert mock_session.execute.call_count >= 1
        assert mock_session.commit.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_get_audit_logs(self, audit_logger):
        """Test audit log retrieval."""
        # Mock database response
        mock_rows = [
            MagicMock(
                id="log-1",
                timestamp=datetime.now(),
                user_id="user-123",
                action="evidence_viewed",
                resource_type="evidence",
                resource_id="evidence-456",
                source_ip="192.168.1.100",
                user_agent="Mozilla/5.0",
                success=True,
                details='{"access_type": "metadata"}',
                session_id="session-789"
            ),
            MagicMock(
                id="log-2",
                timestamp=datetime.now() - timedelta(hours=1),
                user_id="user-123",
                action="evidence_downloaded",
                resource_type="evidence",
                resource_id="evidence-456",
                source_ip="192.168.1.100",
                user_agent="Mozilla/5.0",
                success=True,
                details='{"filename": "test.jpg"}',
                session_id="session-789"
            )
        ]
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        audit_logger.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        audit_logger.async_session.return_value.__aexit__ = AsyncMock()
        
        # Get audit logs
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        filters = {"user_id": "user-123"}
        
        logs = await audit_logger.get_audit_logs(start_time, end_time, filters)
        
        # Verify results
        assert len(logs) == 2
        assert logs[0]["id"] == "log-1"
        assert logs[0]["action"] == "evidence_viewed"
        assert logs[0]["user_id"] == "user-123"
        assert isinstance(logs[0]["details"], dict)
    
    @pytest.mark.asyncio
    async def test_search_audit_logs(self, audit_logger):
        """Test audit log search functionality."""
        # Mock database response
        mock_rows = [
            MagicMock(
                id="log-search-1",
                timestamp=datetime.now(),
                user_id="user-123",
                action="evidence_sealed",
                resource_type="evidence",
                resource_id="evidence-456",
                source_ip="192.168.1.100",
                user_agent="Mozilla/5.0",
                success=True,
                details='{"reason": "legal_proceedings"}',
                session_id="session-789"
            )
        ]
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        audit_logger.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        audit_logger.async_session.return_value.__aexit__ = AsyncMock()
        
        # Search audit logs
        logs = await audit_logger.search_audit_logs("sealed", time_range_hours=24)
        
        # Verify results
        assert len(logs) == 1
        assert logs[0]["action"] == "evidence_sealed"
        assert "legal_proceedings" in str(logs[0]["details"])
    
    @pytest.mark.asyncio
    async def test_export_audit_logs_csv(self, audit_logger):
        """Test audit log export in CSV format."""
        # Mock get_audit_logs method
        mock_logs = [
            {
                "id": "log-1",
                "timestamp": "2024-01-01T12:00:00",
                "user_id": "user-123",
                "action": "evidence_viewed",
                "resource_type": "evidence",
                "resource_id": "evidence-456",
                "source_ip": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "success": True,
                "details": {"access_type": "metadata"},
                "session_id": "session-789"
            }
        ]
        
        audit_logger.get_audit_logs = AsyncMock(return_value=mock_logs)
        
        # Export logs as CSV
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        csv_output = await audit_logger.export_audit_logs(start_time, end_time, format="csv")
        
        # Verify CSV output
        assert isinstance(csv_output, str)
        assert "id,timestamp,user_id,action" in csv_output
        assert "log-1" in csv_output
        assert "evidence_viewed" in csv_output
    
    @pytest.mark.asyncio
    async def test_export_audit_logs_json(self, audit_logger):
        """Test audit log export in JSON format."""
        # Mock get_audit_logs method
        mock_logs = [
            {
                "id": "log-1",
                "timestamp": "2024-01-01T12:00:00",
                "user_id": "user-123",
                "action": "evidence_viewed"
            }
        ]
        
        audit_logger.get_audit_logs = AsyncMock(return_value=mock_logs)
        
        # Export logs as JSON
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        json_output = await audit_logger.export_audit_logs(start_time, end_time, format="json")
        
        # Verify JSON output
        assert isinstance(json_output, str)
        parsed_json = json.loads(json_output)
        assert len(parsed_json) == 1
        assert parsed_json[0]["id"] == "log-1"
    
    @pytest.mark.asyncio
    async def test_get_user_activity_summary(self, audit_logger):
        """Test user activity summary generation."""
        # Mock database responses for different queries
        mock_session = AsyncMock()
        
        # Mock total actions query
        mock_total_result = AsyncMock()
        mock_total_result.scalar.return_value = 25
        
        # Mock actions by type query
        mock_actions_result = AsyncMock()
        mock_actions_result.fetchall.return_value = [
            MagicMock(action="evidence_viewed", count=10),
            MagicMock(action="evidence_downloaded", count=8),
            MagicMock(action="evidence_sealed", count=7)
        ]
        
        # Mock daily activity query
        mock_daily_result = AsyncMock()
        mock_daily_result.fetchall.return_value = [
            MagicMock(date=datetime.now().date(), count=15),
            MagicMock(date=(datetime.now() - timedelta(days=1)).date(), count=10)
        ]
        
        # Mock resources query
        mock_resources_result = AsyncMock()
        mock_resources_result.fetchall.return_value = [
            MagicMock(resource_type="evidence", count=20),
            MagicMock(resource_type="incident", count=5)
        ]
        
        # Set up mock session to return different results for different queries
        mock_session.execute.side_effect = [
            mock_total_result,
            mock_actions_result,
            mock_daily_result,
            mock_resources_result
        ]
        
        audit_logger.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        audit_logger.async_session.return_value.__aexit__ = AsyncMock()
        
        # Get user activity summary
        summary = await audit_logger.get_user_activity_summary("user-123", days=30)
        
        # Verify summary structure
        assert summary["user_id"] == "user-123"
        assert summary["period_days"] == 30
        assert summary["total_actions"] == 25
        assert len(summary["actions_by_type"]) == 3
        assert summary["actions_by_type"][0]["action"] == "evidence_viewed"
        assert summary["actions_by_type"][0]["count"] == 10
        assert len(summary["daily_activity"]) == 2
        assert len(summary["resources_accessed"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])