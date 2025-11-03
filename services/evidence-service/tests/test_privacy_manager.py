"""
Tests for Privacy Manager Implementation.
Tests automatic data purging, anonymization, and retention policies.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from privacy_manager import PrivacyManager
from evidence_store import EvidenceStore
from audit_logger import AuditLogger
from shared.models.evidence import Evidence, EvidenceType, EvidenceStatus


@pytest.fixture
def mock_evidence_store():
    """Mock evidence store for testing."""
    store = MagicMock(spec=EvidenceStore)
    return store


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for testing."""
    logger = MagicMock(spec=AuditLogger)
    logger.log_system_event = AsyncMock()
    logger.log_user_action = AsyncMock()
    return logger


@pytest.fixture
def privacy_manager(mock_evidence_store, mock_audit_logger):
    """Create privacy manager instance for testing."""
    database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
    
    manager = PrivacyManager(
        database_url=database_url,
        evidence_store=mock_evidence_store,
        audit_logger=mock_audit_logger,
        default_retention_hours=24
    )
    
    # Mock the database session
    manager.async_session = AsyncMock()
    
    return manager


@pytest.fixture
def sample_evidence():
    """Sample evidence for testing."""
    return Evidence(
        id='evidence-123',
        type=EvidenceType.IMAGE,
        file_path='2024/01/01/evidence-123.jpg.enc',
        original_filename='test_image.jpg',
        file_size=2048,
        mime_type='image/jpeg',
        hash_sha256='hash123',
        hmac_signature='sig123',
        created_at=datetime.now() - timedelta(hours=2),
        created_by='system',
        camera_id='camera-001',
        incident_id='incident-123',
        status=EvidenceStatus.PENDING,
        metadata={'confidence': 0.8, 'location': 'sector-A'}
    )


class TestPrivacyManager:
    """Test cases for Privacy Manager functionality."""
    
    @pytest.mark.asyncio
    async def test_schedule_automatic_purge_unconfirmed(self, privacy_manager):
        """Test scheduling automatic purge for unconfirmed incidents."""
        # Mock database response for unconfirmed incident
        mock_incident = MagicMock()
        mock_incident.status = 'open'
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = mock_incident
        mock_session.execute.return_value = mock_result
        
        privacy_manager.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        privacy_manager.async_session.return_value.__aexit__ = AsyncMock()
        
        # Mock evidence search
        sample_evidence = Evidence(
            id='evidence-123',
            type=EvidenceType.IMAGE,
            incident_id='incident-123'
        )
        privacy_manager.evidence_store.search_evidence = AsyncMock(return_value=[sample_evidence])
        privacy_manager.evidence_store.schedule_purge = AsyncMock(return_value=True)
        
        # Schedule purge
        success = await privacy_manager.schedule_automatic_purge('incident-123')
        
        # Verify success
        assert success is True
        privacy_manager.evidence_store.schedule_purge.assert_called_once()
        privacy_manager.audit_logger.log_system_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_schedule_automatic_purge_confirmed(self, privacy_manager):
        """Test that confirmed incidents are not scheduled for purge."""
        # Mock database response for confirmed incident
        mock_incident = MagicMock()
        mock_incident.status = 'confirmed'
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = mock_incident
        mock_session.execute.return_value = mock_result
        
        privacy_manager.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        privacy_manager.async_session.return_value.__aexit__ = AsyncMock()
        
        # Schedule purge
        success = await privacy_manager.schedule_automatic_purge('incident-123')
        
        # Verify that purge was not scheduled
        assert success is False
        privacy_manager.evidence_store.schedule_purge.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cancel_automatic_purge(self, privacy_manager):
        """Test canceling automatic purge when incident is confirmed."""
        # Mock database operations
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.rowcount = 2  # 2 evidence items updated
        mock_session.execute.return_value = mock_result
        
        privacy_manager.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        privacy_manager.async_session.return_value.__aexit__ = AsyncMock()
        
        # Mock evidence search
        sample_evidence = [
            Evidence(id='evidence-1', incident_id='incident-123'),
            Evidence(id='evidence-2', incident_id='incident-123')
        ]
        privacy_manager.evidence_store.search_evidence = AsyncMock(return_value=sample_evidence)
        
        # Cancel purge
        success = await privacy_manager.cancel_automatic_purge('incident-123', 'operator-456')
        
        # Verify success
        assert success is True
        assert mock_session.execute.call_count >= 3  # Update + 2 chain of custody entries
        privacy_manager.audit_logger.log_user_action.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_anonymize_image_evidence(self, privacy_manager, sample_evidence):
        """Test image evidence anonymization."""
        # Create test image data
        test_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        import cv2
        _, encoded_image = cv2.imencode('.jpg', test_image_data)
        image_bytes = encoded_image.tobytes()
        
        # Mock evidence store methods
        privacy_manager.evidence_store.retrieve_evidence = AsyncMock(return_value=sample_evidence)
        privacy_manager.evidence_store.get_evidence_file = AsyncMock(return_value=MagicMock(read=lambda: image_bytes))
        privacy_manager.evidence_store.store_evidence = AsyncMock(return_value='anonymized-evidence-123')
        
        # Anonymize evidence
        anonymized_id = await privacy_manager.anonymize_evidence('evidence-123', 'standard')
        
        # Verify anonymization
        assert anonymized_id == 'anonymized-evidence-123'
        privacy_manager.evidence_store.store_evidence.assert_called_once()
        privacy_manager.audit_logger.log_system_event.assert_called_once()
        
        # Verify anonymized metadata
        store_call_args = privacy_manager.evidence_store.store_evidence.call_args
        anonymized_metadata = store_call_args[1]['metadata']
        assert 'anonymized_from' in anonymized_metadata
        assert anonymized_metadata['anonymization_level'] == 'standard'
    
    @pytest.mark.asyncio
    async def test_anonymize_metadata_evidence(self, privacy_manager):
        """Test metadata evidence anonymization."""
        # Create metadata evidence
        metadata_evidence = Evidence(
            id='metadata-123',
            type=EvidenceType.METADATA,
            metadata={'user_id': 'user-456', 'location': 'sector-A', 'confidence': 0.9}
        )
        
        # Create test metadata
        test_metadata = {
            'user_id': 'user-456',
            'operator_id': 'operator-789',
            'location': 'sector-A',
            'confidence': 0.9,
            'timestamp': '2024-01-01T12:00:00'
        }
        metadata_bytes = json.dumps(test_metadata).encode('utf-8')
        
        # Mock evidence store methods
        privacy_manager.evidence_store.retrieve_evidence = AsyncMock(return_value=metadata_evidence)
        privacy_manager.evidence_store.get_evidence_file = AsyncMock(return_value=MagicMock(read=lambda: metadata_bytes))
        privacy_manager.evidence_store.store_evidence = AsyncMock(return_value='anonymized-metadata-123')
        
        # Anonymize evidence
        anonymized_id = await privacy_manager.anonymize_evidence('metadata-123', 'standard')
        
        # Verify anonymization
        assert anonymized_id == 'anonymized-metadata-123'
        
        # Verify anonymized data was stored
        store_call_args = privacy_manager.evidence_store.store_evidence.call_args
        anonymized_data = store_call_args[0][0]  # file_data parameter
        anonymized_metadata = json.loads(anonymized_data.decode('utf-8'))
        
        # Check that sensitive fields were anonymized
        assert anonymized_metadata['user_id'].startswith('anon_')
        assert anonymized_metadata['operator_id'].startswith('anon_')
        assert anonymized_metadata['_anonymized'] is True
        assert anonymized_metadata['confidence'] == 0.9  # Non-sensitive field preserved
    
    @pytest.mark.asyncio
    async def test_apply_retention_policy_purge(self, privacy_manager):
        """Test applying purge retention policy."""
        # Mock evidence matching policy criteria
        old_evidence = Evidence(
            id='old-evidence-123',
            type=EvidenceType.IMAGE,
            created_at=datetime.now() - timedelta(days=2),
            incident_id='incident-123',
            metadata={'confidence': 0.3}  # Low confidence
        )
        
        privacy_manager.evidence_store.search_evidence = AsyncMock(return_value=[old_evidence])
        privacy_manager.evidence_store.schedule_purge = AsyncMock(return_value=True)
        
        # Mock policy criteria check
        privacy_manager._evidence_meets_policy_criteria = AsyncMock(return_value=True)
        
        # Apply retention policy
        affected_evidence = await privacy_manager.apply_retention_policy(
            'low_confidence_detections', 
            {'confidence_threshold': 0.5}
        )
        
        # Verify policy application
        assert len(affected_evidence) == 1
        assert affected_evidence[0] == 'old-evidence-123'
        privacy_manager.evidence_store.schedule_purge.assert_called_once()
        privacy_manager.audit_logger.log_system_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_apply_retention_policy_anonymize(self, privacy_manager):
        """Test applying anonymization retention policy."""
        # Mock evidence for anonymization
        analytics_evidence = Evidence(
            id='analytics-evidence-123',
            type=EvidenceType.METADATA,
            created_at=datetime.now() - timedelta(days=8),
            metadata={'purpose': 'analytics'}
        )
        
        privacy_manager.evidence_store.search_evidence = AsyncMock(return_value=[analytics_evidence])
        privacy_manager._evidence_meets_policy_criteria = AsyncMock(return_value=True)
        privacy_manager.anonymize_evidence = AsyncMock(return_value='anonymized-analytics-123')
        
        # Apply retention policy
        affected_evidence = await privacy_manager.apply_retention_policy(
            'analytics_data',
            {'evidence_type': ['metadata']}
        )
        
        # Verify policy application
        assert len(affected_evidence) == 1
        assert affected_evidence[0] == 'anonymized-analytics-123'
        privacy_manager.anonymize_evidence.assert_called_once_with(
            'analytics-evidence-123', 'aggressive'
        )
    
    @pytest.mark.asyncio
    async def test_evidence_meets_policy_criteria_age(self, privacy_manager):
        """Test policy criteria checking for age-based rules."""
        # Test evidence that meets age criteria
        old_evidence = Evidence(
            id='old-evidence',
            created_at=datetime.now() - timedelta(hours=25)
        )
        
        policy = {
            'criteria': {
                'max_age_hours': 24
            }
        }
        
        # Should meet criteria (older than 24 hours)
        meets_criteria = await privacy_manager._evidence_meets_policy_criteria(old_evidence, policy)
        assert meets_criteria is True
        
        # Test evidence that doesn't meet age criteria
        new_evidence = Evidence(
            id='new-evidence',
            created_at=datetime.now() - timedelta(hours=12)
        )
        
        meets_criteria = await privacy_manager._evidence_meets_policy_criteria(new_evidence, policy)
        assert meets_criteria is False
    
    @pytest.mark.asyncio
    async def test_evidence_meets_policy_criteria_confidence(self, privacy_manager):
        """Test policy criteria checking for confidence-based rules."""
        # Test low confidence evidence
        low_confidence_evidence = Evidence(
            id='low-confidence',
            metadata={'confidence': 0.3}
        )
        
        policy = {
            'criteria': {
                'confidence_threshold': 0.5
            }
        }
        
        # Should meet criteria (confidence below threshold)
        meets_criteria = await privacy_manager._evidence_meets_policy_criteria(low_confidence_evidence, policy)
        assert meets_criteria is True
        
        # Test high confidence evidence
        high_confidence_evidence = Evidence(
            id='high-confidence',
            metadata={'confidence': 0.8}
        )
        
        meets_criteria = await privacy_manager._evidence_meets_policy_criteria(high_confidence_evidence, policy)
        assert meets_criteria is False
    
    @pytest.mark.asyncio
    async def test_generate_privacy_compliance_report(self, privacy_manager):
        """Test privacy compliance report generation."""
        # Mock database responses
        mock_session = AsyncMock()
        
        # Mock evidence statistics
        evidence_stats = [
            MagicMock(type='image', status='pending', count=10, total_size=10240),
            MagicMock(type='video', status='sealed', count=5, total_size=51200)
        ]
        
        # Mock purge statistics
        purge_stats = [
            MagicMock(purge_date=datetime.now().date(), purged_count=3),
            MagicMock(purge_date=(datetime.now() - timedelta(days=1)).date(), purged_count=2)
        ]
        
        # Set up mock session responses
        mock_session.execute.side_effect = [
            MagicMock(fetchall=lambda: evidence_stats),
            MagicMock(fetchall=lambda: purge_stats),
            MagicMock(scalar=lambda: 8)  # anonymization count
        ]
        
        privacy_manager.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        privacy_manager.async_session.return_value.__aexit__ = AsyncMock()
        
        # Generate report
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        report = await privacy_manager.generate_privacy_compliance_report(start_date, end_date)
        
        # Verify report structure
        assert 'report_period' in report
        assert 'evidence_statistics' in report
        assert 'purge_statistics' in report
        assert 'anonymization_count' in report
        assert 'compliance_metrics' in report
        
        # Verify report content
        assert report['anonymization_count'] == 8
        assert len(report['evidence_statistics']) == 2
        assert len(report['purge_statistics']) == 2
        assert report['compliance_metrics']['total_evidence_processed'] == 15
        assert report['compliance_metrics']['total_purged'] == 5
    
    @pytest.mark.asyncio
    async def test_run_privacy_maintenance(self, privacy_manager):
        """Test privacy maintenance task execution."""
        # Mock maintenance tasks
        privacy_manager.evidence_store.purge_expired_evidence = AsyncMock(return_value=['evidence-1', 'evidence-2'])
        privacy_manager.apply_retention_policy = AsyncMock(return_value=['evidence-3'])
        
        # Mock database for unconfirmed incidents
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = [
            MagicMock(id='incident-1'),
            MagicMock(id='incident-2')
        ]
        mock_session.execute.return_value = mock_result
        
        privacy_manager.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        privacy_manager.async_session.return_value.__aexit__ = AsyncMock()
        
        privacy_manager.schedule_automatic_purge = AsyncMock(return_value=True)
        
        # Run maintenance
        summary = await privacy_manager.run_privacy_maintenance()
        
        # Verify maintenance summary
        assert summary['success'] is True
        assert len(summary['tasks_performed']) >= 3
        assert len(summary['errors']) == 0
        
        # Verify tasks were executed
        privacy_manager.evidence_store.purge_expired_evidence.assert_called_once()
        assert privacy_manager.apply_retention_policy.call_count >= 1
        privacy_manager.audit_logger.log_system_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_anonymize_metadata_aggressive(self, privacy_manager):
        """Test aggressive metadata anonymization."""
        test_metadata = {
            'user_id': 'user-123',
            'operator_id': 'operator-456',
            'camera_id': 'camera-789',
            'location': 'sector-A',
            'confidence': 0.9
        }
        
        # Test aggressive anonymization
        anonymized_data = await privacy_manager._anonymize_metadata(
            json.dumps(test_metadata).encode('utf-8'), 
            'aggressive'
        )
        
        anonymized_metadata = json.loads(anonymized_data.decode('utf-8'))
        
        # Verify aggressive anonymization removes fields completely
        assert 'user_id' not in anonymized_metadata
        assert 'operator_id' not in anonymized_metadata
        assert 'camera_id' not in anonymized_metadata
        assert 'confidence' in anonymized_metadata  # Non-sensitive field preserved
        assert anonymized_metadata['_anonymized'] is True
        assert anonymized_metadata['_anonymization_level'] == 'aggressive'
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_evidence(self, privacy_manager):
        """Test error handling for missing evidence during anonymization."""
        # Mock missing evidence
        privacy_manager.evidence_store.retrieve_evidence = AsyncMock(return_value=None)
        
        # Attempt to anonymize missing evidence
        with pytest.raises(ValueError, match="Evidence .* not found"):
            await privacy_manager.anonymize_evidence('nonexistent-evidence')
    
    @pytest.mark.asyncio
    async def test_error_handling_inaccessible_file(self, privacy_manager, sample_evidence):
        """Test error handling for inaccessible evidence files."""
        # Mock inaccessible file
        privacy_manager.evidence_store.retrieve_evidence = AsyncMock(return_value=sample_evidence)
        privacy_manager.evidence_store.get_evidence_file = AsyncMock(return_value=None)
        
        # Attempt to anonymize inaccessible evidence
        with pytest.raises(ValueError, match="Evidence file .* not accessible"):
            await privacy_manager.anonymize_evidence('evidence-123')


if __name__ == "__main__":
    pytest.main([__file__])