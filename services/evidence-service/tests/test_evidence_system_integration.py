"""
Integration Tests for Evidence Management System.
Tests complete workflows from evidence storage to forensics reporting.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json
from io import BytesIO
import base64

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evidence_store import EvidenceStore
from forensics_engine import ForensicsEngine
from privacy_manager import PrivacyManager
from audit_logger import AuditLogger
from shared.models.evidence import Evidence, EvidenceType, EvidenceStatus


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def encryption_key():
    """Generate test encryption key."""
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    return base64.urlsafe_b64encode(key).decode()


@pytest.fixture
def hmac_secret():
    """Generate test HMAC secret."""
    import secrets
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


@pytest.fixture
def evidence_system(temp_storage, encryption_key, hmac_secret):
    """Create complete evidence system for integration testing."""
    database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
    
    # Create evidence store
    evidence_store = EvidenceStore(
        database_url=database_url,
        storage_path=temp_storage,
        encryption_key=encryption_key,
        hmac_secret=hmac_secret
    )
    
    # Create audit logger
    audit_logger = AuditLogger(database_url=database_url)
    
    # Create forensics engine
    forensics_engine = ForensicsEngine(
        database_url=database_url,
        evidence_store=evidence_store,
        output_path=temp_storage
    )
    
    # Create privacy manager
    privacy_manager = PrivacyManager(
        database_url=database_url,
        evidence_store=evidence_store,
        audit_logger=audit_logger
    )
    
    # Mock database sessions for all components
    mock_session = AsyncMock()
    evidence_store.async_session = AsyncMock()
    audit_logger.async_session = AsyncMock()
    forensics_engine.async_session = AsyncMock()
    privacy_manager.async_session = AsyncMock()
    
    return {
        'evidence_store': evidence_store,
        'audit_logger': audit_logger,
        'forensics_engine': forensics_engine,
        'privacy_manager': privacy_manager
    }


class TestEvidenceSystemIntegration:
    """Integration test cases for complete evidence management workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_evidence_lifecycle(self, evidence_system, temp_storage):
        """Test complete evidence lifecycle from creation to purging."""
        evidence_store = evidence_system['evidence_store']
        audit_logger = evidence_system['audit_logger']
        privacy_manager = evidence_system['privacy_manager']
        
        # Mock database operations for evidence storage
        mock_session = AsyncMock()
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # 1. Store evidence
        test_data = b"test evidence data for lifecycle testing"
        metadata = {
            "original_filename": "test_evidence.jpg",
            "mime_type": "image/jpeg",
            "incident_id": "incident-lifecycle-123",
            "camera_id": "camera-001"
        }
        
        evidence_id = await evidence_store.store_evidence(
            file_data=test_data,
            evidence_type=EvidenceType.IMAGE,
            metadata=metadata,
            created_by="test-user"
        )
        
        # Verify evidence was stored
        assert evidence_id is not None
        assert mock_session.execute.call_count >= 2  # Evidence + chain of custody
        
        # 2. Verify integrity
        # Mock retrieve and file access for integrity check
        test_evidence = Evidence(
            id=evidence_id,
            type=EvidenceType.IMAGE,
            file_path="test/path.jpg.enc",
            hash_sha256="",
            hmac_signature=""
        )
        test_evidence.calculate_hash(test_data)
        test_evidence.hmac_signature = evidence_store._generate_hmac_signature(test_data)
        
        evidence_store.retrieve_evidence = AsyncMock(return_value=test_evidence)
        evidence_store.get_evidence_file = AsyncMock(return_value=BytesIO(test_data))
        
        integrity_valid = await evidence_store.verify_integrity(evidence_id)
        assert integrity_valid is True
        
        # 3. Schedule for privacy purging
        privacy_manager.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        privacy_manager.async_session.return_value.__aexit__ = AsyncMock()
        
        # Mock incident as unconfirmed
        mock_incident = MagicMock()
        mock_incident.status = 'open'
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = mock_incident
        mock_session.execute.return_value = mock_result
        
        evidence_store.search_evidence = AsyncMock(return_value=[test_evidence])
        evidence_store.schedule_purge = AsyncMock(return_value=True)
        
        purge_scheduled = await privacy_manager.schedule_automatic_purge("incident-lifecycle-123")
        assert purge_scheduled is True
        
        # 4. Verify audit logging
        audit_logger.log_system_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_forensics_report_generation_workflow(self, evidence_system, temp_storage):
        """Test complete forensics report generation workflow."""
        evidence_store = evidence_system['evidence_store']
        forensics_engine = evidence_system['forensics_engine']
        
        # Mock incident data
        incident_data = {
            'id': 'incident-forensics-123',
            'status': 'closed',
            'priority': 'high',
            'created_at': datetime.now() - timedelta(hours=2),
            'operator_name': 'Test Operator',
            'confidence': 0.95,
            'risk_score': 8.5,
            'camera_id': 'camera-001'
        }
        
        # Mock evidence list
        evidence_list = [
            Evidence(
                id='evidence-forensics-1',
                type=EvidenceType.IMAGE,
                original_filename='crossing_image.jpg',
                file_size=2048,
                created_at=datetime.now() - timedelta(hours=2),
                created_by='system',
                status=EvidenceStatus.SEALED,
                metadata={'confidence': 0.95}
            ),
            Evidence(
                id='evidence-forensics-2',
                type=EvidenceType.VIDEO,
                original_filename='crossing_video.mp4',
                file_size=10240,
                created_at=datetime.now() - timedelta(hours=1),
                created_by='system',
                status=EvidenceStatus.SEALED,
                metadata={'duration': 30}
            )
        ]
        
        # Mock forensics engine dependencies
        forensics_engine._get_incident_data = AsyncMock(return_value=incident_data)
        evidence_store.search_evidence = AsyncMock(return_value=evidence_list)
        evidence_store.get_chain_of_custody = AsyncMock(return_value=None)
        
        # Generate comprehensive report
        report_path = await forensics_engine.generate_report('incident-forensics-123', 'comprehensive')
        
        # Verify report generation
        assert report_path is not None
        assert os.path.exists(report_path)
        assert report_path.endswith('.pdf')
        assert 'incident-forensics-123' in report_path
        
        # Verify report content (file should be non-empty)
        assert os.path.getsize(report_path) > 0
        
        # Generate legal package
        forensics_engine.create_video_summary = AsyncMock(return_value=os.path.join(temp_storage, "test_video.mp4"))
        forensics_engine._generate_chain_of_custody_report = AsyncMock(return_value='{"test": "custody"}')
        forensics_engine._export_incident_csv = AsyncMock(return_value="test,csv,data")
        
        # Create dummy files for legal package
        with open(os.path.join(temp_storage, "test_video.mp4"), 'w') as f:
            f.write("test video")
        
        evidence_store.get_evidence_file = AsyncMock(return_value=BytesIO(b"test evidence data"))
        
        legal_package_path = await forensics_engine.export_legal_package('incident-forensics-123')
        
        # Verify legal package
        assert legal_package_path is not None
        assert os.path.exists(legal_package_path)
        assert legal_package_path.endswith('.zip')
        assert os.path.getsize(legal_package_path) > 0
    
    @pytest.mark.asyncio
    async def test_privacy_anonymization_workflow(self, evidence_system):
        """Test privacy anonymization workflow."""
        evidence_store = evidence_system['evidence_store']
        privacy_manager = evidence_system['privacy_manager']
        audit_logger = evidence_system['audit_logger']
        
        # Create test evidence with sensitive metadata
        sensitive_evidence = Evidence(
            id='sensitive-evidence-123',
            type=EvidenceType.METADATA,
            metadata={
                'user_id': 'user-456',
                'operator_id': 'operator-789',
                'location': 'classified-sector',
                'confidence': 0.8
            }
        )
        
        # Create test metadata content
        sensitive_metadata = {
            'user_id': 'user-456',
            'operator_id': 'operator-789',
            'ip_address': '192.168.1.100',
            'device_id': 'device-123',
            'location': 'classified-sector',
            'confidence': 0.8,
            'timestamp': '2024-01-01T12:00:00'
        }
        metadata_bytes = json.dumps(sensitive_metadata).encode('utf-8')
        
        # Mock evidence store methods
        evidence_store.retrieve_evidence = AsyncMock(return_value=sensitive_evidence)
        evidence_store.get_evidence_file = AsyncMock(return_value=BytesIO(metadata_bytes))
        evidence_store.store_evidence = AsyncMock(return_value='anonymized-evidence-123')
        
        # Perform anonymization
        anonymized_id = await privacy_manager.anonymize_evidence('sensitive-evidence-123', 'standard')
        
        # Verify anonymization
        assert anonymized_id == 'anonymized-evidence-123'
        evidence_store.store_evidence.assert_called_once()
        audit_logger.log_system_event.assert_called_once()
        
        # Verify anonymized data
        store_call_args = evidence_store.store_evidence.call_args
        anonymized_data = store_call_args[0][0]  # file_data parameter
        anonymized_metadata = json.loads(anonymized_data.decode('utf-8'))
        
        # Check anonymization results
        assert anonymized_metadata['user_id'].startswith('anon_')
        assert anonymized_metadata['operator_id'].startswith('anon_')
        assert anonymized_metadata['ip_address'].startswith('anon_')
        assert anonymized_metadata['confidence'] == 0.8  # Non-sensitive preserved
        assert anonymized_metadata['_anonymized'] is True
    
    @pytest.mark.asyncio
    async def test_retention_policy_enforcement(self, evidence_system):
        """Test retention policy enforcement workflow."""
        evidence_store = evidence_system['evidence_store']
        privacy_manager = evidence_system['privacy_manager']
        audit_logger = evidence_system['audit_logger']
        
        # Create old, low-confidence evidence
        old_evidence = Evidence(
            id='old-low-confidence-123',
            type=EvidenceType.IMAGE,
            created_at=datetime.now() - timedelta(days=8),
            metadata={'confidence': 0.3}
        )
        
        # Mock evidence search
        evidence_store.search_evidence = AsyncMock(return_value=[old_evidence])
        evidence_store.schedule_purge = AsyncMock(return_value=True)
        
        # Mock policy criteria check
        privacy_manager._evidence_meets_policy_criteria = AsyncMock(return_value=True)
        
        # Apply retention policy
        affected_evidence = await privacy_manager.apply_retention_policy(
            'low_confidence_detections',
            {'confidence_threshold': 0.5}
        )
        
        # Verify policy enforcement
        assert len(affected_evidence) == 1
        assert affected_evidence[0] == 'old-low-confidence-123'
        evidence_store.schedule_purge.assert_called_once()
        audit_logger.log_system_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evidence_sealing_and_chain_of_custody(self, evidence_system):
        """Test evidence sealing and chain of custody workflow."""
        evidence_store = evidence_system['evidence_store']
        audit_logger = evidence_system['audit_logger']
        
        # Mock database operations for sealing
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # Seal evidence
        success = await evidence_store.seal_evidence('evidence-seal-123', 'operator-456')
        
        # Verify sealing
        assert success is True
        assert mock_session.execute.call_count >= 2  # Update + chain of custody
        assert mock_session.commit.called
        
        # Test custody transfer
        transfer_success = await evidence_store.transfer_custody(
            'evidence-seal-123',
            'operator-456',
            'operator-789',
            'Investigation handover'
        )
        
        # Verify transfer
        assert transfer_success is True
    
    @pytest.mark.asyncio
    async def test_privacy_maintenance_workflow(self, evidence_system):
        """Test complete privacy maintenance workflow."""
        evidence_store = evidence_system['evidence_store']
        privacy_manager = evidence_system['privacy_manager']
        audit_logger = evidence_system['audit_logger']
        
        # Mock maintenance tasks
        evidence_store.purge_expired_evidence = AsyncMock(return_value=['expired-1', 'expired-2'])
        privacy_manager.apply_retention_policy = AsyncMock(return_value=['policy-affected-1'])
        
        # Mock database for unconfirmed incidents
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = [
            MagicMock(id='unconfirmed-incident-1'),
            MagicMock(id='unconfirmed-incident-2')
        ]
        mock_session.execute.return_value = mock_result
        
        privacy_manager.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        privacy_manager.async_session.return_value.__aexit__ = AsyncMock()
        
        privacy_manager.schedule_automatic_purge = AsyncMock(return_value=True)
        
        # Run maintenance
        summary = await privacy_manager.run_privacy_maintenance()
        
        # Verify maintenance execution
        assert summary['success'] is True
        assert len(summary['tasks_performed']) >= 3
        assert len(summary['errors']) == 0
        
        # Verify all maintenance tasks were executed
        evidence_store.purge_expired_evidence.assert_called_once()
        assert privacy_manager.apply_retention_policy.call_count >= 1
        assert privacy_manager.schedule_automatic_purge.call_count == 2
        audit_logger.log_system_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, evidence_system):
        """Test error handling and recovery in evidence workflows."""
        evidence_store = evidence_system['evidence_store']
        forensics_engine = evidence_system['forensics_engine']
        privacy_manager = evidence_system['privacy_manager']
        
        # Test forensics engine error handling
        forensics_engine._get_incident_data = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError, match="Incident .* not found"):
            await forensics_engine.generate_report('nonexistent-incident')
        
        # Test privacy manager error handling
        evidence_store.retrieve_evidence = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError, match="Evidence .* not found"):
            await privacy_manager.anonymize_evidence('nonexistent-evidence')
        
        # Test evidence store integrity verification with tampered data
        test_evidence = Evidence(
            id='tampered-evidence',
            type=EvidenceType.IMAGE,
            hash_sha256='original-hash',
            hmac_signature='original-signature'
        )
        
        evidence_store.retrieve_evidence = AsyncMock(return_value=test_evidence)
        evidence_store.get_evidence_file = AsyncMock(return_value=BytesIO(b"tampered data"))
        
        # Should detect tampering
        integrity_valid = await evidence_store.verify_integrity('tampered-evidence')
        assert integrity_valid is False
    
    @pytest.mark.asyncio
    async def test_concurrent_evidence_operations(self, evidence_system):
        """Test concurrent evidence operations."""
        evidence_store = evidence_system['evidence_store']
        privacy_manager = evidence_system['privacy_manager']
        
        # Mock database operations
        mock_session = AsyncMock()
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # Create multiple concurrent evidence storage operations
        test_data_list = [
            b"evidence data 1",
            b"evidence data 2",
            b"evidence data 3"
        ]
        
        metadata_list = [
            {"original_filename": f"evidence_{i}.jpg", "incident_id": f"incident-{i}"}
            for i in range(3)
        ]
        
        # Execute concurrent operations
        tasks = [
            evidence_store.store_evidence(
                file_data=data,
                evidence_type=EvidenceType.IMAGE,
                metadata=metadata,
                created_by="concurrent-test"
            )
            for data, metadata in zip(test_data_list, metadata_list)
        ]
        
        evidence_ids = await asyncio.gather(*tasks)
        
        # Verify all operations completed
        assert len(evidence_ids) == 3
        assert all(evidence_id is not None for evidence_id in evidence_ids)
        assert len(set(evidence_ids)) == 3  # All IDs should be unique


if __name__ == "__main__":
    pytest.main([__file__])