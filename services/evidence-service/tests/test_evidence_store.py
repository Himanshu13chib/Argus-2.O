"""
Tests for Evidence Store Implementation.
Tests HMAC signing, encryption, and integrity verification.
"""

import pytest
import asyncio
import tempfile
import os
import base64
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evidence_store import EvidenceStore
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
def mock_database():
    """Mock database session."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.fetchone = AsyncMock()
    mock_session.fetchall = AsyncMock()
    return mock_session


@pytest.fixture
def evidence_store(temp_storage, encryption_key, hmac_secret):
    """Create evidence store instance for testing."""
    # Mock database URL for testing
    database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
    
    store = EvidenceStore(
        database_url=database_url,
        storage_path=temp_storage,
        encryption_key=encryption_key,
        hmac_secret=hmac_secret
    )
    
    # Mock the database session
    store.async_session = AsyncMock()
    
    return store


class TestEvidenceStore:
    """Test cases for Evidence Store functionality."""
    
    def test_hmac_signature_generation(self, evidence_store):
        """Test HMAC signature generation and verification."""
        test_data = b"test evidence data"
        
        # Generate signature
        signature = evidence_store._generate_hmac_signature(test_data)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA-256 hex digest length
        
        # Verify signature
        is_valid = evidence_store._verify_hmac_signature(test_data, signature)
        assert is_valid is True
        
        # Test with tampered data
        tampered_data = b"tampered evidence data"
        is_valid_tampered = evidence_store._verify_hmac_signature(tampered_data, signature)
        assert is_valid_tampered is False
    
    def test_data_encryption_decryption(self, evidence_store):
        """Test AES-256 encryption and decryption."""
        test_data = b"sensitive evidence data that needs encryption"
        
        # Encrypt data
        encrypted_data = evidence_store._encrypt_data(test_data)
        assert encrypted_data != test_data
        assert len(encrypted_data) > len(test_data)  # Encrypted data is larger
        
        # Decrypt data
        decrypted_data = evidence_store._decrypt_data(encrypted_data)
        assert decrypted_data == test_data
    
    def test_file_path_generation(self, evidence_store):
        """Test secure file path generation."""
        evidence_id = "test-evidence-123"
        evidence_type = EvidenceType.IMAGE
        
        file_path = evidence_store._generate_file_path(evidence_id, evidence_type)
        
        # Should include date-based directory structure
        assert "/" in file_path
        assert evidence_id in file_path
        assert file_path.endswith(".jpg.enc")
        
        # Test different evidence types
        video_path = evidence_store._generate_file_path(evidence_id, EvidenceType.VIDEO)
        assert video_path.endswith(".mp4.enc")
        
        metadata_path = evidence_store._generate_file_path(evidence_id, EvidenceType.METADATA)
        assert metadata_path.endswith(".json.enc")
    
    @pytest.mark.asyncio
    async def test_store_evidence_local(self, evidence_store, temp_storage):
        """Test evidence storage with local file system."""
        # Mock database operations
        evidence_store.async_session.return_value.__aenter__ = AsyncMock()
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        mock_session = AsyncMock()
        evidence_store.async_session.return_value = mock_session
        
        test_data = b"test image data for evidence storage"
        metadata = {
            "original_filename": "test_image.jpg",
            "mime_type": "image/jpeg",
            "incident_id": "incident-123",
            "camera_id": "camera-456"
        }
        
        # Store evidence
        evidence_id = await evidence_store.store_evidence(
            file_data=test_data,
            evidence_type=EvidenceType.IMAGE,
            metadata=metadata,
            created_by="test-user"
        )
        
        # Verify evidence ID is generated
        assert evidence_id is not None
        assert isinstance(evidence_id, str)
        
        # Verify database operations were called
        assert mock_session.execute.call_count >= 2  # Evidence + chain of custody
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_retrieve_evidence(self, evidence_store):
        """Test evidence retrieval."""
        # Mock database response
        mock_row = MagicMock()
        mock_row.id = "test-evidence-123"
        mock_row.incident_id = "incident-123"
        mock_row.type = "image"
        mock_row.file_path = "2024/01/01/test-evidence-123.jpg.enc"
        mock_row.original_filename = "test.jpg"
        mock_row.file_size = 1024
        mock_row.mime_type = "image/jpeg"
        mock_row.hash_sha256 = "abcd1234"
        mock_row.hmac_signature = "signature123"
        mock_row.encryption_key_id = None
        mock_row.created_at = datetime.now()
        mock_row.created_by = "test-user"
        mock_row.camera_id = "camera-456"
        mock_row.detection_id = None
        mock_row.status = "pending"
        mock_row.retention_until = None
        mock_row.auto_purge = True
        mock_row.metadata = "{}"
        mock_row.tags = []
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # Retrieve evidence
        evidence = await evidence_store.retrieve_evidence("test-evidence-123")
        
        # Verify evidence object
        assert evidence is not None
        assert evidence.id == "test-evidence-123"
        assert evidence.type == EvidenceType.IMAGE
        assert evidence.original_filename == "test.jpg"
        assert evidence.created_by == "test-user"
    
    @pytest.mark.asyncio
    async def test_verify_integrity(self, evidence_store, temp_storage):
        """Test evidence integrity verification."""
        # Create test file with known content
        test_data = b"test data for integrity verification"
        evidence_id = "test-evidence-integrity"
        
        # Create evidence object
        evidence = Evidence(
            id=evidence_id,
            type=EvidenceType.METADATA,
            file_path="test/path.json.enc",
            hash_sha256="",
            hmac_signature=""
        )
        evidence.calculate_hash(test_data)
        evidence.hmac_signature = evidence_store._generate_hmac_signature(test_data)
        
        # Mock retrieve_evidence to return our test evidence
        evidence_store.retrieve_evidence = AsyncMock(return_value=evidence)
        
        # Mock get_evidence_file to return test data
        from io import BytesIO
        evidence_store.get_evidence_file = AsyncMock(return_value=BytesIO(test_data))
        
        # Verify integrity
        is_valid = await evidence_store.verify_integrity(evidence_id)
        assert is_valid is True
        
        # Test with tampered data
        tampered_data = b"tampered test data"
        evidence_store.get_evidence_file = AsyncMock(return_value=BytesIO(tampered_data))
        
        is_valid_tampered = await evidence_store.verify_integrity(evidence_id)
        assert is_valid_tampered is False
    
    @pytest.mark.asyncio
    async def test_seal_evidence(self, evidence_store):
        """Test evidence sealing functionality."""
        evidence_id = "test-evidence-seal"
        operator_id = "operator-123"
        
        # Mock database operations
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # Seal evidence
        success = await evidence_store.seal_evidence(evidence_id, operator_id)
        
        # Verify success
        assert success is True
        assert mock_session.execute.call_count >= 2  # Update + chain of custody
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_transfer_custody(self, evidence_store):
        """Test custody transfer functionality."""
        evidence_id = "test-evidence-transfer"
        from_operator = "operator-1"
        to_operator = "operator-2"
        reason = "Investigation handover"
        
        # Mock database operations
        mock_session = AsyncMock()
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # Transfer custody
        success = await evidence_store.transfer_custody(
            evidence_id, from_operator, to_operator, reason
        )
        
        # Verify success
        assert success is True
        assert mock_session.execute.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_search_evidence(self, evidence_store):
        """Test evidence search functionality."""
        # Mock database response
        mock_rows = [
            MagicMock(
                id="evidence-1",
                incident_id="incident-123",
                type="image",
                file_path="path1.jpg.enc",
                original_filename="image1.jpg",
                file_size=1024,
                mime_type="image/jpeg",
                hash_sha256="hash1",
                hmac_signature="sig1",
                encryption_key_id=None,
                created_at=datetime.now(),
                created_by="user1",
                camera_id="camera1",
                detection_id=None,
                status="pending",
                retention_until=None,
                auto_purge=True,
                metadata="{}",
                tags=[]
            )
        ]
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # Search evidence
        filters = {"incident_id": "incident-123", "type": "image"}
        results = await evidence_store.search_evidence(filters)
        
        # Verify results
        assert len(results) == 1
        assert results[0].id == "evidence-1"
        assert results[0].incident_id == "incident-123"
        assert results[0].type == EvidenceType.IMAGE
    
    @pytest.mark.asyncio
    async def test_schedule_purge(self, evidence_store):
        """Test evidence purge scheduling."""
        evidence_id = "test-evidence-purge"
        purge_date = datetime.now() + timedelta(days=30)
        operator_id = "operator-123"
        
        # Mock database operations
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        evidence_store.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        evidence_store.async_session.return_value.__aexit__ = AsyncMock()
        
        # Schedule purge
        success = await evidence_store.schedule_purge(evidence_id, purge_date, operator_id)
        
        # Verify success
        assert success is True
        assert mock_session.execute.call_count >= 2  # Update + chain of custody
        assert mock_session.commit.called
    
    def test_get_file_extension(self, evidence_store):
        """Test file extension mapping for different evidence types."""
        assert evidence_store._get_file_extension(EvidenceType.IMAGE) == ".jpg"
        assert evidence_store._get_file_extension(EvidenceType.VIDEO) == ".mp4"
        assert evidence_store._get_file_extension(EvidenceType.METADATA) == ".json"
        assert evidence_store._get_file_extension(EvidenceType.AUDIO) == ".wav"
        assert evidence_store._get_file_extension(EvidenceType.SENSOR_DATA) == ".dat"
        assert evidence_store._get_file_extension(EvidenceType.LOG_FILE) == ".log"
        assert evidence_store._get_file_extension(EvidenceType.REPORT) == ".pdf"


if __name__ == "__main__":
    pytest.main([__file__])