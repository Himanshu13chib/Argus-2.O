"""
Tests for Forensics Engine Implementation.
Tests automated report generation and video summary creation.
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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from forensics_engine import ForensicsEngine
from evidence_store import EvidenceStore
from shared.models.evidence import Evidence, EvidenceType, EvidenceStatus


@pytest.fixture
def temp_output():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_evidence_store():
    """Mock evidence store for testing."""
    store = MagicMock(spec=EvidenceStore)
    return store


@pytest.fixture
def forensics_engine(temp_output, mock_evidence_store):
    """Create forensics engine instance for testing."""
    database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
    
    engine = ForensicsEngine(
        database_url=database_url,
        evidence_store=mock_evidence_store,
        output_path=temp_output
    )
    
    # Mock the database session
    engine.async_session = AsyncMock()
    
    return engine


@pytest.fixture
def sample_incident_data():
    """Sample incident data for testing."""
    return {
        'id': 'incident-123',
        'alert_id': 'alert-456',
        'operator_id': 'operator-789',
        'operator_name': 'Test Operator',
        'status': 'closed',
        'priority': 'high',
        'created_at': datetime.now() - timedelta(hours=2),
        'updated_at': datetime.now() - timedelta(hours=1),
        'closed_at': datetime.now(),
        'alert_type': 'border_crossing',
        'severity': 'high',
        'confidence': 0.95,
        'risk_score': 8.5,
        'camera_id': 'camera-001',
        'detection_id': 'detection-123',
        'alert_metadata': {'location': 'sector-A', 'weather': 'clear'}
    }


@pytest.fixture
def sample_evidence_list():
    """Sample evidence list for testing."""
    return [
        Evidence(
            id='evidence-1',
            type=EvidenceType.IMAGE,
            file_path='2024/01/01/evidence-1.jpg.enc',
            original_filename='crossing_image.jpg',
            file_size=2048,
            mime_type='image/jpeg',
            hash_sha256='hash1',
            hmac_signature='sig1',
            created_at=datetime.now() - timedelta(hours=2),
            created_by='system',
            camera_id='camera-001',
            detection_id='detection-123',
            status=EvidenceStatus.PENDING,
            metadata={'confidence': 0.95, 'bbox': [100, 100, 200, 200]}
        ),
        Evidence(
            id='evidence-2',
            type=EvidenceType.VIDEO,
            file_path='2024/01/01/evidence-2.mp4.enc',
            original_filename='crossing_video.mp4',
            file_size=10240,
            mime_type='video/mp4',
            hash_sha256='hash2',
            hmac_signature='sig2',
            created_at=datetime.now() - timedelta(hours=1),
            created_by='system',
            camera_id='camera-001',
            detection_id='detection-123',
            status=EvidenceStatus.PENDING,
            metadata={'duration': 30, 'fps': 30}
        )
    ]


class TestForensicsEngine:
    """Test cases for Forensics Engine functionality."""
    
    def test_setup_custom_styles(self, forensics_engine):
        """Test PDF report style setup."""
        # Verify custom styles are created
        assert 'CustomTitle' in forensics_engine.styles.byName
        assert 'SectionHeader' in forensics_engine.styles.byName
        assert 'EvidenceItem' in forensics_engine.styles.byName
        assert 'Metadata' in forensics_engine.styles.byName
        
        # Verify style properties
        title_style = forensics_engine.styles['CustomTitle']
        assert title_style.fontSize == 18
        assert title_style.alignment == 1  # TA_CENTER
    
    @pytest.mark.asyncio
    async def test_get_incident_data(self, forensics_engine, sample_incident_data):
        """Test incident data retrieval."""
        # Mock database response
        mock_row = MagicMock()
        for key, value in sample_incident_data.items():
            setattr(mock_row, key, value)
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        forensics_engine.async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        forensics_engine.async_session.return_value.__aexit__ = AsyncMock()
        
        # Get incident data
        incident_data = await forensics_engine._get_incident_data('incident-123')
        
        # Verify data
        assert incident_data is not None
        assert incident_data['id'] == 'incident-123'
        assert incident_data['status'] == 'closed'
        assert incident_data['confidence'] == 0.95
    
    @pytest.mark.asyncio
    async def test_generate_report(self, forensics_engine, sample_incident_data, sample_evidence_list, temp_output):
        """Test comprehensive report generation."""
        # Mock dependencies
        forensics_engine._get_incident_data = AsyncMock(return_value=sample_incident_data)
        forensics_engine.evidence_store.search_evidence = AsyncMock(return_value=sample_evidence_list)
        forensics_engine.evidence_store.get_chain_of_custody = AsyncMock(return_value=None)
        
        # Generate report
        report_path = await forensics_engine.generate_report('incident-123', 'comprehensive')
        
        # Verify report file is created
        assert os.path.exists(report_path)
        assert report_path.endswith('.pdf')
        assert 'incident-123' in report_path
        
        # Verify file size (should be non-empty)
        assert os.path.getsize(report_path) > 0
    
    @pytest.mark.asyncio
    async def test_create_video_summary(self, forensics_engine, sample_evidence_list, temp_output):
        """Test video summary creation."""
        # Mock evidence store methods
        forensics_engine.evidence_store.retrieve_evidence.side_effect = sample_evidence_list
        
        # Mock file content for video evidence
        video_content = b"fake video data for testing"
        image_content = b"fake image data for testing"
        
        def mock_get_evidence_file(evidence_id):
            if evidence_id == 'evidence-2':  # Video evidence
                return BytesIO(video_content)
            elif evidence_id == 'evidence-1':  # Image evidence
                return BytesIO(image_content)
            return None
        
        forensics_engine.evidence_store.get_evidence_file.side_effect = mock_get_evidence_file
        
        # Mock video compilation method to avoid OpenCV dependency in tests
        forensics_engine._create_video_compilation = AsyncMock()
        
        # Create video summary
        video_path = await forensics_engine.create_video_summary(['evidence-1', 'evidence-2'])
        
        # Verify video summary path
        assert video_path is not None
        assert video_path.endswith('.mp4')
        assert 'video_summary' in video_path
        
        # Verify video compilation was called
        forensics_engine._create_video_compilation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_export_legal_package(self, forensics_engine, sample_incident_data, sample_evidence_list, temp_output):
        """Test legal package export."""
        # Mock dependencies
        forensics_engine._get_incident_data = AsyncMock(return_value=sample_incident_data)
        forensics_engine.evidence_store.search_evidence = AsyncMock(return_value=sample_evidence_list)
        forensics_engine.evidence_store.get_evidence_file = AsyncMock(return_value=BytesIO(b"test evidence data"))
        forensics_engine.evidence_store.get_chain_of_custody = AsyncMock(return_value=None)
        forensics_engine.generate_report = AsyncMock(return_value=os.path.join(temp_output, "test_report.pdf"))
        forensics_engine.create_video_summary = AsyncMock(return_value=os.path.join(temp_output, "test_video.mp4"))
        forensics_engine._generate_chain_of_custody_report = AsyncMock(return_value='{"test": "data"}')
        forensics_engine._export_incident_csv = AsyncMock(return_value="test,csv,data")
        
        # Create dummy files for mocked methods
        with open(os.path.join(temp_output, "test_report.pdf"), 'w') as f:
            f.write("test report")
        with open(os.path.join(temp_output, "test_video.mp4"), 'w') as f:
            f.write("test video")
        
        # Export legal package
        package_path = await forensics_engine.export_legal_package('incident-123')
        
        # Verify package file is created
        assert os.path.exists(package_path)
        assert package_path.endswith('.zip')
        assert 'legal_package' in package_path
        assert 'incident-123' in package_path
        
        # Verify package size (should be non-empty)
        assert os.path.getsize(package_path) > 0
    
    @pytest.mark.asyncio
    async def test_add_incident_summary(self, forensics_engine, sample_incident_data):
        """Test incident summary section generation."""
        story_elements = await forensics_engine._add_incident_summary(sample_incident_data)
        
        # Verify story elements are generated
        assert len(story_elements) > 0
        
        # Check for expected elements (Paragraph, Table, Spacer)
        element_types = [type(element).__name__ for element in story_elements]
        assert 'Paragraph' in element_types
        assert 'Table' in element_types
        assert 'Spacer' in element_types
    
    @pytest.mark.asyncio
    async def test_add_evidence_section(self, forensics_engine, sample_evidence_list):
        """Test evidence section generation."""
        story_elements = await forensics_engine._add_evidence_section(sample_evidence_list)
        
        # Verify story elements are generated
        assert len(story_elements) > 0
        
        # Check for expected elements
        element_types = [type(element).__name__ for element in story_elements]
        assert 'Paragraph' in element_types
        assert 'Table' in element_types
    
    @pytest.mark.asyncio
    async def test_add_timeline_section(self, forensics_engine, sample_incident_data, sample_evidence_list):
        """Test timeline section generation."""
        story_elements = await forensics_engine._add_timeline_section(sample_incident_data, sample_evidence_list)
        
        # Verify story elements are generated
        assert len(story_elements) > 0
        
        # Check for timeline table
        element_types = [type(element).__name__ for element in story_elements]
        assert 'Table' in element_types
    
    @pytest.mark.asyncio
    async def test_export_incident_csv(self, forensics_engine, sample_incident_data, sample_evidence_list):
        """Test incident CSV export."""
        # Mock dependencies
        forensics_engine._get_incident_data = AsyncMock(return_value=sample_incident_data)
        forensics_engine.evidence_store.search_evidence = AsyncMock(return_value=sample_evidence_list)
        
        # Export CSV
        csv_data = await forensics_engine._export_incident_csv('incident-123')
        
        # Verify CSV content
        assert isinstance(csv_data, str)
        assert len(csv_data) > 0
        assert 'Incident Data' in csv_data
        assert 'Evidence Data' in csv_data
        assert 'incident-123' in csv_data
    
    def test_format_file_size(self, forensics_engine):
        """Test file size formatting."""
        assert forensics_engine._format_file_size(0) == "0 B"
        assert forensics_engine._format_file_size(1024) == "1.0 KB"
        assert forensics_engine._format_file_size(1024 * 1024) == "1.0 MB"
        assert forensics_engine._format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert forensics_engine._format_file_size(512) == "512.0 B"
        assert forensics_engine._format_file_size(1536) == "1.5 KB"
    
    @pytest.mark.asyncio
    async def test_generate_chain_of_custody_report(self, forensics_engine):
        """Test chain of custody report generation."""
        from shared.models.evidence import ChainOfCustody
        
        # Create test chain of custody
        chain = ChainOfCustody(evidence_id='evidence-123')
        chain.entries = [
            {
                'timestamp': datetime.now().isoformat(),
                'action': 'created',
                'operator_id': 'operator-1',
                'details': 'Evidence created',
                'entry_id': 'entry-1'
            },
            {
                'timestamp': (datetime.now() + timedelta(hours=1)).isoformat(),
                'action': 'sealed',
                'operator_id': 'operator-2',
                'details': 'Evidence sealed for legal proceedings',
                'entry_id': 'entry-2'
            }
        ]
        
        # Generate report
        report_json = await forensics_engine._generate_chain_of_custody_report(chain)
        
        # Verify report content
        assert isinstance(report_json, str)
        report_data = json.loads(report_json)
        assert report_data['evidence_id'] == 'evidence-123'
        assert len(report_data['entries']) == 2
        assert report_data['total_entries'] == 2
        assert 'generated_at' in report_data
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_incident(self, forensics_engine):
        """Test error handling for missing incident."""
        # Mock missing incident
        forensics_engine._get_incident_data = AsyncMock(return_value=None)
        
        # Attempt to generate report
        with pytest.raises(ValueError, match="Incident .* not found"):
            await forensics_engine.generate_report('nonexistent-incident')
    
    @pytest.mark.asyncio
    async def test_error_handling_empty_evidence(self, forensics_engine, sample_incident_data):
        """Test handling of incidents with no evidence."""
        # Mock incident with no evidence
        forensics_engine._get_incident_data = AsyncMock(return_value=sample_incident_data)
        forensics_engine.evidence_store.search_evidence = AsyncMock(return_value=[])
        forensics_engine.evidence_store.get_chain_of_custody = AsyncMock(return_value=None)
        
        # Generate report
        report_path = await forensics_engine.generate_report('incident-123')
        
        # Verify report is still generated
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 0


if __name__ == "__main__":
    pytest.main([__file__])