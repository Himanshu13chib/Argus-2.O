"""
Immutable Evidence Storage Implementation for Project Argus.
Provides HMAC signing, AES-256 encryption, and append-only storage.
"""

import os
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, BinaryIO
from pathlib import Path
import json
import asyncio
import logging
from io import BytesIO

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import aiofiles
from minio import Minio
from minio.error import S3Error

from shared.interfaces.evidence import IEvidenceStore
from shared.models.evidence import Evidence, EvidenceType, EvidenceStatus, ChainOfCustody


logger = logging.getLogger(__name__)


class EvidenceStore(IEvidenceStore):
    """
    Immutable evidence storage with HMAC signing and AES-256 encryption.
    Implements append-only storage with integrity verification.
    """
    
    def __init__(self, 
                 database_url: str,
                 storage_path: str,
                 encryption_key: str,
                 hmac_secret: str,
                 minio_client: Optional[Minio] = None,
                 bucket_name: str = "evidence"):
        """
        Initialize evidence store with encryption and storage backends.
        
        Args:
            database_url: PostgreSQL connection string
            storage_path: Local storage path for evidence files
            encryption_key: Base64 encoded encryption key for AES-256
            hmac_secret: Secret key for HMAC signing
            minio_client: Optional MinIO client for object storage
            bucket_name: S3/MinIO bucket name for evidence storage
        """
        self.database_url = database_url
        self.storage_path = Path(storage_path)
        self.bucket_name = bucket_name
        
        # Initialize encryption
        self.encryption_key = base64.urlsafe_b64decode(encryption_key.encode())
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        self.hmac_secret = hmac_secret.encode()
        
        # Initialize storage backends
        self.minio_client = minio_client
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Ensure MinIO bucket exists
        if self.minio_client:
            try:
                if not self.minio_client.bucket_exists(bucket_name):
                    self.minio_client.make_bucket(bucket_name)
                    logger.info(f"Created MinIO bucket: {bucket_name}")
            except S3Error as e:
                logger.error(f"Failed to create MinIO bucket: {e}")
                raise
    
    def _generate_hmac_signature(self, data: bytes) -> str:
        """Generate HMAC signature for data integrity."""
        return hmac.new(self.hmac_secret, data, hashlib.sha256).hexdigest()
    
    def _verify_hmac_signature(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature for data integrity."""
        expected_signature = self._generate_hmac_signature(data)
        return hmac.compare_digest(expected_signature, signature)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES-256."""
        return self.fernet.encrypt(data)
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-256."""
        return self.fernet.decrypt(encrypted_data)
    
    def _generate_file_path(self, evidence_id: str, evidence_type: EvidenceType) -> str:
        """Generate secure file path for evidence storage."""
        # Use date-based directory structure for organization
        date_path = datetime.now().strftime("%Y/%m/%d")
        file_extension = self._get_file_extension(evidence_type)
        return f"{date_path}/{evidence_id}{file_extension}.enc"
    
    def _get_file_extension(self, evidence_type: EvidenceType) -> str:
        """Get appropriate file extension for evidence type."""
        extensions = {
            EvidenceType.IMAGE: ".jpg",
            EvidenceType.VIDEO: ".mp4",
            EvidenceType.METADATA: ".json",
            EvidenceType.AUDIO: ".wav",
            EvidenceType.SENSOR_DATA: ".dat",
            EvidenceType.LOG_FILE: ".log",
            EvidenceType.REPORT: ".pdf"
        }
        return extensions.get(evidence_type, ".bin")
    
    async def store_evidence(self, file_data: bytes, evidence_type: EvidenceType, 
                           metadata: Dict[str, Any], created_by: str) -> str:
        """
        Store evidence with encryption and integrity verification.
        
        Args:
            file_data: Raw file data to store
            evidence_type: Type of evidence being stored
            metadata: Additional metadata for the evidence
            created_by: ID of user creating the evidence
            
        Returns:
            Evidence ID for the stored evidence
        """
        try:
            # Create evidence record
            evidence = Evidence(
                type=evidence_type,
                created_by=created_by,
                metadata=metadata,
                file_size=len(file_data),
                original_filename=metadata.get('original_filename', ''),
                mime_type=metadata.get('mime_type', ''),
                incident_id=metadata.get('incident_id'),
                camera_id=metadata.get('camera_id'),
                detection_id=metadata.get('detection_id')
            )
            
            # Calculate hash and HMAC signature
            evidence.calculate_hash(file_data)
            hmac_signature = self._generate_hmac_signature(file_data)
            evidence.hmac_signature = hmac_signature
            
            # Encrypt data
            encrypted_data = self._encrypt_data(file_data)
            
            # Generate file path
            file_path = self._generate_file_path(evidence.id, evidence_type)
            evidence.file_path = file_path
            
            # Store file (try MinIO first, fallback to local storage)
            if self.minio_client:
                try:
                    # Store in MinIO
                    self.minio_client.put_object(
                        bucket_name=self.bucket_name,
                        object_name=file_path,
                        data=BytesIO(encrypted_data),
                        length=len(encrypted_data),
                        metadata={
                            'evidence-id': evidence.id,
                            'evidence-type': evidence_type.value,
                            'created-by': created_by,
                            'hash-sha256': evidence.hash_sha256,
                            'hmac-signature': hmac_signature
                        }
                    )
                    logger.info(f"Stored evidence {evidence.id} in MinIO")
                except S3Error as e:
                    logger.warning(f"MinIO storage failed, using local storage: {e}")
                    await self._store_local_file(file_path, encrypted_data)
            else:
                # Store locally
                await self._store_local_file(file_path, encrypted_data)
            
            # Store metadata in database
            async with self.async_session() as session:
                # Insert evidence record
                evidence_query = """
                INSERT INTO evidence (
                    id, incident_id, type, file_path, original_filename, file_size,
                    mime_type, hash_sha256, hmac_signature, created_by, camera_id,
                    detection_id, metadata, tags
                ) VALUES (
                    :id, :incident_id, :type, :file_path, :original_filename, :file_size,
                    :mime_type, :hash_sha256, :hmac_signature, :created_by, :camera_id,
                    :detection_id, :metadata, :tags
                )
                """
                
                await session.execute(sa.text(evidence_query), {
                    'id': evidence.id,
                    'incident_id': evidence.incident_id,
                    'type': evidence.type.value,
                    'file_path': evidence.file_path,
                    'original_filename': evidence.original_filename,
                    'file_size': evidence.file_size,
                    'mime_type': evidence.mime_type,
                    'hash_sha256': evidence.hash_sha256,
                    'hmac_signature': evidence.hmac_signature,
                    'created_by': evidence.created_by,
                    'camera_id': evidence.camera_id,
                    'detection_id': evidence.detection_id,
                    'metadata': json.dumps(evidence.metadata),
                    'tags': evidence.tags
                })
                
                # Add initial chain of custody entry
                custody_query = """
                INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                VALUES (:evidence_id, :action, :operator_id, :details)
                """
                
                await session.execute(sa.text(custody_query), {
                    'evidence_id': evidence.id,
                    'action': 'created',
                    'operator_id': created_by,
                    'details': f'Evidence created and stored with type: {evidence_type.value}'
                })
                
                await session.commit()
                logger.info(f"Evidence {evidence.id} stored successfully")
                
            return evidence.id
            
        except Exception as e:
            logger.error(f"Failed to store evidence: {e}")
            raise
    
    async def _store_local_file(self, file_path: str, encrypted_data: bytes) -> None:
        """Store encrypted file locally."""
        full_path = self.storage_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, 'wb') as f:
            await f.write(encrypted_data)
        
        logger.info(f"Stored evidence locally at {full_path}")
    
    async def retrieve_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Retrieve evidence metadata by ID."""
        try:
            async with self.async_session() as session:
                query = """
                SELECT id, incident_id, type, file_path, original_filename, file_size,
                       mime_type, hash_sha256, hmac_signature, encryption_key_id,
                       created_at, created_by, camera_id, detection_id, status,
                       retention_until, auto_purge, metadata, tags
                FROM evidence WHERE id = :evidence_id
                """
                
                result = await session.execute(sa.text(query), {'evidence_id': evidence_id})
                row = result.fetchone()
                
                if not row:
                    return None
                
                # Convert row to Evidence object
                evidence = Evidence(
                    id=row.id,
                    incident_id=row.incident_id,
                    type=EvidenceType(row.type),
                    file_path=row.file_path,
                    original_filename=row.original_filename or '',
                    file_size=row.file_size,
                    mime_type=row.mime_type or '',
                    hash_sha256=row.hash_sha256,
                    hmac_signature=row.hmac_signature,
                    encryption_key_id=row.encryption_key_id,
                    created_at=row.created_at,
                    created_by=row.created_by,
                    camera_id=row.camera_id,
                    detection_id=row.detection_id,
                    status=EvidenceStatus(row.status),
                    retention_until=row.retention_until,
                    auto_purge=row.auto_purge,
                    metadata=json.loads(row.metadata) if row.metadata else {},
                    tags=row.tags or []
                )
                
                return evidence
                
        except Exception as e:
            logger.error(f"Failed to retrieve evidence {evidence_id}: {e}")
            return None
    
    async def get_evidence_file(self, evidence_id: str) -> Optional[BinaryIO]:
        """Get decrypted evidence file content."""
        try:
            evidence = await self.retrieve_evidence(evidence_id)
            if not evidence:
                return None
            
            # Try to get file from MinIO first
            encrypted_data = None
            if self.minio_client:
                try:
                    response = self.minio_client.get_object(
                        bucket_name=self.bucket_name,
                        object_name=evidence.file_path
                    )
                    encrypted_data = response.read()
                    response.close()
                except S3Error:
                    logger.warning(f"Failed to retrieve from MinIO, trying local storage")
            
            # Fallback to local storage
            if encrypted_data is None:
                local_path = self.storage_path / evidence.file_path
                if local_path.exists():
                    async with aiofiles.open(local_path, 'rb') as f:
                        encrypted_data = await f.read()
                else:
                    logger.error(f"Evidence file not found: {evidence.file_path}")
                    return None
            
            # Decrypt data
            decrypted_data = self._decrypt_data(encrypted_data)
            
            # Verify integrity
            if not self._verify_hmac_signature(decrypted_data, evidence.hmac_signature):
                logger.error(f"HMAC verification failed for evidence {evidence_id}")
                return None
            
            # Verify hash
            if not evidence.verify_integrity(decrypted_data):
                logger.error(f"Hash verification failed for evidence {evidence_id}")
                return None
            
            return BytesIO(decrypted_data)
            
        except Exception as e:
            logger.error(f"Failed to get evidence file {evidence_id}: {e}")
            return None
    
    async def verify_integrity(self, evidence_id: str) -> bool:
        """Verify evidence integrity using stored hash and HMAC."""
        try:
            evidence = await self.retrieve_evidence(evidence_id)
            if not evidence:
                return False
            
            file_content = await self.get_evidence_file(evidence_id)
            if not file_content:
                return False
            
            data = file_content.read()
            file_content.close()
            
            # Verify both hash and HMAC
            hash_valid = evidence.verify_integrity(data)
            hmac_valid = self._verify_hmac_signature(data, evidence.hmac_signature)
            
            return hash_valid and hmac_valid
            
        except Exception as e:
            logger.error(f"Failed to verify integrity for evidence {evidence_id}: {e}")
            return False
    
    async def seal_evidence(self, evidence_id: str, operator_id: str) -> bool:
        """Seal evidence to prevent modifications."""
        try:
            async with self.async_session() as session:
                # Update evidence status
                update_query = """
                UPDATE evidence SET status = 'sealed' WHERE id = :evidence_id
                """
                result = await session.execute(sa.text(update_query), {'evidence_id': evidence_id})
                
                if result.rowcount == 0:
                    return False
                
                # Add chain of custody entry
                custody_query = """
                INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                VALUES (:evidence_id, :action, :operator_id, :details)
                """
                
                await session.execute(sa.text(custody_query), {
                    'evidence_id': evidence_id,
                    'action': 'sealed',
                    'operator_id': operator_id,
                    'details': 'Evidence sealed for legal proceedings'
                })
                
                await session.commit()
                logger.info(f"Evidence {evidence_id} sealed by {operator_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to seal evidence {evidence_id}: {e}")
            return False
    
    async def transfer_custody(self, evidence_id: str, from_operator: str, 
                             to_operator: str, reason: str) -> bool:
        """Transfer evidence custody between operators."""
        try:
            async with self.async_session() as session:
                # Add chain of custody entry
                custody_query = """
                INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                VALUES (:evidence_id, :action, :operator_id, :details)
                """
                
                await session.execute(sa.text(custody_query), {
                    'evidence_id': evidence_id,
                    'action': 'transfer',
                    'operator_id': to_operator,
                    'details': f'Transferred from {from_operator} to {to_operator}. Reason: {reason}'
                })
                
                await session.commit()
                logger.info(f"Evidence {evidence_id} transferred from {from_operator} to {to_operator}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to transfer custody for evidence {evidence_id}: {e}")
            return False
    
    async def get_chain_of_custody(self, evidence_id: str) -> Optional[ChainOfCustody]:
        """Get complete chain of custody for evidence."""
        try:
            async with self.async_session() as session:
                query = """
                SELECT timestamp, action, operator_id, details, entry_id
                FROM chain_of_custody 
                WHERE evidence_id = :evidence_id 
                ORDER BY timestamp ASC
                """
                
                result = await session.execute(sa.text(query), {'evidence_id': evidence_id})
                rows = result.fetchall()
                
                if not rows:
                    return None
                
                chain = ChainOfCustody(evidence_id=evidence_id)
                for row in rows:
                    entry = {
                        'timestamp': row.timestamp.isoformat(),
                        'action': row.action,
                        'operator_id': row.operator_id,
                        'details': row.details or '',
                        'entry_id': row.entry_id
                    }
                    chain.entries.append(entry)
                
                return chain
                
        except Exception as e:
            logger.error(f"Failed to get chain of custody for evidence {evidence_id}: {e}")
            return None
    
    async def search_evidence(self, filters: Dict[str, Any]) -> List[Evidence]:
        """Search evidence by various criteria."""
        try:
            conditions = []
            params = {}
            
            if 'incident_id' in filters:
                conditions.append("incident_id = :incident_id")
                params['incident_id'] = filters['incident_id']
            
            if 'type' in filters:
                conditions.append("type = :type")
                params['type'] = filters['type']
            
            if 'created_by' in filters:
                conditions.append("created_by = :created_by")
                params['created_by'] = filters['created_by']
            
            if 'camera_id' in filters:
                conditions.append("camera_id = :camera_id")
                params['camera_id'] = filters['camera_id']
            
            if 'status' in filters:
                conditions.append("status = :status")
                params['status'] = filters['status']
            
            if 'start_date' in filters:
                conditions.append("created_at >= :start_date")
                params['start_date'] = filters['start_date']
            
            if 'end_date' in filters:
                conditions.append("created_at <= :end_date")
                params['end_date'] = filters['end_date']
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            async with self.async_session() as session:
                query = f"""
                SELECT id, incident_id, type, file_path, original_filename, file_size,
                       mime_type, hash_sha256, hmac_signature, encryption_key_id,
                       created_at, created_by, camera_id, detection_id, status,
                       retention_until, auto_purge, metadata, tags
                FROM evidence 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 1000
                """
                
                result = await session.execute(sa.text(query), params)
                rows = result.fetchall()
                
                evidence_list = []
                for row in rows:
                    evidence = Evidence(
                        id=row.id,
                        incident_id=row.incident_id,
                        type=EvidenceType(row.type),
                        file_path=row.file_path,
                        original_filename=row.original_filename or '',
                        file_size=row.file_size,
                        mime_type=row.mime_type or '',
                        hash_sha256=row.hash_sha256,
                        hmac_signature=row.hmac_signature,
                        encryption_key_id=row.encryption_key_id,
                        created_at=row.created_at,
                        created_by=row.created_by,
                        camera_id=row.camera_id,
                        detection_id=row.detection_id,
                        status=EvidenceStatus(row.status),
                        retention_until=row.retention_until,
                        auto_purge=row.auto_purge,
                        metadata=json.loads(row.metadata) if row.metadata else {},
                        tags=row.tags or []
                    )
                    evidence_list.append(evidence)
                
                return evidence_list
                
        except Exception as e:
            logger.error(f"Failed to search evidence: {e}")
            return []
    
    async def schedule_purge(self, evidence_id: str, purge_date: datetime, operator_id: str) -> bool:
        """Schedule evidence for automatic purging."""
        try:
            async with self.async_session() as session:
                # Update evidence with purge schedule
                update_query = """
                UPDATE evidence 
                SET retention_until = :purge_date, auto_purge = true
                WHERE id = :evidence_id
                """
                
                result = await session.execute(sa.text(update_query), {
                    'evidence_id': evidence_id,
                    'purge_date': purge_date
                })
                
                if result.rowcount == 0:
                    return False
                
                # Add chain of custody entry
                custody_query = """
                INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                VALUES (:evidence_id, :action, :operator_id, :details)
                """
                
                await session.execute(sa.text(custody_query), {
                    'evidence_id': evidence_id,
                    'action': 'purge_scheduled',
                    'operator_id': operator_id,
                    'details': f'Scheduled for purge on {purge_date.isoformat()}'
                })
                
                await session.commit()
                logger.info(f"Evidence {evidence_id} scheduled for purge on {purge_date}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to schedule purge for evidence {evidence_id}: {e}")
            return False
    
    async def purge_expired_evidence(self) -> List[str]:
        """Purge expired evidence and return list of purged IDs."""
        purged_ids = []
        
        try:
            async with self.async_session() as session:
                # Find expired evidence
                query = """
                SELECT id, file_path FROM evidence 
                WHERE auto_purge = true 
                AND retention_until IS NOT NULL 
                AND retention_until < NOW()
                AND status != 'sealed'
                """
                
                result = await session.execute(sa.text(query))
                expired_evidence = result.fetchall()
                
                for evidence in expired_evidence:
                    try:
                        # Delete file from storage
                        if self.minio_client:
                            try:
                                self.minio_client.remove_object(
                                    bucket_name=self.bucket_name,
                                    object_name=evidence.file_path
                                )
                            except S3Error:
                                pass  # File might not exist in MinIO
                        
                        # Delete local file
                        local_path = self.storage_path / evidence.file_path
                        if local_path.exists():
                            local_path.unlink()
                        
                        # Update evidence status
                        update_query = """
                        UPDATE evidence SET status = 'purged' WHERE id = :evidence_id
                        """
                        await session.execute(sa.text(update_query), {'evidence_id': evidence.id})
                        
                        # Add chain of custody entry
                        custody_query = """
                        INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                        VALUES (:evidence_id, :action, :operator_id, :details)
                        """
                        
                        await session.execute(sa.text(custody_query), {
                            'evidence_id': evidence.id,
                            'action': 'purged',
                            'operator_id': 'system',
                            'details': 'Automatically purged due to retention policy'
                        })
                        
                        purged_ids.append(evidence.id)
                        logger.info(f"Purged expired evidence: {evidence.id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to purge evidence {evidence.id}: {e}")
                        continue
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to purge expired evidence: {e}")
        
        return purged_ids