"""
Key Management System for Project Argus Evidence Store.
Handles encryption key generation, rotation, and secure storage.
"""

import os
import secrets
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


logger = logging.getLogger(__name__)


class KeyManager:
    """
    Secure key management for evidence encryption.
    Handles key generation, rotation, and secure storage.
    """
    
    def __init__(self, database_url: str, key_storage_path: str, master_key: str):
        """
        Initialize key manager.
        
        Args:
            database_url: PostgreSQL connection string
            key_storage_path: Path to store encrypted keys
            master_key: Master key for encrypting stored keys
        """
        self.database_url = database_url
        self.key_storage_path = Path(key_storage_path)
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize master key encryption
        master_key_bytes = base64.urlsafe_b64decode(master_key.encode())
        self.master_fernet = Fernet(base64.urlsafe_b64encode(master_key_bytes))
        
        # Initialize database
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Initialize key storage table
        self._init_key_storage_table()
    
    def _init_key_storage_table(self):
        """Initialize key storage table if it doesn't exist."""
        # This would typically be done in a migration, but for completeness:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS encryption_keys (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            key_id VARCHAR(255) UNIQUE NOT NULL,
            key_type VARCHAR(50) NOT NULL,
            encrypted_key TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP,
            active BOOLEAN DEFAULT TRUE,
            metadata JSONB DEFAULT '{}'
        );
        
        CREATE INDEX IF NOT EXISTS idx_encryption_keys_key_id ON encryption_keys(key_id);
        CREATE INDEX IF NOT EXISTS idx_encryption_keys_active ON encryption_keys(active);
        """
        
        # Note: In production, this should be handled by proper database migrations
        logger.info("Key storage table initialization would be handled by migrations")
    
    def generate_encryption_key(self) -> str:
        """Generate a new AES-256 encryption key."""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()
    
    def generate_hmac_secret(self) -> str:
        """Generate a new HMAC secret key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    async def store_key(self, key_id: str, key_data: str, key_type: str = "aes256", 
                       expires_at: Optional[datetime] = None, 
                       metadata: Optional[Dict] = None) -> bool:
        """
        Store an encryption key securely.
        
        Args:
            key_id: Unique identifier for the key
            key_data: The key data to store
            key_type: Type of key (aes256, hmac, etc.)
            expires_at: Optional expiration date
            metadata: Additional metadata for the key
            
        Returns:
            True if key was stored successfully
        """
        try:
            # Encrypt the key with master key
            encrypted_key = self.master_fernet.encrypt(key_data.encode()).decode()
            
            async with self.async_session() as session:
                query = """
                INSERT INTO encryption_keys (key_id, key_type, encrypted_key, expires_at, metadata)
                VALUES (:key_id, :key_type, :encrypted_key, :expires_at, :metadata)
                ON CONFLICT (key_id) DO UPDATE SET
                    encrypted_key = EXCLUDED.encrypted_key,
                    expires_at = EXCLUDED.expires_at,
                    metadata = EXCLUDED.metadata,
                    active = TRUE
                """
                
                await session.execute(sa.text(query), {
                    'key_id': key_id,
                    'key_type': key_type,
                    'encrypted_key': encrypted_key,
                    'expires_at': expires_at,
                    'metadata': json.dumps(metadata or {})
                })
                
                await session.commit()
                logger.info(f"Stored encryption key: {key_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store key {key_id}: {e}")
            return False
    
    async def retrieve_key(self, key_id: str) -> Optional[str]:
        """
        Retrieve and decrypt an encryption key.
        
        Args:
            key_id: Unique identifier for the key
            
        Returns:
            Decrypted key data or None if not found
        """
        try:
            async with self.async_session() as session:
                query = """
                SELECT encrypted_key FROM encryption_keys 
                WHERE key_id = :key_id AND active = TRUE
                AND (expires_at IS NULL OR expires_at > NOW())
                """
                
                result = await session.execute(sa.text(query), {'key_id': key_id})
                row = result.fetchone()
                
                if not row:
                    return None
                
                # Decrypt the key
                decrypted_key = self.master_fernet.decrypt(row.encrypted_key.encode()).decode()
                return decrypted_key
                
        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            return None
    
    async def rotate_key(self, old_key_id: str, new_key_id: str) -> bool:
        """
        Rotate an encryption key by creating a new one and marking the old one as inactive.
        
        Args:
            old_key_id: ID of the key to rotate
            new_key_id: ID for the new key
            
        Returns:
            True if rotation was successful
        """
        try:
            # Get the old key metadata
            async with self.async_session() as session:
                query = """
                SELECT key_type, metadata FROM encryption_keys 
                WHERE key_id = :key_id AND active = TRUE
                """
                
                result = await session.execute(sa.text(query), {'key_id': old_key_id})
                row = result.fetchone()
                
                if not row:
                    logger.error(f"Key {old_key_id} not found for rotation")
                    return False
                
                key_type = row.key_type
                metadata = json.loads(row.metadata) if row.metadata else {}
                
                # Generate new key
                if key_type == "aes256":
                    new_key_data = self.generate_encryption_key()
                elif key_type == "hmac":
                    new_key_data = self.generate_hmac_secret()
                else:
                    logger.error(f"Unsupported key type for rotation: {key_type}")
                    return False
                
                # Store new key
                metadata['rotated_from'] = old_key_id
                metadata['rotation_date'] = datetime.now().isoformat()
                
                success = await self.store_key(
                    key_id=new_key_id,
                    key_data=new_key_data,
                    key_type=key_type,
                    metadata=metadata
                )
                
                if not success:
                    return False
                
                # Mark old key as inactive
                update_query = """
                UPDATE encryption_keys 
                SET active = FALSE, expires_at = NOW()
                WHERE key_id = :key_id
                """
                
                await session.execute(sa.text(update_query), {'key_id': old_key_id})
                await session.commit()
                
                logger.info(f"Successfully rotated key from {old_key_id} to {new_key_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rotate key {old_key_id}: {e}")
            return False
    
    async def list_keys(self, active_only: bool = True) -> List[Dict]:
        """
        List all stored keys with their metadata.
        
        Args:
            active_only: If True, only return active keys
            
        Returns:
            List of key information dictionaries
        """
        try:
            async with self.async_session() as session:
                where_clause = "WHERE active = TRUE" if active_only else ""
                query = f"""
                SELECT key_id, key_type, created_at, expires_at, active, metadata
                FROM encryption_keys 
                {where_clause}
                ORDER BY created_at DESC
                """
                
                result = await session.execute(sa.text(query))
                rows = result.fetchall()
                
                keys = []
                for row in rows:
                    key_info = {
                        'key_id': row.key_id,
                        'key_type': row.key_type,
                        'created_at': row.created_at.isoformat(),
                        'expires_at': row.expires_at.isoformat() if row.expires_at else None,
                        'active': row.active,
                        'metadata': json.loads(row.metadata) if row.metadata else {}
                    }
                    keys.append(key_info)
                
                return keys
                
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []
    
    async def cleanup_expired_keys(self) -> List[str]:
        """
        Clean up expired keys from storage.
        
        Returns:
            List of cleaned up key IDs
        """
        cleaned_keys = []
        
        try:
            async with self.async_session() as session:
                # Find expired keys
                query = """
                SELECT key_id FROM encryption_keys 
                WHERE expires_at IS NOT NULL 
                AND expires_at < NOW()
                AND active = FALSE
                """
                
                result = await session.execute(sa.text(query))
                expired_keys = [row.key_id for row in result.fetchall()]
                
                # Delete expired keys
                for key_id in expired_keys:
                    delete_query = """
                    DELETE FROM encryption_keys WHERE key_id = :key_id
                    """
                    
                    await session.execute(sa.text(delete_query), {'key_id': key_id})
                    cleaned_keys.append(key_id)
                    logger.info(f"Cleaned up expired key: {key_id}")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
        
        return cleaned_keys
    
    async def backup_keys(self, backup_path: str) -> bool:
        """
        Create an encrypted backup of all active keys.
        
        Args:
            backup_path: Path to store the backup file
            
        Returns:
            True if backup was successful
        """
        try:
            keys = await self.list_keys(active_only=True)
            
            # Create backup data structure
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'keys': []
            }
            
            # Retrieve and include key data
            for key_info in keys:
                key_data = await self.retrieve_key(key_info['key_id'])
                if key_data:
                    backup_key = key_info.copy()
                    backup_key['key_data'] = key_data
                    backup_data['keys'].append(backup_key)
            
            # Encrypt backup data
            backup_json = json.dumps(backup_data, indent=2)
            encrypted_backup = self.master_fernet.encrypt(backup_json.encode())
            
            # Write to file
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_file, 'wb') as f:
                f.write(encrypted_backup)
            
            logger.info(f"Created key backup at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup keys: {e}")
            return False
    
    async def restore_keys(self, backup_path: str) -> bool:
        """
        Restore keys from an encrypted backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if restore was successful
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Read and decrypt backup
            with open(backup_file, 'rb') as f:
                encrypted_backup = f.read()
            
            decrypted_backup = self.master_fernet.decrypt(encrypted_backup)
            backup_data = json.loads(decrypted_backup.decode())
            
            # Restore keys
            restored_count = 0
            for key_info in backup_data['keys']:
                success = await self.store_key(
                    key_id=key_info['key_id'],
                    key_data=key_info['key_data'],
                    key_type=key_info['key_type'],
                    expires_at=datetime.fromisoformat(key_info['expires_at']) if key_info['expires_at'] else None,
                    metadata=key_info['metadata']
                )
                
                if success:
                    restored_count += 1
            
            logger.info(f"Restored {restored_count} keys from backup")
            return restored_count > 0
            
        except Exception as e:
            logger.error(f"Failed to restore keys from backup: {e}")
            return False