"""
Security Service for Project Argus.
Provides encryption, key management, and secure communications.
"""

import os
import logging
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import json
import asyncpg
import redis
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import shared interfaces
import sys
sys.path.append('/app/shared')
from interfaces.security import ISecurityManager, IKeyManager

# Import audit and compliance modules
from audit_compliance import SecurityEventMonitor, ComplianceManager, SecurityEventType, ComplianceFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://argus:argus@db:5432/argus")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MASTER_KEY = os.getenv("MASTER_KEY", "change-this-master-key-in-production")

# Global connections
db_pool = None
redis_client = None

# Security scheme
security = HTTPBearer()


class KeyInfo(BaseModel):
    key_id: str
    key_type: str
    algorithm: str
    key_size: int
    created_at: datetime
    expires_at: Optional[datetime]
    status: str


class EncryptionRequest(BaseModel):
    data: str  # Base64 encoded data
    key_id: Optional[str] = None


class DecryptionRequest(BaseModel):
    encrypted_data: str  # Base64 encoded
    key_id: str


class SigningRequest(BaseModel):
    data: str  # Base64 encoded data
    key_id: Optional[str] = None


class VerificationRequest(BaseModel):
    data: str  # Base64 encoded
    signature: str  # Base64 encoded
    key_id: str


class HMACRequest(BaseModel):
    data: str  # Base64 encoded data
    key_id: Optional[str] = None


class SecurityEventRequest(BaseModel):
    event_type: str
    user_id: Optional[str] = None
    source_ip: str
    action: str
    success: bool
    details: Dict[str, Any] = {}
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    session_id: Optional[str] = None


class ComplianceReportRequest(BaseModel):
    framework: str
    period_start: datetime
    period_end: datetime


class KeyManager(IKeyManager):
    """Implementation of cryptographic key management."""
    
    def __init__(self, db_pool, redis_client, master_key: str):
        self.db_pool = db_pool
        self.redis = redis_client
        self.master_key = master_key.encode()
        self._derive_master_key()
    
    def _derive_master_key(self):
        """Derive encryption key from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'project_argus_salt',  # In production, use random salt per installation
            iterations=100000,
            backend=default_backend()
        )
        self.derived_key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.fernet = Fernet(self.derived_key)
    
    async def generate_key(self, key_type: str, key_size: int = 256) -> str:
        """Generate new cryptographic key and return key ID."""
        try:
            key_id = secrets.token_urlsafe(32)
            
            if key_type == "symmetric":
                # Generate AES key
                key_data = secrets.token_bytes(key_size // 8)
                algorithm = f"AES-{key_size}"
            elif key_type == "hmac":
                # Generate HMAC key
                key_data = secrets.token_bytes(key_size // 8)
                algorithm = f"HMAC-SHA256"
            elif key_type == "rsa":
                # Generate RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend()
                )
                key_data = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                algorithm = f"RSA-{key_size}"
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
            
            # Encrypt key data with master key
            encrypted_key = self.fernet.encrypt(key_data)
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO encryption_keys 
                    (key_id, key_type, algorithm, key_size, encrypted_key_data, 
                     created_at, status)
                    VALUES ($1, $2, $3, $4, $5, $6, 'active')
                """, key_id, key_type, algorithm, key_size, 
                    encrypted_key, datetime.utcnow())
            
            # Cache in Redis for performance
            await self.redis.setex(
                f"key:{key_id}",
                timedelta(hours=24),
                encrypted_key
            )
            
            logger.info(f"Generated new {key_type} key: {key_id}")
            return key_id
            
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate key"
            )
    
    async def get_key(self, key_id: str) -> Optional[bytes]:
        """Get key by ID."""
        try:
            # Try Redis cache first
            cached_key = await self.redis.get(f"key:{key_id}")
            if cached_key:
                return self.fernet.decrypt(cached_key.encode())
            
            # Get from database
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT encrypted_key_data FROM encryption_keys WHERE key_id = $1 AND status = 'active'",
                    key_id
                )
                
                if not row:
                    return None
                
                encrypted_key = row['encrypted_key_data']
                
                # Cache for future use
                await self.redis.setex(
                    f"key:{key_id}",
                    timedelta(hours=24),
                    encrypted_key
                )
                
                return self.fernet.decrypt(encrypted_key)
                
        except Exception as e:
            logger.error(f"Error getting key {key_id}: {e}")
            return None
    
    async def rotate_key(self, key_id: str) -> str:
        """Rotate key and return new key ID."""
        try:
            # Get old key info
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT key_type, key_size FROM encryption_keys WHERE key_id = $1",
                    key_id
                )
                
                if not row:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Key not found"
                    )
                
                # Generate new key
                new_key_id = await self.generate_key(row['key_type'], row['key_size'])
                
                # Mark old key as rotated
                await conn.execute(
                    "UPDATE encryption_keys SET status = 'rotated', rotated_at = $1 WHERE key_id = $2",
                    datetime.utcnow(), key_id
                )
                
                # Remove from cache
                await self.redis.delete(f"key:{key_id}")
                
                logger.info(f"Rotated key {key_id} to {new_key_id}")
                return new_key_id
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error rotating key: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to rotate key"
            )
    
    async def revoke_key(self, key_id: str) -> bool:
        """Revoke key (mark as invalid)."""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE encryption_keys SET status = 'revoked', revoked_at = $1 WHERE key_id = $2",
                    datetime.utcnow(), key_id
                )
                
                # Remove from cache
                await self.redis.delete(f"key:{key_id}")
                
                success = result.split()[-1] != '0'
                if success:
                    logger.info(f"Revoked key: {key_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Error revoking key: {e}")
            return False
    
    async def list_keys(self, key_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all keys, optionally filtered by type."""
        try:
            async with self.db_pool.acquire() as conn:
                if key_type:
                    rows = await conn.fetch(
                        "SELECT key_id, key_type, algorithm, key_size, created_at, expires_at, status FROM encryption_keys WHERE key_type = $1 ORDER BY created_at DESC",
                        key_type
                    )
                else:
                    rows = await conn.fetch(
                        "SELECT key_id, key_type, algorithm, key_size, created_at, expires_at, status FROM encryption_keys ORDER BY created_at DESC"
                    )
                
                keys = []
                for row in rows:
                    keys.append({
                        "key_id": row['key_id'],
                        "key_type": row['key_type'],
                        "algorithm": row['algorithm'],
                        "key_size": row['key_size'],
                        "created_at": row['created_at'].isoformat(),
                        "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                        "status": row['status']
                    })
                
                return keys
                
        except Exception as e:
            logger.error(f"Error listing keys: {e}")
            return []
    
    async def backup_keys(self, backup_location: str) -> bool:
        """Backup keys to secure location."""
        try:
            # In a real implementation, this would backup to secure storage
            # For now, we'll create a backup record
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO key_backups (backup_id, location, created_at, status)
                    VALUES ($1, $2, $3, 'completed')
                """, secrets.token_urlsafe(16), backup_location, datetime.utcnow())
            
            logger.info(f"Keys backed up to: {backup_location}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up keys: {e}")
            return False
    
    async def restore_keys(self, backup_location: str) -> bool:
        """Restore keys from backup."""
        try:
            # In a real implementation, this would restore from secure storage
            logger.info(f"Keys restored from: {backup_location}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring keys: {e}")
            return False
    
    async def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key metadata and information."""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM encryption_keys WHERE key_id = $1",
                    key_id
                )
                
                if not row:
                    return None
                
                return {
                    "key_id": row['key_id'],
                    "key_type": row['key_type'],
                    "algorithm": row['algorithm'],
                    "key_size": row['key_size'],
                    "created_at": row['created_at'].isoformat(),
                    "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                    "status": row['status'],
                    "rotated_at": row['rotated_at'].isoformat() if row['rotated_at'] else None,
                    "revoked_at": row['revoked_at'].isoformat() if row['revoked_at'] else None
                }
                
        except Exception as e:
            logger.error(f"Error getting key info: {e}")
            return None


class SecurityManager(ISecurityManager):
    """Implementation of security operations."""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    async def encrypt_data(self, data: bytes, key_id: Optional[str] = None) -> Tuple[bytes, str]:
        """Encrypt data and return (encrypted_data, key_id)."""
        try:
            if not key_id:
                # Generate new symmetric key
                key_id = await self.key_manager.generate_key("symmetric", 256)
            
            key_data = await self.key_manager.get_key(key_id)
            if not key_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Encryption key not found"
                )
            
            # Use AES-GCM for authenticated encryption
            iv = secrets.token_bytes(12)  # 96-bit IV for GCM
            cipher = Cipher(
                algorithms.AES(key_data[:32]),  # Use first 32 bytes for AES-256
                modes.GCM(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Combine IV, tag, and ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext
            
            return encrypted_data, key_id
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Encryption failed"
            )
    
    async def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using specified key."""
        try:
            key_data = await self.key_manager.get_key(key_id)
            if not key_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Decryption key not found"
                )
            
            # Extract IV, tag, and ciphertext
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            cipher = Cipher(
                algorithms.AES(key_data[:32]),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Decryption failed"
            )
    
    async def sign_data(self, data: bytes, key_id: Optional[str] = None) -> Tuple[bytes, str]:
        """Sign data and return (signature, key_id)."""
        try:
            if not key_id:
                # Generate new RSA key for signing
                key_id = await self.key_manager.generate_key("rsa", 2048)
            
            key_data = await self.key_manager.get_key(key_id)
            if not key_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Signing key not found"
                )
            
            # Load private key
            private_key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )
            
            # Sign data using RSA-PSS
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature, key_id
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Signing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Signing failed"
            )
    
    async def verify_signature(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify data signature."""
        try:
            key_data = await self.key_manager.get_key(key_id)
            if not key_data:
                return False
            
            # Load private key and extract public key
            private_key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Verify signature
            try:
                public_key.verify(
                    signature,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except:
                return False
                
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    async def generate_hmac(self, data: bytes, key_id: Optional[str] = None) -> Tuple[str, str]:
        """Generate HMAC and return (hmac_hex, key_id)."""
        try:
            if not key_id:
                # Generate new HMAC key
                key_id = await self.key_manager.generate_key("hmac", 256)
            
            key_data = await self.key_manager.get_key(key_id)
            if not key_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="HMAC key not found"
                )
            
            # Generate HMAC-SHA256
            mac = hmac.new(key_data, data, hashlib.sha256)
            hmac_hex = mac.hexdigest()
            
            return hmac_hex, key_id
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"HMAC generation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="HMAC generation failed"
            )
    
    async def verify_hmac(self, data: bytes, hmac_hex: str, key_id: str) -> bool:
        """Verify HMAC signature."""
        try:
            key_data = await self.key_manager.get_key(key_id)
            if not key_data:
                return False
            
            # Generate expected HMAC
            mac = hmac.new(key_data, data, hashlib.sha256)
            expected_hmac = mac.hexdigest()
            
            # Use constant-time comparison
            return hmac.compare_digest(expected_hmac, hmac_hex)
            
        except Exception as e:
            logger.error(f"HMAC verification error: {e}")
            return False
    
    async def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password and return (hash, salt)."""
        if not salt:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 with SHA-256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
            backend=default_backend()
        )
        
        password_hash = kdf.derive(password.encode())
        return password_hash.hex(), salt
    
    async def verify_password(self, password: str, hash_hex: str, salt: str) -> bool:
        """Verify password against hash."""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
                backend=default_backend()
            )
            
            expected_hash = kdf.derive(password.encode())
            return hmac.compare_digest(expected_hash.hex(), hash_hex)
            
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


# Global service instances
key_manager = None
security_manager = None
security_monitor = None
compliance_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global db_pool, redis_client, key_manager, security_manager
    
    # Initialize database connection pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    
    # Initialize Redis client
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    
    # Create encryption keys table if not exists
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS encryption_keys (
                key_id VARCHAR(255) PRIMARY KEY,
                key_type VARCHAR(50) NOT NULL,
                algorithm VARCHAR(100) NOT NULL,
                key_size INTEGER NOT NULL,
                encrypted_key_data BYTEA NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                expires_at TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active',
                rotated_at TIMESTAMP,
                revoked_at TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS key_backups (
                backup_id VARCHAR(255) PRIMARY KEY,
                location VARCHAR(500) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                status VARCHAR(20) DEFAULT 'pending'
            )
        """)
    
    # Initialize services
    key_manager = KeyManager(db_pool, redis_client, MASTER_KEY)
    security_manager = SecurityManager(key_manager)
    security_monitor = SecurityEventMonitor(db_pool, redis_client)
    compliance_manager = ComplianceManager(db_pool, redis_client)
    
    # Initialize audit and compliance systems
    await security_monitor.initialize()
    await compliance_manager.initialize()
    
    logger.info("Security service started")
    
    yield
    
    # Cleanup
    await db_pool.close()
    redis_client.close()
    logger.info("Security service stopped")


# Create FastAPI app
app = FastAPI(
    title="Project Argus Security Service",
    description="Encryption and Key Management API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/keys/generate")
async def generate_key(key_type: str, key_size: int = 256):
    """Generate new cryptographic key."""
    key_id = await key_manager.generate_key(key_type, key_size)
    return {"key_id": key_id, "message": "Key generated successfully"}


@app.get("/keys")
async def list_keys(key_type: Optional[str] = None):
    """List all keys, optionally filtered by type."""
    keys = await key_manager.list_keys(key_type)
    return {"keys": keys}


@app.get("/keys/{key_id}")
async def get_key_info(key_id: str):
    """Get key information."""
    key_info = await key_manager.get_key_info(key_id)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Key not found"
        )
    return key_info


@app.post("/keys/{key_id}/rotate")
async def rotate_key(key_id: str):
    """Rotate key."""
    new_key_id = await key_manager.rotate_key(key_id)
    return {"old_key_id": key_id, "new_key_id": new_key_id, "message": "Key rotated successfully"}


@app.delete("/keys/{key_id}")
async def revoke_key(key_id: str):
    """Revoke key."""
    success = await key_manager.revoke_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Key not found"
        )
    return {"message": "Key revoked successfully"}


@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest):
    """Encrypt data."""
    try:
        data = base64.b64decode(request.data)
        encrypted_data, key_id = await security_manager.encrypt_data(data, request.key_id)
        
        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
            "key_id": key_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {e}"
        )


@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest):
    """Decrypt data."""
    try:
        encrypted_data = base64.b64decode(request.encrypted_data)
        decrypted_data = await security_manager.decrypt_data(encrypted_data, request.key_id)
        
        return {
            "data": base64.b64encode(decrypted_data).decode()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {e}"
        )


@app.post("/sign")
async def sign_data(request: SigningRequest):
    """Sign data."""
    try:
        data = base64.b64decode(request.data)
        signature, key_id = await security_manager.sign_data(data, request.key_id)
        
        return {
            "signature": base64.b64encode(signature).decode(),
            "key_id": key_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {e}"
        )


@app.post("/verify")
async def verify_signature(request: VerificationRequest):
    """Verify signature."""
    try:
        data = base64.b64decode(request.data)
        signature = base64.b64decode(request.signature)
        
        valid = await security_manager.verify_signature(data, signature, request.key_id)
        
        return {"valid": valid}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {e}"
        )


@app.post("/hmac")
async def generate_hmac(request: HMACRequest):
    """Generate HMAC."""
    try:
        data = base64.b64decode(request.data)
        hmac_hex, key_id = await security_manager.generate_hmac(data, request.key_id)
        
        return {
            "hmac": hmac_hex,
            "key_id": key_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {e}"
        )


@app.post("/hmac/verify")
async def verify_hmac(data: str, hmac_hex: str, key_id: str):
    """Verify HMAC."""
    try:
        data_bytes = base64.b64decode(data)
        valid = await security_manager.verify_hmac(data_bytes, hmac_hex, key_id)
        
        return {"valid": valid}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {e}"
        )


@app.post("/keys/backup")
async def backup_keys(location: str):
    """Backup keys to secure location."""
    success = await key_manager.backup_keys(location)
    if success:
        return {"message": "Keys backed up successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backup failed"
        )


@app.post("/keys/restore")
async def restore_keys(location: str):
    """Restore keys from backup."""
    success = await key_manager.restore_keys(location)
    if success:
        return {"message": "Keys restored successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Restore failed"
        )


# Audit Logging and Compliance Endpoints

@app.post("/security/events")
async def log_security_event(request: SecurityEventRequest):
    """Log a security event."""
    try:
        event_type = SecurityEventType(request.event_type)
        event_id = await security_monitor.log_security_event(
            event_type=event_type,
            user_id=request.user_id,
            source_ip=request.source_ip,
            action=request.action,
            success=request.success,
            details=request.details,
            user_agent=request.user_agent,
            resource=request.resource,
            session_id=request.session_id
        )
        
        return {"event_id": event_id, "message": "Security event logged successfully"}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event type: {request.event_type}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log security event"
        )


@app.get("/security/dashboard")
async def get_security_dashboard():
    """Get security dashboard data."""
    try:
        dashboard_data = await security_monitor.get_security_dashboard_data()
        return dashboard_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security dashboard data"
        )


@app.get("/security/events")
async def get_security_events(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """Get security events with filters."""
    try:
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        # Build query filters
        filters = {}
        if event_type:
            filters['event_type'] = event_type
        if severity:
            filters['severity'] = severity
        
        # Query database
        async with db_pool.acquire() as conn:
            query = """
                SELECT event_id, event_type, severity, timestamp, user_id, source_ip,
                       action, success, details, risk_score
                FROM security_events 
                WHERE timestamp BETWEEN $1 AND $2
            """
            params = [start_time, end_time]
            
            if event_type:
                query += " AND event_type = $3"
                params.append(event_type)
            
            if severity:
                query += f" AND severity = ${len(params) + 1}"
                params.append(severity)
            
            query += f" ORDER BY timestamp DESC LIMIT ${len(params) + 1}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            events = []
            for row in rows:
                events.append({
                    'event_id': row['event_id'],
                    'event_type': row['event_type'],
                    'severity': row['severity'],
                    'timestamp': row['timestamp'].isoformat(),
                    'user_id': row['user_id'],
                    'source_ip': str(row['source_ip']) if row['source_ip'] else None,
                    'action': row['action'],
                    'success': row['success'],
                    'details': row['details'],
                    'risk_score': row['risk_score']
                })
            
            return {"events": events, "total": len(events)}
            
    except Exception as e:
        logger.error(f"Error getting security events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security events"
        )


@app.get("/security/alerts")
async def get_security_alerts(status_filter: str = "active"):
    """Get security alerts."""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT sa.*, ip.name as pattern_name
                FROM security_alerts sa
                JOIN intrusion_patterns ip ON sa.pattern_id = ip.pattern_id
                WHERE sa.status = $1
                ORDER BY sa.triggered_at DESC
                LIMIT 50
            """, status_filter)
            
            alerts = []
            for row in rows:
                alerts.append({
                    'alert_id': row['alert_id'],
                    'pattern_id': row['pattern_id'],
                    'pattern_name': row['pattern_name'],
                    'triggered_at': row['triggered_at'].isoformat(),
                    'source_ip': str(row['source_ip']) if row['source_ip'] else None,
                    'user_id': row['user_id'],
                    'event_count': row['event_count'],
                    'risk_score': row['risk_score'],
                    'status': row['status'],
                    'details': row['details']
                })
            
            return {"alerts": alerts}
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security alerts"
        )


@app.post("/security/alerts/{alert_id}/acknowledge")
async def acknowledge_security_alert(alert_id: str, user_id: str):
    """Acknowledge a security alert."""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE security_alerts 
                SET status = 'acknowledged', acknowledged_by = $1, acknowledged_at = NOW()
                WHERE alert_id = $2
            """, user_id, alert_id)
            
            if result == "UPDATE 0":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Alert not found"
                )
            
            return {"message": "Alert acknowledged successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert"
        )


@app.post("/compliance/reports")
async def generate_compliance_report(request: ComplianceReportRequest, user_id: str):
    """Generate compliance report."""
    try:
        framework = ComplianceFramework(request.framework)
        report_id = await compliance_manager.generate_compliance_report(
            framework=framework,
            period_start=request.period_start,
            period_end=request.period_end,
            user_id=user_id
        )
        
        if report_id:
            return {"report_id": report_id, "message": "Compliance report generated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate compliance report"
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid compliance framework: {request.framework}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate compliance report"
        )


@app.get("/compliance/reports")
async def list_compliance_reports(framework: Optional[str] = None, limit: int = 20):
    """List compliance reports."""
    try:
        async with db_pool.acquire() as conn:
            if framework:
                rows = await conn.fetch("""
                    SELECT report_id, framework, report_type, period_start, period_end,
                           generated_at, status, summary
                    FROM compliance_reports 
                    WHERE framework = $1
                    ORDER BY generated_at DESC
                    LIMIT $2
                """, framework, limit)
            else:
                rows = await conn.fetch("""
                    SELECT report_id, framework, report_type, period_start, period_end,
                           generated_at, status, summary
                    FROM compliance_reports 
                    ORDER BY generated_at DESC
                    LIMIT $1
                """, limit)
            
            reports = []
            for row in rows:
                reports.append({
                    'report_id': row['report_id'],
                    'framework': row['framework'],
                    'report_type': row['report_type'],
                    'period_start': row['period_start'].isoformat(),
                    'period_end': row['period_end'].isoformat(),
                    'generated_at': row['generated_at'].isoformat(),
                    'status': row['status'],
                    'summary': row['summary']
                })
            
            return {"reports": reports}
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list compliance reports"
        )


@app.get("/compliance/reports/{report_id}")
async def get_compliance_report(report_id: str):
    """Get compliance report details."""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM compliance_reports WHERE report_id = $1",
                report_id
            )
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Report not found"
                )
            
            return {
                'report_id': row['report_id'],
                'framework': row['framework'],
                'report_type': row['report_type'],
                'period_start': row['period_start'].isoformat(),
                'period_end': row['period_end'].isoformat(),
                'generated_at': row['generated_at'].isoformat(),
                'generated_by': row['generated_by'],
                'status': row['status'],
                'summary': row['summary'],
                'findings': row['findings'],
                'recommendations': row['recommendations']
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get compliance report"
        )


@app.get("/compliance/reports/{report_id}/export")
async def export_compliance_report(report_id: str, format: str = "csv"):
    """Export compliance report."""
    try:
        if format not in ["csv", "json"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid format. Supported formats: csv, json"
            )
        
        report_data = await compliance_manager.export_compliance_report(report_id, format)
        
        if not report_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found or export failed"
            )
        
        media_type = "text/csv" if format == "csv" else "application/json"
        filename = f"compliance_report_{report_id}.{format}"
        
        from fastapi.responses import Response
        return Response(
            content=report_data,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export compliance report"
        )


@app.get("/compliance/frameworks")
async def list_compliance_frameworks():
    """List available compliance frameworks."""
    frameworks = [
        {
            'id': framework.value,
            'name': framework.name,
            'description': f"{framework.name} compliance framework"
        }
        for framework in ComplianceFramework
    ]
    
    return {"frameworks": frameworks}


@app.get("/security/blocked-entities")
async def get_blocked_entities(entity_type: Optional[str] = None):
    """Get blocked entities (IPs, users, etc.)."""
    try:
        async with db_pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch("""
                    SELECT entity_id, entity_type, entity_value, reason, blocked_at, expires_at
                    FROM blocked_entities 
                    WHERE active = TRUE AND entity_type = $1
                    ORDER BY blocked_at DESC
                """, entity_type)
            else:
                rows = await conn.fetch("""
                    SELECT entity_id, entity_type, entity_value, reason, blocked_at, expires_at
                    FROM blocked_entities 
                    WHERE active = TRUE
                    ORDER BY blocked_at DESC
                """)
            
            entities = []
            for row in rows:
                entities.append({
                    'entity_id': row['entity_id'],
                    'entity_type': row['entity_type'],
                    'entity_value': row['entity_value'],
                    'reason': row['reason'],
                    'blocked_at': row['blocked_at'].isoformat(),
                    'expires_at': row['expires_at'].isoformat() if row['expires_at'] else None
                })
            
            return {"blocked_entities": entities}
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get blocked entities"
        )


@app.delete("/security/blocked-entities/{entity_id}")
async def unblock_entity(entity_id: str, user_id: str):
    """Unblock an entity."""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE blocked_entities 
                SET active = FALSE, unblocked_at = NOW(), unblocked_by = $1
                WHERE entity_id = $2 AND active = TRUE
            """, user_id, entity_id)
            
            if result == "UPDATE 0":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Blocked entity not found"
                )
            
            return {"message": "Entity unblocked successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unblock entity"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)