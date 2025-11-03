"""
Security and access control interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from ..models import User, Permission, UserRole


class ISecurityManager(ABC):
    """Interface for security operations."""
    
    @abstractmethod
    def encrypt_data(self, data: bytes, key_id: Optional[str] = None) -> Tuple[bytes, str]:
        """Encrypt data and return (encrypted_data, key_id)."""
        pass
    
    @abstractmethod
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using specified key."""
        pass
    
    @abstractmethod
    def sign_data(self, data: bytes, key_id: Optional[str] = None) -> Tuple[bytes, str]:
        """Sign data and return (signature, key_id)."""
        pass
    
    @abstractmethod
    def verify_signature(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify data signature."""
        pass
    
    @abstractmethod
    def generate_hmac(self, data: bytes, key_id: Optional[str] = None) -> Tuple[str, str]:
        """Generate HMAC and return (hmac_hex, key_id)."""
        pass
    
    @abstractmethod
    def verify_hmac(self, data: bytes, hmac_hex: str, key_id: str) -> bool:
        """Verify HMAC signature."""
        pass
    
    @abstractmethod
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password and return (hash, salt)."""
        pass
    
    @abstractmethod
    def verify_password(self, password: str, hash: str, salt: str) -> bool:
        """Verify password against hash."""
        pass


class IKeyManager(ABC):
    """Interface for cryptographic key management."""
    
    @abstractmethod
    def generate_key(self, key_type: str, key_size: int = 256) -> str:
        """Generate new cryptographic key and return key ID."""
        pass
    
    @abstractmethod
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get key by ID."""
        pass
    
    @abstractmethod
    def rotate_key(self, key_id: str) -> str:
        """Rotate key and return new key ID."""
        pass
    
    @abstractmethod
    def revoke_key(self, key_id: str) -> bool:
        """Revoke key (mark as invalid)."""
        pass
    
    @abstractmethod
    def list_keys(self, key_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all keys, optionally filtered by type."""
        pass
    
    @abstractmethod
    def backup_keys(self, backup_location: str) -> bool:
        """Backup keys to secure location."""
        pass
    
    @abstractmethod
    def restore_keys(self, backup_location: str) -> bool:
        """Restore keys from backup."""
        pass
    
    @abstractmethod
    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key metadata and information."""
        pass


class IAccessController(ABC):
    """Interface for access control and authorization."""
    
    @abstractmethod
    def authenticate_user(self, username: str, password: str, 
                         mfa_token: Optional[str] = None) -> Optional[User]:
        """Authenticate user and return user object if successful."""
        pass
    
    @abstractmethod
    def authorize_action(self, user: User, permission: Permission, 
                        resource_id: Optional[str] = None) -> bool:
        """Check if user is authorized to perform action."""
        pass
    
    @abstractmethod
    def create_session(self, user: User, client_info: Dict[str, Any]) -> str:
        """Create user session and return session token."""
        pass
    
    @abstractmethod
    def validate_session(self, session_token: str) -> Optional[User]:
        """Validate session token and return user if valid."""
        pass
    
    @abstractmethod
    def revoke_session(self, session_token: str) -> bool:
        """Revoke user session."""
        pass
    
    @abstractmethod
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all permissions for user."""
        pass
    
    @abstractmethod
    def assign_role(self, user_id: str, role: UserRole) -> bool:
        """Assign role to user."""
        pass
    
    @abstractmethod
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant specific permission to user."""
        pass
    
    @abstractmethod
    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Revoke specific permission from user."""
        pass
    
    @abstractmethod
    def audit_access(self, user_id: str, action: str, resource: str, 
                    success: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Log access attempt for audit."""
        pass
    
    @abstractmethod
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active sessions, optionally filtered by user."""
        pass
    
    @abstractmethod
    def enforce_password_policy(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against policy, return (valid, error_messages)."""
        pass