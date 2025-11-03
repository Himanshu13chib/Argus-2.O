"""
Storage and caching interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic
from datetime import datetime

T = TypeVar('T')


class IRepository(ABC, Generic[T]):
    """Generic repository interface for data persistence."""
    
    @abstractmethod
    def create(self, entity: T) -> str:
        """Create new entity and return ID."""
        pass
    
    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def update(self, entity_id: str, updates: Dict[str, Any]) -> Optional[T]:
        """Update entity and return updated version."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    def find_by_criteria(self, criteria: Dict[str, Any], 
                        limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """Find entities matching criteria."""
        pass
    
    @abstractmethod
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching criteria."""
        pass
    
    @abstractmethod
    def exists(self, entity_id: str) -> bool:
        """Check if entity exists."""
        pass


class ICacheManager(ABC):
    """Interface for caching operations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries, optionally matching pattern."""
        pass
    
    @abstractmethod
    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key in seconds."""
        pass
    
    @abstractmethod
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend TTL for existing key."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass
    
    @abstractmethod
    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        pass
    
    @abstractmethod
    def set_multiple(self, key_value_pairs: Dict[str, Any], 
                    ttl_seconds: Optional[int] = None) -> None:
        """Set multiple key-value pairs in cache."""
        pass