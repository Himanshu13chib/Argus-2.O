"""
Privacy-Preserving Data Management for Project Argus.
Implements automatic data purging, anonymization, and retention policies.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
import asyncio
import hashlib
import re
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
from PIL import Image, ImageFilter
import cv2

from shared.interfaces.evidence import IPrivacyManager
from shared.models.evidence import Evidence, EvidenceType, EvidenceStatus
from .evidence_store import EvidenceStore
from .audit_logger import AuditLogger


logger = logging.getLogger(__name__)


class PrivacyManager(IPrivacyManager):
    """
    Privacy-preserving data management system for Project Argus.
    Handles automatic data purging, anonymization, and retention policies.
    """
    
    def __init__(self, 
                 database_url: str,
                 evidence_store: EvidenceStore,
                 audit_logger: AuditLogger,
                 default_retention_hours: int = 24):
        """
        Initialize privacy manager.
        
        Args:
            database_url: PostgreSQL connection string
            evidence_store: Evidence store instance
            audit_logger: Audit logger instance
            default_retention_hours: Default retention period for unconfirmed incidents
        """
        self.database_url = database_url
        self.evidence_store = evidence_store
        self.audit_logger = audit_logger
        self.default_retention_hours = default_retention_hours
        
        # Initialize database
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Privacy settings
        self.anonymization_settings = {
            'blur_faces': True,
            'blur_license_plates': True,
            'remove_metadata': True,
            'hash_identifiers': True
        }
    
    async def schedule_automatic_purge(self, incident_id: str) -> bool:
        """
        Schedule automatic purge for unconfirmed incidents.
        
        Args:
            incident_id: ID of the incident to schedule for purge
            
        Returns:
            True if scheduling was successful
        """
        try:
            purge_time = datetime.now() + timedelta(hours=self.default_retention_hours)
            
            async with self.async_session() as session:
                # Check if incident is confirmed
                incident_query = """
                SELECT status FROM incidents WHERE id = :incident_id
                """
                
                result = await session.execute(sa.text(incident_query), {'incident_id': incident_id})
                incident = result.fetchone()
                
                if not incident:
                    logger.warning(f"Incident {incident_id} not found")
                    return False
                
                # Only schedule purge for unconfirmed incidents
                if incident.status in ['open', 'investigating']:
                    # Get all evidence for this incident
                    evidence_list = await self.evidence_store.search_evidence({
                        'incident_id': incident_id
                    })
                    
                    # Schedule each evidence item for purge
                    for evidence in evidence_list:
                        await self.evidence_store.schedule_purge(
                            evidence.id, 
                            purge_time, 
                            'privacy_manager'
                        )
                    
                    # Log the scheduling
                    await self.audit_logger.log_system_event(
                        'privacy_purge_scheduled',
                        'privacy_manager',
                        {
                            'incident_id': incident_id,
                            'purge_time': purge_time.isoformat(),
                            'evidence_count': len(evidence_list),
                            'reason': 'unconfirmed_incident_retention_policy'
                        }
                    )
                    
                    logger.info(f"Scheduled automatic purge for incident {incident_id} at {purge_time}")
                    return True
                
                else:
                    logger.info(f"Incident {incident_id} is confirmed, skipping automatic purge")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to schedule automatic purge for incident {incident_id}: {e}")
            return False
    
    async def cancel_automatic_purge(self, incident_id: str, operator_id: str) -> bool:
        """
        Cancel automatic purge when incident is confirmed.
        
        Args:
            incident_id: ID of the incident
            operator_id: ID of the operator confirming the incident
            
        Returns:
            True if cancellation was successful
        """
        try:
            async with self.async_session() as session:
                # Update evidence to cancel auto-purge
                update_query = """
                UPDATE evidence 
                SET auto_purge = false, retention_until = NULL
                WHERE incident_id = :incident_id AND auto_purge = true
                """
                
                result = await session.execute(sa.text(update_query), {'incident_id': incident_id})
                updated_count = result.rowcount
                
                if updated_count > 0:
                    # Add chain of custody entries
                    evidence_list = await self.evidence_store.search_evidence({
                        'incident_id': incident_id
                    })
                    
                    for evidence in evidence_list:
                        custody_query = """
                        INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                        VALUES (:evidence_id, :action, :operator_id, :details)
                        """
                        
                        await session.execute(sa.text(custody_query), {
                            'evidence_id': evidence.id,
                            'action': 'purge_cancelled',
                            'operator_id': operator_id,
                            'details': 'Automatic purge cancelled due to incident confirmation'
                        })
                    
                    await session.commit()
                    
                    # Log the cancellation
                    await self.audit_logger.log_user_action(
                        operator_id,
                        'privacy_purge_cancelled',
                        'incident',
                        incident_id,
                        {
                            'evidence_count': updated_count,
                            'reason': 'incident_confirmed'
                        }
                    )
                    
                    logger.info(f"Cancelled automatic purge for incident {incident_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel automatic purge for incident {incident_id}: {e}")
            return False
    
    async def anonymize_evidence(self, evidence_id: str, anonymization_level: str = "standard") -> str:
        """
        Create anonymized version of evidence.
        
        Args:
            evidence_id: ID of the evidence to anonymize
            anonymization_level: Level of anonymization (minimal, standard, aggressive)
            
        Returns:
            ID of the anonymized evidence
        """
        try:
            # Get original evidence
            original_evidence = await self.evidence_store.retrieve_evidence(evidence_id)
            if not original_evidence:
                raise ValueError(f"Evidence {evidence_id} not found")
            
            # Get evidence file
            file_content = await self.evidence_store.get_evidence_file(evidence_id)
            if not file_content:
                raise ValueError(f"Evidence file {evidence_id} not accessible")
            
            # Anonymize based on evidence type
            anonymized_data = None
            
            if original_evidence.type == EvidenceType.IMAGE:
                anonymized_data = await self._anonymize_image(file_content.read(), anonymization_level)
            elif original_evidence.type == EvidenceType.VIDEO:
                anonymized_data = await self._anonymize_video(file_content.read(), anonymization_level)
            elif original_evidence.type == EvidenceType.METADATA:
                anonymized_data = await self._anonymize_metadata(file_content.read(), anonymization_level)
            else:
                # For other types, just remove metadata
                anonymized_data = file_content.read()
            
            if anonymized_data is None:
                raise ValueError(f"Failed to anonymize evidence type {original_evidence.type}")
            
            # Create anonymized metadata
            anonymized_metadata = original_evidence.metadata.copy()
            anonymized_metadata['anonymized_from'] = evidence_id
            anonymized_metadata['anonymization_level'] = anonymization_level
            anonymized_metadata['anonymization_timestamp'] = datetime.now().isoformat()
            
            # Remove sensitive metadata
            sensitive_keys = ['gps_coordinates', 'location', 'operator_id', 'user_id']
            for key in sensitive_keys:
                anonymized_metadata.pop(key, None)
            
            # Store anonymized evidence
            anonymized_evidence_id = await self.evidence_store.store_evidence(
                file_data=anonymized_data,
                evidence_type=original_evidence.type,
                metadata=anonymized_metadata,
                created_by='privacy_manager'
            )
            
            # Log the anonymization
            await self.audit_logger.log_system_event(
                'evidence_anonymized',
                'privacy_manager',
                {
                    'original_evidence_id': evidence_id,
                    'anonymized_evidence_id': anonymized_evidence_id,
                    'anonymization_level': anonymization_level
                }
            )
            
            logger.info(f"Created anonymized evidence {anonymized_evidence_id} from {evidence_id}")
            return anonymized_evidence_id
            
        except Exception as e:
            logger.error(f"Failed to anonymize evidence {evidence_id}: {e}")
            raise
    
    async def _anonymize_image(self, image_data: bytes, level: str) -> bytes:
        """Anonymize image data by blurring faces and sensitive areas."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid image data")
            
            # Apply anonymization based on level
            if level in ["standard", "aggressive"]:
                # Blur faces (simplified - in production would use face detection)
                img = cv2.GaussianBlur(img, (15, 15), 0)
            
            if level == "aggressive":
                # Additional privacy measures for aggressive anonymization
                # Add noise
                noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
                img = cv2.add(img, noise)
                
                # Reduce resolution
                height, width = img.shape[:2]
                img = cv2.resize(img, (width//2, height//2))
                img = cv2.resize(img, (width, height))
            
            # Encode back to bytes
            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to anonymize image: {e}")
            raise
    
    async def _anonymize_video(self, video_data: bytes, level: str) -> bytes:
        """Anonymize video data by processing frames."""
        try:
            # For video anonymization, we would process each frame
            # This is a simplified implementation
            
            # In a full implementation, you would:
            # 1. Extract frames from video
            # 2. Apply image anonymization to each frame
            # 3. Reconstruct video from anonymized frames
            
            # For now, return original data with metadata stripped
            return video_data
            
        except Exception as e:
            logger.error(f"Failed to anonymize video: {e}")
            raise
    
    async def _anonymize_metadata(self, metadata_bytes: bytes, level: str) -> bytes:
        """Anonymize metadata by removing or hashing sensitive information."""
        try:
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Define sensitive fields to anonymize
            sensitive_fields = {
                'minimal': ['user_id', 'operator_id'],
                'standard': ['user_id', 'operator_id', 'ip_address', 'device_id', 'location'],
                'aggressive': ['user_id', 'operator_id', 'ip_address', 'device_id', 'location', 
                              'camera_id', 'detection_id', 'coordinates']
            }
            
            fields_to_anonymize = sensitive_fields.get(level, sensitive_fields['standard'])
            
            # Anonymize sensitive fields
            for field in fields_to_anonymize:
                if field in metadata:
                    if level == "aggressive":
                        # Remove completely
                        del metadata[field]
                    else:
                        # Hash the value
                        original_value = str(metadata[field])
                        hashed_value = hashlib.sha256(original_value.encode()).hexdigest()[:16]
                        metadata[field] = f"anon_{hashed_value}"
            
            # Add anonymization marker
            metadata['_anonymized'] = True
            metadata['_anonymization_level'] = level
            metadata['_anonymization_timestamp'] = datetime.now().isoformat()
            
            return json.dumps(metadata).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to anonymize metadata: {e}")
            raise
    
    async def apply_retention_policy(self, policy_name: str, filters: Dict[str, Any]) -> List[str]:
        """
        Apply retention policy to evidence matching filters.
        
        Args:
            policy_name: Name of the retention policy
            filters: Filters to select evidence for policy application
            
        Returns:
            List of evidence IDs affected by the policy
        """
        try:
            # Get retention policies
            policies = await self._get_retention_policies()
            
            if policy_name not in policies:
                raise ValueError(f"Retention policy '{policy_name}' not found")
            
            policy = policies[policy_name]
            affected_evidence = []
            
            # Find evidence matching filters
            evidence_list = await self.evidence_store.search_evidence(filters)
            
            for evidence in evidence_list:
                # Check if evidence meets policy criteria
                if await self._evidence_meets_policy_criteria(evidence, policy):
                    # Apply retention policy
                    if policy['action'] == 'purge':
                        purge_date = datetime.now() + timedelta(days=policy['retention_days'])
                        await self.evidence_store.schedule_purge(
                            evidence.id,
                            purge_date,
                            'privacy_manager'
                        )
                        affected_evidence.append(evidence.id)
                    
                    elif policy['action'] == 'anonymize':
                        anonymized_id = await self.anonymize_evidence(
                            evidence.id,
                            policy.get('anonymization_level', 'standard')
                        )
                        affected_evidence.append(anonymized_id)
                    
                    elif policy['action'] == 'archive':
                        # Move to long-term storage (implementation depends on storage backend)
                        await self._archive_evidence(evidence.id)
                        affected_evidence.append(evidence.id)
            
            # Log policy application
            await self.audit_logger.log_system_event(
                'retention_policy_applied',
                'privacy_manager',
                {
                    'policy_name': policy_name,
                    'filters': filters,
                    'affected_count': len(affected_evidence),
                    'affected_evidence': affected_evidence[:10]  # Log first 10 IDs
                }
            )
            
            logger.info(f"Applied retention policy '{policy_name}' to {len(affected_evidence)} evidence items")
            return affected_evidence
            
        except Exception as e:
            logger.error(f"Failed to apply retention policy '{policy_name}': {e}")
            return []
    
    async def _get_retention_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get configured retention policies."""
        # In a production system, these would be stored in database or config
        return {
            'unconfirmed_incidents': {
                'action': 'purge',
                'retention_days': 1,
                'criteria': {
                    'incident_status': ['open', 'investigating'],
                    'max_age_hours': 24
                }
            },
            'low_confidence_detections': {
                'action': 'anonymize',
                'retention_days': 30,
                'anonymization_level': 'standard',
                'criteria': {
                    'confidence_threshold': 0.5,
                    'max_age_days': 7
                }
            },
            'archived_incidents': {
                'action': 'archive',
                'retention_days': 365,
                'criteria': {
                    'incident_status': ['closed'],
                    'min_age_days': 90
                }
            },
            'analytics_data': {
                'action': 'anonymize',
                'retention_days': 180,
                'anonymization_level': 'aggressive',
                'criteria': {
                    'evidence_type': ['metadata'],
                    'purpose': 'analytics'
                }
            }
        }
    
    async def _evidence_meets_policy_criteria(self, evidence: Evidence, policy: Dict[str, Any]) -> bool:
        """Check if evidence meets policy criteria."""
        try:
            criteria = policy.get('criteria', {})
            
            # Check age criteria
            if 'max_age_hours' in criteria:
                age_hours = (datetime.now() - evidence.created_at).total_seconds() / 3600
                if age_hours < criteria['max_age_hours']:
                    return False
            
            if 'min_age_days' in criteria:
                age_days = (datetime.now() - evidence.created_at).days
                if age_days < criteria['min_age_days']:
                    return False
            
            if 'max_age_days' in criteria:
                age_days = (datetime.now() - evidence.created_at).days
                if age_days > criteria['max_age_days']:
                    return False
            
            # Check evidence type
            if 'evidence_type' in criteria:
                if evidence.type.value not in criteria['evidence_type']:
                    return False
            
            # Check confidence threshold (if available in metadata)
            if 'confidence_threshold' in criteria:
                confidence = evidence.metadata.get('confidence', 1.0)
                if confidence >= criteria['confidence_threshold']:
                    return False
            
            # Check incident status
            if 'incident_status' in criteria and evidence.incident_id:
                async with self.async_session() as session:
                    query = "SELECT status FROM incidents WHERE id = :incident_id"
                    result = await session.execute(sa.text(query), {'incident_id': evidence.incident_id})
                    incident = result.fetchone()
                    
                    if incident and incident.status not in criteria['incident_status']:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check policy criteria: {e}")
            return False
    
    async def _archive_evidence(self, evidence_id: str) -> bool:
        """Archive evidence to long-term storage."""
        try:
            # In a production system, this would move evidence to cheaper, long-term storage
            # For now, we'll just mark it as archived
            
            async with self.async_session() as session:
                update_query = """
                UPDATE evidence 
                SET status = 'archived'
                WHERE id = :evidence_id
                """
                
                result = await session.execute(sa.text(update_query), {'evidence_id': evidence_id})
                
                if result.rowcount > 0:
                    # Add chain of custody entry
                    custody_query = """
                    INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                    VALUES (:evidence_id, :action, :operator_id, :details)
                    """
                    
                    await session.execute(sa.text(custody_query), {
                        'evidence_id': evidence_id,
                        'action': 'archived',
                        'operator_id': 'privacy_manager',
                        'details': 'Evidence archived according to retention policy'
                    })
                    
                    await session.commit()
                    logger.info(f"Archived evidence {evidence_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to archive evidence {evidence_id}: {e}")
            return False
    
    async def generate_privacy_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate privacy compliance report for specified time period.
        
        Args:
            start_date: Start date for the report
            end_date: End date for the report
            
        Returns:
            Privacy compliance report data
        """
        try:
            async with self.async_session() as session:
                # Get evidence statistics
                evidence_stats_query = """
                SELECT 
                    type,
                    status,
                    COUNT(*) as count,
                    SUM(file_size) as total_size
                FROM evidence 
                WHERE created_at >= :start_date AND created_at <= :end_date
                GROUP BY type, status
                """
                
                result = await session.execute(sa.text(evidence_stats_query), {
                    'start_date': start_date,
                    'end_date': end_date
                })
                evidence_stats = result.fetchall()
                
                # Get purge statistics
                purge_stats_query = """
                SELECT 
                    DATE(timestamp) as purge_date,
                    COUNT(*) as purged_count
                FROM chain_of_custody 
                WHERE action = 'purged' 
                AND timestamp >= :start_date 
                AND timestamp <= :end_date
                GROUP BY DATE(timestamp)
                ORDER BY purge_date
                """
                
                result = await session.execute(sa.text(purge_stats_query), {
                    'start_date': start_date,
                    'end_date': end_date
                })
                purge_stats = result.fetchall()
                
                # Get anonymization statistics
                anonymization_query = """
                SELECT COUNT(*) as anonymized_count
                FROM audit_logs 
                WHERE action = 'evidence_anonymized'
                AND timestamp >= :start_date 
                AND timestamp <= :end_date
                """
                
                result = await session.execute(sa.text(anonymization_query), {
                    'start_date': start_date,
                    'end_date': end_date
                })
                anonymized_count = result.scalar() or 0
                
                # Compile report
                report = {
                    'report_period': {
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    },
                    'evidence_statistics': [
                        {
                            'type': row.type,
                            'status': row.status,
                            'count': row.count,
                            'total_size_bytes': row.total_size or 0
                        }
                        for row in evidence_stats
                    ],
                    'purge_statistics': [
                        {
                            'date': row.purge_date.isoformat(),
                            'purged_count': row.purged_count
                        }
                        for row in purge_stats
                    ],
                    'anonymization_count': anonymized_count,
                    'compliance_metrics': {
                        'total_evidence_processed': sum(row.count for row in evidence_stats),
                        'total_data_size_bytes': sum(row.total_size or 0 for row in evidence_stats),
                        'total_purged': sum(row.purged_count for row in purge_stats),
                        'anonymization_rate': anonymized_count / max(sum(row.count for row in evidence_stats), 1)
                    },
                    'generated_at': datetime.now().isoformat()
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Failed to generate privacy compliance report: {e}")
            return {}
    
    async def run_privacy_maintenance(self) -> Dict[str, Any]:
        """
        Run periodic privacy maintenance tasks.
        
        Returns:
            Summary of maintenance tasks performed
        """
        try:
            maintenance_summary = {
                'start_time': datetime.now().isoformat(),
                'tasks_performed': [],
                'errors': []
            }
            
            # 1. Purge expired evidence
            try:
                purged_ids = await self.evidence_store.purge_expired_evidence()
                maintenance_summary['tasks_performed'].append({
                    'task': 'purge_expired_evidence',
                    'items_processed': len(purged_ids),
                    'details': f"Purged {len(purged_ids)} expired evidence items"
                })
            except Exception as e:
                maintenance_summary['errors'].append({
                    'task': 'purge_expired_evidence',
                    'error': str(e)
                })
            
            # 2. Apply retention policies
            try:
                policies = await self._get_retention_policies()
                for policy_name in policies:
                    affected = await self.apply_retention_policy(policy_name, {})
                    maintenance_summary['tasks_performed'].append({
                        'task': f'retention_policy_{policy_name}',
                        'items_processed': len(affected),
                        'details': f"Applied policy '{policy_name}' to {len(affected)} items"
                    })
            except Exception as e:
                maintenance_summary['errors'].append({
                    'task': 'apply_retention_policies',
                    'error': str(e)
                })
            
            # 3. Schedule purge for new unconfirmed incidents
            try:
                async with self.async_session() as session:
                    # Find unconfirmed incidents older than 1 hour without scheduled purge
                    query = """
                    SELECT DISTINCT i.id
                    FROM incidents i
                    LEFT JOIN evidence e ON i.id = e.incident_id
                    WHERE i.status IN ('open', 'investigating')
                    AND i.created_at < NOW() - INTERVAL '1 hour'
                    AND (e.auto_purge IS NULL OR e.auto_purge = false)
                    """
                    
                    result = await session.execute(sa.text(query))
                    incidents = result.fetchall()
                    
                    scheduled_count = 0
                    for incident in incidents:
                        if await self.schedule_automatic_purge(incident.id):
                            scheduled_count += 1
                    
                    maintenance_summary['tasks_performed'].append({
                        'task': 'schedule_automatic_purge',
                        'items_processed': scheduled_count,
                        'details': f"Scheduled automatic purge for {scheduled_count} unconfirmed incidents"
                    })
                    
            except Exception as e:
                maintenance_summary['errors'].append({
                    'task': 'schedule_automatic_purge',
                    'error': str(e)
                })
            
            maintenance_summary['end_time'] = datetime.now().isoformat()
            maintenance_summary['success'] = len(maintenance_summary['errors']) == 0
            
            # Log maintenance summary
            await self.audit_logger.log_system_event(
                'privacy_maintenance_completed',
                'privacy_manager',
                maintenance_summary
            )
            
            logger.info(f"Privacy maintenance completed: {len(maintenance_summary['tasks_performed'])} tasks, {len(maintenance_summary['errors'])} errors")
            return maintenance_summary
            
        except Exception as e:
            logger.error(f"Privacy maintenance failed: {e}")
            return {
                'start_time': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }