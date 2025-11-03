"""
Comprehensive Audit Logger Implementation for Project Argus.
Provides detailed audit trails for all system operations.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import csv
import io

from shared.interfaces.evidence import IAuditLogger


logger = logging.getLogger(__name__)


class AuditLogger(IAuditLogger):
    """
    Comprehensive audit logging system for Project Argus.
    Records all user actions, system events, and security events.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize audit logger with database connection.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def log_user_action(self, user_id: str, action: str, resource_type: str, 
                            resource_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log user action for audit trail."""
        try:
            async with self.async_session() as session:
                query = """
                INSERT INTO audit_logs (user_id, action, resource_type, resource_id, details)
                VALUES (:user_id, :action, :resource_type, :resource_id, :details)
                """
                
                await session.execute(sa.text(query), {
                    'user_id': user_id,
                    'action': action,
                    'resource_type': resource_type,
                    'resource_id': resource_id,
                    'details': json.dumps(details or {})
                })
                
                await session.commit()
                logger.debug(f"Logged user action: {user_id} - {action} on {resource_type}:{resource_id}")
                
        except Exception as e:
            logger.error(f"Failed to log user action: {e}")
    
    async def log_system_event(self, event_type: str, component: str, 
                             details: Dict[str, Any]) -> None:
        """Log system event for audit trail."""
        try:
            async with self.async_session() as session:
                query = """
                INSERT INTO audit_logs (action, resource_type, details)
                VALUES (:action, :resource_type, :details)
                """
                
                await session.execute(sa.text(query), {
                    'action': event_type,
                    'resource_type': f"system:{component}",
                    'details': json.dumps(details)
                })
                
                await session.commit()
                logger.debug(f"Logged system event: {event_type} in {component}")
                
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
    
    async def log_security_event(self, event_type: str, user_id: Optional[str], 
                               source_ip: str, details: Dict[str, Any]) -> None:
        """Log security-related event."""
        try:
            async with self.async_session() as session:
                query = """
                INSERT INTO audit_logs (user_id, action, resource_type, source_ip, details)
                VALUES (:user_id, :action, :resource_type, :source_ip, :details)
                """
                
                await session.execute(sa.text(query), {
                    'user_id': user_id,
                    'action': event_type,
                    'resource_type': 'security',
                    'source_ip': source_ip,
                    'details': json.dumps(details)
                })
                
                await session.commit()
                logger.info(f"Logged security event: {event_type} from {source_ip}")
                
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    async def log_evidence_access(self, evidence_id: str, user_id: str, 
                                action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log evidence access for chain of custody."""
        try:
            # Log in audit logs
            await self.log_user_action(
                user_id=user_id,
                action=f"evidence_{action}",
                resource_type="evidence",
                resource_id=evidence_id,
                details=details
            )
            
            # Also log in chain of custody if it's an access action
            if action in ['viewed', 'downloaded', 'exported']:
                async with self.async_session() as session:
                    custody_query = """
                    INSERT INTO chain_of_custody (evidence_id, action, operator_id, details)
                    VALUES (:evidence_id, :action, :operator_id, :details)
                    """
                    
                    await session.execute(sa.text(custody_query), {
                        'evidence_id': evidence_id,
                        'action': action,
                        'operator_id': user_id,
                        'details': json.dumps(details or {})
                    })
                    
                    await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log evidence access: {e}")
    
    async def get_audit_logs(self, start_time: datetime, end_time: datetime, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve audit logs for time range with optional filters."""
        try:
            conditions = ["timestamp >= :start_time", "timestamp <= :end_time"]
            params = {'start_time': start_time, 'end_time': end_time}
            
            if filters:
                if 'user_id' in filters:
                    conditions.append("user_id = :user_id")
                    params['user_id'] = filters['user_id']
                
                if 'action' in filters:
                    conditions.append("action ILIKE :action")
                    params['action'] = f"%{filters['action']}%"
                
                if 'resource_type' in filters:
                    conditions.append("resource_type = :resource_type")
                    params['resource_type'] = filters['resource_type']
                
                if 'success' in filters:
                    conditions.append("success = :success")
                    params['success'] = filters['success']
            
            where_clause = " AND ".join(conditions)
            
            async with self.async_session() as session:
                query = f"""
                SELECT id, timestamp, user_id, action, resource_type, resource_id,
                       source_ip, user_agent, success, details, session_id
                FROM audit_logs 
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT 10000
                """
                
                result = await session.execute(sa.text(query), params)
                rows = result.fetchall()
                
                logs = []
                for row in rows:
                    log_entry = {
                        'id': row.id,
                        'timestamp': row.timestamp.isoformat(),
                        'user_id': row.user_id,
                        'action': row.action,
                        'resource_type': row.resource_type,
                        'resource_id': row.resource_id,
                        'source_ip': str(row.source_ip) if row.source_ip else None,
                        'user_agent': row.user_agent,
                        'success': row.success,
                        'details': json.loads(row.details) if row.details else {},
                        'session_id': row.session_id
                    }
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
    
    async def search_audit_logs(self, query: str, time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Search audit logs by text query."""
        try:
            start_time = datetime.now() - timedelta(hours=time_range_hours)
            end_time = datetime.now()
            
            async with self.async_session() as session:
                search_query = """
                SELECT id, timestamp, user_id, action, resource_type, resource_id,
                       source_ip, user_agent, success, details, session_id
                FROM audit_logs 
                WHERE timestamp >= :start_time 
                AND timestamp <= :end_time
                AND (
                    action ILIKE :query 
                    OR resource_type ILIKE :query 
                    OR resource_id ILIKE :query
                    OR details::text ILIKE :query
                )
                ORDER BY timestamp DESC
                LIMIT 1000
                """
                
                result = await session.execute(sa.text(search_query), {
                    'start_time': start_time,
                    'end_time': end_time,
                    'query': f"%{query}%"
                })
                rows = result.fetchall()
                
                logs = []
                for row in rows:
                    log_entry = {
                        'id': row.id,
                        'timestamp': row.timestamp.isoformat(),
                        'user_id': row.user_id,
                        'action': row.action,
                        'resource_type': row.resource_type,
                        'resource_id': row.resource_id,
                        'source_ip': str(row.source_ip) if row.source_ip else None,
                        'user_agent': row.user_agent,
                        'success': row.success,
                        'details': json.loads(row.details) if row.details else {},
                        'session_id': row.session_id
                    }
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Failed to search audit logs: {e}")
            return []
    
    async def export_audit_logs(self, start_time: datetime, end_time: datetime, 
                              format: str = "csv") -> str:
        """Export audit logs to file."""
        try:
            logs = await self.get_audit_logs(start_time, end_time)
            
            if format.lower() == "csv":
                output = io.StringIO()
                if logs:
                    fieldnames = logs[0].keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for log in logs:
                        # Convert complex fields to strings for CSV
                        csv_log = log.copy()
                        if 'details' in csv_log and isinstance(csv_log['details'], dict):
                            csv_log['details'] = json.dumps(csv_log['details'])
                        writer.writerow(csv_log)
                
                return output.getvalue()
            
            elif format.lower() == "json":
                return json.dumps(logs, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export audit logs: {e}")
            return ""
    
    async def get_user_activity_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get summary of user activity over time period."""
        try:
            start_time = datetime.now() - timedelta(days=days)
            end_time = datetime.now()
            
            async with self.async_session() as session:
                # Get total actions
                total_query = """
                SELECT COUNT(*) as total_actions
                FROM audit_logs 
                WHERE user_id = :user_id 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
                """
                
                result = await session.execute(sa.text(total_query), {
                    'user_id': user_id,
                    'start_time': start_time,
                    'end_time': end_time
                })
                total_actions = result.scalar()
                
                # Get actions by type
                actions_query = """
                SELECT action, COUNT(*) as count
                FROM audit_logs 
                WHERE user_id = :user_id 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
                GROUP BY action
                ORDER BY count DESC
                LIMIT 10
                """
                
                result = await session.execute(sa.text(actions_query), {
                    'user_id': user_id,
                    'start_time': start_time,
                    'end_time': end_time
                })
                actions_by_type = [{'action': row.action, 'count': row.count} for row in result.fetchall()]
                
                # Get daily activity
                daily_query = """
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM audit_logs 
                WHERE user_id = :user_id 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                """
                
                result = await session.execute(sa.text(daily_query), {
                    'user_id': user_id,
                    'start_time': start_time,
                    'end_time': end_time
                })
                daily_activity = [{'date': row.date.isoformat(), 'count': row.count} for row in result.fetchall()]
                
                # Get resource types accessed
                resources_query = """
                SELECT resource_type, COUNT(*) as count
                FROM audit_logs 
                WHERE user_id = :user_id 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
                AND resource_type IS NOT NULL
                GROUP BY resource_type
                ORDER BY count DESC
                """
                
                result = await session.execute(sa.text(resources_query), {
                    'user_id': user_id,
                    'start_time': start_time,
                    'end_time': end_time
                })
                resources_accessed = [{'resource_type': row.resource_type, 'count': row.count} for row in result.fetchall()]
                
                return {
                    'user_id': user_id,
                    'period_days': days,
                    'start_date': start_time.isoformat(),
                    'end_date': end_time.isoformat(),
                    'total_actions': total_actions,
                    'actions_by_type': actions_by_type,
                    'daily_activity': daily_activity,
                    'resources_accessed': resources_accessed
                }
                
        except Exception as e:
            logger.error(f"Failed to get user activity summary: {e}")
            return {}