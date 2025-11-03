"""
Evidence Service Main Application for Project Argus.
Handles evidence storage, retrieval, and forensics operations.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
from minio import Minio
from minio.error import S3Error

from evidence_store import EvidenceStore
from audit_logger import AuditLogger
from key_manager import KeyManager
from shared.models.evidence import Evidence, EvidenceType, EvidenceStatus
from shared.interfaces.evidence import IEvidenceStore, IAuditLogger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global service instances
evidence_store: Optional[IEvidenceStore] = None
audit_logger: Optional[IAuditLogger] = None
key_manager: Optional[KeyManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global evidence_store, audit_logger, key_manager
    
    # Get configuration from environment
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/argus")
    storage_path = os.getenv("EVIDENCE_STORAGE_PATH", "./data/evidence")
    key_storage_path = os.getenv("KEY_STORAGE_PATH", "./data/keys")
    encryption_key = os.getenv("EVIDENCE_ENCRYPTION_KEY", "")
    hmac_secret = os.getenv("EVIDENCE_HMAC_SECRET", "")
    master_key = os.getenv("MASTER_ENCRYPTION_KEY", "")
    
    # MinIO configuration (optional)
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    minio_bucket = os.getenv("MINIO_BUCKET", "evidence")
    
    if not encryption_key or not hmac_secret or not master_key:
        logger.error("Missing required encryption keys in environment variables")
        raise RuntimeError("Missing encryption configuration")
    
    # Initialize MinIO client if configured
    minio_client = None
    if minio_endpoint and minio_access_key and minio_secret_key:
        try:
            minio_client = Minio(
                endpoint=minio_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=os.getenv("MINIO_SECURE", "true").lower() == "true"
            )
            logger.info("MinIO client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize MinIO client: {e}")
    
    # Initialize services
    try:
        # Initialize key manager
        key_manager = KeyManager(
            database_url=database_url,
            key_storage_path=key_storage_path,
            master_key=master_key
        )
        
        # Initialize audit logger
        audit_logger = AuditLogger(database_url=database_url)
        
        # Initialize evidence store
        evidence_store = EvidenceStore(
            database_url=database_url,
            storage_path=storage_path,
            encryption_key=encryption_key,
            hmac_secret=hmac_secret,
            minio_client=minio_client,
            bucket_name=minio_bucket
        )
        
        logger.info("Evidence service started successfully")
        
        # Log service startup
        await audit_logger.log_system_event(
            event_type="service_started",
            component="evidence_service",
            details={"version": "1.0.0", "storage_backend": "minio" if minio_client else "local"}
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize evidence service: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    if audit_logger:
        await audit_logger.log_system_event(
            event_type="service_stopped",
            component="evidence_service",
            details={"shutdown_reason": "normal"}
        )
    
    logger.info("Evidence service shutting down")


# Create FastAPI app
app = FastAPI(
    title="Project Argus Evidence Service",
    description="Secure evidence storage and management service with HMAC signing and AES-256 encryption",
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


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user ID from JWT token (simplified for demo)."""
    # In production, this would validate JWT and extract user ID
    return "demo-user-id"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "evidence-service",
        "version": "1.0.0",
        "features": ["encryption", "hmac_signing", "audit_logging", "key_management"]
    }


@app.post("/evidence", response_model=Dict[str, str])
async def store_evidence(
    file: UploadFile = File(...),
    evidence_type: str = Form(...),
    incident_id: Optional[str] = Form(None),
    camera_id: Optional[str] = Form(None),
    detection_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}"),
    current_user: str = Depends(get_current_user)
):
    """Store new evidence file with encryption and integrity verification."""
    try:
        # Read file data
        file_data = await file.read()
        
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata) if metadata else {}
        metadata_dict.update({
            'original_filename': file.filename,
            'mime_type': file.content_type,
            'incident_id': incident_id,
            'camera_id': camera_id,
            'detection_id': detection_id
        })
        
        # Store evidence
        evidence_id = await evidence_store.store_evidence(
            file_data=file_data,
            evidence_type=EvidenceType(evidence_type),
            metadata=metadata_dict,
            created_by=current_user
        )
        
        # Log the action
        await audit_logger.log_user_action(
            user_id=current_user,
            action="evidence_stored",
            resource_type="evidence",
            resource_id=evidence_id,
            details={
                "evidence_type": evidence_type,
                "file_size": len(file_data),
                "incident_id": incident_id,
                "camera_id": camera_id
            }
        )
        
        return {"evidence_id": evidence_id, "status": "stored"}
        
    except Exception as e:
        logger.error(f"Failed to store evidence: {e}")
        await audit_logger.log_user_action(
            user_id=current_user,
            action="evidence_store_failed",
            resource_type="evidence",
            resource_id="unknown",
            details={"error": str(e), "evidence_type": evidence_type}
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evidence/{evidence_id}")
async def get_evidence(
    evidence_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get evidence metadata."""
    try:
        evidence = await evidence_store.retrieve_evidence(evidence_id)
        if not evidence:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        # Log evidence access
        await audit_logger.log_evidence_access(
            evidence_id=evidence_id,
            user_id=current_user,
            action="viewed",
            details={"access_type": "metadata"}
        )
        
        return evidence.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evidence/{evidence_id}/file")
async def get_evidence_file(
    evidence_id: str,
    current_user: str = Depends(get_current_user)
):
    """Download evidence file with integrity verification."""
    try:
        from fastapi.responses import StreamingResponse
        
        file_content = await evidence_store.get_evidence_file(evidence_id)
        if not file_content:
            raise HTTPException(status_code=404, detail="Evidence file not found")
        
        evidence = await evidence_store.retrieve_evidence(evidence_id)
        filename = evidence.original_filename if evidence else f"{evidence_id}.bin"
        
        # Log evidence file access
        await audit_logger.log_evidence_access(
            evidence_id=evidence_id,
            user_id=current_user,
            action="downloaded",
            details={"filename": filename, "file_size": evidence.file_size if evidence else 0}
        )
        
        return StreamingResponse(
            file_content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evidence file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evidence/{evidence_id}/seal")
async def seal_evidence(
    evidence_id: str,
    current_user: str = Depends(get_current_user)
):
    """Seal evidence for legal proceedings."""
    try:
        success = await evidence_store.seal_evidence(evidence_id, current_user)
        if not success:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        # Log sealing action
        await audit_logger.log_user_action(
            user_id=current_user,
            action="evidence_sealed",
            resource_type="evidence",
            resource_id=evidence_id,
            details={"reason": "legal_proceedings"}
        )
        
        return {"status": "sealed", "evidence_id": evidence_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to seal evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evidence/{evidence_id}/verify")
async def verify_evidence(
    evidence_id: str,
    current_user: str = Depends(get_current_user)
):
    """Verify evidence integrity using HMAC and hash verification."""
    try:
        is_valid = await evidence_store.verify_integrity(evidence_id)
        
        # Log verification action
        await audit_logger.log_user_action(
            user_id=current_user,
            action="evidence_verified",
            resource_type="evidence",
            resource_id=evidence_id,
            details={"integrity_valid": is_valid}
        )
        
        return {"evidence_id": evidence_id, "integrity_valid": is_valid}
        
    except Exception as e:
        logger.error(f"Failed to verify evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evidence/{evidence_id}/chain-of-custody")
async def get_chain_of_custody(
    evidence_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get complete chain of custody for evidence."""
    try:
        chain = await evidence_store.get_chain_of_custody(evidence_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        # Log chain of custody access
        await audit_logger.log_evidence_access(
            evidence_id=evidence_id,
            user_id=current_user,
            action="chain_of_custody_viewed",
            details={"entries_count": len(chain.entries)}
        )
        
        return chain.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chain of custody: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evidence/{evidence_id}/transfer")
async def transfer_custody(
    evidence_id: str,
    to_operator: str = Form(...),
    reason: str = Form(...),
    current_user: str = Depends(get_current_user)
):
    """Transfer evidence custody between operators."""
    try:
        success = await evidence_store.transfer_custody(
            evidence_id=evidence_id,
            from_operator=current_user,
            to_operator=to_operator,
            reason=reason
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        # Log custody transfer
        await audit_logger.log_user_action(
            user_id=current_user,
            action="custody_transferred",
            resource_type="evidence",
            resource_id=evidence_id,
            details={"to_operator": to_operator, "reason": reason}
        )
        
        return {"status": "transferred", "evidence_id": evidence_id, "to_operator": to_operator}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to transfer custody: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evidence/search")
async def search_evidence(
    incident_id: Optional[str] = None,
    evidence_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    created_by: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """Search evidence by various criteria."""
    try:
        filters = {}
        if incident_id:
            filters['incident_id'] = incident_id
        if evidence_type:
            filters['type'] = evidence_type
        if camera_id:
            filters['camera_id'] = camera_id
        if created_by:
            filters['created_by'] = created_by
        if start_date:
            filters['start_date'] = datetime.fromisoformat(start_date)
        if end_date:
            filters['end_date'] = datetime.fromisoformat(end_date)
        
        evidence_list = await evidence_store.search_evidence(filters)
        
        # Log search action
        await audit_logger.log_user_action(
            user_id=current_user,
            action="evidence_searched",
            resource_type="evidence",
            resource_id="multiple",
            details={"filters": filters, "results_count": len(evidence_list)}
        )
        
        return [evidence.to_dict() for evidence in evidence_list]
        
    except Exception as e:
        logger.error(f"Failed to search evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evidence/{evidence_id}/schedule-purge")
async def schedule_purge(
    evidence_id: str,
    purge_date: str = Form(...),
    current_user: str = Depends(get_current_user)
):
    """Schedule evidence for automatic purging."""
    try:
        purge_datetime = datetime.fromisoformat(purge_date)
        success = await evidence_store.schedule_purge(evidence_id, purge_datetime, current_user)
        
        if not success:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        # Log purge scheduling
        await audit_logger.log_user_action(
            user_id=current_user,
            action="purge_scheduled",
            resource_type="evidence",
            resource_id=evidence_id,
            details={"purge_date": purge_date}
        )
        
        return {"status": "scheduled", "evidence_id": evidence_id, "purge_date": purge_date}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule purge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/purge-expired")
async def purge_expired_evidence(current_user: str = Depends(get_current_user)):
    """Purge expired evidence (admin only)."""
    try:
        purged_ids = await evidence_store.purge_expired_evidence()
        
        # Log purge operation
        await audit_logger.log_user_action(
            user_id=current_user,
            action="expired_evidence_purged",
            resource_type="evidence",
            resource_id="multiple",
            details={"purged_count": len(purged_ids), "purged_ids": purged_ids}
        )
        
        return {"status": "completed", "purged_count": len(purged_ids), "purged_ids": purged_ids}
        
    except Exception as e:
        logger.error(f"Failed to purge expired evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/logs")
async def get_audit_logs(
    start_date: str,
    end_date: str,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """Get audit logs for specified time range."""
    try:
        start_time = datetime.fromisoformat(start_date)
        end_time = datetime.fromisoformat(end_date)
        
        filters = {}
        if user_id:
            filters['user_id'] = user_id
        if action:
            filters['action'] = action
        
        logs = await audit_logger.get_audit_logs(start_time, end_time, filters)
        
        return {"logs": logs, "count": len(logs)}
        
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )