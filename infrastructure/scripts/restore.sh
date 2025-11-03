#!/bin/bash

# Project Argus Restore Script
# Restores database, evidence store, and configuration from backup

set -e

# Check if backup file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    echo "Available backups:"
    ls -la /backups/argus_backup_*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="/tmp/argus_restore_$(date +%s)"

# Verify backup file exists
if [ ! -f "${BACKUP_FILE}" ]; then
    echo "Error: Backup file ${BACKUP_FILE} not found"
    exit 1
fi

# Verify checksum if available
if [ -f "${BACKUP_FILE}.sha256" ]; then
    echo "Verifying backup integrity..."
    if ! sha256sum -c "${BACKUP_FILE}.sha256"; then
        echo "Error: Backup integrity check failed"
        exit 1
    fi
    echo "Backup integrity verified"
fi

echo "Starting restore from: ${BACKUP_FILE}"

# Extract backup
echo "Extracting backup..."
mkdir -p "${RESTORE_DIR}"
tar -xzf "${BACKUP_FILE}" -C "${RESTORE_DIR}"

# Find the backup directory
BACKUP_NAME=$(basename "${BACKUP_FILE}" .tar.gz)
BACKUP_PATH="${RESTORE_DIR}/${BACKUP_NAME}"

if [ ! -d "${BACKUP_PATH}" ]; then
    echo "Error: Backup directory not found in archive"
    exit 1
fi

# Read backup metadata
if [ -f "${BACKUP_PATH}/metadata.json" ]; then
    echo "Backup metadata:"
    cat "${BACKUP_PATH}/metadata.json"
    echo ""
fi

# Confirm restore
read -p "Are you sure you want to restore from this backup? This will overwrite existing data. (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    rm -rf "${RESTORE_DIR}"
    exit 0
fi

# Database credentials
DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}
DB_NAME=${POSTGRES_DB:-project_argus}
DB_USER=${POSTGRES_USER:-argus_user}
DB_PASSWORD=${POSTGRES_PASSWORD}

# MinIO credentials
MINIO_ENDPOINT=${MINIO_ENDPOINT:-localhost:9000}
MINIO_ACCESS_KEY=${MINIO_ROOT_USER:-argus_admin}
MINIO_SECRET_KEY=${MINIO_ROOT_PASSWORD}

# Restore PostgreSQL database
if [ -f "${BACKUP_PATH}/database.dump" ]; then
    echo "Restoring PostgreSQL database..."
    
    # Drop and recreate database
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS ${DB_NAME};"
    
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d postgres \
        -c "CREATE DATABASE ${DB_NAME};"
    
    # Restore database
    PGPASSWORD="${DB_PASSWORD}" pg_restore \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --verbose \
        --no-password \
        "${BACKUP_PATH}/database.dump"
    
    echo "Database restored successfully"
else
    echo "Warning: No database backup found"
fi

# Restore MinIO evidence store
if [ -d "${BACKUP_PATH}/evidence" ]; then
    echo "Restoring MinIO evidence store..."
    
    # Configure MinIO client
    mc alias set argus-minio "http://${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}"
    
    # Remove existing bucket contents (with confirmation)
    mc rm --recursive --force argus-minio/argus-evidence/ || true
    
    # Restore evidence files
    mc mirror "${BACKUP_PATH}/evidence/" argus-minio/argus-evidence/
    
    echo "Evidence store restored successfully"
else
    echo "Warning: No evidence backup found"
fi

# Restore configuration files
if [ -d "${BACKUP_PATH}/config" ]; then
    echo "Restoring configuration files..."
    
    # Backup current config
    if [ -f ".env" ]; then
        cp .env .env.backup.$(date +%s)
    fi
    
    # Restore configuration
    cp -r "${BACKUP_PATH}/config/"* . 2>/dev/null || true
    
    echo "Configuration files restored successfully"
    echo "Note: Previous .env file backed up with timestamp"
else
    echo "Warning: No configuration backup found"
fi

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "${RESTORE_DIR}"

echo "Restore completed successfully!"
echo "Please restart all services to apply the restored configuration."