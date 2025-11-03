#!/bin/bash

# Project Argus Backup Script
# Creates backups of database, evidence store, and configuration

set -e

# Configuration
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="argus_backup_${DATE}"
RETENTION_DAYS=30

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

echo "Starting Project Argus backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup PostgreSQL database
echo "Backing up PostgreSQL database..."
PGPASSWORD="${DB_PASSWORD}" pg_dump \
    -h "${DB_HOST}" \
    -p "${DB_PORT}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    --verbose \
    --no-password \
    --format=custom \
    --file="${BACKUP_DIR}/${BACKUP_NAME}/database.dump"

# Backup MinIO evidence store
echo "Backing up MinIO evidence store..."
mc alias set argus-minio "http://${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}"
mc mirror argus-minio/argus-evidence "${BACKUP_DIR}/${BACKUP_NAME}/evidence/"

# Backup configuration files
echo "Backing up configuration files..."
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/config"
cp -r .env "${BACKUP_DIR}/${BACKUP_NAME}/config/" 2>/dev/null || true
cp -r edge/config/ "${BACKUP_DIR}/${BACKUP_NAME}/config/edge/" 2>/dev/null || true
cp -r infrastructure/ "${BACKUP_DIR}/${BACKUP_NAME}/config/infrastructure/" 2>/dev/null || true

# Create backup metadata
echo "Creating backup metadata..."
cat > "${BACKUP_DIR}/${BACKUP_NAME}/metadata.json" << EOF
{
    "backup_name": "${BACKUP_NAME}",
    "timestamp": "$(date -Iseconds)",
    "database_host": "${DB_HOST}",
    "database_name": "${DB_NAME}",
    "minio_endpoint": "${MINIO_ENDPOINT}",
    "backup_size": "$(du -sh ${BACKUP_DIR}/${BACKUP_NAME} | cut -f1)",
    "version": "1.0.0"
}
EOF

# Compress backup
echo "Compressing backup..."
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"
rm -rf "${BACKUP_NAME}/"

# Calculate checksum
echo "Calculating checksum..."
sha256sum "${BACKUP_NAME}.tar.gz" > "${BACKUP_NAME}.tar.gz.sha256"

# Clean up old backups
echo "Cleaning up old backups (older than ${RETENTION_DAYS} days)..."
find "${BACKUP_DIR}" -name "argus_backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}" -name "argus_backup_*.tar.gz.sha256" -mtime +${RETENTION_DAYS} -delete

echo "Backup completed successfully: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "Backup size: $(du -sh ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz | cut -f1)"