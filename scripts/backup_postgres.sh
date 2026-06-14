#!/usr/bin/env sh
set -eu

ENV_FILE="${1:-.env.docker.prod.local}"
BACKUP_DIR="${BACKUP_DIR:-backups/postgres}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="${BACKUP_DIR}/postgres-${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

docker compose --env-file "$ENV_FILE" -f docker-compose.prod.yml exec -T postgres \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  | gzip > "$OUT_FILE"

echo "Wrote $OUT_FILE"
