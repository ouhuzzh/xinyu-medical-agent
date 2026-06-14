#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: sh scripts/restore_postgres.sh <backup.sql.gz|backup.sql> [env-file] [--yes]"
  exit 1
fi

BACKUP_FILE="$1"
ENV_FILE="${2:-.env.docker.prod.local}"
CONFIRM_FLAG="${3:-}"

if [ ! -f "$BACKUP_FILE" ]; then
  echo "ERROR: backup file not found: $BACKUP_FILE"
  exit 1
fi

if [ "$CONFIRM_FLAG" != "--yes" ] && [ "${CONFIRM_RESTORE:-}" != "yes" ]; then
  echo "Refusing to restore without confirmation."
  echo "Re-run with: sh scripts/restore_postgres.sh \"$BACKUP_FILE\" \"$ENV_FILE\" --yes"
  exit 1
fi

case "$BACKUP_FILE" in
  *.gz)
    RESTORE_CMD="gunzip -c \"$BACKUP_FILE\""
    ;;
  *)
    RESTORE_CMD="cat \"$BACKUP_FILE\""
    ;;
esac

sh -c "$RESTORE_CMD" | docker compose --env-file "$ENV_FILE" -f docker-compose.prod.yml exec -T postgres \
  sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

echo "Restore completed from $BACKUP_FILE"
