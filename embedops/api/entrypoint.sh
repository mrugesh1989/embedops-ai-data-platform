mkdir -p embedops/api
cat > embedops/api/entrypoint.sh << 'EOF'
#!/usr/bin/env sh
set -eu

CHUNK_STORE_PATH="${CHUNK_STORE_PATH:-/app/data/processed/chunks.jsonl}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

echo "[EmbedOps API] Waiting for chunk store: ${CHUNK_STORE_PATH} (timeout=${WAIT_SECONDS}s)"

i=0
while [ "$i" -lt "$WAIT_SECONDS" ]; do
  if [ -f "$CHUNK_STORE_PATH" ]; then
    echo "[EmbedOps API] Chunk store found. Starting API..."
    exec uvicorn embedops.api.main:app --host 0.0.0.0 --port 8000
  fi
  i=$((i+1))
  sleep 1
done

echo "[EmbedOps API] ERROR: chunk store not found after ${WAIT_SECONDS}s: ${CHUNK_STORE_PATH}"
exit 1
EOF