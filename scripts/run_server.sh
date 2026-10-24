#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/dkd twinloop-server --port uvicorn app.server:app --host 0.0.0.0 --port 8000
for i in $(seq 1 30); do
  if out=$(curl -s http://127.0.0.1:8000/api/health); then echo "$out"; exit 0; fi
  sleep 1
done
echo "server did not start"; docker logs --tail 50 twinloop-server; exit 1
