#!/bin/sh
set -eu

exec /opt/venv/bin/lmcache server \
  --host 0.0.0.0 \
  --port "${LMCACHE_MP_PORT:?LMCACHE_MP_PORT is required}" \
  --http-port "${LMCACHE_HTTP_PORT:?LMCACHE_HTTP_PORT is required}" \
  --chunk-size 400 \
  --l1-size-gb 256 \
  --eviction-policy LRU \
  --max-workers 4 \
  --supported-transfer-mode lmcache_driven
