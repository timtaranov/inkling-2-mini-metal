#!/bin/sh
# Build (if needed) and start the Inkling-2-Mini server on http://127.0.0.1:8080
cd "$(dirname "$0")" || exit 1
make -s inkling-server || exit 1
exec ./inkling-server "$@"
