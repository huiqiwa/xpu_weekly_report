#!/bin/bash
# Kill all GPU processes owned by the current user

USER=$(whoami)
PIDS=$(xpu-smi ps 2>/dev/null | grep -v xpu-smi | grep -v "^PID" | awk '{print $1}' | sort -u)

if [ -z "$PIDS" ]; then
    echo "No GPU processes found."
    exit 0
fi

OWNED_PIDS=""
for pid in $PIDS; do
    owner=$(ps -p "$pid" -o user= 2>/dev/null)
    if [ "$owner" = "$USER" ]; then
        OWNED_PIDS="$OWNED_PIDS $pid"
    fi
done

if [ -z "$OWNED_PIDS" ]; then
    echo "No GPU processes owned by $USER."
    exit 0
fi

echo "Killing GPU processes owned by $USER: $OWNED_PIDS"
kill -9 $OWNED_PIDS
echo "Done."
