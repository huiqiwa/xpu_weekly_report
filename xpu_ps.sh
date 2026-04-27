#!/bin/bash
# Show processes occupying XPU devices along with user/command info.
# Usage: bash xpu_ps.sh

pids=$(sudo xpu-smi ps 2>/dev/null | awk 'NR>1 && $2!="xpu-smi" {pids[$1]=1} END {for(p in pids) printf p","}' | sed 's/,$//')

if [[ -z "$pids" ]]; then
    echo "No user processes on XPU."
    echo ""
    echo "=== Device Utilization ==="
    sudo xpu-smi dump -m 0,5,31 -n1 2>/dev/null
    exit 0
fi

# Device-level summary (deduplicated per PID)
echo "=== XPU Device Usage ==="
sudo xpu-smi ps 2>/dev/null | awk 'NR==1{print;next} $2!="xpu-smi"{print}'
echo ""

# Build PID -> devices mapping
declare -A pid_devices
while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $1}')
    dev=$(echo "$line" | awk '{print $3}')
    if [[ -n "${pid_devices[$pid]}" ]]; then
        pid_devices[$pid]="${pid_devices[$pid]},$dev"
    else
        pid_devices[$pid]="$dev"
    fi
done < <(sudo xpu-smi ps 2>/dev/null | awk 'NR>1 && $2!="xpu-smi"{print}')

# Process details with device info
echo "=== Process Details ==="
printf "%-10s %-10s %-10s %-12s %-12s %s\n" "PID" "USER" "STARTED" "ELAPSED" "DEVICES" "CMD"
for pid in $(echo "$pids" | tr ',' '\n'); do
    info=$(ps -o pid=,user=,start=,etime=,cmd= -p "$pid" 2>/dev/null)
    [[ -z "$info" ]] && continue
    p=$(echo "$info" | awk '{print $1}')
    u=$(echo "$info" | awk '{print $2}')
    s=$(echo "$info" | awk '{print $3}')
    e=$(echo "$info" | awk '{print $4}')
    c=$(echo "$info" | awk '{for(i=5;i<=NF;i++) printf $i" "; print ""}')
    devs="${pid_devices[$p]:-?}"
    printf "%-10s %-10s %-10s %-12s %-12s %s\n" "$p" "$u" "$s" "$e" "$devs" "$c"
done

# Device utilization snapshot
echo ""
echo "=== Device Utilization ==="
sudo xpu-smi dump -m 0,5,31 -n1 2>/dev/null
