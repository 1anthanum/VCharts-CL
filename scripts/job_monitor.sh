#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VTBench Job Monitor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage:
#   bash scripts/job_monitor.sh              # one-shot status
#   bash scripts/job_monitor.sh --watch      # auto-refresh every 60s
#   bash scripts/job_monitor.sh --watch 30   # refresh every 30s
#   bash scripts/job_monitor.sh 11508915     # specific job ID

set -euo pipefail

# ── Args ──
WATCH=false
INTERVAL=60
JOBID=""

for arg in "$@"; do
    case "$arg" in
        --watch) WATCH=true ;;
        [0-9]*)
            if [ -z "$JOBID" ] && [ ${#arg} -gt 4 ]; then
                JOBID="$arg"
            else
                INTERVAL="$arg"
            fi
            ;;
    esac
done

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

format_duration() {
    local secs=$1
    local h=$((secs / 3600))
    local m=$(( (secs % 3600) / 60 ))
    if [ $h -gt 0 ]; then
        printf "%dh%02dm" $h $m
    else
        printf "%dm" $m
    fi
}

print_status() {
    clear 2>/dev/null || true
    local now
    now=$(date "+%Y-%m-%d %H:%M:%S")

    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD} VTBench Job Monitor  ${DIM}$now${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

    # ── Find jobs ──
    local jobs
    if [ -n "$JOBID" ]; then
        jobs=$(squeue -j "$JOBID" -u "$USER" -o "%.12i %.4t %.12M %.12l %.20S %.6C %.10m %R" --noheader 2>/dev/null || true)
    else
        jobs=$(squeue -u "$USER" -o "%.12i %.4t %.12M %.12l %.20S %.6C %.10m %R" --noheader 2>/dev/null || true)
    fi

    if [ -z "$jobs" ]; then
        echo -e "\n${YELLOW} No active jobs found.${RESET}"

        # Check recent completed jobs
        echo -e "\n${DIM} Recent completed jobs (last 24h):${RESET}"
        sacct -u "$USER" --starttime "$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -v-24H '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || echo '2026-03-31')" \
            --format="JobID,JobName%20,State%12,Elapsed,Start,End,ExitCode" --noheader 2>/dev/null | head -10
        echo ""
        return 1
    fi

    # ── Display each job ──
    echo ""
    while IFS= read -r line; do
        local jid state elapsed timelimit start_time cpus mem node
        jid=$(echo "$line" | awk '{print $1}')
        state=$(echo "$line" | awk '{print $2}')
        elapsed=$(echo "$line" | awk '{print $3}')
        timelimit=$(echo "$line" | awk '{print $4}')
        start_time=$(echo "$line" | awk '{print $5}')
        cpus=$(echo "$line" | awk '{print $6}')
        mem=$(echo "$line" | awk '{print $7}')
        node=$(echo "$line" | awk '{print $8}')

        # State color
        local state_color state_label
        case "$state" in
            R)  state_color=$GREEN;  state_label="RUNNING" ;;
            PD) state_color=$YELLOW; state_label="PENDING (waiting for resources)" ;;
            CG) state_color=$CYAN;   state_label="COMPLETING" ;;
            *)  state_color=$RED;    state_label="$state" ;;
        esac

        echo -e " Job ID:    ${BOLD}$jid${RESET}"
        echo -e " Status:    ${state_color}${BOLD}$state_label${RESET}"
        echo -e " Node:      $node"
        echo -e " Resources: ${cpus} CPUs, ${mem} RAM, 1 GPU"
        echo -e " Elapsed:   $elapsed / $timelimit"

        # Calculate wait time for pending jobs
        if [ "$state" = "PD" ]; then
            local submit_time
            submit_time=$(sacct -j "$jid" --format="Submit" --noheader 2>/dev/null | head -1 | xargs)
            if [ -n "$submit_time" ]; then
                local submit_epoch now_epoch wait_secs
                submit_epoch=$(date -d "$submit_time" +%s 2>/dev/null || echo "0")
                now_epoch=$(date +%s)
                if [ "$submit_epoch" -gt 0 ]; then
                    wait_secs=$((now_epoch - submit_epoch))
                    echo -e " Waiting:   ${YELLOW}$(format_duration $wait_secs)${RESET}"
                fi
            fi

            # Estimated start time
            local est_start
            est_start=$(squeue -j "$jid" --start -o "%.20S" --noheader 2>/dev/null | xargs)
            if [ -n "$est_start" ] && [ "$est_start" != "N/A" ]; then
                echo -e " Est Start: ${CYAN}$est_start${RESET}"
            fi
        fi

        # ── Experiment progress (only for running jobs) ──
        if [ "$state" = "R" ]; then
            echo ""
            echo -e " ${BOLD}Experiment Progress:${RESET}"

            local logfile="$HOME/logs/${jid}_out.txt"
            if [ -f "$logfile" ]; then
                # Extract current phase
                local current_phase
                current_phase=$(grep -oP "PHASE \K[A-Z0-9]+(?=:)" "$logfile" 2>/dev/null | tail -1)
                if [ -n "$current_phase" ]; then
                    echo -e "   Current phase: ${CYAN}${BOLD}$current_phase${RESET}"
                fi

                # Count completed experiments from state file
                local state_file="$HOME/myproject/results/orchestrator_state.json"
                if [ -f "$state_file" ]; then
                    local completed failed
                    completed=$(python3 -c "import json; d=json.load(open('$state_file')); print(len(d.get('completed',{})))" 2>/dev/null || echo "?")
                    failed=$(python3 -c "import json; d=json.load(open('$state_file')); print(len(d.get('failed',{})))" 2>/dev/null || echo "?")
                    echo -e "   Completed: ${GREEN}$completed${RESET}/23    Failed: ${RED}$failed${RESET}"
                fi

                # Last 3 meaningful log lines
                echo -e "\n   ${DIM}Recent log:${RESET}"
                grep -E "START:|DONE:|FAIL|PHASE|SKIP|Accuracy|ERROR" "$logfile" 2>/dev/null | tail -5 | while IFS= read -r logline; do
                    # Colorize
                    if echo "$logline" | grep -q "DONE:"; then
                        echo -e "   ${GREEN}$logline${RESET}"
                    elif echo "$logline" | grep -q "FAIL\|ERROR"; then
                        echo -e "   ${RED}$logline${RESET}"
                    elif echo "$logline" | grep -q "START:\|PHASE"; then
                        echo -e "   ${CYAN}$logline${RESET}"
                    else
                        echo -e "   ${DIM}$logline${RESET}"
                    fi
                done
            else
                echo -e "   ${DIM}Log not yet available: $logfile${RESET}"
            fi

            # ── GPU status ──
            echo ""
            echo -e " ${BOLD}GPU (node $node):${RESET}"
            # If running on the same node, nvidia-smi works; otherwise need ssh
            if [ "$(hostname)" = "$node" ]; then
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                    --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r util mem_used mem_total temp; do
                    echo -e "   Utilization: ${util}%  Memory: ${mem_used}/${mem_total} MB  Temp: ${temp}°C"
                done
            else
                echo -e "   ${DIM}(Run on $node to see live GPU stats)${RESET}"
            fi
        fi

        echo -e "\n${DIM}─────────────────────────────────────────────────${RESET}"

    done <<< "$jobs"

    if $WATCH; then
        echo -e "${DIM} Refreshing every ${INTERVAL}s — Ctrl+C to stop${RESET}"
    fi
}

# ── Main ──
if $WATCH; then
    while true; do
        print_status || true
        sleep "$INTERVAL"
    done
else
    print_status
fi
