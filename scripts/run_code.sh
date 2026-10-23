#!/bin/bash
export PATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")/.venv/bin:$PATH"

# Function to calculate execution time from .dat timestamps (excluding trailing zero-memory samples)
calculate_execution_time() {
    local dat_file=$1
    awk 'NR>1 && /^MEM/ && $2 != "0.000000" {
        if (min == "") min = $3
        max = $3
    }
    END {
        printf "%.4f\n", max - min
    }' "$dat_file"
}

# Function to calculate memory usage in MB*seconds
calculate_memory_usage() {
    local dat_file=$1
    awk 'BEGIN {
        prev_time = 0;
        prev_mem_mb = 0;
        mem_time_mb_s = 0;
    }
    NR>1 {
        mem_in_mb = $2;
        timestamp = $3;

        if (prev_time > 0) {
            time_interval_s = timestamp - prev_time;
            mem_time_mb_s += (prev_mem_mb + mem_in_mb) / 2 * time_interval_s;
        }

        prev_time = timestamp;
        prev_mem_mb = mem_in_mb;
    }
    END {
        printf "%.2f\n", mem_time_mb_s;
    }' "$dat_file"
}

# BFS to collect all descendant PIDs of a given root PID
collect_all_pids() {
    local root=$1
    local out=$2
    local queue=("$root")
    while [ ${#queue[@]} -gt 0 ]; do
        local pid="${queue[0]}"
        queue=("${queue[@]:1}")
        echo "$pid" >> "$out"
        while IFS= read -r child; do
            [ -n "$child" ] && queue+=("$child")
        done < <(ps --ppid "$pid" -o pid= 2>/dev/null | tr -d ' ')
    done
}


completion_file="$1"
completion_dat_file="$2"
max_execution_time="$3"
stage="${4:-unknown}"

# PID files
pid_file="${completion_dat_file%.dat}.pid"
pids_file="${completion_dat_file%.dat}_${stage}.pids"

# Inject PID capture at top of script
instrumented_file=$(mktemp --suffix=.py)
echo "import os; open('$pid_file', 'w').write(str(os.getpid()))" > "$instrumented_file"
cat "$completion_file" >> "$instrumented_file"

echo "Executing $completion_file"
error_output=$(mktemp)
rm -f "$completion_dat_file"

# Run mprof in background so we can monitor its full process tree
> "$pids_file"
timeout "$max_execution_time" mprof run --interval 0.001 --output "$completion_dat_file" "$instrumented_file" 2> "$error_output" &
MPROF_PID=$!

# Background monitor: poll the process tree every 50ms
(
    while kill -0 $MPROF_PID 2>/dev/null; do
        collect_all_pids $MPROF_PID "$pids_file"
        sleep 0.05
    done
    # Final pass to catch any remaining children
    collect_all_pids $MPROF_PID "$pids_file"
) &
MONITOR_PID=$!

wait $MPROF_PID
exit_status=$?
wait $MONITOR_PID 2>/dev/null

# Deduplicate PIDs
sort -un "$pids_file" -o "$pids_file"

# Check execution status
echo "Execution status: $exit_status"
if [ $exit_status -ne 0 ] || [ -s "$error_output" ]; then
    echo "Execution failed or errors were reported for $completion_file. Removing .dat file."
    rm -f "$completion_dat_file"
elif [ -f "$completion_dat_file" ]; then
    PID=$(cat "$pid_file" 2>/dev/null)
    mem_usage_mb_s=$(calculate_memory_usage "$completion_dat_file")
    exec_time=$(calculate_execution_time "$completion_dat_file")
    echo "PID: $PID"
    echo "All PIDs (stage=$stage): $(cat $pids_file | tr '\n' ' ')"
    echo "Execution time (s): $exec_time"
    echo "Memory usage (MB*seconds): $mem_usage_mb_s"
else
    echo "Execution completed but no .dat file found for $completion_file."
fi

# Clean up
rm -f "$error_output" "$instrumented_file"
