#!/bin/bash
# install_task_deps.sh (DEPRECATED)
# This script is a thin wrapper around install_deps.py.
# Use install_deps.py directly instead.
#
# Usage: source install_task_deps.sh "task1,task2,task3" [eval_dir]

echo "WARNING: install_task_deps.sh is deprecated. Use install_deps.py instead." >&2

install_task_dependencies() {
    local tasks_string="$1"
    local eval_dir="${2:-$(pwd)}"

    python "${eval_dir}/examples/install_deps.py" \
        --tasks "${tasks_string}" \
        --eval-dir "${eval_dir}" \
        --skip-base \
        --skip-fixups
}

# If script is sourced with arguments, run the function
if [ -n "$1" ]; then
    install_task_dependencies "$1" "$2"
fi
