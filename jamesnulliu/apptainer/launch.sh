#!/bin/bash
set -e

export HF_HOME=${HF_HOME:-"/home/shared"}

source .venv/bin/activate

TARGET_FILE=${TARGET_FILE:-"ds-r1-d-qwen2-1.5b.py"}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET_FILE="$2"; shift ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
    shift
done

python jamesnulliu/$TARGET_FILE