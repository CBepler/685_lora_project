#!/bin/bash
# Script to cancel all LoRA training jobs for the current user

echo "========================================="
echo "Cancel LoRA Training Jobs"
echo "========================================="
echo ""

# Get current user
USER=$(whoami)

# Get list of job IDs
JOB_IDS=$(squeue -u $USER -h -o "%i" | grep -E "lora|sparse|qlora|hira")

if [ -z "$JOB_IDS" ]; then
    echo "No active LoRA training jobs found for user: $USER"
    exit 0
fi

# Count jobs
JOB_COUNT=$(echo "$JOB_IDS" | wc -l)

echo "Found $JOB_COUNT active LoRA training job(s):"
squeue -u $USER --format="%.10i %.20j %.8T %.10M" | grep -E "JOBID|lora|sparse|qlora|hira"

echo ""
read -p "Are you sure you want to cancel all these jobs? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancellation aborted."
    exit 0
fi

echo ""
echo "Cancelling jobs..."

for JOB_ID in $JOB_IDS; do
    JOB_NAME=$(squeue -j $JOB_ID -h -o "%j")
    scancel $JOB_ID
    echo "  Cancelled: Job $JOB_ID ($JOB_NAME)"
done

echo ""
echo "========================================="
echo "All LoRA training jobs cancelled."
echo "========================================="
