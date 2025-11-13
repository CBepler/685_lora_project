#!/bin/bash
# Script to check the status of all LoRA training jobs

echo "========================================="
echo "LoRA Training Jobs Status"
echo "========================================="
echo ""

# Get current user
USER=$(whoami)

# Check if there are any jobs
JOB_COUNT=$(squeue -u $USER | grep -E "lora|sparse|qlora|hira" | wc -l)

if [ $JOB_COUNT -eq 0 ]; then
    echo "No active LoRA training jobs found for user: $USER"
    echo ""
    echo "Recent job history:"
    sacct -u $USER --format=JobID,JobName%20,State,Elapsed,Start,End -S $(date -d '1 day ago' +%Y-%m-%d) | grep -E "lora|sparse|qlora|hira" | head -20
else
    echo "Active jobs for user: $USER"
    echo ""
    squeue -u $USER --format="%.10i %.20j %.8T %.10M %.9l %.6D %R" | grep -E "JOBID|lora|sparse|qlora|hira"

    echo ""
    echo "Job Summary:"
    echo "  Total active jobs: $JOB_COUNT"
    echo "  Running: $(squeue -u $USER -t RUNNING | grep -E "lora|sparse|qlora|hira" | wc -l)"
    echo "  Pending: $(squeue -u $USER -t PENDING | grep -E "lora|sparse|qlora|hira" | wc -l)"
fi

echo ""
echo "========================================="
echo ""

# Display recent completed jobs
echo "Recently completed jobs (last 24 hours):"
sacct -u $USER --format=JobID,JobName%20,State,Elapsed,End -S $(date -d '1 day ago' +%Y-%m-%d) -s COMPLETED,FAILED,CANCELLED | grep -E "lora|sparse|qlora|hira" | tail -20

echo ""
echo "========================================="
echo "Commands:"
echo "  View job details: scontrol show job <job_id>"
echo "  View output log: tail -f logs/<job_name>_<job_id>.out"
echo "  View error log: tail -f logs/<job_name>_<job_id>.err"
echo "  Cancel a job: scancel <job_id>"
echo "  Cancel all jobs: bash scripts/slurm/cancel_all_jobs.sh"
echo "========================================="
