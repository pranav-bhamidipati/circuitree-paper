#!/bin/bash --login

# Submit all arrayjobs in the directory `arrayjobs` to the queue, waiting 1 second between each submission
script_dir=/home/pbhamidi/git/circuitree/models/oscillation/scripts/
job_scripts=$(ls $script_dir/arrayjobs)
i=0
for job_script in $job_scripts; do
    echo "Submitting arrayjob $job_script"
    sbatch $script_dir/arrayjobs/$job_script
    ((++i < 2)) || break
    sleep 1
done

echo "Done"