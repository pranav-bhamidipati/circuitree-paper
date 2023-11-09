#!/bin/bash --login

#SBATCH --time=03:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per job in array (i.e. tasks)
#SBATCH --array=0-__arraylimit__
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "__jobname__-__jobnum__"   # job name
#SBATCH --mail-user=pbhamidi@usc.edu   # email address
#SBATCH --mail-type=FAIL

#SBATCH -o /home/pbhamidi/slurm_out/slurm.%N.%j.out # STDOUT
#SBATCH -e /home/pbhamidi/slurm_out/slurm.%N.%j.err # STDERR


#======START===============================

module purge
module load openssl/3.0.3 python3/3.10.7

eval "$(conda shell.bash hook)"
mamba init
source ~/.bashrc
source ~/mambaforge/etc/profile.d/mamba.sh

echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "A total of $SLURM_NTASKS tasks is used"

# Since only 1000 jobs are allowed in an array job, this is added to the 
# SLURM_ARRAY_TASK_ID to get the actual job index
export JOB_QUEUE_FIRST_INDEX=__index__

echo "Environment Variables"
env
echo ""

echo "Activating mamba environment"

mamba deactivate
mamba activate /home/pbhamidi/git/circuitree/env
echo ""

CMD="/home/pbhamidi/git/circuitree/env/bin/python /home/pbhamidi/git/circuitree/models/oscillation/run_bfs_arrayjob.py"
echo $CMD
$CMD
echo "Done with simulation"

NEXT_JOB_NAME="__jobname__-__next_jobnum__"
NEXT_JOB_SCRIPT="__next_script__"
SLURM_N_SUBMITTED_TASKS=$(($(squeue -r -u pbhamidi | wc -l) - 1))
echo "Total number of tasks in SLURM job queue: $SLURM_N_SUBMITTED_TASKS"
SLURM_N_PENDING_TASKS_IN_ARRAY=$(($(squeue -r -u pbhamidi -t PENDING -n "__jobname__-__jobnum__" | wc -l) - 1))

if [ ! -z $NEXT_JOB_SCRIPT ]; then

    # Submit the next job if there is space in the queue or if this is the last task in the array job
    echo "Next job script: $NEXT_JOB_SCRIPT"
    echo "Number of running tasks in this array: $SLURM_N_PENDING_TASKS_IN_ARRAY"
    SUBMIT_NEXT_JOB=$(($SLURM_N_SUBMITTED_TASKS < __tasklimit__ || $SLURM_N_PENDING_TASKS_IN_ARRAY == 0))

else
    echo "Next job script: None"
    SUBMIT_NEXT_JOB=0
fi

echo "Submit next job? $SUBMIT_NEXT_JOB"f

if [ $SUBMIT_NEXT_JOB -eq 1 ]; then

    echo "Attempting submission of job $NEXT_JOB_NAME with script: $NEXT_JOB_SCRIPT"

    while true; do
        if (( $(squeue -u pbhamidi -n $NEXT_JOB_NAME | wc -l) > 1 )); then 
            echo "Job $NEXT_JOB_NAME is already in the queue. Aborting..."
            break
        elif ln -s $NEXT_JOB_SCRIPT "$NEXT_JOB_SCRIPT.lock" 2>/dev/null; then
            echo "Lock acquired"

            # This job might have been submitted while we were waiting for the lock, 
            # so just in case, we wait long enough that such a job will enter the queue
            sleep 2

            # Look for a .submitted file. If it exists, then the job has already been submitted
            if [ -f "$NEXT_JOB_SCRIPT.submitted" ]; then
                echo "Job $NEXT_JOB_NAME has already been submitted."
                echo "Releasing lock and aborting..."
                rm "$NEXT_JOB_SCRIPT.lock"
                break
            elif (( $(squeue -u pbhamidi -n $NEXT_JOB_NAME | wc -l) > 1 )); then 
                echo "Job $NEXT_JOB_NAME is already in the queue. Aborting..."
            else
                echo "Submitting job $NEXT_JOB_NAME"
                sbatch $NEXT_JOB_SCRIPT
                echo "Creating submission file: $NEXT_JOB_SCRIPT.submitted"
                touch "$NEXT_JOB_SCRIPT.submitted"
            fi
            echo "Releasing lock"
            rm "$NEXT_JOB_SCRIPT.lock"
            break
        else
            sleep 0.5
        fi
    done
fi

echo "Exiting"

#======END================================= 


