#!/bin/bash --login

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per job in array (i.e. tasks)
#SBATCH --array=0-39
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH -J "Circuitree-MCTS-Celery"   # job name
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

echo "Environment Variables (excluding BASH_FUNC variables)"
echo "====================================================="
env | grep -v '^BASH_FUNC\|^ \|}' # Skips BASH_FUNC variables, which clutter the output
echo "======================================================"
echo ""
echo ""

echo "Activating mamba environment"

mamba deactivate
mamba activate /home/pbhamidi/git/circuitree-paper/env
echo ""

WORKSPACE="/home/pbhamidi/git/circuitree-paper/oscillation"
echo "Changing dir to workspace: $WORKSPACE"
cd $WORKSPACE
echo ""

# Submit the next job array. Each task in the array will run once the corresponding task 
# in this array exits. To make sure it's submitted only once, we only submit from the first array task.
JOBSUBMIT="sbatch --dependency=aftercorr:$SLURM_JOB_ID --kill-on-invalid-dep=yes /home/pbhamidi/git/circuitree-paper/oscillation/submit_celery_worker.sh"
if [ "$SLURM_ARRAY_TASK_ID" == "0" ]; then
    echo ""
    echo "Submitting next job..."
    echo "$JOBSUBMIT"
    $JOBSUBMIT
fi

echo "Launching celery worker"
CMD="celery -A oscillation_parallel_celery.app worker -P processes"
echo $CMD
$CMD

echo "Exiting"

#======END================================= 
