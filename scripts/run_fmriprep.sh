#!/bin/bash
#SBATCH --job-name=adni_fmriprep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=19G
#SBATCH --time=12:00:00
#SBATCH --array=1-2
# For the full run after the 2-session pilot test, replace line above with e.g. #SBATCH --array=3-3218%200
#SBATCH --output=/projects/adresearch/moghaddami/gnn-connectivity-matrix/slurm_logs/fmriprep_%A_%a.out
#SBATCH --error=/projects/adresearch/moghaddami/gnn-connectivity-matrix/slurm_logs/fmriprep_%A_%a.err
#SBATCH --mail-user=moghaddami@oakland.edu
#SBATCH --mail-type=END,FAIL

# 1. Load Singularity 3.9.6
module purge
module load DefaultModules
module load Singularity/3.9.6

set -euo pipefail

# 2. Define paths (Code/Logs in /projects, Heavy Data in /scratch)
REPO_DIR="/projects/adresearch/moghaddami/gnn-connectivity-matrix"
BIDS_DIR="/scratch/users/moghaddami/fmriprep-ready"
OUT_ROOT="/scratch/users/moghaddami/fmriprep-output"
SIF_IMAGE="${OUT_ROOT}/fmriprep-25.2.7.sif"
FS_LICENSE="${REPO_DIR}/freesurfer_license.txt"
TEMPLATEFLOW_DIR="${OUT_ROOT}/templateflow"

# 3. Get Subject & Session for this Slurm array task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${REPO_DIR}/sessions_list.tsv")
SUB=$(echo "$LINE" | awk '{print $1}')
SES=$(echo "$LINE" | awk '{print $2}')

SESSION_TAG="sub-${SUB}_ses-${SES}"
echo "=== Starting Task ${SLURM_ARRAY_TASK_ID}: ${SESSION_TAG} on node $(hostname) at $(date) ==="

# 4. Create isolated per-session directories on /scratch
SES_DERIV="${OUT_ROOT}/derivatives/${SESSION_TAG}"
SES_FSDIR="${OUT_ROOT}/freesurfer/${SESSION_TAG}"
WORK_DIR="${OUT_ROOT}/work/${SESSION_TAG}"

mkdir -p "${SES_DERIV}" "${SES_FSDIR}" "${WORK_DIR}"
trap 'rm -rf "${WORK_DIR}"' EXIT

# 5. Write BIDS filter JSON restricting both T1w and BOLD to this single session
FILTER_FILE="${WORK_DIR}/bids_filter.json"
cat <<JSON > "${FILTER_FILE}"
{
  "t1w": {"datatype": "anat", "session": "${SES}", "suffix": "T1w"},
  "bold": {"datatype": "func", "session": "${SES}", "suffix": "bold"}
}
JSON

# 6. Run fMRIPrep (with SliceTiming correction AND FreeSurfer recon-all enabled!)
export SINGULARITYENV_TEMPLATEFLOW_HOME="/opt/templateflow"

singularity run --cleanenv \
  -B "${BIDS_DIR}:/data:ro" \
  -B "${SES_DERIV}:/out" \
  -B "${SES_FSDIR}:/fsdir" \
  -B "${WORK_DIR}:/work" \
  -B "${FS_LICENSE}:/opt/freesurfer/license.txt:ro" \
  -B "${TEMPLATEFLOW_DIR}:/opt/templateflow" \
  "${SIF_IMAGE}" \
  /data /out participant \
  --participant-label "${SUB}" \
  --session-label "${SES}" \
  --bids-filter-file /work/bids_filter.json \
  --skip-bids-validation \
  --fs-subjects-dir /fsdir \
  --work-dir /work \
  --nprocs 4 \
  --omp-nthreads 4 \
  --mem-mb 18000 \
  --output-spaces MNI152NLin2009cAsym:res-2 \
  --cifti-output 91k \
  --notrack

# 7. Permanently back up FreeSurfer stats, Euler log, and fMRIPrep confounds TSV to /projects
STATS_BACKUP="${REPO_DIR}/extracted_freesurfer_stats/${SESSION_TAG}"
mkdir -p "${STATS_BACKUP}"

cp -r "${SES_FSDIR}"/sub-*/stats "${STATS_BACKUP}/"
cp "${SES_FSDIR}"/sub-*/scripts/recon-all.log "${STATS_BACKUP}/" 2>/dev/null || true
cp "${SES_DERIV}/sub-${SUB}/ses-${SES}/func/"*_desc-confounds_timeseries.tsv "${STATS_BACKUP}/"

# 8. Archive SES_FSDIR into a single .tar file on /scratch after verifying stats/aseg.stats was backed up safely
if [ -s "${STATS_BACKUP}/stats/aseg.stats" ]; then
  tar -cf "${SES_FSDIR}.tar" -C "${OUT_ROOT}/freesurfer" "${SESSION_TAG}" && rm -rf "${SES_FSDIR}"
else
  echo "WARNING: aseg.stats not found in backup; keeping ${SES_FSDIR} unarchived!"
fi

echo "=== Completed Task ${SLURM_ARRAY_TASK_ID}: ${SESSION_TAG} at $(date) ==="
