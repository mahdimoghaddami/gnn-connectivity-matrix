# HPC `fMRIPrep` Preprocessing Pipeline (Matilda Cluster)

This directory contains [`run_fmriprep.sh`](run_fmriprep.sh), the Slurm array script used to preprocess all **`3,218` longitudinal ADNI sessions** in parallel on Oakland University's **Matilda HPC cluster** using containerized **`fMRIPrep 25.2.7`** (`Singularity/3.9.6`) with FreeSurfer `recon-all` and slice-timing correction enabled.

---

## Why Each Session Is Run in Isolation

By default, passing a multi-session subject to `fMRIPrep` merges all longitudinal T1w scans of that participant across years into a single cross-time anatomical template and runs FreeSurfer `recon-all` once on that average. While appropriate for static cross-sectional studies, doing so would erase **visit-specific cortical thinning and hippocampal atrophy** needed by our longitudinal GNN + RNN.

[`run_fmriprep.sh`](run_fmriprep.sh) solves this by indexing `(subject, session)` pairs from `../sessions_list.tsv` and running each visit with:
- `--participant-label "${SUB}"` and `--session-label "${SES}"`
- An isolated per-session `--bids-filter-file` restricting both `T1w` and `bold` to `"session": "${SES}"`
- A dedicated per-session `--fs-subjects-dir` (`${OUT_ROOT}/freesurfer/sub-${SUB}_ses-${SES}`)

Every visit therefore receives its own independent FreeSurfer `stats/` tables, surface reconstructions, and co-registered BOLD derivatives.

---

## HPC Storage Layout (`/projects` vs `/scratch`)

| Purpose | Variable / Path on Matilda | Filesystem & Retention |
| :--- | :--- | :--- |
| Code, job list & logs | `REPO_DIR="/projects/adresearch/moghaddami/gnn-connectivity-matrix"` | `/projects` (permanent) |
| FreeSurfer license | `FS_LICENSE="${REPO_DIR}/freesurfer_license.txt"` | `/projects` (gitignored) |
| Permanent stats & confounds backup | `STATS_BACKUP="${REPO_DIR}/extracted_freesurfer_stats/${SESSION_TAG}"` | `/projects` (gitignored) |
| Slurm stdout / stderr logs | `${REPO_DIR}/slurm_logs/fmriprep_%A_%a.{out,err}` | `/projects` (gitignored) |
| Input BIDS dataset (`3,218` sessions) | `BIDS_DIR="/scratch/users/moghaddami/fmriprep-ready"` | `/scratch` (`10 TB` / `10.24M` file quota) |
| Output root | `OUT_ROOT="/scratch/users/moghaddami/fmriprep-output"` | `/scratch` |
| `fMRIPrep` container image | `SIF_IMAGE="${OUT_ROOT}/fmriprep-25.2.7.sif"` | `/scratch` |
| Pre-cached brain templates | `TEMPLATEFLOW_DIR="${OUT_ROOT}/templateflow"` | `/scratch` |
| Preprocessed volumetric + CIFTI derivatives | `SES_DERIV="${OUT_ROOT}/derivatives/${SESSION_TAG}"` | `/scratch` |
| Archived FreeSurfer reconstructions | `${OUT_ROOT}/freesurfer/${SESSION_TAG}.tar` | `/scratch` (`~450` files collapsed to `1` `.tar`) |
| Transient Nipype work directory | `WORK_DIR="${OUT_ROOT}/work/${SESSION_TAG}"` | `/scratch` (auto-deleted via `trap` on exit) |

---

## One-Time Setup on the Matilda Login Node

Perform these steps once before submitting `sbatch scripts/run_fmriprep.sh` (assuming `data/fmriprep-ready/` has already been uploaded to `/scratch/users/moghaddami/fmriprep-ready`).

### Step 1: Create Directories & Generate `sessions_list.tsv`

Generate `sessions_list.tsv` directly from the uploaded BIDS directories so every `(subject, session)` entry matches the exact folder names on `/scratch`:

```bash
# 1. Create output & Singularity cache directories on /scratch
mkdir -p /scratch/users/moghaddami/fmriprep-output/{derivatives,freesurfer,work,templateflow,.singularity_cache}

# 2. Create log and permanent backup folders inside the repo on /projects
cd /projects/adresearch/moghaddami/gnn-connectivity-matrix
mkdir -p slurm_logs scripts extracted_freesurfer_stats

# 3. Scan /scratch/users/moghaddami/fmriprep-ready for all (sub, ses) pairs
find /scratch/users/moghaddami/fmriprep-ready -mindepth 2 -maxdepth 2 -type d -name "ses-*" | \
  awk -F'/' '{
    sub_id = $(NF-1); gsub(/^sub-/, "", sub_id);
    ses_id = $NF;     gsub(/^ses-/, "", ses_id);
    print sub_id "\t" ses_id
  }' | sort > /projects/adresearch/moghaddami/gnn-connectivity-matrix/sessions_list.tsv

# 4. Verify (expected: 3218 lines, e.g. "002S0295\t20120510")
wc -l sessions_list.tsv
head -n 5 sessions_list.tsv
```

### Step 2: Add Your FreeSurfer License

FreeSurfer `recon-all` requires a valid FreeSurfer license file (excluded from Git via [`.gitignore`](../.gitignore)):

```bash
nano /projects/adresearch/moghaddami/gnn-connectivity-matrix/freesurfer_license.txt
```

Paste your FreeSurfer license lines, save, and verify Unix `LF` line endings (FreeSurfer's license parser rejects Windows `CRLF` endings):

```bash
sed -i 's/\r$//' /projects/adresearch/moghaddami/gnn-connectivity-matrix/freesurfer_license.txt
```

### Step 3: Pull `fMRIPrep 25.2.7` & Cache TemplateFlow on `/scratch`

Compute nodes on Matilda should not rely on internet access or concurrent template downloads across 100+ parallel jobs. Pull the container and pre-cache all required TemplateFlow spaces **from inside the Singularity image** (avoiding host `pip --user` pollution in `/home`):

```bash
# 1. Load Singularity and keep its cache on /scratch (never /home)
module load Singularity/3.9.6
export SINGULARITY_CACHEDIR="/scratch/users/moghaddami/fmriprep-output/.singularity_cache"

# 2. Pull fMRIPrep 25.2.7 container image
singularity pull /scratch/users/moghaddami/fmriprep-output/fmriprep-25.2.7.sif docker://nipreps/fmriprep:25.2.7

# 3. Pre-download TemplateFlow templates using the Python environment inside the .sif container
singularity exec --cleanenv \
  -B /scratch/users/moghaddami/fmriprep-output/templateflow:/opt/templateflow \
  /scratch/users/moghaddami/fmriprep-output/fmriprep-25.2.7.sif \
  python -c "
import os
os.environ['TEMPLATEFLOW_HOME'] = '/opt/templateflow'
import templateflow.api as tfa
for t in ['MNI152NLin2009cAsym', 'MNI152NLin6Asym', 'OASIS30ANTs', 'fsLR', 'fsaverage']:
    print(f'Caching {t}...')
    tfa.get(t)
print('All templates cached successfully!')
"
```

Verify both the `.sif` image and cached templates exist before submitting jobs:

```bash
ls -lh /scratch/users/moghaddami/fmriprep-output/fmriprep-25.2.7.sif
ls -lh /scratch/users/moghaddami/fmriprep-output/templateflow/
```

---

## Inside `run_fmriprep.sh`

[`run_fmriprep.sh`](run_fmriprep.sh) follows an 8-step workflow per array task:

1. **Module Loading (`module purge` / `Singularity/3.9.6`)** runs *before* `set -euo pipefail` so HPC module shell functions do not trip over `set -u`.
2. **Path Configuration** points code/logs to `/projects` and heavy data/caches to `/scratch`.
3. **Session Lookup** extracts `SUB` and `SES` from line `${SLURM_ARRAY_TASK_ID}` of `sessions_list.tsv` and forms `SESSION_TAG="sub-${SUB}_ses-${SES}"`.
4. **Isolated Workspace & Crash-Safe Cleanup** creates `SES_DERIV`, `SES_FSDIR`, and `WORK_DIR` on `/scratch`, registering `trap 'rm -rf "${WORK_DIR}"' EXIT` so failed or timed-out jobs never leave orphaned 15–25 GB temporary directories behind.
5. **BIDS Filter Generation** writes `${WORK_DIR}/bids_filter.json` restricting `t1w` (`anat`) and `bold` (`func`) to `session == "${SES}"`.
6. **`fMRIPrep 25.2.7` Execution** runs inside `singularity run --cleanenv` with `SINGULARITYENV_TEMPLATEFLOW_HOME="/opt/templateflow"`:

   | Flag | Rationale |
   | :--- | :--- |
   | `--participant-label "${SUB}"` & `--session-label "${SES}"` | Scopes PyBIDS indexing to a single subject/session for fast startup and clean per-visit workflow naming |
   | `--bids-filter-file /work/bids_filter.json` | Guarantees only the current session's `T1w` and `bold` are processed together |
   | `--skip-bids-validation` | Avoids startup failures from `master_data_file.csv` / custom `"SliceTimingSource"` JSON keys and prevents 120 parallel jobs from crawling 12,872 files simultaneously |
   | `--fs-subjects-dir /fsdir` | Saves visit-specific FreeSurfer `recon-all` surfaces and morphometry tables without overwriting other visits |
   | `--nprocs 4 --omp-nthreads 4` | Matches `#SBATCH --cpus-per-task=4` (optimal throughput since FreeSurfer `recon-all` is mostly 1–2-threaded) |
   | `--mem-mb 18000` | Caps Nipype below Slurm's `#SBATCH --mem=19G` allocation to prevent OOM kills |
   | `--output-spaces MNI152NLin2009cAsym:res-2` | Writes preprocessed 4D BOLD in standard 2 mm MNI volumetric space (omits redundant native `T1w` 4D BOLD resampling) |
   | `--cifti-output 91k` | Outputs HCP `fsLR 91k` surface grayordinate BOLD time series (`.dtseries.nii`) |
   | `--notrack` | Disables internet telemetry pings on offline compute nodes |
   | *(omits `--ignore slicetiming`)* | Ensures `fMRIPrep` applies AFNI `3dTshift` slice-timing correction using the `SliceTiming` metadata populated in Notebook `06` |

7. **Permanent `/projects` Backup** copies `"${SES_FSDIR}"/sub-*/stats`, `recon-all.log`, and `*_desc-confounds_timeseries.tsv` into `extracted_freesurfer_stats/${SESSION_TAG}/`. (Using the `sub-*` wildcard supports both `sub-${SUB}` and `sub-${SUB}_ses-${SES}` FreeSurfer folder naming.)
8. **Conditional `.tar` Archiving** verifies `${STATS_BACKUP}/stats/aseg.stats` is non-empty (`-s`), archives `${SES_FSDIR}` to `${SES_FSDIR}.tar`, and removes the uncompressed directory—preserving full cortical meshes/surfaces for future atlas projections (`mris_anatomical_stats`) while collapsing ~450 files into `1` `.tar` file per session.

### `#SBATCH` Resource Settings

| Directive | Value | Rationale |
| :--- | :--- | :--- |
| `--nodes=1`, `--ntasks=1` | `1` node, `1` task | `fMRIPrep` uses shared-memory multithreading within a single container |
| `--cpus-per-task` | `4` | Doubles cluster concurrency compared to 8 cores while finishing typical sessions in ~5–6 h |
| `--mem` | `19G` | Pilot peak RSS was `~4.6 GB`; `19G` provides safe headroom for 976-volume multiband `ADNI3` scans |
| `--time` | `12:00:00` | Routes cleanly to Matilda's `general-long` partition (7-day limit) and protects long `recon-all` runs |
| `--array` | `3-3218%120` | Submits sessions `3–3218` in one job (`MaxArraySize = 5001`), throttled to `120` concurrent tasks (`480` CPUs) so transient `work/` directories stay safely within the `10 TB` quota |
| `--mail-type` | `END` | Sends a single array-completion email without spamming per-task failure emails across 3,216 jobs |

---

## Pilot Verification & Running the Full Cohort

### Pilot Run Summary (Job `410309`, Tasks `1–2`)

The initial 2-session pilot (`sub-002S0295_ses-20120510` and `sub-002S0413_ses-20120515`) passed every check:
- **Slurm status:** Both completed (`State = COMPLETED`, `ExitCode = 0:0`), runtimes `05:19:05` and `05:13:41`, peak RAM (`MaxRSS`) `~4.6 GB`, `.err` logs `0` bytes.
- **Slice-timing confirmed:** Logs confirmed `BOLD series will be slice-timing corrected to an offset of 1.47s` using our reconstructed Philips `SliceTiming`.
- **Outputs & cleanup:** FreeSurfer `stats/` (`aseg.stats`, `aparc.stats`, `DKTatlas.stats`, `a2009s.stats`), `recon-all.log`, and `*_desc-confounds_timeseries.tsv` were backed up cleanly (~540 KB/session), `/scratch/users/moghaddami/fmriprep-output/freesurfer/` held `~770 MB` `.tar` archives, and `work/` was completely empty.
- **Cluster limit check:** Matilda `MaxArraySize = 5001`, allowing tasks `3–3218` to be submitted in a single array.

### Submitting Sessions `3–3218`

On the Matilda login node:

```bash
cd /projects/adresearch/moghaddami/gnn-connectivity-matrix
git pull

# Ensure Unix LF line endings and execute permission before submitting
sed -i 's/\r$//' scripts/run_fmriprep.sh
chmod +x scripts/run_fmriprep.sh

sbatch scripts/run_fmriprep.sh
```

### Monitoring & Post-Run Audit Commands

While running:

```bash
# Active/pending tasks
squeue -u $USER

# Follow a specific task log (replace <JOBID> and <TASKID>)
tail -f slurm_logs/fmriprep_<JOBID>_<TASKID>.out
```

After completion, run a full audit on exit codes, backups, `/scratch` outputs, and quotas:

```bash
# 1. Check summary of failed, timed-out, or OOM tasks (replace <JOBID>)
sacct -j <JOBID> --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED --format=JobID%18,State,ExitCode,Elapsed,MaxRSS

# 2. Verify permanent backups in /projects (expected: 3,218 each when complete)
ls -1 /projects/adresearch/moghaddami/gnn-connectivity-matrix/extracted_freesurfer_stats/*/stats/aseg.stats | wc -l
ls -1 /projects/adresearch/moghaddami/gnn-connectivity-matrix/extracted_freesurfer_stats/*/*_desc-confounds_timeseries.tsv | wc -l

# 3. Verify /scratch derivatives, FreeSurfer .tar archives, and empty work/
ls -1 /scratch/users/moghaddami/fmriprep-output/derivatives/*/sub-*/ses-*/func/*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz | wc -l
ls -1 /scratch/users/moghaddami/fmriprep-output/freesurfer/*.tar | wc -l
ls -A /scratch/users/moghaddami/fmriprep-output/work/

# 4. Check /scratch disk & inode quota
lfs quota -h -u $USER /scratch
```

### Selectively Rerunning Failed Tasks

If any individual tasks fail or hit walltime, extract their array indices from `sacct` and resubmit only those task IDs without modifying the script:

```bash
sbatch --array=145,892,1204 scripts/run_fmriprep.sh
```

*(Note: Because each session writes to isolated `SES_DERIV`, `SES_FSDIR`, and `WORK_DIR` folders, you may want to remove a partial failed `work/<SESSION_TAG>` or `freesurfer/<SESSION_TAG>` directory before resubmitting if a job crashed mid-run.)*

---

## Empirical Storage & Resource Budget (`3,218` Sessions)

Measured per-session sizes from pilot job `410309` (note that `976`-volume multiband `ADNI3` scans have larger 4D BOLD derivatives than `140`-volume single-band scans):

| Storage Location | Contents | Approx. Per Session | Projected Total (`3,218` Sessions) |
| :--- | :--- | :---: | :---: |
| `/scratch` (`fmriprep-ready/`) | Input BIDS NIfTIs + JSONs | `~65 MB` | **`~0.21 TB`** |
| `/scratch` (`derivatives/`) | Volumetric MNI BOLD + `91k` CIFTI + anat surfaces/reports | `~0.9–1.0 GB` | **`~2.9–3.2 TB`** |
| `/scratch` (`freesurfer/*.tar`) | Archived FreeSurfer `recon-all` directories | `~0.77 GB` | **`~2.48 TB`** (`3,218` files) |
| `/scratch` (`work/`) | Active Nipype scratch (auto-cleaned on exit) | `~15–20 GB` active | **`~1.5–2.5 TB` peak** at `%120` |
| **Total Persistent `/scratch`** | *(Quota: `10 TB` / `10.24M` files)* | — | **`~5.8 TB`** (leaves `~4.2 TB` headroom) |
| `/projects` (`extracted_freesurfer_stats/`) | `stats/`, `recon-all.log`, `confounds_timeseries.tsv` | `~540 KB` | **`~1.7 GB`** (permanent) |

---

## Troubleshooting & Notes

- **Windows `CRLF` Line Endings:** Editing `scripts/run_fmriprep.sh` or `freesurfer_license.txt` on Windows can introduce `\r\n` line endings that cause `sbatch` (`#!/bin/bash^M: bad interpreter`) or FreeSurfer license errors. Always ensure Unix `LF` endings on Matilda (`sed -i 's/\r$//' ...`).
- **Keep Large Output Folders Out of Git:** `slurm_logs/` and `extracted_freesurfer_stats/` are listed in [`.gitignore`](../.gitignore). Across all `3,218` sessions, those two folders contain `~74,000` files (`~3+ GB`), which would exceed GitHub's repository limits. Keep them on `/projects` and export compiled tabular summaries or transfer them via `tar`/`scp` after completion.
- **Downstream Files Actually Used:**
  - **Edges (2D Functional Connectivity):** `derivatives/.../func/*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz` (or `*_space-fsLR_den-91k_bold.dtseries.nii`) + `*_desc-confounds_timeseries.tsv`
  - **Structural Node Features & Euler QC:** `extracted_freesurfer_stats/<SESSION_TAG>/stats/{lh,rh}.aparc*.stats` and `aseg.stats` (plus `.tar` FreeSurfer folders on `/scratch` if extracting custom surface parcellations with `mris_anatomical_stats` later)
  - **Ignored:** Transient `work/` intermediates, redundant HTML report SVGs after visual spot-checks, and unused nuisance columns in the confounds TSV.
