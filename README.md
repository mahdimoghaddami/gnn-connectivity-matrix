# gnn-connectivity-matrix

Predicting longitudinal Alzheimer's disease progression (`CN` / `MCI` / `AD`) with a multimodal **Spatio-Temporal Graph Neural Network (GNN + RNN)** trained on resting-state fMRI functional connectivity and structural MRI brain morphometry from ADNI (`ADNIGO`, `ADNI2`, `ADNI3`, and `ADNI4`).

> **Work in Progress:** Dataset curation, BIDS standardization, metadata reconstruction, pre-`fMRIPrep` quality control, and the Slurm/Singularity HPC preprocessing pipeline are complete and validated on pilot runs. Full-cohort `fMRIPrep` + FreeSurfer preprocessing is currently configured for execution on HPC, while post-`fMRIPrep` connectome extraction and GNN + RNN modeling are upcoming.

---

## Project Overview

Alzheimer's disease involves coupled **functional disconnection** and **structural neurodegeneration** that evolve over years of clinical follow-up. Rather than collapsing each subject to a baseline diagnosis or a single static scan, this project constructs a longitudinal sequence of multimodal brain graphs per participant:

- **Graph Topology & Edge Weights (Resting-State fMRI):** Regional functional connectivity matrices computed from denoised resting-state BOLD time series parcellated in standard brain space (`MNI152NLin2009cAsym:res-2` and `fsLR 91k` CIFTI).
- **Multimodal Node Features (Structural T1w MRI via FreeSurfer `recon-all`):** Per-visit cortical thickness (`ThickAvg`, `ThickStd`), normalized gray-matter volumes (`GrayVol / eTIV`), subcortical volumes (e.g., bilateral hippocampus, amygdala, ventricles), surface area, and cortical curvature/folding metrics—allowing each visit to capture progressive brain atrophy independently.
- **Temporal Modeling (GNN + RNN):** Visit-level brain graphs aligned to consensus per-visit clinical diagnoses (`CN`, `MCI`, `AD`) from ADNI's `DXSUM` table to model longitudinal conversion and progression trajectories.

---

## Project Status

| Stage | Status | Details |
| :--- | :--- | :--- |
| **1. Raw ADNI Curation & Pairing** | **Completed** | Filtered LONI fMRI metadata to resting-state BOLD (`00`), paired each scan with its same-visit/nearest 3D T1w scan (`01`) |
| **2. BIDS Conversion & Salvage** | **Completed** | Converted DICOMs via `dcm2niix` (`02`), salvaged broken/mismatched scans (`03`), standardized `participants.tsv` and BIDS structure (`05`) |
| **3. Visit-Level Clinical Labeling** | **Completed** | Aligned exact per-visit `CN` / `MCI` / `AD` diagnoses from `DXSUM` (`04`) across all longitudinal visits (`0` missing labels) |
| **4. `SliceTiming` & Scanner QC** | **Completed** | Reconstructed missing `SliceTiming` for `1,278` scans (`06`), filtered `49` defective scans via adapted Table 2 criteria (`07`), built `data/fmriprep-ready/` (`08`) |
| **5. HPC Preprocessing (`fMRIPrep` + FreeSurfer)** | **In Progress** | Session-isolated Slurm array pipeline (`fMRIPrep 25.2.7` + `recon-all`) validated on 2-session pilot (`0` errors); configured for full `3–3218` run ([`scripts/README.md`](scripts/README.md)) |
| **6. Post-`fMRIPrep` QC & Connectomes** | **Planned** | Filter high-motion / poor-surface sessions (`framewise_displacement`, FreeSurfer Euler number), denoise BOLD, and extract 2D connectivity matrices with `nilearn` |
| **7. Spatio-Temporal GNN + RNN** | **Planned** | Build multimodal dynamic brain graphs and train/evaluate longitudinal progression models |

---

## Dataset Summary

### Curation & Attrition Funnel

Starting from the raw ADNI fMRI metadata export, strict pairing, conversion salvage, and scanner-parameter QC yield **`3,218` clean multimodal sessions** across **`1,322` unique participants**:

| Pipeline Stage | Sessions Remaining | Removed / Outcome | Reference |
| :--- | :---: | :--- | :--- |
| Raw LONI fMRI export (`all_fmri_10_19_2026.csv`) | `9,789` records | — | [`00-keep_only_fmri.ipynb`](src/gnn_connectivity_matrix/00-keep_only_fmri.ipynb) |
| Filtered resting-state BOLD (`only_bold_fmri.csv`) | `3,365` records | Removed non-BOLD / duplicate/unwanted series | [`00-keep_only_fmri.ipynb`](src/gnn_connectivity_matrix/00-keep_only_fmri.ipynb) |
| Deduplicated fMRI–T1w pairs (`clean_dedup_fmri_to_t1_master.csv`) | `3,278` sessions | Paired each fMRI with matched 3D T1w scan | [`01-smri-download-image-ids.ipynb`](src/gnn_connectivity_matrix/01-smri-download-image-ids.ipynb) |
| BIDS conversion + failed-scan salvage | `3,267` sessions | `3,242` converted first pass; replaced 1 misassigned T1w localizer (`I232236 → I232233`); salvaged `24` fMRI scans; excluded `11` unrecoverable truncated DICOMs | [`02`](src/gnn_connectivity_matrix/02-organize-in-bids-format.ipynb), [`03`](src/gnn_connectivity_matrix/03-analyze-failed-conversions.ipynb), [`05`](src/gnn_connectivity_matrix/05-finalize-bids-and-master-manifest.ipynb) |
| Pre-`fMRIPrep` scanner QC (`49` excluded) | `3,218` sessions | Excluded `28` slow TR (`≥ 5.0 s`), `15` partial-brain depth (`< 140 mm` / 36 slices), `4` aborted scans (`< 250 s`), `2` body-coil scans (`BODY` / `Q-Body`) | [`07-pre-fmriprep-scanner-qc.ipynb`](src/gnn_connectivity_matrix/07-pre-fmriprep-scanner-qc.ipynb) |
| **Final `fmriprep-ready` dataset (`master_data_file.csv`)** | **`3,218` sessions** | **`1,322` unique subjects; `12,872` scan/sidecar files (`~210 GB` hard-linked)** | [`08-create-fmriprep-ready-dataset.ipynb`](src/gnn_connectivity_matrix/08-create-fmriprep-ready-dataset.ipynb) |

### Final `3,218`-Session Cohort Breakdown

- **Per-Visit Clinical Diagnosis (`DXSUM.DIAGNOSIS`, `0` missing):**
  - `CN` (Cognitively Normal): **`1,516`** (`47.1%`)
  - `MCI` (Mild Cognitive Impairment): **`1,286`** (`40.0%`)
  - `AD` (Alzheimer's Disease / Dementia): **`416`** (`12.9%`)
  - *Why per-visit `DXSUM` over LONI `Group`:* Baseline `Group` never updates longitudinally (`0 / 1,329` subjects change `Group` across visits) and misses **`452` (`13.80%`)** clinical conversions and reversions across the `3,278` candidate sessions. `DXSUM` covers all ADNI phases through 2026 (`100.00%` agreement with `ADNIMERGE.DX` on the `1,897` overlapping visits where `ADNIMERGE.DX` is non-null).
- **ADNI Protocol Phases (`MRIProtocolPhase`):**
  - `ADNI3`: **`2,051`**
  - `ADNIGO/ADNI2`: **`858`**
  - `ADNI4`: **`307`**
  - Untagged in `MRIQC`: **`2`**
- **Scanner Manufacturers:**
  - `Philips`: **`1,232`**
  - `Siemens`: **`1,214`**
  - `GE`: **`770`**
  - Untagged in `MRIQC`: **`2`**

---

## Repository Structure

```text
gnn-connectivity-matrix/
├── README.md
├── pyproject.toml                       # uv / Python >=3.14 & PyTorch cu132 project config
├── uv.lock                              # Locked Python dependencies
├── .python-version
├── .gitignore                           # Excludes data/, freesurfer_license.txt, slurm_logs/, extracted_freesurfer_stats/
├── sessions_list.tsv                    # 3,218 (BIDS subject, session) pairs for Slurm array indexing
├── results/
│   └── dataset_creation/
│       ├── only_bold_fmri.csv                   # Filtered resting-state BOLD fMRI table (3,365 rows)
│       ├── clean_dedup_fmri_to_t1_master.csv    # Paired & diagnosed fMRI-T1w master table (3,278 rows)
│       ├── bids_conversion_manifest.csv         # Per-session BIDS conversion, salvage & QC status
│       ├── final_bids_dataset_manifest.csv      # Unified 56-column BIDS + QC manifest (3,278 rows)
│       └── master_data_file.csv                 # Streamlined 25-column manifest for the 3,218 ready sessions
├── scripts/
│   ├── README.md                        # Matilda HPC setup, Slurm execution, monitoring & storage guide
│   └── run_fmriprep.sh                  # Slurm array job script (fMRIPrep 25.2.7 + FreeSurfer + .tar backup)
└── src/
    └── gnn_connectivity_matrix/
        ├── __init__.py
        ├── 00-keep_only_fmri.ipynb
        ├── 01-smri-download-image-ids.ipynb
        ├── 02-organize-in-bids-format.ipynb
        ├── 03-analyze-failed-conversions.ipynb
        ├── 04-add-diagnosis.ipynb
        ├── 05-finalize-bids-and-master-manifest.ipynb
        ├── 06-populate-missing-slicetiming.ipynb
        ├── 07-pre-fmriprep-scanner-qc.ipynb
        └── 08-create-fmriprep-ready-dataset.ipynb
```

*Gitignored local/HPC directories:* `data/` (`raw_fmri/`, `raw_smri/`, `raw_bids/`, `fmriprep-ready/`, and ADNI CSV tables), `freesurfer_license.txt`, `slurm_logs/`, and `extracted_freesurfer_stats/`.

---

## Data Curation & BIDS Pipeline

Each stage of dataset preparation is documented and executed sequentially in [`src/gnn_connectivity_matrix/`](src/gnn_connectivity_matrix/):

| Notebook | Summary of Work |
| :--- | :--- |
| [`00-keep_only_fmri.ipynb`](src/gnn_connectivity_matrix/00-keep_only_fmri.ipynb) | Filters LONI fMRI metadata (`all_fmri_10_19_2026.csv`) to resting-state BOLD acquisitions (`only_bold_fmri.csv`). |
| [`01-smri-download-image-ids.ipynb`](src/gnn_connectivity_matrix/01-smri-download-image-ids.ipynb) | Pairs every resting-state fMRI scan with its corresponding 3D T1w scan, deduplicates repeats, and outputs `clean_dedup_fmri_to_t1_master.csv` (`3,278` sessions). |
| [`02-organize-in-bids-format.ipynb`](src/gnn_connectivity_matrix/02-organize-in-bids-format.ipynb) | Converts ADNI DICOMs into BIDS NIfTI + JSON (`data/raw_bids/`) via `dcm2niix`, using short Windows temporary junctions (`_winapi.CreateJunction`) to avoid `260`-char `MAX_PATH` errors and injecting `"TaskName": "rest"`. |
| [`03-analyze-failed-conversions.ipynb`](src/gnn_connectivity_matrix/03-analyze-failed-conversions.ipynb) | Replaces misassigned T1w scout `I232236` with true MPRAGE `I232233` (`100%` T1w conversion), salvages `24` of `35` failed fMRI conversions via slice interpolation / incomplete trailing-volume removal, documents `11` unrecoverable series, and logs provenance in `Comments` for future ablation studies. |
| [`04-add-diagnosis.ipynb`](src/gnn_connectivity_matrix/04-add-diagnosis.ipynb) | Assigns exact per-visit clinical diagnoses (`CN`, `MCI`, `AD`) via hierarchical `DXSUM` matching (with `CDR` fallback on 4 edge cases) and quantifies cross-source agreement against `ADNIMERGE`, `CDR`, and static baseline `Group`. |
| [`05-finalize-bids-and-master-manifest.ipynb`](src/gnn_connectivity_matrix/05-finalize-bids-and-master-manifest.ipynb) | Removes duplicate non-BIDS `_T1wa` files and the `11` `anat`-only incomplete folders, generates BIDS `participants.tsv`/`.json`, and builds `final_bids_dataset_manifest.csv`. |
| [`06-populate-missing-slicetiming.ipynb`](src/gnn_connectivity_matrix/06-populate-missing-slicetiming.ipynb) | Reconstructs `SliceTiming` across all **`1,278` missing BOLD JSON sidecars** (`1,154` `MAYO_NFQ_Exact` from `MAYOADIRL_MRI_FMRI_NFQ_29Oct2026.csv`, `47` `ADNI_Protocol_SingleBand_Inferred`, and `77` `ADNI_Protocol_MultiBand_Inferred` `ADNI4` Philips MB=3 scans with `MultibandAccelerationFactor: 3`), and trims the empty 49th MOSAIC padding tile on Siemens scan `sub-114S6039/ses-20170721` (`I879211`). Verified `100%` (`3,267 / 3,267`) sidecar compliance before QC. |
| [`07-pre-fmriprep-scanner-qc.ipynb`](src/gnn_connectivity_matrix/07-pre-fmriprep-scanner-qc.ipynb) | Adapts Rutherford et al. Table 2 scanner-parameter checks so valid `ADNI4` multiband scans (`TR = 1.5 s`, `51 × 3.0 mm = 153 mm`, `295.5 s`) are preserved while removing `49` genuinely defective sessions, leaving `3,218` QC-passed sessions across `1,322` subjects. |
| [`08-create-fmriprep-ready-dataset.ipynb`](src/gnn_connectivity_matrix/08-create-fmriprep-ready-dataset.ipynb) | Populates standalone `data/fmriprep-ready/` (`12,872` files across `3,218` sessions) using NTFS hard links for `.nii.gz` (`0` extra disk space for `~210 GB`) and copied JSON/TSV metadata, plus the clean 25-column [`master_data_file.csv`](results/dataset_creation/master_data_file.csv). |

---

## HPC Preprocessing (`fMRIPrep 25.2.7` + FreeSurfer `recon-all`)

Preprocessing is orchestrated on Oakland University's **Matilda HPC cluster** using Slurm job arrays and Singularity (`scripts/run_fmriprep.sh`):

- **Longitudinal Visit Isolation:** Every `(sub, ses)` pair is processed independently via `--participant-label`, `--session-label`, and a per-session `--bids-filter-file` with an isolated `--fs-subjects-dir`. This prevents `fMRIPrep` from merging longitudinal T1w scans into a single cross-time subject template, preserving per-visit cortical thickness and subcortical atrophy features.
- **Slice-Timing & Multimodal Outputs:** `SliceTiming` correction (`3dTshift`) is active (all `3,218` JSON sidecars populated), FreeSurfer `recon-all` is enabled, and functional derivatives are written in both volumetric (`MNI152NLin2009cAsym:res-2`) and cortical surface (`--cifti-output 91k`) spaces.
- **Quota-Safe Storage Design:** Lightweight FreeSurfer `stats/`, `recon-all.log`, and `*_desc-confounds_timeseries.tsv` are backed up permanently to `/projects`, while full per-session FreeSurfer folders are archived into single `.tar` files on `/scratch` and `work/` directories are auto-cleaned via `trap ... EXIT`.

See **[`scripts/README.md`](scripts/README.md)** for full setup commands, `#SBATCH` settings, monitoring/verification checks, and storage math.

---

## Planned Downstream Pipeline

1. **Post-`fMRIPrep` Quality Control:** Filter sessions using `framewise_displacement` (`FD`) motion thresholds from `*_desc-confounds_timeseries.tsv` and FreeSurfer surface reconstruction **Euler number** (`2 - 2 × (lhSurfaceHoles + rhSurfaceHoles)`) from `aseg.stats`.
2. **BOLD Denoising & Connectome Extraction:** Apply confound regression (motion, aCompCor/WM/CSF, scrubbing/censoring) and bandpass filtering in `nilearn` to extract regional BOLD time series and compute 2D functional connectivity matrices.
3. **Multimodal Graph & Sequence Construction:** Combine functional connectivity edge matrices with regional FreeSurfer structural features (`ThickAvg`, `GrayVol / eTIV`, `SurfArea / eTIV`, curvature), spatial coordinates, and visit-level clinical covariates (`Age`, `Sex`, `Δt`, `eTIV`-normalized ventricle volume) ordered chronologically per subject.
4. **Spatio-Temporal GNN + RNN Training:** Train and evaluate longitudinal graph-recurrent models for diagnostic classification (`CN` / `MCI` / `AD`) and future clinical progression prediction, including ablation studies (e.g., salvaged/interpolated scan sensitivity tracked via `Comments`).

---

## Local Environment & Data Access

- **Python Environment (`uv`):** Managed with [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) (`Python >= 3.14`). Key packages include `dcm2niix`, `pydicom`, `pandas`, `numpy`, and `torch` / `torchvision` (`pytorch-cu132`).
  ```bash
  uv sync
  ```
- **ADNI Data Access:** Raw DICOMs, NIfTIs, and clinical tables (`DXSUM`, `MRIQC`, `MAYOADIRL_MRI_FMRI_NFQ`, etc.) are subject to the ADNI Data Use Agreement and are excluded from version control via [`.gitignore`](.gitignore). Data must be downloaded directly from the [LONI Image & Data Archive (IDA)](https://ida.loni.usc.edu/) under an approved ADNI account.
