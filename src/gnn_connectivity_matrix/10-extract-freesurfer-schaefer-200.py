#!/usr/bin/env python3
"""
Extract per-visit FreeSurfer structural morphometry features for the Schaefer 2018
parcellation (`schaefer200_subcort14` by default: 200 cortical parcels + 14
`aseg.stats` subcortical gray-matter nuclei = 214 ROIs), aligned 1-to-1 with the
functional connectivity matrices produced by `09-extract-fc-matrices.py`.

Why this script is needed
-------------------------
FreeSurfer `recon-all` computes cortical surface morphometry by default for only
three built-in atlases (`aparc` Desikan-Killiany 68, `aparc.a2009s` Destrieux 148,
and `aparc.DKTatlas` 62). Because `run_fmriprep.sh` archived every visit's full
`recon-all` folder into `<freesurfer-dir>/sub-*_ses-*.tar` on `/scratch`, we can
project the CBIG `fsaverage` Schaefer 2018 `.annot` onto each visit's native
surface mesh (`sphere.reg`) via `mri_surf2surf`, compute exact tetrahedral
(`-th3`) cortical morphometry via `mris_anatomical_stats` (plus optional cortical
gray-white intensity contrast via `mri_segstats` on `?h.w-g.pct.mgh`), and combine
those cortical features with the 14 subcortical gray-matter nuclei and global/QC
metrics (`eTIV`, Euler number, ventricle volume) from `aseg.stats`.

Per-session outputs (`<output-dir>/`, default `extracted_freesurfer_stats/schaefer200_subcort14`)
--------------------------------------------------------------------------------------------------
1. `<session_tag>.npz`:
   - `node_features` [214, 16] float32 — GNN-ready structural node feature matrix
     aligned row-for-row with `extracted_fc_matrices/schaefer200_subcort14/<session_tag>.npz`
   - `cortical_features` [200, 12] float32 — NumVert, SurfArea, GrayVol, ThickAvg,
     ThickStd, MeanCurv, GausCurv, FoldInd, CurvInd, WGPctMean, WGPctStd, WGPctSNR
   - `subcortical_features` [14, 7] float32 — NVoxels, Volume_mm3, normMean,
     normStdDev, normMin, normMax, normRange from `aseg.stats`
   - `roi_names` [214], `valid_roi_mask` [214], `node_feature_names` [16],
     `etiv_mm3`, `euler_number`, `lh_euler_number`, `rh_euler_number`,
     `lh_surface_holes`, `rh_surface_holes`, `ventricle_norm_etiv`, etc.
2. `stats/<session_tag>/{lh,rh}.Schaefer2018_200Parcels_7Networks_order.stats`
   (and `.w-g.pct.stats`), also synced into `extracted_freesurfer_stats/<session_tag>/stats/`
   when that directory exists.
3. `atlas_schaefer200_subcort14_roi_metadata.tsv` and `freesurfer_extraction_manifest.csv`.

Recommended Matilda HPC Workflow
--------------------------------
1. Login node (once, with internet access, before submitting Slurm jobs):
   uv run src/gnn_connectivity_matrix/10-extract-freesurfer-schaefer-200.py --prepare-atlas-only

2. Pilot test on 3 sessions (compute node or interactive `srun` with `Singularity/3.9.6`):
   module load Singularity/3.9.6
   uv run src/gnn_connectivity_matrix/10-extract-freesurfer-schaefer-200.py --limit 3 --workers 3

3a. Single multi-core Slurm job (~3,215 sessions in ~60-90 min on 16-32 CPUs):
   # Tip: wrapping the whole Python invocation in one `singularity exec` avoids
   # per-session container startup overhead, though running via `uv run` on the
   # host (auto-calling `singularity exec` per session) is also fully supported:
   module load Singularity/3.9.6
   uv run src/gnn_connectivity_matrix/10-extract-freesurfer-schaefer-200.py --workers 16

3b. Or Slurm array sharding (e.g., 8 shards x 8 CPUs -> ~20 min walltime):
   module load Singularity/3.9.6
   uv run src/gnn_connectivity_matrix/10-extract-freesurfer-schaefer-200.py \
       --workers 8 --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 8

4. Archive self-contained output folder (~100 MB) for transfer to local PC:
   tar -cf extracted_freesurfer_schaefer200_subcort14.tar \
       extracted_freesurfer_stats/schaefer200_subcort14/
"""

from __future__ import annotations

import os

for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
import multiprocessing as mp
from pathlib import Path
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Any
import urllib.request

import numpy as np
import pandas as pd


DEFAULT_REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS_LIST = DEFAULT_REPO_DIR / "sessions_list.tsv"
DEFAULT_FREESURFER_DIR = Path("/scratch/users/moghaddami/fmriprep-output/freesurfer")
DEFAULT_SIF_IMAGE = Path("/scratch/users/moghaddami/fmriprep-output/fmriprep-25.2.7.sif")
DEFAULT_FS_LICENSE = DEFAULT_REPO_DIR / "freesurfer_license.txt"
DEFAULT_STATS_BACKUP_DIR = DEFAULT_REPO_DIR / "extracted_freesurfer_stats"
DEFAULT_OUTPUT_ROOT = DEFAULT_REPO_DIR / "extracted_freesurfer_stats"
DEFAULT_ATLAS_CACHE_DIR = DEFAULT_REPO_DIR / "data" / "atlases"

CBIG_ANNOT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage/label/"
    "{hemi}.Schaefer2018_{n_cortical}Parcels_7Networks_order.annot"
)

CBIG_LUT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/MNI/Schaefer2018_{n_cortical}Parcels_7Networks_order.txt"
)

SUPPORTED_ATLASES: dict[str, dict[str, Any]] = {
    "schaefer100_subcort14": {
        "family": "schaefer",
        "n_cortical": 100,
        "include_subcortical": True,
        "stats_stem": "Schaefer2018_100Parcels_7Networks_order",
    },
    "schaefer200": {
        "family": "schaefer",
        "n_cortical": 200,
        "include_subcortical": False,
        "stats_stem": "Schaefer2018_200Parcels_7Networks_order",
    },
    "schaefer200_subcort14": {
        "family": "schaefer",
        "n_cortical": 200,
        "include_subcortical": True,
        "stats_stem": "Schaefer2018_200Parcels_7Networks_order",
    },
    "schaefer400_subcort14": {
        "family": "schaefer",
        "n_cortical": 400,
        "include_subcortical": True,
        "stats_stem": "Schaefer2018_400Parcels_7Networks_order",
    },
    "destrieux148": {
        "family": "destrieux",
        "n_cortical": 148,
        "include_subcortical": False,
        "stats_stem": "aparc.a2009s",
    },
    "destrieux148_subcort14": {
        "family": "destrieux",
        "n_cortical": 148,
        "include_subcortical": True,
        "stats_stem": "aparc.a2009s",
    },
}

# Exact 14 bilateral subcortical gray-matter nuclei matching 09-extract-fc-matrices.py
SUBCORTICAL_SPEC: list[tuple[str, str]] = [
    ("LH", "Left-Thalamus"),
    ("LH", "Left-Caudate"),
    ("LH", "Left-Putamen"),
    ("LH", "Left-Pallidum"),
    ("LH", "Left-Hippocampus"),
    ("LH", "Left-Amygdala"),
    ("LH", "Left-Accumbens-area"),
    ("RH", "Right-Thalamus"),
    ("RH", "Right-Caudate"),
    ("RH", "Right-Putamen"),
    ("RH", "Right-Pallidum"),
    ("RH", "Right-Hippocampus"),
    ("RH", "Right-Amygdala"),
    ("RH", "Right-Accumbens-area"),
]

# Handle FreeSurfer 5/6 vs FreeSurfer 7 Thalamus naming transparently if ever encountered
ASEG_STRUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "Left-Thalamus": ("Left-Thalamus", "Left-Thalamus-Proper"),
    "Right-Thalamus": ("Right-Thalamus", "Right-Thalamus-Proper"),
}

APARC_METRIC_COLS: tuple[str, ...] = (
    "NumVert",
    "SurfArea",
    "GrayVol",
    "ThickAvg",
    "ThickStd",
    "MeanCurv",
    "GausCurv",
    "FoldInd",
    "CurvInd",
)

WG_PCT_METRIC_COLS: tuple[str, ...] = (
    "WGPctMean",
    "WGPctStd",
    "WGPctSNR",
)

CORTICAL_FEATURE_NAMES: tuple[str, ...] = APARC_METRIC_COLS + WG_PCT_METRIC_COLS

SUBCORTICAL_FEATURE_NAMES: tuple[str, ...] = (
    "NVoxels",
    "Volume_mm3",
    "normMean",
    "normStdDev",
    "normMin",
    "normMax",
    "normRange",
)

NODE_FEATURE_NAMES: tuple[str, ...] = (
    "gray_vol_mm3",
    "gray_vol_norm_etiv",
    "surf_area_mm2",
    "surf_area_norm_etiv",
    "surf_area_norm_etiv23",
    "thick_avg_mm",
    "thick_std_mm",
    "mean_curv",
    "gaus_curv",
    "fold_ind",
    "curv_ind",
    "wg_pct_mean",
    "wg_pct_snr",
    "aseg_norm_mean",
    "aseg_norm_std",
    "is_subcortical",
)

IGNORED_PARCEL_NAMES: frozenset[str] = frozenset(
    {
        "unknown",
        "medial_wall",
        "background",
        "background+freesurfer_defined_medial_wall",
        "corpuscallosum",
    }
)

REQUIRED_MRI_FILENAMES: frozenset[str] = frozenset(
    {
        "wm.mgz",
        "orig.mgz",
        "ribbon.mgz",
        "lh.ribbon.mgz",
        "rh.ribbon.mgz",
        "aseg.presurf.mgz",
        "aseg.mgz",
        "brainmask.mgz",
        "norm.mgz",
    }
)

# Shell snippet matching FreeSurfer 7.3.2's own recon-all invocations (verified against
# lh.aparc.a2009s.stats and lh.w-g.pct.stats from fMRIPrep 25.2.7 on Matilda HPC).
FREESURFER_EXTRACT_SHELL = r"""
set -euo pipefail
export FREESURFER_HOME="${FREESURFER_HOME:-/opt/freesurfer}"
export FS_LICENSE="${FS_LICENSE_PATH}"
export SUBJECTS_DIR="${FS_SUBJECTS_DIR}"
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

SUB="${FS_SUBJ_ID}"
ANNOT_DIR="${FS_ANNOT_DIR}"
OUT_DIR="${FS_OUT_DIR}"
STATS_STEM="${FS_STATS_STEM}"
RUN_WG_PCT="${FS_RUN_WG_PCT}"

if [ ! -e "${SUBJECTS_DIR}/fsaverage" ]; then
  ln -s "${FREESURFER_HOME}/subjects/fsaverage" "${SUBJECTS_DIR}/fsaverage"
fi

mkdir -p "${SUBJECTS_DIR}/${SUB}/scripts" "${SUBJECTS_DIR}/${SUB}/label" "${OUT_DIR}"
cd "${SUBJECTS_DIR}/${SUB}/scripts"

for H in lh rh; do
  "${FREESURFER_HOME}/bin/mri_surf2surf" \
    --srcsubject fsaverage \
    --trgsubject "${SUB}" \
    --hemi "${H}" \
    --sval-annot "${ANNOT_DIR}/${H}.${STATS_STEM}.annot" \
    --tval "${SUBJECTS_DIR}/${SUB}/label/${H}.${STATS_STEM}.annot" \
    > /dev/null

  "${FREESURFER_HOME}/bin/mris_anatomical_stats" \
    -th3 -mgz -noglobal \
    -cortex "../label/${H}.cortex.label" \
    -f "${OUT_DIR}/${H}.${STATS_STEM}.stats" \
    -b -a "../label/${H}.${STATS_STEM}.annot" \
    -c "../label/${H}.${STATS_STEM}.ctab" \
    "${SUB}" "${H}" white \
    > /dev/null

  if [ "${RUN_WG_PCT}" = "1" ] && [ -s "${SUBJECTS_DIR}/${SUB}/surf/${H}.w-g.pct.mgh" ]; then
    "${FREESURFER_HOME}/bin/mri_segstats" \
      --in "${SUBJECTS_DIR}/${SUB}/surf/${H}.w-g.pct.mgh" \
      --annot "${SUB}" "${H}" "${STATS_STEM}" \
      --sum "${OUT_DIR}/${H}.${STATS_STEM}.w-g.pct.stats" \
      --snr \
      > /dev/null
  fi
done
"""

_WORKER_STATE: dict[str, Any] = {}


def _decode_text(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    return str(val).strip()


def _parse_schaefer_label(raw_label: str) -> tuple[str, str]:
    hemi = "Unknown"
    network = "Unknown"
    if "_LH_" in raw_label:
        hemi = "LH"
        after_hemi = raw_label.split("_LH_", 1)[1]
    elif "_RH_" in raw_label:
        hemi = "RH"
        after_hemi = raw_label.split("_RH_", 1)[1]
    else:
        after_hemi = raw_label
    if "_" in after_hemi:
        network = after_hemi.split("_", 1)[0]
    elif after_hemi:
        network = after_hemi
    return hemi, network


def _safe_float(token: str, default: float = 0.0) -> float:
    try:
        val = float(token)
        return val if math.isfinite(val) else default
    except (ValueError, TypeError):
        return default


def prepare_schaefer_annots(n_cortical: int, atlas_cache_dir: Path) -> Path:
    """
    Download (if not already cached) and validate CBIG fsaverage LH/RH `.annot`
    files for `Schaefer2018_{n_cortical}Parcels_7Networks_order`.
    """
    annot_dir = atlas_cache_dir / "schaefer_freesurfer_annot"
    annot_dir.mkdir(parents=True, exist_ok=True)
    stats_stem = f"Schaefer2018_{n_cortical}Parcels_7Networks_order"

    for hemi in ("lh", "rh"):
        dest = annot_dir / f"{hemi}.{stats_stem}.annot"
        if not dest.exists() or dest.stat().st_size < 1024:
            url = CBIG_ANNOT_URL_TEMPLATE.format(hemi=hemi, n_cortical=n_cortical)
            tmp_dest = annot_dir / f".{hemi}.{stats_stem}.{os.getpid()}.tmp.annot"
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, str(tmp_dest))
            if tmp_dest.stat().st_size < 1024:
                tmp_dest.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded annotation file appears truncated: {url}")
            os.replace(tmp_dest, dest)

    return annot_dir


def load_or_build_roi_metadata(
    atlas_name: str,
    atlas_cache_dir: Path,
    annot_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load ROI metadata table matching `09-extract-fc-matrices.py` so node ordering
    is guaranteed 1-to-1 with extracted FC matrices, ensuring `fs_struct_name`
    and `aseg_struct_name` are populated for every ROI.
    """
    if atlas_name not in SUPPORTED_ATLASES:
        raise ValueError(f"Unsupported atlas '{atlas_name}'. Choose from: {sorted(SUPPORTED_ATLASES)}")

    cfg = SUPPORTED_ATLASES[atlas_name]
    family: str = cfg["family"]
    n_cortical: int = cfg["n_cortical"]
    include_subcortical: bool = cfg["include_subcortical"]

    candidate_tsvs = [
        atlas_cache_dir / f"{atlas_name}_roi_metadata.tsv",
        DEFAULT_REPO_DIR / "extracted_fc_matrices" / atlas_name / f"atlas_{atlas_name}_roi_metadata.tsv",
    ]
    if atlas_name == "schaefer200":
        candidate_tsvs.extend(
            [
                atlas_cache_dir / "schaefer200_subcort14_roi_metadata.tsv",
                DEFAULT_REPO_DIR
                / "extracted_fc_matrices"
                / "schaefer200_subcort14"
                / "atlas_schaefer200_subcort14_roi_metadata.tsv",
            ]
        )

    meta_df: pd.DataFrame | None = None
    for candidate in candidate_tsvs:
        if candidate.exists():
            df = pd.read_csv(candidate, sep="\t")
            if not include_subcortical and "network" in df.columns:
                df = df.loc[df["network"] != "Subcortical"].reset_index(drop=True)
            meta_df = df
            break

    if meta_df is None:
        if family != "schaefer":
            raise FileNotFoundError(
                f"ROI metadata TSV for '{atlas_name}' not found in {atlas_cache_dir}. "
                f"Run `09-extract-fc-matrices.py --atlas {atlas_name} --prepare-atlas-only` first."
            )
        meta_df = _build_schaefer_roi_metadata_fallback(
            n_cortical=n_cortical,
            include_subcortical=include_subcortical,
            atlas_cache_dir=atlas_cache_dir,
            annot_dir=annot_dir,
        )

    if "fs_struct_name" not in meta_df.columns:
        meta_df["fs_struct_name"] = ""
    if "aseg_struct_name" not in meta_df.columns:
        meta_df["aseg_struct_name"] = ""

    meta_df["fs_struct_name"] = meta_df["fs_struct_name"].fillna("").astype(str)
    meta_df["aseg_struct_name"] = meta_df["aseg_struct_name"].fillna("").astype(str)

    is_subcort = meta_df["network"].astype(str) == "Subcortical"
    if family == "schaefer":
        cort_mask = ~is_subcort & (meta_df["fs_struct_name"].str.len() == 0)
        meta_df.loc[cort_mask, "fs_struct_name"] = meta_df.loc[cort_mask, "roi_name"]
    elif family == "destrieux":
        cort_mask = ~is_subcort & (meta_df["fs_struct_name"].str.len() == 0)
        meta_df.loc[cort_mask, "fs_struct_name"] = (
            meta_df.loc[cort_mask, "roi_name"]
            .astype(str)
            .str.replace(r"^(LH_|RH_)", "", regex=True)
        )

    sub_missing_aseg = is_subcort & (meta_df["aseg_struct_name"].str.len() == 0)
    meta_df.loc[sub_missing_aseg, "aseg_struct_name"] = (
        meta_df.loc[sub_missing_aseg, "roi_name"].astype(str).str.removeprefix("Subcort_")
    )
    sub_missing_fs = is_subcort & (meta_df["fs_struct_name"].str.len() == 0)
    meta_df.loc[sub_missing_fs, "fs_struct_name"] = meta_df.loc[sub_missing_fs, "aseg_struct_name"]

    expected_total = n_cortical + (len(SUBCORTICAL_SPEC) if include_subcortical else 0)
    if len(meta_df) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} ROIs in '{atlas_name}' metadata, found {len(meta_df)}."
        )

    # Validate against downloaded FreeSurfer `.annot` color tables when nibabel + annots exist.
    if family == "schaefer" and annot_dir is not None:
        _validate_annot_against_metadata(meta_df, annot_dir, cfg["stats_stem"])

    return meta_df


def _build_schaefer_roi_metadata_fallback(
    n_cortical: int,
    include_subcortical: bool,
    atlas_cache_dir: Path,
    annot_dir: Path | None,
) -> pd.DataFrame:
    """Build Schaefer ROI ordering from `.annot` ctab or CBIG LUT if NIfTI cache is absent."""
    labels: list[str] = []
    stats_stem = f"Schaefer2018_{n_cortical}Parcels_7Networks_order"

    if annot_dir is not None:
        try:
            from nibabel.freesurfer import io as fsio

            for hemi in ("lh", "rh"):
                annot_path = annot_dir / f"{hemi}.{stats_stem}.annot"
                if annot_path.exists():
                    _, _, names = fsio.read_annot(str(annot_path))
                    for raw in names:
                        decoded = _decode_text(raw)
                        if decoded and decoded.lower() not in IGNORED_PARCEL_NAMES:
                            labels.append(decoded)
        except Exception:
            labels = []

    if len(labels) != n_cortical:
        labels = []
        lut_dir = atlas_cache_dir / "cbig_schaefer_raw"
        lut_dir.mkdir(parents=True, exist_ok=True)
        lut_path = lut_dir / f"Schaefer2018_{n_cortical}Parcels_7Networks_order.txt"
        if not lut_path.exists():
            url = CBIG_LUT_URL_TEMPLATE.format(n_cortical=n_cortical)
            urllib.request.urlretrieve(url, str(lut_path))
        by_idx: dict[int, str] = {}
        with lut_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    by_idx[int(parts[0])] = parts[1]
        labels = [by_idx[i] for i in range(1, n_cortical + 1)]

    rows: list[dict[str, Any]] = []
    for idx, roi_name in enumerate(labels, start=1):
        hemi, network = _parse_schaefer_label(roi_name)
        rows.append(
            {
                "roi_index": idx,
                "roi_name": roi_name,
                "atlas_source": f"Schaefer2018_{n_cortical}Parcels_7Networks",
                "hemisphere": hemi,
                "network": network,
                "fs_struct_name": roi_name,
                "aseg_struct_name": "",
                "mni_x": np.nan,
                "mni_y": np.nan,
                "mni_z": np.nan,
                "template_voxel_count": 0,
            }
        )

    if include_subcortical:
        for offset, (hemi, aseg_name) in enumerate(SUBCORTICAL_SPEC, start=1):
            rows.append(
                {
                    "roi_index": n_cortical + offset,
                    "roi_name": f"Subcort_{aseg_name}",
                    "atlas_source": "HarvardOxford_sub_maxprob_thr25_2mm",
                    "hemisphere": hemi,
                    "network": "Subcortical",
                    "fs_struct_name": aseg_name,
                    "aseg_struct_name": aseg_name,
                    "mni_x": np.nan,
                    "mni_y": np.nan,
                    "mni_z": np.nan,
                    "template_voxel_count": 0,
                }
            )

    return pd.DataFrame(rows)


def _validate_annot_against_metadata(
    meta_df: pd.DataFrame,
    annot_dir: Path,
    stats_stem: str,
) -> None:
    """Verify `.annot` parcel names match the cortical `fs_struct_name` entries in `meta_df`."""
    lh_annot = annot_dir / f"lh.{stats_stem}.annot"
    rh_annot = annot_dir / f"rh.{stats_stem}.annot"
    if not (lh_annot.exists() and rh_annot.exists()):
        return
    try:
        from nibabel.freesurfer import io as fsio
    except ImportError:
        return

    for hemi_code, annot_path in (("LH", lh_annot), ("RH", rh_annot)):
        _, _, raw_names = fsio.read_annot(str(annot_path))
        annot_names = {
            _decode_text(n)
            for n in raw_names
            if _decode_text(n) and _decode_text(n).lower() not in IGNORED_PARCEL_NAMES
        }
        expected = set(
            meta_df.loc[
                (meta_df["network"] != "Subcortical") & (meta_df["hemisphere"] == hemi_code),
                "fs_struct_name",
            ].astype(str)
        )
        missing = sorted(expected - annot_names)
        if missing:
            raise RuntimeError(
                f"{annot_path.name} is missing {len(missing)} expected {hemi_code} parcels "
                f"(sample: {missing[:5]})."
            )


def load_sessions_list(sessions_path: Path) -> list[tuple[str, str]]:
    """Read whitespace/tab-delimited (subject, session) pairs from sessions_list.tsv."""
    sessions: list[tuple[str, str]] = []
    with sessions_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            sub_id = parts[0].removeprefix("sub-")
            ses_id = parts[1].removeprefix("ses-")
            if sub_id.lower() in {"subject", "sub", "participant_id"}:
                continue
            sessions.append((sub_id, ses_id))
    return sessions


def select_shard(
    sessions: list[tuple[str, str]],
    shard_index: int,
    num_shards: int,
) -> list[tuple[str, str]]:
    """Return 1-indexed shard slice of sessions using round-robin striping for load balance."""
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (1 <= shard_index <= num_shards):
        raise ValueError(f"--shard-index must be between 1 and {num_shards}, got {shard_index}.")
    return [ses for idx, ses in enumerate(sessions) if (idx % num_shards) == (shard_index - 1)]


def _classify_fs_relpath(member_name: str) -> tuple[str, str] | None:
    """
    Given a tar member or relative directory path (e.g.
    `sub-002S0295_ses-20120510/sub-002S0295_ses-20120510/surf/lh.white` or
    `sub-002S0295_ses-20120510/surf/lh.white`), return `(subj_id, rel_inside_subj)`
    if and only if this file is needed by `mri_surf2surf`, `mris_anatomical_stats`,
    `mri_segstats`, or `aseg.stats` parsing.
    """
    norm = posixpath.normpath(member_name.replace("\\", "/")).lstrip("/")
    if not norm or norm == "." or norm.startswith("../") or "/../" in f"/{norm}/":
        return None
    parts = [p for p in norm.split("/") if p and p != "."]
    if len(parts) < 3 or "fsaverage" in parts:
        return None

    # Locate the FreeSurfer standard subdirectory (`surf`, `label`, `mri`, `stats`)
    fs_subdirs = {"surf", "label", "mri", "stats"}
    subdir_idx: int | None = None
    for idx in range(len(parts) - 2, 0, -1):
        if parts[idx] in fs_subdirs:
            subdir_idx = idx
            break
    if subdir_idx is None:
        return None

    subj_id = parts[subdir_idx - 1]
    if not subj_id or subj_id == "fsaverage":
        return None

    subdir = parts[subdir_idx]
    rel_sub = parts[subdir_idx + 1 :]
    if not rel_sub:
        return None

    if subdir == "surf":
        # Extract all surface meshes, registrations, curvatures, thickness, w-g.pct,
        # and internal symlinks (crucially `lh.pial -> lh.pial.T1` in FreeSurfer 7!).
        if len(rel_sub) == 1:
            return subj_id, f"surf/{rel_sub[0]}"
        return None

    if subdir == "label":
        if len(rel_sub) == 1 and rel_sub[0] in {"lh.cortex.label", "rh.cortex.label"}:
            return subj_id, f"label/{rel_sub[0]}"
        return None

    if subdir == "mri":
        if len(rel_sub) == 1 and rel_sub[0] in REQUIRED_MRI_FILENAMES:
            return subj_id, f"mri/{rel_sub[0]}"
        if len(rel_sub) == 2 and rel_sub[0] == "transforms":
            return subj_id, f"mri/transforms/{rel_sub[1]}"
        return None

    if subdir == "stats":
        if len(rel_sub) == 1 and rel_sub[0] == "aseg.stats":
            return subj_id, "stats/aseg.stats"
        return None

    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ensure_pial_fallbacks(subj_dir: Path) -> None:
    """Ensure `?h.pial` resolves cleanly if FreeSurfer 7 stored `?h.pial.T1`."""
    surf_dir = subj_dir / "surf"
    for hemi in ("lh", "rh"):
        pial = surf_dir / f"{hemi}.pial"
        pial_t1 = surf_dir / f"{hemi}.pial.T1"
        if not pial.exists() and pial_t1.exists():
            if pial.is_symlink():
                pial.unlink()
            try:
                os.symlink(pial_t1.name, pial)
            except OSError:
                shutil.copy2(pial_t1, pial)


def stage_subject_from_tar(tar_path: Path, subjects_dir: Path) -> str:
    """
    Stream-extract only the required FreeSurfer surface/label/mri/stats files from
    `<session_tag>.tar` into `subjects_dir/<subj_id>/`, preserving internal symlinks
    (`lh.pial -> lh.pial.T1`) and skipping `fsaverage`.
    """
    subjects_dir.mkdir(parents=True, exist_ok=True)
    detected_subj: str | None = None
    deferred_symlinks: list[tuple[Path, Path, str]] = []
    deferred_hardlinks: list[tuple[Path, str]] = []

    with tarfile.open(tar_path, mode="r:*") as tf:
        for member in tf:
            classified = _classify_fs_relpath(member.name)
            if classified is None:
                continue
            subj_id, rel_inside = classified
            if detected_subj is None:
                detected_subj = subj_id
            elif subj_id != detected_subj:
                continue

            subj_root = (subjects_dir / subj_id).resolve()
            dest_path = (subj_root / rel_inside).resolve()
            if not str(dest_path).startswith(str(subj_root)):
                continue

            if member.isfile():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_f = tf.extractfile(member)
                if extracted_f is not None:
                    with extracted_f, dest_path.open("wb") as out_f:
                        shutil.copyfileobj(extracted_f, out_f, length=1024 * 1024)
            elif member.issym():
                deferred_symlinks.append((subj_root, dest_path, member.linkname))
            elif member.islnk():
                deferred_hardlinks.append((dest_path, member.linkname))

    if detected_subj is None:
        raise RuntimeError(f"No FreeSurfer subject directory found inside {tar_path.name}.")

    subj_dir = subjects_dir / detected_subj
    for subj_root, dest_path, link_target in deferred_symlinks:
        if posixpath.isabs(link_target):
            continue
        resolved_target = (dest_path.parent / link_target).resolve()
        if not str(resolved_target).startswith(str(subj_root)):
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() or dest_path.is_symlink():
            dest_path.unlink()
        try:
            os.symlink(link_target, dest_path)
        except OSError:
            if resolved_target.exists():
                shutil.copy2(resolved_target, dest_path)

    for dest_path, hard_target_name in deferred_hardlinks:
        target_cls = _classify_fs_relpath(hard_target_name)
        if target_cls is None or target_cls[0] != detected_subj:
            continue
        src_existing = subjects_dir / detected_subj / target_cls[1]
        if src_existing.exists():
            _link_or_copy(src_existing, dest_path)

    _ensure_pial_fallbacks(subj_dir)
    _verify_staged_subject(subj_dir)
    return detected_subj


def stage_subject_from_dir(raw_session_dir: Path, subjects_dir: Path) -> str:
    """Stage required files from an unarchived FreeSurfer directory without mutating it."""
    candidates: list[Path] = []
    if (raw_session_dir / "surf").is_dir() and raw_session_dir.name != "fsaverage":
        candidates.append(raw_session_dir)
    else:
        for child in sorted(raw_session_dir.iterdir()):
            if child.is_dir() and child.name != "fsaverage" and (child / "surf").is_dir():
                candidates.append(child)

    if not candidates:
        raise RuntimeError(f"No FreeSurfer subject directory with surf/ in {raw_session_dir}.")

    src_subj = candidates[0]
    subj_id = src_subj.name
    dst_subj = subjects_dir / subj_id
    for subdir in ("surf", "label", "mri", "stats"):
        src_sub = src_subj / subdir
        if not src_sub.is_dir():
            continue
        for item in src_sub.rglob("*"):
            if item.is_dir() and not item.is_symlink():
                continue
            rel = f"{subj_id}/{item.relative_to(src_subj).as_posix()}"
            classified = _classify_fs_relpath(rel)
            if classified is None:
                continue
            dst_item = dst_subj / classified[1]
            resolved = item.resolve()
            if resolved.is_file():
                _link_or_copy(resolved, dst_item)

    _ensure_pial_fallbacks(dst_subj)
    _verify_staged_subject(dst_subj)
    return subj_id


def _verify_staged_subject(subj_dir: Path) -> None:
    required = [
        subj_dir / "surf" / "lh.white",
        subj_dir / "surf" / "rh.white",
        subj_dir / "surf" / "lh.pial",
        subj_dir / "surf" / "rh.pial",
        subj_dir / "surf" / "lh.sphere.reg",
        subj_dir / "surf" / "rh.sphere.reg",
        subj_dir / "surf" / "lh.thickness",
        subj_dir / "surf" / "rh.thickness",
        subj_dir / "surf" / "lh.curv",
        subj_dir / "surf" / "rh.curv",
        subj_dir / "label" / "lh.cortex.label",
        subj_dir / "label" / "rh.cortex.label",
        subj_dir / "mri" / "wm.mgz",
    ]
    missing = [str(p.relative_to(subj_dir)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Incomplete FreeSurfer directory '{subj_dir.name}', missing: {missing}")


def detect_execution_backend(
    sif_image: Path,
    fs_license: Path,
    prefer_backend: str = "auto",
) -> tuple[str, str | None]:
    """
    Determine whether FreeSurfer binaries can be executed directly ('native', e.g.
    when this script itself is invoked inside `singularity exec fmriprep-25.2.7.sif`)
    or via Singularity/Apptainer ('container').
    """
    fs_home = Path(os.environ.get("FREESURFER_HOME", "/opt/freesurfer"))
    has_native = (
        (fs_home / "bin" / "mri_surf2surf").is_file()
        and (fs_home / "bin" / "mris_anatomical_stats").is_file()
        and (fs_home / "subjects" / "fsaverage").exists()
    ) or (
        shutil.which("mri_surf2surf") is not None
        and shutil.which("mris_anatomical_stats") is not None
    )

    if prefer_backend in ("auto", "native") and has_native:
        return "native", None
    if prefer_backend == "native":
        raise RuntimeError(
            "Requested --backend=native, but FreeSurfer `mri_surf2surf` / `mris_anatomical_stats` "
            "were not found in $FREESURFER_HOME/bin or PATH."
        )

    container_bin = shutil.which("singularity") or shutil.which("apptainer")
    if container_bin is None:
        raise RuntimeError(
            "Neither native FreeSurfer binaries nor `singularity`/`apptainer` were found on PATH. "
            "On Matilda HPC, run `module load Singularity/3.9.6` before launching."
        )
    if not sif_image.is_file():
        raise FileNotFoundError(f"Singularity image not found: {sif_image}")
    if not fs_license.is_file():
        raise FileNotFoundError(f"FreeSurfer license file not found: {fs_license}")

    return "container", container_bin


def run_freesurfer_surface_stats(
    subjects_dir: Path,
    subj_id: str,
    annot_dir: Path,
    tmp_out_dir: Path,
    stats_stem: str,
    include_wg_pct: bool,
    backend: str,
    container_bin: str | None,
    sif_image: Path,
    fs_license: Path,
) -> None:
    """Execute `mri_surf2surf` + `mris_anatomical_stats` (+ optional `w-g.pct` `mri_segstats`)."""
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    wg_flag = "1" if include_wg_pct else "0"

    if backend == "native":
        fs_home = os.environ.get("FREESURFER_HOME")
        if not fs_home:
            surf2surf_bin = shutil.which("mri_surf2surf")
            if surf2surf_bin:
                fs_home = str(Path(surf2surf_bin).resolve().parents[1])
            else:
                fs_home = "/opt/freesurfer"
        env = os.environ.copy()
        env.update(
            {
                "FREESURFER_HOME": fs_home,
                "FS_LICENSE_PATH": str(fs_license.resolve()),
                "FS_SUBJECTS_DIR": str(subjects_dir.resolve()),
                "FS_SUBJ_ID": subj_id,
                "FS_ANNOT_DIR": str(annot_dir.resolve()),
                "FS_OUT_DIR": str(tmp_out_dir.resolve()),
                "FS_STATS_STEM": stats_stem,
                "FS_RUN_WG_PCT": wg_flag,
            }
        )
        cmd = ["bash", "-c", FREESURFER_EXTRACT_SHELL]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    else:
        assert container_bin is not None
        # Bind onto `/fsdir`, `/data`, `/out`, and `/opt/freesurfer/license.txt` — the exact
        # mount points validated across all 3,218 fMRIPrep jobs in `scripts/run_fmriprep.sh`.
        cmd = [
            container_bin,
            "exec",
            "--cleanenv",
            "-B",
            f"{subjects_dir.resolve()}:/fsdir",
            "-B",
            f"{annot_dir.resolve()}:/data:ro",
            "-B",
            f"{tmp_out_dir.resolve()}:/out",
            "-B",
            f"{fs_license.resolve()}:/opt/freesurfer/license.txt:ro",
            "--env",
            "FS_LICENSE_PATH=/opt/freesurfer/license.txt",
            "--env",
            "FS_SUBJECTS_DIR=/fsdir",
            "--env",
            f"FS_SUBJ_ID={subj_id}",
            "--env",
            "FS_ANNOT_DIR=/data",
            "--env",
            "FS_OUT_DIR=/out",
            "--env",
            f"FS_STATS_STEM={stats_stem}",
            "--env",
            f"FS_RUN_WG_PCT={wg_flag}",
            str(sif_image.resolve()),
            "bash",
            "-c",
            FREESURFER_EXTRACT_SHELL,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"FreeSurfer command exited with {proc.returncode}: {err_tail}")


def parse_aparc_stats_file(stats_path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    Parse a FreeSurfer `mris_anatomical_stats` `.stats` file.
    Returns `(header_measures, {StructName: float64[9]})` for:
    `NumVert, SurfArea, GrayVol, ThickAvg, ThickStd, MeanCurv, GausCurv, FoldInd, CurvInd`.
    """
    measures: dict[str, float] = {}
    parcels: dict[str, np.ndarray] = {}

    with stats_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# Measure "):
                parts = [p.strip() for p in line[len("# Measure ") :].split(",")]
                if len(parts) >= 4:
                    val = _safe_float(parts[3], default=np.nan)
                    measures[parts[1]] = val
                    measures[f"{parts[0]}.{parts[1]}"] = val
                continue
            if line.startswith("#"):
                continue

            cols = line.split()
            if len(cols) < 10:
                continue
            struct_name = cols[0]
            if struct_name.lower() in IGNORED_PARCEL_NAMES:
                continue
            vals = np.array([_safe_float(c, 0.0) for c in cols[1:10]], dtype=np.float64)
            parcels[struct_name] = vals

    return measures, parcels


def parse_wg_pct_stats_file(stats_path: Path) -> dict[str, np.ndarray]:
    """
    Parse a FreeSurfer `mri_segstats --snr` gray-white contrast `.w-g.pct.stats` file.
    Columns: `Index SegId NVertices Area_mm2 StructName Mean StdDev Min Max Range SNR`
    Returns `{StructName: float64[3]}` -> `(WGPctMean, WGPctStd, WGPctSNR)`.
    """
    parcels: dict[str, np.ndarray] = {}
    if not stats_path.is_file() or stats_path.stat().st_size == 0:
        return parcels

    with stats_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 11:
                continue
            struct_name = cols[4]
            if struct_name.lower() in IGNORED_PARCEL_NAMES:
                continue
            mean_val = _safe_float(cols[5], 0.0)
            std_val = _safe_float(cols[6], 0.0)
            snr_val = _safe_float(cols[10], 0.0)
            parcels[struct_name] = np.array([mean_val, std_val, snr_val], dtype=np.float64)

    return parcels


def parse_aseg_stats_file(aseg_path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    Parse FreeSurfer `aseg.stats`.
    Returns `(global_measures, {StructName: float64[7]})` where the 7 columns are:
    `NVoxels, Volume_mm3, normMean, normStdDev, normMin, normMax, normRange`.
    """
    measures: dict[str, float] = {}
    structures: dict[str, np.ndarray] = {}

    with aseg_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# Measure "):
                parts = [p.strip() for p in line[len("# Measure ") :].split(",")]
                if len(parts) >= 4:
                    val = _safe_float(parts[3], default=np.nan)
                    measures[parts[0]] = val
                    measures[parts[1]] = val
                continue
            if line.startswith("#"):
                continue

            cols = line.split()
            if len(cols) < 10:
                continue
            # Columns: Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
            struct_name = cols[4]
            vals = np.array(
                [
                    _safe_float(cols[2], 0.0),
                    _safe_float(cols[3], 0.0),
                    _safe_float(cols[5], 0.0),
                    _safe_float(cols[6], 0.0),
                    _safe_float(cols[7], 0.0),
                    _safe_float(cols[8], 0.0),
                    _safe_float(cols[9], 0.0),
                ],
                dtype=np.float64,
            )
            structures[struct_name] = vals

    return measures, structures


def _lookup_cortical_parcel(
    hemi_parcels: dict[str, np.ndarray],
    fs_struct_name: str,
    hemi: str,
) -> np.ndarray | None:
    if fs_struct_name in hemi_parcels:
        return hemi_parcels[fs_struct_name]
    pref = f"{hemi.lower()}_{fs_struct_name}"
    if pref in hemi_parcels:
        return hemi_parcels[pref]
    return None


def assemble_session_features(
    meta_df: pd.DataFrame,
    lh_stats_path: Path,
    rh_stats_path: Path,
    aseg_stats_path: Path,
    lh_wg_path: Path | None = None,
    rh_wg_path: Path | None = None,
) -> dict[str, Any]:
    """
    Align LH/RH cortical stats + optional w-g.pct stats + aseg.stats onto the exact
    ROI row order of `meta_df` (matching `09-extract-fc-matrices.py` 1-to-1).
    """
    lh_meas, lh_parcels = parse_aparc_stats_file(lh_stats_path)
    rh_meas, rh_parcels = parse_aparc_stats_file(rh_stats_path)
    aseg_meas, aseg_structs = parse_aseg_stats_file(aseg_stats_path)

    lh_wg = parse_wg_pct_stats_file(lh_wg_path) if lh_wg_path is not None else {}
    rh_wg = parse_wg_pct_stats_file(rh_wg_path) if rh_wg_path is not None else {}
    wg_pct_available = bool(lh_wg and rh_wg)

    etiv_mm3 = float(
        aseg_meas.get("eTIV", aseg_meas.get("EstimatedTotalIntraCranialVol", np.nan))
    )
    if not math.isfinite(etiv_mm3) or etiv_mm3 <= 0.0:
        raise ValueError(f"Invalid or missing eTIV in {aseg_stats_path}: {etiv_mm3}")
    etiv_area_scale = etiv_mm3 ** (2.0 / 3.0)

    lh_holes = int(round(aseg_meas.get("lhSurfaceHoles", 0.0)))
    rh_holes = int(round(aseg_meas.get("rhSurfaceHoles", 0.0)))
    total_holes = int(round(aseg_meas.get("SurfaceHoles", lh_holes + rh_holes)))
    lh_euler = 2 - 2 * lh_holes
    rh_euler = 2 - 2 * rh_holes
    euler_total = 2 - 2 * (lh_holes + rh_holes)

    n_total = len(meta_df)
    cort_mask_df = meta_df["network"].astype(str) != "Subcortical"
    n_cortical = int(cort_mask_df.sum())
    n_subcort = n_total - n_cortical

    expected_lh = int(((meta_df["hemisphere"] == "LH") & cort_mask_df).sum())
    expected_rh = int(((meta_df["hemisphere"] == "RH") & cort_mask_df).sum())
    if len(lh_parcels) < max(1, int(0.9 * expected_lh)) or len(rh_parcels) < max(1, int(0.9 * expected_rh)):
        raise RuntimeError(
            f"Too few cortical parcels in stats files: lh={len(lh_parcels)} (expected {expected_lh}), "
            f"rh={len(rh_parcels)} (expected {expected_rh})."
        )

    cortical_features = np.zeros((n_cortical, len(CORTICAL_FEATURE_NAMES)), dtype=np.float32)
    subcortical_features = np.zeros((n_subcort, len(SUBCORTICAL_FEATURE_NAMES)), dtype=np.float32)
    node_features = np.zeros((n_total, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    valid_roi_mask = np.zeros(n_total, dtype=bool)

    cort_row_idx = 0
    sub_row_idx = 0
    n_valid_cort = 0
    n_valid_sub = 0

    for node_idx, row in enumerate(meta_df.itertuples(index=False)):
        hemi = str(row.hemisphere)
        is_sub = str(row.network) == "Subcortical"

        if not is_sub:
            fs_name = str(row.fs_struct_name)
            hemi_map = lh_parcels if hemi == "LH" else rh_parcels
            hemi_wg = lh_wg if hemi == "LH" else rh_wg

            aparc_vec = _lookup_cortical_parcel(hemi_map, fs_name, hemi)
            wg_vec = _lookup_cortical_parcel(hemi_wg, fs_name, hemi) if wg_pct_available else None

            if aparc_vec is not None and aparc_vec[0] > 0:
                valid_roi_mask[node_idx] = True
                n_valid_cort += 1
                cortical_features[cort_row_idx, :9] = aparc_vec.astype(np.float32)
                if wg_vec is not None:
                    cortical_features[cort_row_idx, 9:12] = wg_vec.astype(np.float32)

                surf_area = float(aparc_vec[1])
                gray_vol = float(aparc_vec[2])
                wg_mean = float(wg_vec[0]) if wg_vec is not None else 0.0
                wg_snr = float(wg_vec[2]) if wg_vec is not None else 0.0

                node_features[node_idx, :] = np.array(
                    [
                        gray_vol,
                        gray_vol / etiv_mm3,
                        surf_area,
                        surf_area / etiv_mm3,
                        surf_area / etiv_area_scale,
                        float(aparc_vec[3]),  # ThickAvg
                        float(aparc_vec[4]),  # ThickStd
                        float(aparc_vec[5]),  # MeanCurv
                        float(aparc_vec[6]),  # GausCurv
                        float(aparc_vec[7]),  # FoldInd
                        float(aparc_vec[8]),  # CurvInd
                        wg_mean,
                        wg_snr,
                        0.0,  # aseg_norm_mean
                        0.0,  # aseg_norm_std
                        0.0,  # is_subcortical
                    ],
                    dtype=np.float32,
                )
            cort_row_idx += 1
        else:
            aseg_name = str(row.aseg_struct_name)
            candidates = ASEG_STRUCT_ALIASES.get(aseg_name, (aseg_name,))
            aseg_vec = next((aseg_structs[c] for c in candidates if c in aseg_structs), None)
            if aseg_vec is None:
                raise KeyError(f"Subcortical ROI '{aseg_name}' missing from {aseg_stats_path}.")

            subcortical_features[sub_row_idx, :] = aseg_vec.astype(np.float32)
            vol_mm3 = float(aseg_vec[1])
            norm_mean = float(aseg_vec[2])
            norm_std = float(aseg_vec[3])
            if vol_mm3 > 0.0:
                valid_roi_mask[node_idx] = True
                n_valid_sub += 1

            node_features[node_idx, :] = np.array(
                [
                    vol_mm3,
                    vol_mm3 / etiv_mm3,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    norm_mean,
                    norm_std,
                    1.0,  # is_subcortical
                ],
                dtype=np.float32,
            )
            sub_row_idx += 1

    ventricle_vol = float(aseg_meas.get("VentricleChoroidVol", np.nan))
    ventricle_norm_etiv = float(ventricle_vol / etiv_mm3) if math.isfinite(ventricle_vol) else np.nan

    return {
        "node_features": node_features,
        "cortical_features": cortical_features,
        "subcortical_features": subcortical_features,
        "valid_roi_mask": valid_roi_mask,
        "wg_pct_available": wg_pct_available,
        "n_valid_rois": int(np.sum(valid_roi_mask)),
        "n_valid_cortical_rois": n_valid_cort,
        "n_valid_subcortical_rois": n_valid_sub,
        "etiv_mm3": etiv_mm3,
        "euler_number": euler_total,
        "lh_euler_number": lh_euler,
        "rh_euler_number": rh_euler,
        "lh_surface_holes": lh_holes,
        "rh_surface_holes": rh_holes,
        "total_surface_holes": total_holes,
        "brainseg_vol_mm3": float(aseg_meas.get("BrainSegVol", np.nan)),
        "brainseg_not_vent_mm3": float(aseg_meas.get("BrainSegVolNotVent", np.nan)),
        "ventricle_choroid_vol_mm3": ventricle_vol,
        "ventricle_norm_etiv": ventricle_norm_etiv,
        "cortex_vol_mm3": float(aseg_meas.get("CortexVol", np.nan)),
        "cerebral_wm_vol_mm3": float(aseg_meas.get("CerebralWhiteMatterVol", np.nan)),
        "subcort_gray_vol_mm3": float(aseg_meas.get("SubCortGrayVol", np.nan)),
        "total_gray_vol_mm3": float(aseg_meas.get("TotalGrayVol", np.nan)),
        "lh_mean_thickness_mm": float(lh_meas.get("MeanThickness", np.nan)),
        "rh_mean_thickness_mm": float(rh_meas.get("MeanThickness", np.nan)),
        "lh_white_surf_area_mm2": float(lh_meas.get("WhiteSurfArea", np.nan)),
        "rh_white_surf_area_mm2": float(rh_meas.get("WhiteSurfArea", np.nan)),
    }


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _find_existing_stats_pair(
    session_tag: str,
    stats_stem: str,
    output_dir: Path,
    stats_backup_dir: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Locate existing LH/RH `.stats` and optional `.w-g.pct.stats` if already extracted."""
    search_dirs: list[Path] = [output_dir / "stats" / session_tag]
    if stats_backup_dir is not None:
        search_dirs.append(stats_backup_dir / session_tag / "stats")

    for d in search_dirs:
        lh_s = d / f"lh.{stats_stem}.stats"
        rh_s = d / f"rh.{stats_stem}.stats"
        if lh_s.is_file() and lh_s.stat().st_size > 0 and rh_s.is_file() and rh_s.stat().st_size > 0:
            lh_wg_candidates = [d / f"lh.{stats_stem}.w-g.pct.stats"]
            rh_wg_candidates = [d / f"rh.{stats_stem}.w-g.pct.stats"]
            if stats_stem == "aparc":
                lh_wg_candidates.append(d / "lh.w-g.pct.stats")
                rh_wg_candidates.append(d / "rh.w-g.pct.stats")
            lh_wg = next((p for p in lh_wg_candidates if p.is_file() and p.stat().st_size > 0), None)
            rh_wg = next((p for p in rh_wg_candidates if p.is_file() and p.stat().st_size > 0), None)
            return lh_s, rh_s, lh_wg, rh_wg

    return None, None, None, None


def _init_worker(
    meta_df: pd.DataFrame,
    atlas_name: str,
    stats_stem: str,
    family: str,
    freesurfer_dir: str,
    stats_backup_dir: str | None,
    output_dir: str,
    annot_dir: str | None,
    sif_image: str,
    fs_license: str,
    backend: str | None,
    container_bin: str | None,
    tmp_dir: str | None,
    include_wg_pct: bool,
    sync_backup_stats: bool,
    overwrite: bool,
) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "meta_df": meta_df,
            "roi_names_arr": np.asarray(meta_df["roi_name"].astype(str).tolist(), dtype=str),
            "atlas_name": atlas_name,
            "stats_stem": stats_stem,
            "family": family,
            "freesurfer_dir": Path(freesurfer_dir),
            "stats_backup_dir": Path(stats_backup_dir) if stats_backup_dir else None,
            "output_dir": Path(output_dir),
            "annot_dir": Path(annot_dir) if annot_dir else None,
            "sif_image": Path(sif_image),
            "fs_license": Path(fs_license),
            "backend": backend,
            "container_bin": container_bin,
            "tmp_dir": Path(tmp_dir) if tmp_dir else None,
            "include_wg_pct": include_wg_pct,
            "sync_backup_stats": sync_backup_stats,
            "overwrite": overwrite,
        }
    )


def process_single_session(sub_id: str, ses_id: str) -> dict[str, Any]:
    """Extract Schaefer FreeSurfer stats and build the FC-aligned `.npz` for one session."""
    t0 = time.monotonic()
    session_tag = f"sub-{sub_id}_ses-{ses_id}"
    output_dir: Path = _WORKER_STATE["output_dir"]
    stats_stem: str = _WORKER_STATE["stats_stem"]
    family: str = _WORKER_STATE["family"]
    stats_backup_dir: Path | None = _WORKER_STATE["stats_backup_dir"]
    overwrite: bool = _WORKER_STATE["overwrite"]

    npz_path = output_dir / f"{session_tag}.npz"
    out_session_stats_dir = output_dir / "stats" / session_tag

    base_record: dict[str, Any] = {
        "session_tag": session_tag,
        "BIDS_Subject": f"sub-{sub_id}",
        "BIDS_Session": f"ses-{ses_id}",
        "status": "error",
        "atlas": _WORKER_STATE["atlas_name"],
        "n_valid_rois": 0,
        "n_valid_cortical_rois": 0,
        "n_valid_subcortical_rois": 0,
        "wg_pct_available": False,
        "euler_number": np.nan,
        "lh_surface_holes": np.nan,
        "rh_surface_holes": np.nan,
        "etiv_mm3": np.nan,
        "ventricle_norm_etiv": np.nan,
        "lh_mean_thickness_mm": np.nan,
        "rh_mean_thickness_mm": np.nan,
        "elapsed_s": 0.0,
        "npz_path": str(npz_path),
        "error_message": "",
    }

    if npz_path.exists() and not overwrite:
        try:
            with np.load(npz_path, allow_pickle=False) as existing:
                base_record.update(
                    {
                        "status": "skipped_existing",
                        "n_valid_rois": int(np.sum(existing["valid_roi_mask"])),
                        "n_valid_cortical_rois": int(existing["n_valid_cortical_rois"]),
                        "n_valid_subcortical_rois": int(existing["n_valid_subcortical_rois"]),
                        "wg_pct_available": bool(existing["wg_pct_available"]),
                        "euler_number": int(existing["euler_number"]),
                        "lh_surface_holes": int(existing["lh_surface_holes"]),
                        "rh_surface_holes": int(existing["rh_surface_holes"]),
                        "etiv_mm3": round(float(existing["etiv_mm3"]), 2),
                        "ventricle_norm_etiv": round(float(existing["ventricle_norm_etiv"]), 6),
                        "lh_mean_thickness_mm": round(float(existing["lh_mean_thickness_mm"]), 4),
                        "rh_mean_thickness_mm": round(float(existing["rh_mean_thickness_mm"]), 4),
                        "elapsed_s": round(time.monotonic() - t0, 2),
                    }
                )
            return base_record
        except Exception:
            pass

    lh_stats, rh_stats, lh_wg, rh_wg = (None, None, None, None)
    if not overwrite or family == "destrieux":
        lh_stats, rh_stats, lh_wg, rh_wg = _find_existing_stats_pair(
            session_tag=session_tag,
            stats_stem=stats_stem,
            output_dir=output_dir,
            stats_backup_dir=stats_backup_dir,
        )

    aseg_backup = (
        stats_backup_dir / session_tag / "stats" / "aseg.stats"
        if stats_backup_dir is not None
        else None
    )
    aseg_path: Path | None = aseg_backup if (aseg_backup and aseg_backup.is_file()) else None

    tmp_base: Path | None = _WORKER_STATE["tmp_dir"]
    if tmp_base is None:
        for env_tmp in ("SLURM_TMPDIR", "TMPDIR"):
            env_val = os.environ.get(env_tmp)
            if env_val and Path(env_val).is_dir():
                tmp_base = Path(env_val)
                break

    tmp_root: Path | None = None
    ran_surface_projection = False

    try:
        need_projection = lh_stats is None or rh_stats is None or aseg_path is None
        if need_projection:
            if family != "schaefer":
                base_record["status"] = "missing_freesurfer"
                base_record["error_message"] = (
                    f"Existing {stats_stem}.stats or aseg.stats not found for {session_tag}."
                )
                base_record["elapsed_s"] = round(time.monotonic() - t0, 2)
                return base_record

            fs_dir: Path = _WORKER_STATE["freesurfer_dir"]
            tar_path = fs_dir / f"{session_tag}.tar"
            raw_dir = fs_dir / session_tag
            if not tar_path.is_file() and not raw_dir.is_dir():
                base_record["status"] = "missing_freesurfer"
                base_record["error_message"] = (
                    f"Neither {tar_path.name} nor directory {session_tag}/ found in {fs_dir}."
                )
                base_record["elapsed_s"] = round(time.monotonic() - t0, 2)
                return base_record

            backend: str | None = _WORKER_STATE["backend"]
            if backend is None:
                raise RuntimeError(
                    "Surface projection required, but FreeSurfer/Singularity backend is unavailable."
                )

            if tmp_base is not None:
                tmp_base.mkdir(parents=True, exist_ok=True)
            tmp_root = Path(
                tempfile.mkdtemp(
                    prefix=f"fs_schaefer_{session_tag}_",
                    dir=str(tmp_base) if tmp_base is not None else None,
                )
            )
            subjects_dir = tmp_root / "subjects"
            tmp_out_dir = tmp_root / "out_stats"

            if tar_path.is_file():
                subj_id = stage_subject_from_tar(tar_path, subjects_dir)
            else:
                subj_id = stage_subject_from_dir(raw_dir, subjects_dir)

            if aseg_path is None:
                staged_aseg = subjects_dir / subj_id / "stats" / "aseg.stats"
                if staged_aseg.is_file():
                    aseg_path = staged_aseg

            if aseg_path is None or not aseg_path.is_file():
                raise FileNotFoundError(f"aseg.stats not found in backup or archive for {session_tag}.")

            if lh_stats is None or rh_stats is None:
                annot_dir: Path | None = _WORKER_STATE["annot_dir"]
                if annot_dir is None:
                    raise RuntimeError("Annotation directory is not configured.")
                run_freesurfer_surface_stats(
                    subjects_dir=subjects_dir,
                    subj_id=subj_id,
                    annot_dir=annot_dir,
                    tmp_out_dir=tmp_out_dir,
                    stats_stem=stats_stem,
                    include_wg_pct=_WORKER_STATE["include_wg_pct"],
                    backend=backend,
                    container_bin=_WORKER_STATE["container_bin"],
                    sif_image=_WORKER_STATE["sif_image"],
                    fs_license=_WORKER_STATE["fs_license"],
                )
                ran_surface_projection = True

                # Persist generated `.stats` files atomically into `<output_dir>/stats/<session_tag>/`
                # and optionally into `extracted_freesurfer_stats/<session_tag>/stats/`.
                backup_stats_subdir = (
                    stats_backup_dir / session_tag / "stats"
                    if (_WORKER_STATE["sync_backup_stats"] and stats_backup_dir is not None)
                    else None
                )
                can_sync_backup = backup_stats_subdir is not None and backup_stats_subdir.is_dir()

                for generated in sorted(tmp_out_dir.glob(f"*.{stats_stem}*.stats")):
                    if generated.stat().st_size == 0:
                        continue
                    dest_primary = out_session_stats_dir / generated.name
                    _copy_atomic(generated, dest_primary)
                    if can_sync_backup and backup_stats_subdir is not None:
                        dest_backup = backup_stats_subdir / generated.name
                        if dest_backup.resolve() != dest_primary.resolve():
                            _copy_atomic(generated, dest_backup)

                lh_stats = out_session_stats_dir / f"lh.{stats_stem}.stats"
                rh_stats = out_session_stats_dir / f"rh.{stats_stem}.stats"
                cand_lh_wg = out_session_stats_dir / f"lh.{stats_stem}.w-g.pct.stats"
                cand_rh_wg = out_session_stats_dir / f"rh.{stats_stem}.w-g.pct.stats"
                lh_wg = cand_lh_wg if cand_lh_wg.is_file() else None
                rh_wg = cand_rh_wg if cand_rh_wg.is_file() else None

        assert lh_stats is not None and rh_stats is not None and aseg_path is not None
        assembled = assemble_session_features(
            meta_df=_WORKER_STATE["meta_df"],
            lh_stats_path=lh_stats,
            rh_stats_path=rh_stats,
            aseg_stats_path=aseg_path,
            lh_wg_path=lh_wg,
            rh_wg_path=rh_wg,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_npz = output_dir / f".{session_tag}.{os.getpid()}.tmp.npz"
        np.savez_compressed(
            tmp_npz,
            node_features=assembled["node_features"],
            node_feature_names=np.array(NODE_FEATURE_NAMES),
            cortical_features=assembled["cortical_features"],
            cortical_feature_names=np.array(CORTICAL_FEATURE_NAMES),
            subcortical_features=assembled["subcortical_features"],
            subcortical_feature_names=np.array(SUBCORTICAL_FEATURE_NAMES),
            roi_names=_WORKER_STATE["roi_names_arr"],
            valid_roi_mask=assembled["valid_roi_mask"],
            wg_pct_available=np.bool_(assembled["wg_pct_available"]),
            n_valid_cortical_rois=np.int32(assembled["n_valid_cortical_rois"]),
            n_valid_subcortical_rois=np.int32(assembled["n_valid_subcortical_rois"]),
            subject=np.array(f"sub-{sub_id}"),
            session=np.array(f"ses-{ses_id}"),
            atlas=np.array(_WORKER_STATE["atlas_name"]),
            etiv_mm3=np.float64(assembled["etiv_mm3"]),
            euler_number=np.int32(assembled["euler_number"]),
            lh_euler_number=np.int32(assembled["lh_euler_number"]),
            rh_euler_number=np.int32(assembled["rh_euler_number"]),
            lh_surface_holes=np.int32(assembled["lh_surface_holes"]),
            rh_surface_holes=np.int32(assembled["rh_surface_holes"]),
            total_surface_holes=np.int32(assembled["total_surface_holes"]),
            brainseg_vol_mm3=np.float64(assembled["brainseg_vol_mm3"]),
            brainseg_not_vent_mm3=np.float64(assembled["brainseg_not_vent_mm3"]),
            ventricle_choroid_vol_mm3=np.float64(assembled["ventricle_choroid_vol_mm3"]),
            ventricle_norm_etiv=np.float32(assembled["ventricle_norm_etiv"]),
            cortex_vol_mm3=np.float64(assembled["cortex_vol_mm3"]),
            cerebral_wm_vol_mm3=np.float64(assembled["cerebral_wm_vol_mm3"]),
            subcort_gray_vol_mm3=np.float64(assembled["subcort_gray_vol_mm3"]),
            total_gray_vol_mm3=np.float64(assembled["total_gray_vol_mm3"]),
            lh_mean_thickness_mm=np.float32(assembled["lh_mean_thickness_mm"]),
            rh_mean_thickness_mm=np.float32(assembled["rh_mean_thickness_mm"]),
            lh_white_surf_area_mm2=np.float32(assembled["lh_white_surf_area_mm2"]),
            rh_white_surf_area_mm2=np.float32(assembled["rh_white_surf_area_mm2"]),
        )
        os.replace(tmp_npz, npz_path)

        base_record.update(
            {
                "status": "completed" if ran_surface_projection else "assembled_from_existing_stats",
                "n_valid_rois": assembled["n_valid_rois"],
                "n_valid_cortical_rois": assembled["n_valid_cortical_rois"],
                "n_valid_subcortical_rois": assembled["n_valid_subcortical_rois"],
                "wg_pct_available": assembled["wg_pct_available"],
                "euler_number": assembled["euler_number"],
                "lh_surface_holes": assembled["lh_surface_holes"],
                "rh_surface_holes": assembled["rh_surface_holes"],
                "etiv_mm3": round(assembled["etiv_mm3"], 2),
                "ventricle_norm_etiv": round(assembled["ventricle_norm_etiv"], 6),
                "lh_mean_thickness_mm": round(assembled["lh_mean_thickness_mm"], 4),
                "rh_mean_thickness_mm": round(assembled["rh_mean_thickness_mm"], 4),
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
        )
    except Exception as exc:
        base_record["status"] = "failed"
        base_record["error_message"] = f"{type(exc).__name__}: {exc}"
        base_record["elapsed_s"] = round(time.monotonic() - t0, 2)
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)

    return base_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-visit FreeSurfer Schaefer 200 cortical morphometry + aseg "
            "subcortical features aligned 1-to-1 with 09-extract-fc-matrices.py."
        )
    )
    parser.add_argument(
        "--sessions-list",
        type=Path,
        default=DEFAULT_SESSIONS_LIST,
        help="Path to sessions_list.tsv containing (sub, ses) pairs.",
    )
    parser.add_argument(
        "--freesurfer-dir",
        type=Path,
        default=DEFAULT_FREESURFER_DIR,
        help="Directory holding per-session FreeSurfer `.tar` archives (or folders) on HPC.",
    )
    parser.add_argument(
        "--stats-backup-dir",
        type=Path,
        default=DEFAULT_STATS_BACKUP_DIR,
        help="Directory holding backed-up `sub-*_ses-*/stats/aseg.stats` folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to <repo>/extracted_freesurfer_stats/<atlas>).",
    )
    parser.add_argument(
        "--atlas",
        choices=sorted(SUPPORTED_ATLASES),
        default="schaefer200_subcort14",
        help="Parcellation atlas matching 09-extract-fc-matrices.py (default: schaefer200_subcort14).",
    )
    parser.add_argument(
        "--atlas-cache-dir",
        type=Path,
        default=DEFAULT_ATLAS_CACHE_DIR,
        help="Directory holding cached ROI metadata TSV and downloaded fsaverage `.annot` files.",
    )
    parser.add_argument(
        "--prepare-atlas-only",
        action="store_true",
        help="Download/validate CBIG fsaverage `.annot` files into --atlas-cache-dir and exit.",
    )
    parser.add_argument(
        "--sif-image",
        type=Path,
        default=DEFAULT_SIF_IMAGE,
        help="Path to fMRIPrep Singularity `.sif` image containing FreeSurfer 7.3.2.",
    )
    parser.add_argument(
        "--fs-license",
        type=Path,
        default=DEFAULT_FS_LICENSE,
        help="Path to FreeSurfer license file.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "container"),
        default="auto",
        help="Execution backend for FreeSurfer CLI tools (default: auto).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Temporary directory for staged surfaces (defaults to $SLURM_TMPDIR, $TMPDIR, or /tmp).",
    )
    parser.add_argument(
        "--no-wg-pct",
        action="store_true",
        help="Skip extracting cortical gray-white intensity contrast (`?h.w-g.pct.mgh`).",
    )
    parser.add_argument(
        "--no-sync-backup-stats",
        action="store_true",
        help="Do not copy generated `.stats` files into --stats-backup-dir/<session>/stats/.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel worker processes (default: min(8, cpu_count)).",
    )
    parser.add_argument("--shard-index", type=int, default=1, help="1-indexed Slurm array shard ID.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of Slurm shards.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of sessions to process (for pilot testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run surface projection and overwrite existing `.stats` and `.npz` files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SUPPORTED_ATLASES[args.atlas]
    family: str = cfg["family"]
    n_cortical: int = cfg["n_cortical"]
    stats_stem: str = cfg["stats_stem"]
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.atlas)

    annot_dir: Path | None = None
    if family == "schaefer":
        annot_dir = prepare_schaefer_annots(n_cortical=n_cortical, atlas_cache_dir=args.atlas_cache_dir)

    meta_df = load_or_build_roi_metadata(
        atlas_name=args.atlas,
        atlas_cache_dir=args.atlas_cache_dir,
        annot_dir=annot_dir,
    )
    print(f"Atlas metadata ready: {args.atlas} ({len(meta_df)} total ROIs, stem={stats_stem})")
    if annot_dir is not None:
        print(f"FreeSurfer fsaverage annotations ready: {annot_dir}")

    if args.prepare_atlas_only:
        return

    backend: str | None = None
    container_bin: str | None = None
    if family == "schaefer":
        try:
            backend, container_bin = detect_execution_backend(
                sif_image=args.sif_image,
                fs_license=args.fs_license,
                prefer_backend=args.backend,
            )
            print(f"FreeSurfer execution backend: {backend}")
        except Exception as exc:
            # Allow pure-Python assembly if `.stats` files were already generated previously;
            # sessions needing surface projection will report the backend error clearly.
            if args.backend != "auto":
                raise
            print(
                f"Note: FreeSurfer/Singularity backend unavailable ({exc}); "
                "only sessions with pre-existing `.stats` files can be assembled."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    dest_atlas_tsv = output_dir / f"atlas_{args.atlas}_roi_metadata.tsv"
    meta_df.to_csv(dest_atlas_tsv, sep="\t", index=False)

    all_sessions = load_sessions_list(args.sessions_list)
    shard_sessions = select_shard(all_sessions, args.shard_index, args.num_shards)
    if args.limit is not None:
        shard_sessions = shard_sessions[: args.limit]

    print(
        f"Processing {len(shard_sessions)} sessions "
        f"(shard {args.shard_index}/{args.num_shards}, workers={args.workers}, "
        f"atlas={args.atlas}) -> {output_dir}"
    )

    init_args = (
        meta_df,
        args.atlas,
        stats_stem,
        family,
        str(args.freesurfer_dir),
        str(args.stats_backup_dir) if args.stats_backup_dir else None,
        str(output_dir),
        str(annot_dir) if annot_dir else None,
        str(args.sif_image),
        str(args.fs_license),
        backend,
        container_bin,
        str(args.tmp_dir) if args.tmp_dir else None,
        not args.no_wg_pct,
        not args.no_sync_backup_stats,
        args.overwrite,
    )

    results: list[dict[str, Any]] = []
    t_start = time.monotonic()

    if args.workers <= 1:
        _init_worker(*init_args)
        for idx, (sub_id, ses_id) in enumerate(shard_sessions, start=1):
            rec = process_single_session(sub_id, ses_id)
            results.append(rec)
            if idx % 25 == 0 or idx == len(shard_sessions) or rec["status"] == "failed":
                err_suffix = f" | {rec['error_message'][:160]}" if rec["error_message"] else ""
                print(
                    f"[{idx}/{len(shard_sessions)}] {rec['session_tag']} -> {rec['status']} "
                    f"({rec['elapsed_s']:.1f}s){err_suffix}",
                    flush=True,
                )
    else:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=init_args,
        ) as executor:
            futures = {
                executor.submit(process_single_session, sub_id, ses_id): (sub_id, ses_id)
                for sub_id, ses_id in shard_sessions
            }
            for idx, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                results.append(rec)
                if idx % 25 == 0 or idx == len(shard_sessions) or rec["status"] == "failed":
                    err_suffix = f" | {rec['error_message'][:160]}" if rec["error_message"] else ""
                    print(
                        f"[{idx}/{len(shard_sessions)}] {rec['session_tag']} -> {rec['status']} "
                        f"({rec['elapsed_s']:.1f}s){err_suffix}",
                        flush=True,
                    )

    results.sort(key=lambda r: r["session_tag"])
    manifest_name = (
        f"freesurfer_extraction_manifest_shard_{args.shard_index:04d}.csv"
        if args.num_shards > 1
        else "freesurfer_extraction_manifest.csv"
    )
    manifest_path = output_dir / manifest_name
    if results:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    status_counts: dict[str, int] = {}
    for rec in results:
        status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1

    print(
        f"Done in {time.monotonic() - t_start:.1f}s | Status counts: {status_counts} | "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
