#!/usr/bin/env python3
"""
Extract denoised ROI BOLD time series, 2D functional connectivity matrices
(Pearson and Ledoit-Wolf partial correlation), and regional ALFF/fALFF from
fMRIPrep MNI152NLin2009cAsym:res-2 outputs on the Matilda HPC cluster.

Default Atlas: `schaefer200_subcort14` (214 whole-brain ROIs)
- Labels 1..200:   Schaefer 2018 200-parcel, 7-network cortical parcellation (2 mm MNI)
- Labels 201..214: 14 bilateral subcortical gray-matter nuclei from Harvard-Oxford
                   (sub-maxprob-thr25-2mm: Thalamus, Caudate, Putamen, Pallidum,
                    Hippocampus, Amygdala, Accumbens), matched directly to
                   FreeSurfer aseg.stats StructName identifiers.

Recommended Matilda HPC Workflow
--------------------------------
1. Login node (once, with internet access, before submitting Slurm jobs):
   python src/gnn_connectivity_matrix/09-extract-fc-matrices.py --prepare-atlas-only

2a. Single multi-core Slurm job (~3,215 sessions in ~45-90 min on 16-32 CPUs):
   python src/gnn_connectivity_matrix/09-extract-fc-matrices.py --workers 16

2b. Or Slurm array sharding (e.g., 16 shards x 4 CPUs):
   python src/gnn_connectivity_matrix/09-extract-fc-matrices.py \
       --workers 4 --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 16

3. Archive extracted .npz matrices for fast transfer to local PC:
   tar -cf extracted_fc_matrices_schaefer200_subcort14.tar extracted_fc_matrices/schaefer200_subcort14/
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP threads before importing numpy/scipy/sklearn to avoid
# CPU oversubscription when running ProcessPoolExecutor workers on HPC nodes.
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
import time
import urllib.request
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, signal, sparse
from sklearn.covariance import LedoitWolf


DEFAULT_REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS_LIST = DEFAULT_REPO_DIR / "sessions_list.tsv"
DEFAULT_DERIVATIVES_DIR = Path("/scratch/users/moghaddami/fmriprep-output/derivatives")
DEFAULT_OUTPUT_ROOT = DEFAULT_REPO_DIR / "extracted_fc_matrices"
DEFAULT_ATLAS_CACHE_DIR = DEFAULT_REPO_DIR / "data" / "atlases"

SUPPORTED_ATLASES: dict[str, int] = {
    "schaefer100_subcort14": 100,
    "schaefer200_subcort14": 200,
    "schaefer400_subcort14": 400,
}

# Canonical 14 bilateral subcortical gray-matter nuclei in Harvard-Oxford
# subcortical atlas, mapped to exact FreeSurfer aseg.stats StructName values.
SUBCORTICAL_SPEC: list[tuple[str, str, str]] = [
    ("left thalamus", "LH", "Left-Thalamus"),
    ("left caudate", "LH", "Left-Caudate"),
    ("left putamen", "LH", "Left-Putamen"),
    ("left pallidum", "LH", "Left-Pallidum"),
    ("left hippocampus", "LH", "Left-Hippocampus"),
    ("left amygdala", "LH", "Left-Amygdala"),
    ("left accumbens", "LH", "Left-Accumbens-area"),
    ("right thalamus", "RH", "Right-Thalamus"),
    ("right caudate", "RH", "Right-Caudate"),
    ("right putamen", "RH", "Right-Putamen"),
    ("right pallidum", "RH", "Right-Pallidum"),
    ("right hippocampus", "RH", "Right-Hippocampus"),
    ("right amygdala", "RH", "Right-Amygdala"),
    ("right accumbens", "RH", "Right-Accumbens-area"),
]

MOTION_BASE_COLS = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
PHYSIO_BASE_COLS = ("csf", "white_matter")
GLOBAL_BASE_COLS = ("global_signal",)
CONFOUND_SUFFIXES = ("", "_derivative1", "_power2", "_derivative1_power2")

_WORKER_STATE: dict[str, Any] = {}


def _decode_label(label: Any) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8", errors="replace").strip()
    return str(label).strip()


def _normalize_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").replace("-", " ").split())


def _parse_schaefer_label(raw_label: str) -> tuple[str, str]:
    """Extract hemisphere ('LH'/'RH') and Yeo-7 network name from Schaefer label."""
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


def resample_labels_nearest(
    src_data: np.ndarray,
    src_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor resample an integer 3D label volume to a target grid."""
    src_int = np.asarray(src_data, dtype=np.int32)
    if src_int.shape == target_shape and np.allclose(src_affine, target_affine, atol=1e-4):
        return src_int
    vox_to_vox = np.linalg.inv(src_affine) @ target_affine
    resampled = ndimage.affine_transform(
        src_int,
        matrix=vox_to_vox[:3, :3],
        offset=vox_to_vox[:3, 3],
        output_shape=target_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return resampled.astype(np.int32, copy=False)


def _fetch_schaefer_raw(n_cortical: int, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch Schaefer 2018 7-network 2mm MNI NIfTI + ordered ROI labels."""
    nilearn_dir = cache_dir / "nilearn_data"
    nilearn_dir.mkdir(parents=True, exist_ok=True)
    try:
        from nilearn import datasets

        bunch = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_cortical,
            yeo_networks=7,
            resolution_mm=2,
            data_dir=str(nilearn_dir),
            verbose=0,
        )
        img = nib.load(str(bunch.maps))
        data = np.asarray(img.dataobj, dtype=np.int32)
        labels = [_decode_label(lbl) for lbl in bunch.labels]
        # nilearn >=0.11 prepends 'Background' at index 0; older versions omit it.
        if len(labels) == n_cortical + 1 and _normalize_label(labels[0]) == "background":
            labels = labels[1:]
        if len(labels) != n_cortical:
            raise RuntimeError(
                f"Expected {n_cortical} Schaefer labels, got {len(labels)} from nilearn."
            )
        return data, img.affine, labels
    except ImportError:
        raw_dir = cache_dir / "cbig_schaefer_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        nii_name = f"Schaefer2018_{n_cortical}Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
        lut_name = f"Schaefer2018_{n_cortical}Parcels_7Networks_order.txt"
        base_url = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
            "Parcellations/MNI"
        )
        nii_path = raw_dir / nii_name
        lut_path = raw_dir / lut_name
        if not nii_path.exists():
            urllib.request.urlretrieve(f"{base_url}/{nii_name}", nii_path)
        if not lut_path.exists():
            urllib.request.urlretrieve(f"{base_url}/{lut_name}", lut_path)
        img = nib.load(str(nii_path))
        data = np.asarray(img.dataobj, dtype=np.int32)
        labels_by_idx: dict[int, str] = {}
        with lut_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    labels_by_idx[int(parts[0])] = parts[1]
        labels = [labels_by_idx[i] for i in range(1, n_cortical + 1)]
        return data, img.affine, labels


def _fetch_harvard_oxford_subcort(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch Harvard-Oxford sub-maxprob-thr25-2mm NIfTI + 0-indexed label list."""
    nilearn_dir = cache_dir / "nilearn_data"
    nilearn_dir.mkdir(parents=True, exist_ok=True)
    try:
        from nilearn import datasets

        bunch = datasets.fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-2mm",
            data_dir=str(nilearn_dir),
            verbose=0,
        )
        img = bunch.maps if isinstance(bunch.maps, nib.Nifti1Image) else nib.load(str(bunch.maps))
        data = np.asarray(img.dataobj, dtype=np.int32)
        labels = [_decode_label(lbl) for lbl in bunch.labels]
        return data, img.affine, labels
    except ImportError as exc:
        raise RuntimeError(
            "nilearn is required to fetch Harvard-Oxford subcortical atlas when not yet cached. "
            "Install nilearn (`pip install nilearn`) or pre-build the atlas cache with --prepare-atlas-only."
        ) from exc


def build_or_load_combined_atlas(
    atlas_name: str,
    atlas_cache_dir: Path,
) -> tuple[Path, Path, pd.DataFrame]:
    """
    Build (if not already cached) and return paths + metadata DataFrame for the
    combined Schaefer cortical + 14 Harvard-Oxford subcortical 2mm MNI atlas.
    """
    if atlas_name not in SUPPORTED_ATLASES:
        raise ValueError(f"Unsupported atlas '{atlas_name}'. Choose from: {list(SUPPORTED_ATLASES)}")

    atlas_cache_dir.mkdir(parents=True, exist_ok=True)
    atlas_nii_path = atlas_cache_dir / f"{atlas_name}_2mm.nii.gz"
    atlas_tsv_path = atlas_cache_dir / f"{atlas_name}_roi_metadata.tsv"

    if atlas_nii_path.exists() and atlas_tsv_path.exists():
        meta_df = pd.read_csv(atlas_tsv_path, sep="\t")
        return atlas_nii_path, atlas_tsv_path, meta_df

    n_cortical = SUPPORTED_ATLASES[atlas_name]
    schaefer_data, schaefer_affine, schaefer_labels = _fetch_schaefer_raw(n_cortical, atlas_cache_dir)
    ho_data, ho_affine, ho_labels = _fetch_harvard_oxford_subcort(atlas_cache_dir)

    ho_on_schaefer = resample_labels_nearest(
        ho_data,
        ho_affine,
        target_shape=schaefer_data.shape[:3],
        target_affine=schaefer_affine,
    )

    ho_norm_to_idx = {_normalize_label(lbl): idx for idx, lbl in enumerate(ho_labels)}

    combined = np.zeros(schaefer_data.shape[:3], dtype=np.int32)
    cortical_mask = (schaefer_data >= 1) & (schaefer_data <= n_cortical)
    combined[cortical_mask] = schaefer_data[cortical_mask]

    rows: list[dict[str, Any]] = []
    for roi_idx in range(1, n_cortical + 1):
        roi_name = schaefer_labels[roi_idx - 1]
        hemi, network = _parse_schaefer_label(roi_name)
        ijk = np.argwhere(combined == roi_idx)
        if ijk.size == 0:
            raise RuntimeError(f"Empty cortical parcel {roi_idx} ({roi_name}) in atlas.")
        xyz = nib.affines.apply_affine(schaefer_affine, ijk.mean(axis=0))
        rows.append(
            {
                "roi_index": roi_idx,
                "roi_name": roi_name,
                "atlas_source": f"Schaefer2018_{n_cortical}Parcels_7Networks",
                "hemisphere": hemi,
                "network": network,
                "aseg_struct_name": "",
                "mni_x": round(float(xyz[0]), 2),
                "mni_y": round(float(xyz[1]), 2),
                "mni_z": round(float(xyz[2]), 2),
                "template_voxel_count": int(ijk.shape[0]),
            }
        )

    for offset, (ho_key, hemi, aseg_name) in enumerate(SUBCORTICAL_SPEC, start=1):
        roi_idx = n_cortical + offset
        if ho_key not in ho_norm_to_idx:
            raise RuntimeError(
                f"Expected subcortical label '{ho_key}' not found in Harvard-Oxford labels: {ho_labels}"
            )
        ho_label_id = ho_norm_to_idx[ho_key]
        # Keep cortical ribbon labels intact if a deep nucleus boundary touches cortex.
        sub_mask = (ho_on_schaefer == ho_label_id) & (combined == 0)
        if not np.any(sub_mask):
            raise RuntimeError(f"Empty subcortical ROI {roi_idx} ({aseg_name}) after cortical masking.")
        combined[sub_mask] = roi_idx
        ijk = np.argwhere(combined == roi_idx)
        xyz = nib.affines.apply_affine(schaefer_affine, ijk.mean(axis=0))
        rows.append(
            {
                "roi_index": roi_idx,
                "roi_name": f"Subcort_{aseg_name}",
                "atlas_source": "HarvardOxford_sub_maxprob_thr25_2mm",
                "hemisphere": hemi,
                "network": "Subcortical",
                "aseg_struct_name": aseg_name,
                "mni_x": round(float(xyz[0]), 2),
                "mni_y": round(float(xyz[1]), 2),
                "mni_z": round(float(xyz[2]), 2),
                "template_voxel_count": int(ijk.shape[0]),
            }
        )

    meta_df = pd.DataFrame(rows)
    out_img = nib.Nifti1Image(combined.astype(np.int16), schaefer_affine)
    out_img.set_data_dtype(np.int16)

    tmp_nii = atlas_cache_dir / f".{atlas_name}_2mm.tmp.nii.gz"
    tmp_tsv = atlas_cache_dir / f".{atlas_name}_roi_metadata.tmp.tsv"
    nib.save(out_img, str(tmp_nii))
    meta_df.to_csv(tmp_tsv, sep="\t", index=False)
    os.replace(tmp_nii, atlas_nii_path)
    os.replace(tmp_tsv, atlas_tsv_path)

    return atlas_nii_path, atlas_tsv_path, meta_df


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


def find_session_inputs(
    derivatives_dir: Path,
    sub_id: str,
    ses_id: str,
    confounds_dir: Path | None = None,
) -> dict[str, Path | None]:
    """
    Locate preprocessed MNI BOLD NIfTI, brain mask, BOLD JSON, and confounds TSV
    across both Matilda session-isolated derivatives and standard BIDS derivatives.
    """
    session_tag = f"sub-{sub_id}_ses-{ses_id}"
    candidate_func_dirs = [
        derivatives_dir / session_tag / f"sub-{sub_id}" / f"ses-{ses_id}" / "func",
        derivatives_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "func",
    ]
    func_dir = next((d for d in candidate_func_dirs if d.is_dir()), None)
    if func_dir is None:
        return {}

    bold_matches = sorted(
        func_dir.glob("*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii*")
    )
    mask_matches = sorted(
        func_dir.glob("*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii*")
    )
    json_matches = sorted(
        func_dir.glob("*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json")
    )
    conf_matches = sorted(func_dir.glob("*_desc-confounds_timeseries.tsv"))

    if not conf_matches and confounds_dir is not None:
        backup_dir = confounds_dir / session_tag
        if backup_dir.is_dir():
            conf_matches = sorted(backup_dir.glob("*_desc-confounds_timeseries.tsv"))

    if not bold_matches or not mask_matches or not conf_matches:
        return {}

    return {
        "bold_nii": bold_matches[0],
        "mask_nii": mask_matches[0],
        "bold_json": json_matches[0] if json_matches else None,
        "confounds_tsv": conf_matches[0],
    }


def resolve_tr_seconds(bold_img: nib.Nifti1Image, bold_json: Path | None) -> float:
    """Resolve repetition time (TR, seconds) from fMRIPrep JSON sidecar or NIfTI zooms."""
    if bold_json is not None and bold_json.exists():
        with bold_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        tr = meta.get("RepetitionTime")
        if tr is not None and float(tr) > 0:
            return float(tr)
    zooms = bold_img.header.get_zooms()
    if len(zooms) >= 4 and float(zooms[3]) > 0:
        return float(zooms[3])
    raise ValueError("Unable to determine valid positive RepetitionTime (TR) from JSON or NIfTI.")


def extract_roi_timeseries_chunked(
    bold_img: nib.Nifti1Image,
    mask_data: np.ndarray,
    atlas_resampled: np.ndarray,
    n_rois: int,
    template_voxel_counts: np.ndarray,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-bounded vectorized extraction of mean BOLD timeseries per integer ROI.
    Returns (ts_raw [T, N], roi_voxel_counts [N], roi_coverage [N]).
    """
    if bold_img.ndim != 4:
        raise ValueError(f"Expected 4D BOLD image, got shape {bold_img.shape}.")
    if mask_data.shape != bold_img.shape[:3]:
        raise ValueError(
            f"Brain mask shape {mask_data.shape} does not match BOLD spatial shape {bold_img.shape[:3]}."
        )

    valid_atlas = np.where(mask_data, atlas_resampled, 0).astype(np.int32, copy=False)
    flat_atlas = valid_atlas.ravel()
    roi_voxel_counts = np.bincount(flat_atlas, minlength=n_rois + 1)[1 : n_rois + 1].astype(
        np.int32, copy=False
    )
    roi_coverage = (
        roi_voxel_counts.astype(np.float32) / np.maximum(template_voxel_counts.astype(np.float32), 1.0)
    ).astype(np.float32)

    voxel_indices = np.flatnonzero((flat_atlas >= 1) & (flat_atlas <= n_rois))
    if voxel_indices.size == 0:
        raise ValueError("Intersection of brain mask and atlas contains 0 voxels.")
    voxel_roi_zero_based = flat_atlas[voxel_indices] - 1
    weights = np.where(
        roi_voxel_counts[voxel_roi_zero_based] > 0,
        1.0 / roi_voxel_counts[voxel_roi_zero_based],
        0.0,
    )
    roi_proj = sparse.csr_matrix(
        (weights, (voxel_roi_zero_based, np.arange(voxel_indices.size))),
        shape=(n_rois, voxel_indices.size),
        dtype=np.float64,
    )

    n_vols = bold_img.shape[3]
    ts_raw = np.zeros((n_vols, n_rois), dtype=np.float64)

    for start in range(0, n_vols, chunk_size):
        stop = min(start + chunk_size, n_vols)
        chunk = np.asarray(bold_img.dataobj[..., start:stop], dtype=np.float32)
        chunk_flat = chunk.reshape(-1, stop - start)[voxel_indices, :]
        ts_raw[start:stop, :] = (roi_proj @ chunk_flat).T

    return ts_raw, roi_voxel_counts, roi_coverage


def build_confound_matrix(
    conf_df: pd.DataFrame,
    strategy: str = "32p",
    min_clean_volumes: int = 30,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """
    Summarize head motion, detect leading non-steady-state dummy volumes, and build
    standardized nuisance regressor matrix for confound regression.
    """
    if "framewise_displacement" in conf_df.columns:
        fd_vals = pd.to_numeric(conf_df["framewise_displacement"], errors="coerce").dropna().to_numpy()
        if fd_vals.size > 0:
            motion_summary = {
                "mean_fd": float(np.mean(fd_vals)),
                "max_fd": float(np.max(fd_vals)),
                "median_fd": float(np.median(fd_vals)),
                "pct_fd_gt_0_5mm": float(np.mean(fd_vals > 0.5) * 100.0),
            }
        else:
            motion_summary = {"mean_fd": np.nan, "max_fd": np.nan, "median_fd": np.nan, "pct_fd_gt_0_5mm": np.nan}
    else:
        motion_summary = {"mean_fd": np.nan, "max_fd": np.nan, "median_fd": np.nan, "pct_fd_gt_0_5mm": np.nan}

    nss_cols = sorted(c for c in conf_df.columns if c.startswith("non_steady_state_outlier"))
    n_dummy = 0
    if nss_cols:
        nss_arr = conf_df[nss_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        is_nss = np.any(nss_arr >= 0.5, axis=1)
        while n_dummy < len(is_nss) and is_nss[n_dummy] and (len(is_nss) - n_dummy) > min_clean_volumes:
            n_dummy += 1

    conf_sub = conf_df.iloc[n_dummy:].copy()
    if len(conf_sub) < min_clean_volumes:
        raise ValueError(
            f"Insufficient clean volumes after removing {n_dummy} dummy frames: {len(conf_sub)} < {min_clean_volumes}."
        )

    if strategy == "24p":
        base_cols = MOTION_BASE_COLS
    elif strategy == "32p":
        base_cols = MOTION_BASE_COLS + PHYSIO_BASE_COLS
    elif strategy == "36p":
        base_cols = MOTION_BASE_COLS + PHYSIO_BASE_COLS + GLOBAL_BASE_COLS
    else:
        raise ValueError(f"Unsupported confound strategy '{strategy}'. Choose from: 24p, 32p, 36p.")

    selected_cols: list[str] = []
    for base in base_cols:
        for suffix in CONFOUND_SUFFIXES:
            col = f"{base}{suffix}"
            if col in conf_sub.columns:
                selected_cols.append(col)

    # Keep any interior non-steady-state spike columns if present after n_dummy.
    for col in nss_cols:
        vals = pd.to_numeric(conf_sub[col], errors="coerce").fillna(0.0).to_numpy()
        if np.any(vals >= 0.5):
            selected_cols.append(col)

    if not selected_cols:
        return np.zeros((len(conf_sub), 0), dtype=np.float64), n_dummy, motion_summary

    reg_cols: list[np.ndarray] = []
    for col in selected_cols:
        arr = pd.to_numeric(conf_sub[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(np.isnan(arr)):
            if "_derivative1" in col or "_power2" in col or col.startswith("non_steady_state_outlier"):
                arr = np.nan_to_num(arr, nan=0.0)
            else:
                col_mean = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else 0.0
                arr = np.nan_to_num(arr, nan=col_mean)
        std = float(np.std(arr))
        if std > 1e-12:
            reg_cols.append((arr - float(np.mean(arr))) / std)

    if not reg_cols:
        return np.zeros((len(conf_sub), 0), dtype=np.float64), n_dummy, motion_summary

    conf_mat = np.column_stack(reg_cols).astype(np.float64, copy=False)
    return conf_mat, n_dummy, motion_summary


def _regress_out_confounds(ts: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    """Linearly project confounds + intercept out of timeseries [T, K]."""
    design = np.ones((ts.shape[0], 1), dtype=np.float64)
    if confounds.shape[1] > 0:
        design = np.column_stack([confounds, design])
    betas, _, _, _ = np.linalg.lstsq(design, ts, rcond=None)
    return ts - design @ betas


def _bandpass_detrended(
    arr: np.ndarray,
    tr_s: float,
    high_pass: float,
    low_pass: float,
    order: int = 5,
) -> np.ndarray:
    """Linearly detrend and apply zero-phase Butterworth bandpass filter along axis=0."""
    if arr.shape[1] == 0:
        return arr.copy()
    detrended = signal.detrend(arr, axis=0, type="linear")
    nyq = 0.5 / tr_s
    eff_low = min(low_pass, 0.95 * nyq)
    if not (0.0 < high_pass < eff_low < nyq):
        raise ValueError(
            f"Invalid bandpass range [{high_pass}, {eff_low}] Hz for TR={tr_s:.4f}s (Nyquist={nyq:.4f} Hz)."
        )
    # Reduce filter order cleanly if a session has unusually few volumes for default padlen.
    eff_order = order
    while eff_order > 1 and (3 * (2 * eff_order + 1)) >= detrended.shape[0]:
        eff_order -= 1
    sos = signal.butter(eff_order, [high_pass, eff_low], btype="bandpass", fs=1.0 / tr_s, output="sos")
    padlen = min(3 * (2 * eff_order + 1), detrended.shape[0] - 1)
    return signal.sosfiltfilt(sos, detrended, axis=0, padlen=padlen)


def denoise_and_compute_fc(
    ts_raw: np.ndarray,
    confounds: np.ndarray,
    roi_voxel_counts: np.ndarray,
    tr_s: float,
    high_pass: float = 0.01,
    low_pass: float = 0.08,
) -> dict[str, Any]:
    """
    Compute regional ALFF/fALFF, denoised z-scored ROI time series (via simultaneous
    linear detrending + bandpass filtering of signals & confounds before regression),
    Pearson correlation FC, and Ledoit-Wolf shrinkage partial correlation FC.
    """
    n_vols, n_rois = ts_raw.shape
    initial_active = (roi_voxel_counts > 0) & (np.std(ts_raw, axis=0) > 1e-10)
    if int(np.sum(initial_active)) < 2:
        raise ValueError(f"Too few non-constant ROIs with brain coverage: {int(np.sum(initial_active))}.")

    ts_act = ts_raw[:, initial_active].astype(np.float64, copy=True)

    # 1. ALFF / fALFF prior to temporal bandpass & variance normalization
    ts_broad = signal.detrend(ts_act, axis=0, type="linear")
    conf_broad = signal.detrend(confounds, axis=0, type="linear") if confounds.shape[1] > 0 else confounds
    ts_broad_resid = _regress_out_confounds(ts_broad, conf_broad)

    freqs = np.fft.rfftfreq(n_vols, d=tr_s)
    amp_spec = np.abs(np.fft.rfft(ts_broad_resid, axis=0)) * (2.0 / n_vols)
    nyq = 0.5 / tr_s
    eff_low = min(low_pass, 0.95 * nyq)
    band_mask = (freqs >= high_pass) & (freqs <= eff_low)
    pos_mask = freqs > 0.0

    roi_alff = np.zeros(n_rois, dtype=np.float32)
    roi_falff = np.zeros(n_rois, dtype=np.float32)
    if np.any(band_mask) and np.any(pos_mask):
        alff_act = np.mean(amp_spec[band_mask, :], axis=0)
        total_amp_act = np.sum(amp_spec[pos_mask, :], axis=0)
        falff_act = np.divide(
            np.sum(amp_spec[band_mask, :], axis=0),
            total_amp_act,
            out=np.zeros_like(total_amp_act),
            where=total_amp_act > 1e-12,
        )
        roi_alff[initial_active] = alff_act.astype(np.float32)
        roi_falff[initial_active] = falff_act.astype(np.float32)

    # 2. Simultaneous detrending & zero-phase Butterworth filtering (Lindquist et al. 2019)
    ts_bp = _bandpass_detrended(ts_act, tr_s=tr_s, high_pass=high_pass, low_pass=low_pass)
    conf_bp = (
        _bandpass_detrended(confounds, tr_s=tr_s, high_pass=high_pass, low_pass=low_pass)
        if confounds.shape[1] > 0
        else confounds
    )
    ts_resid = _regress_out_confounds(ts_bp, conf_bp)

    resid_std = np.std(ts_resid, axis=0, ddof=1)
    active_sub = resid_std > 1e-10
    valid_roi_mask = np.zeros(n_rois, dtype=bool)
    active_indices = np.flatnonzero(initial_active)[active_sub]
    valid_roi_mask[active_indices] = True

    if active_indices.size < 2:
        raise ValueError(f"Too few non-constant ROIs after denoising: {active_indices.size}.")

    ts_act_clean = ts_resid[:, active_sub]
    ts_act_clean = (ts_act_clean - np.mean(ts_act_clean, axis=0)) / np.std(ts_act_clean, axis=0, ddof=1)

    ts_full = np.zeros((n_vols, n_rois), dtype=np.float32)
    ts_full[:, valid_roi_mask] = ts_act_clean.astype(np.float32)

    # 3. Pearson correlation FC
    corr_act = np.corrcoef(ts_act_clean, rowvar=False)
    corr_act = np.nan_to_num(corr_act, nan=0.0)
    np.clip(corr_act, -1.0, 1.0, out=corr_act)
    np.fill_diagonal(corr_act, 1.0)

    pearson_fc = np.eye(n_rois, dtype=np.float32)
    pearson_fc[np.ix_(valid_roi_mask, valid_roi_mask)] = corr_act.astype(np.float32)

    # 4. Regularized partial correlation via Ledoit-Wolf precision matrix
    lw = LedoitWolf(store_precision=True, assume_centered=True)
    lw.fit(ts_act_clean)
    prec = lw.precision_
    diag = np.sqrt(np.maximum(np.diag(prec), 1e-12))
    partial_act = -prec / np.outer(diag, diag)
    partial_act = np.nan_to_num(partial_act, nan=0.0)
    np.clip(partial_act, -1.0, 1.0, out=partial_act)
    np.fill_diagonal(partial_act, 1.0)

    partial_fc = np.eye(n_rois, dtype=np.float32)
    partial_fc[np.ix_(valid_roi_mask, valid_roi_mask)] = partial_act.astype(np.float32)

    return {
        "timeseries": ts_full,
        "pearson_fc": pearson_fc,
        "partial_fc": partial_fc,
        "roi_alff": roi_alff,
        "roi_falff": roi_falff,
        "valid_roi_mask": valid_roi_mask,
    }


def _init_worker(
    atlas_nii_path: str,
    template_voxel_counts: np.ndarray,
    atlas_name: str,
    derivatives_dir: str,
    confounds_dir: str | None,
    output_dir: str,
    confound_strategy: str,
    high_pass: float,
    low_pass: float,
    overwrite: bool,
) -> None:
    atlas_img = nib.load(atlas_nii_path)
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "atlas_data": np.asarray(atlas_img.dataobj, dtype=np.int32),
            "atlas_affine": atlas_img.affine.copy(),
            "template_voxel_counts": template_voxel_counts.astype(np.int32, copy=True),
            "n_rois": int(len(template_voxel_counts)),
            "atlas_name": atlas_name,
            "derivatives_dir": Path(derivatives_dir),
            "confounds_dir": Path(confounds_dir) if confounds_dir else None,
            "output_dir": Path(output_dir),
            "confound_strategy": confound_strategy,
            "high_pass": high_pass,
            "low_pass": low_pass,
            "overwrite": overwrite,
            "resampled_cache": {},
        }
    )


def _get_resampled_atlas(target_shape: tuple[int, int, int], target_affine: np.ndarray) -> np.ndarray:
    cache_key = (target_shape, tuple(np.round(target_affine.ravel(), 5).tolist()))
    cache: dict[Any, np.ndarray] = _WORKER_STATE["resampled_cache"]
    if cache_key not in cache:
        cache[cache_key] = resample_labels_nearest(
            _WORKER_STATE["atlas_data"],
            _WORKER_STATE["atlas_affine"],
            target_shape=target_shape,
            target_affine=target_affine,
        )
    return cache[cache_key]


def process_single_session(sub_id: str, ses_id: str) -> dict[str, Any]:
    """Process one (sub_id, ses_id) session and write its compressed .npz bundle."""
    t0 = time.monotonic()
    session_tag = f"sub-{sub_id}_ses-{ses_id}"
    output_dir: Path = _WORKER_STATE["output_dir"]
    npz_path = output_dir / f"{session_tag}.npz"

    base_record: dict[str, Any] = {
        "session_tag": session_tag,
        "BIDS_Subject": f"sub-{sub_id}",
        "BIDS_Session": f"ses-{ses_id}",
        "status": "error",
        "atlas": _WORKER_STATE["atlas_name"],
        "confound_strategy": _WORKER_STATE["confound_strategy"],
        "tr_s": np.nan,
        "n_volumes_raw": 0,
        "n_dummy_removed": 0,
        "n_volumes_clean": 0,
        "n_valid_rois": 0,
        "mean_fd": np.nan,
        "max_fd": np.nan,
        "pct_fd_gt_0_5mm": np.nan,
        "elapsed_s": 0.0,
        "npz_path": str(npz_path),
        "error_message": "",
    }

    if npz_path.exists() and not _WORKER_STATE["overwrite"]:
        try:
            with np.load(npz_path, allow_pickle=False) as existing:
                base_record.update(
                    {
                        "status": "skipped_existing",
                        "tr_s": float(existing["tr_s"]),
                        "n_volumes_raw": int(existing["n_volumes_raw"]),
                        "n_dummy_removed": int(existing["n_dummy_removed"]),
                        "n_volumes_clean": int(existing["n_volumes_clean"]),
                        "n_valid_rois": int(np.sum(existing["valid_roi_mask"])),
                        "mean_fd": float(existing["mean_fd"]),
                        "max_fd": float(existing["max_fd"]),
                        "pct_fd_gt_0_5mm": float(existing["pct_fd_gt_0_5mm"]),
                        "elapsed_s": round(time.monotonic() - t0, 2),
                    }
                )
            return base_record
        except Exception:
            # Re-extract corrupted existing archive rather than skipping it.
            pass

    inputs = find_session_inputs(
        _WORKER_STATE["derivatives_dir"],
        sub_id,
        ses_id,
        confounds_dir=_WORKER_STATE["confounds_dir"],
    )
    if not inputs:
        base_record["status"] = "missing_bold"
        base_record["error_message"] = "Preprocessed MNI BOLD, brain mask, or confounds TSV not found."
        base_record["elapsed_s"] = round(time.monotonic() - t0, 2)
        return base_record

    try:
        bold_img = nib.load(str(inputs["bold_nii"]))
        mask_img = nib.load(str(inputs["mask_nii"]))
        conf_df = pd.read_csv(inputs["confounds_tsv"], sep="\t", na_values=["n/a", "NA", ""])

        n_vols_raw = int(bold_img.shape[3])
        if len(conf_df) != n_vols_raw:
            raise ValueError(
                f"Volume mismatch: BOLD has {n_vols_raw} frames, confounds TSV has {len(conf_df)} rows."
            )

        tr_s = resolve_tr_seconds(bold_img, inputs.get("bold_json"))
        conf_mat, n_dummy, motion_summary = build_confound_matrix(
            conf_df,
            strategy=_WORKER_STATE["confound_strategy"],
        )

        mask_data = np.asarray(mask_img.dataobj) > 0
        atlas_resampled = _get_resampled_atlas(bold_img.shape[:3], bold_img.affine)

        ts_raw, roi_voxel_counts, roi_coverage = extract_roi_timeseries_chunked(
            bold_img=bold_img,
            mask_data=mask_data,
            atlas_resampled=atlas_resampled,
            n_rois=_WORKER_STATE["n_rois"],
            template_voxel_counts=_WORKER_STATE["template_voxel_counts"],
        )
        ts_clean_raw = ts_raw[n_dummy:, :]

        fc_out = denoise_and_compute_fc(
            ts_raw=ts_clean_raw,
            confounds=conf_mat,
            roi_voxel_counts=roi_voxel_counts,
            tr_s=tr_s,
            high_pass=_WORKER_STATE["high_pass"],
            low_pass=_WORKER_STATE["low_pass"],
        )

        n_vols_clean = int(fc_out["timeseries"].shape[0])
        n_valid_rois = int(np.sum(fc_out["valid_roi_mask"]))

        tmp_npz = output_dir / f".{session_tag}.{os.getpid()}.tmp.npz"
        np.savez_compressed(
            tmp_npz,
            pearson_fc=fc_out["pearson_fc"],
            partial_fc=fc_out["partial_fc"],
            timeseries=fc_out["timeseries"],
            roi_alff=fc_out["roi_alff"],
            roi_falff=fc_out["roi_falff"],
            roi_voxel_counts=roi_voxel_counts,
            roi_coverage=roi_coverage,
            valid_roi_mask=fc_out["valid_roi_mask"],
            subject=np.array(f"sub-{sub_id}"),
            session=np.array(f"ses-{ses_id}"),
            atlas=np.array(_WORKER_STATE["atlas_name"]),
            confound_strategy=np.array(_WORKER_STATE["confound_strategy"]),
            tr_s=np.float32(tr_s),
            n_volumes_raw=np.int32(n_vols_raw),
            n_dummy_removed=np.int32(n_dummy),
            n_volumes_clean=np.int32(n_vols_clean),
            mean_fd=np.float32(motion_summary["mean_fd"]),
            max_fd=np.float32(motion_summary["max_fd"]),
            pct_fd_gt_0_5mm=np.float32(motion_summary["pct_fd_gt_0_5mm"]),
        )
        os.replace(tmp_npz, npz_path)

        base_record.update(
            {
                "status": "completed",
                "tr_s": round(tr_s, 4),
                "n_volumes_raw": n_vols_raw,
                "n_dummy_removed": n_dummy,
                "n_volumes_clean": n_vols_clean,
                "n_valid_rois": n_valid_rois,
                "mean_fd": round(motion_summary["mean_fd"], 4),
                "max_fd": round(motion_summary["max_fd"], 4),
                "pct_fd_gt_0_5mm": round(motion_summary["pct_fd_gt_0_5mm"], 2),
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
        )
    except Exception as exc:
        base_record["status"] = "failed"
        base_record["error_message"] = f"{type(exc).__name__}: {exc}"
        base_record["elapsed_s"] = round(time.monotonic() - t0, 2)

    return base_record


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract denoised ROI timeseries and 2D FC matrices from fMRIPrep derivatives."
    )
    parser.add_argument(
        "--sessions-list",
        type=Path,
        default=DEFAULT_SESSIONS_LIST,
        help="Path to sessions_list.tsv containing (sub, ses) pairs.",
    )
    parser.add_argument(
        "--derivatives-dir",
        type=Path,
        default=DEFAULT_DERIVATIVES_DIR,
        help="Root fMRIPrep derivatives directory on HPC.",
    )
    parser.add_argument(
        "--confounds-dir",
        type=Path,
        default=None,
        help="Optional fallback directory containing sub-*_ses-* folders with confounds TSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to <repo>/extracted_fc_matrices/<atlas>).",
    )
    parser.add_argument(
        "--atlas",
        choices=sorted(SUPPORTED_ATLASES),
        default="schaefer200_subcort14",
        help="Whole-brain cortical+subcortical parcellation atlas (default: schaefer200_subcort14).",
    )
    parser.add_argument(
        "--atlas-cache-dir",
        type=Path,
        default=DEFAULT_ATLAS_CACHE_DIR,
        help="Directory holding cached combined atlas NIfTI + ROI metadata TSV.",
    )
    parser.add_argument(
        "--prepare-atlas-only",
        action="store_true",
        help="Download/build combined atlas into --atlas-cache-dir and exit (run once on login node).",
    )
    parser.add_argument(
        "--confound-strategy",
        choices=("24p", "32p", "36p"),
        default="32p",
        help="Nuisance regression strategy: 24p (motion), 32p (motion+WM+CSF, default), or 36p (+GSR).",
    )
    parser.add_argument("--high-pass", type=float, default=0.01, help="Bandpass high-pass Hz (default: 0.01).")
    parser.add_argument("--low-pass", type=float, default=0.08, help="Bandpass low-pass Hz (default: 0.08).")
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
        help="Overwrite existing per-session .npz files instead of resuming/skipping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.atlas)

    atlas_nii_path, atlas_tsv_path, meta_df = build_or_load_combined_atlas(
        args.atlas,
        args.atlas_cache_dir,
    )
    print(f"Atlas ready: {atlas_nii_path} ({len(meta_df)} ROIs)")

    if args.prepare_atlas_only:
        print(f"Atlas metadata TSV saved at: {atlas_tsv_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy ROI metadata table alongside extracted matrices so the archive is self-contained.
    dest_atlas_tsv = output_dir / f"atlas_{args.atlas}_roi_metadata.tsv"
    meta_df.to_csv(dest_atlas_tsv, sep="\t", index=False)

    all_sessions = load_sessions_list(args.sessions_list)
    shard_sessions = select_shard(all_sessions, args.shard_index, args.num_shards)
    if args.limit is not None:
        shard_sessions = shard_sessions[: args.limit]

    print(
        f"Processing {len(shard_sessions)} sessions "
        f"(shard {args.shard_index}/{args.num_shards}, workers={args.workers}, "
        f"atlas={args.atlas}, confounds={args.confound_strategy}) -> {output_dir}"
    )

    template_voxel_counts = meta_df["template_voxel_count"].to_numpy(dtype=np.int32)
    init_kwargs = (
        str(atlas_nii_path),
        template_voxel_counts,
        args.atlas,
        str(args.derivatives_dir),
        str(args.confounds_dir) if args.confounds_dir else None,
        str(output_dir),
        args.confound_strategy,
        args.high_pass,
        args.low_pass,
        args.overwrite,
    )

    results: list[dict[str, Any]] = []
    t_start = time.monotonic()

    if args.workers <= 1:
        _init_worker(*init_kwargs)
        for idx, (sub_id, ses_id) in enumerate(shard_sessions, start=1):
            rec = process_single_session(sub_id, ses_id)
            results.append(rec)
            if idx % 25 == 0 or idx == len(shard_sessions):
                print(
                    f"[{idx}/{len(shard_sessions)}] {rec['session_tag']} -> {rec['status']} "
                    f"({rec['elapsed_s']:.1f}s)"
                )
    else:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=init_kwargs,
        ) as executor:
            futures = {
                executor.submit(process_single_session, sub_id, ses_id): (sub_id, ses_id)
                for sub_id, ses_id in shard_sessions
            }
            for idx, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                results.append(rec)
                if idx % 25 == 0 or idx == len(shard_sessions):
                    print(
                        f"[{idx}/{len(shard_sessions)}] {rec['session_tag']} -> {rec['status']} "
                        f"({rec['elapsed_s']:.1f}s)"
                    )

    results.sort(key=lambda r: r["session_tag"])
    manifest_name = (
        f"fc_extraction_manifest_shard_{args.shard_index:04d}.csv"
        if args.num_shards > 1
        else "fc_extraction_manifest.csv"
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
