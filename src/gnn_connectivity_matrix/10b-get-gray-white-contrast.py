#!/usr/bin/env python3
"""
Extract cortical gray-white matter intensity contrast (`WGPctMean`, `WGPctStd`,
`WGPctSNR`) for the Destrieux 2009 (`aparc.a2009s`, 148 cortical parcels) atlas
from archived fMRIPrep FreeSurfer `recon-all` outputs on the Matilda HPC cluster.

Why this script is needed
-------------------------
During default `recon-all`, FreeSurfer runs `mri_segstats --snr` on
`?h.w-g.pct.mgh` only for the Desikan-Killiany (`aparc`) parcellation
(`{lh,rh}.w-g.pct.stats`), not for Destrieux (`aparc.a2009s`). Because
`run_fmriprep.sh` archived each visit's full `recon-all` directory to
`/scratch/users/moghaddami/fmriprep-output/freesurfer/sub-*_ses-*.tar`, we can
stage just the 6-8 surface/annotation files needed per visit and run
`mri_segstats --in .../{lh,rh}.w-g.pct.mgh --annot ... aparc.a2009s --snr`
inside `fmriprep-25.2.7.sif` (FreeSurfer 7.3.2).

Per-session outputs (`<output-dir>/`, default `extracted_freesurfer_stats/destrieux148_subcort14`)
--------------------------------------------------------------------------------------------------
- `stats/<session_tag>/{lh,rh}.aparc.a2009s.w-g.pct.stats`
  (also synced into `extracted_freesurfer_stats/<session_tag>/stats/` when that directory exists)
- `destrieux_wg_pct_manifest.csv`

Recommended Matilda HPC Workflow
--------------------------------
1. Pilot test on 3 sessions:
   module load Singularity/3.9.6
   uv run src/gnn_connectivity_matrix/10b-get-gray-white-contrast.py --limit 3 --workers 3

2a. Standard host run (~5-10 min for 3,215 sessions with 16 workers):
   module load Singularity/3.9.6
   uv run src/gnn_connectivity_matrix/10b-get-gray-white-contrast.py --workers 16

2b. Fast single-container run (~2 min, avoids per-session container startup overhead):
   module load Singularity/3.9.6
   singularity exec --cleanenv \
     -B /projects:/projects \
     -B /scratch:/scratch \
     /scratch/users/moghaddami/fmriprep-output/fmriprep-25.2.7.sif \
     python3 /projects/adresearch/moghaddami/gnn-connectivity-matrix/src/gnn_connectivity_matrix/10b-get-gray-white-contrast.py \
     --backend native --workers 16

3. Archive lightweight stats folder for transfer back to local PC:
   tar -cf extracted_freesurfer_destrieux148_wg_pct.tar \
       extracted_freesurfer_stats/destrieux148_subcort14/
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import multiprocessing as mp
import os
from pathlib import Path
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Any


for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
):
    os.environ.setdefault(_env_var, "1")


DEFAULT_REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS_LIST = DEFAULT_REPO_DIR / "sessions_list.tsv"
DEFAULT_FREESURFER_DIR = Path("/scratch/users/moghaddami/fmriprep-output/freesurfer")
DEFAULT_STATS_BACKUP_DIR = DEFAULT_REPO_DIR / "extracted_freesurfer_stats"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_DIR / "extracted_freesurfer_stats" / "destrieux148_subcort14"
DEFAULT_SIF_IMAGE = Path("/scratch/users/moghaddami/fmriprep-output/fmriprep-25.2.7.sif")
DEFAULT_FS_LICENSE = DEFAULT_REPO_DIR / "freesurfer_license.txt"

IGNORED_PARCEL_NAMES: frozenset[str] = frozenset(
    {
        "unknown",
        "medial_wall",
        "background",
        "background+freesurfer_defined_medial_wall",
        "corpuscallosum",
    }
)

MIN_PARCELS_PER_HEMI = 50

WG_PCT_SEGSTATS_SHELL = r"""
set -euo pipefail
export FREESURFER_HOME="${FREESURFER_HOME:-/opt/freesurfer}"
export FS_LICENSE="${FS_LICENSE_PATH}"
export SUBJECTS_DIR="${FS_SUBJECTS_DIR}"
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

SUB="${FS_SUBJ_ID}"
ANNOT_NAME="${FS_ANNOT_NAME}"
OUT_DIR="${FS_OUT_DIR}"

mkdir -p "${OUT_DIR}"

for H in lh rh; do
  "${FREESURFER_HOME}/bin/mri_segstats" \
    --in "${SUBJECTS_DIR}/${SUB}/surf/${H}.w-g.pct.mgh" \
    --annot "${SUB}" "${H}" "${ANNOT_NAME}" \
    --sum "${OUT_DIR}/${H}.${ANNOT_NAME}.w-g.pct.stats" \
    --snr \
    > /dev/null
done
"""

_WORKER_STATE: dict[str, Any] = {}


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


def _needed_files_for_annot(annot_name: str) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    surf_files = frozenset(
        {
            "lh.w-g.pct.mgh",
            "rh.w-g.pct.mgh",
            "lh.white",
            "rh.white",
            "lh.area",
            "rh.area",
        }
    )
    label_files = frozenset({f"lh.{annot_name}.annot", f"rh.{annot_name}.annot"})
    required_rel = frozenset(
        {
            "surf/lh.w-g.pct.mgh",
            "surf/rh.w-g.pct.mgh",
            "surf/lh.white",
            "surf/rh.white",
            f"label/lh.{annot_name}.annot",
            f"label/rh.{annot_name}.annot",
        }
    )
    return surf_files, label_files, required_rel


def _classify_wg_pct_relpath(
    member_name: str,
    surf_files: frozenset[str],
    label_files: frozenset[str],
) -> tuple[str, str] | None:
    """
    Match only the minimal surf/ and label/ files needed for `mri_segstats --annot --snr`,
    handling both 1-level (`<subjid>/surf/...`) and 2-level (`<session_tag>/<subjid>/surf/...`)
    Matilda `.tar` layouts while rejecting `fsaverage` and path traversal.
    """
    norm = posixpath.normpath(member_name.replace("\\", "/")).lstrip("/")
    if not norm or norm == "." or norm.startswith("../") or "/../" in f"/{norm}/":
        return None
    parts = [p for p in norm.split("/") if p and p != "."]
    if len(parts) < 3 or "fsaverage" in parts:
        return None

    subdir_idx: int | None = None
    for idx in range(len(parts) - 2, 0, -1):
        if parts[idx] in {"surf", "label"}:
            subdir_idx = idx
            break
    if subdir_idx is None:
        return None

    subj_id = parts[subdir_idx - 1]
    if not subj_id or subj_id == "fsaverage":
        return None

    subdir = parts[subdir_idx]
    rel_sub = parts[subdir_idx + 1 :]
    if len(rel_sub) != 1:
        return None

    fname = rel_sub[0]
    if subdir == "surf" and fname in surf_files:
        return subj_id, f"surf/{fname}"
    if subdir == "label" and fname in label_files:
        return subj_id, f"label/{fname}"
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _verify_staged_subject(subj_dir: Path, required_rel: frozenset[str]) -> None:
    missing = sorted(rel for rel in required_rel if not (subj_dir / rel).is_file())
    if missing:
        raise RuntimeError(f"Incomplete FreeSurfer directory '{subj_dir.name}', missing: {missing}")


def stage_subject_from_tar(
    tar_path: Path,
    subjects_dir: Path,
    annot_name: str,
) -> str:
    """Stream-extract only the needed surf/ and label/ files from `<session_tag>.tar`."""
    subjects_dir.mkdir(parents=True, exist_ok=True)
    surf_files, label_files, required_rel = _needed_files_for_annot(annot_name)
    detected_subj: str | None = None
    deferred_symlinks: list[tuple[Path, Path, str]] = []
    deferred_hardlinks: list[tuple[Path, str]] = []

    with tarfile.open(tar_path, mode="r:*") as tf:
        for member in tf:
            classified = _classify_wg_pct_relpath(member.name, surf_files, label_files)
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
        target_cls = _classify_wg_pct_relpath(hard_target_name, surf_files, label_files)
        if target_cls is None or target_cls[0] != detected_subj:
            continue
        src_existing = subj_dir / target_cls[1]
        if src_existing.exists():
            _link_or_copy(src_existing, dest_path)

    _verify_staged_subject(subj_dir, required_rel)
    return detected_subj


def stage_subject_from_dir(
    raw_session_dir: Path,
    subjects_dir: Path,
    annot_name: str,
) -> str:
    """Stage required surf/ and label/ files from an unarchived FreeSurfer directory."""
    surf_files, label_files, required_rel = _needed_files_for_annot(annot_name)
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
    for subdir in ("surf", "label"):
        src_sub = src_subj / subdir
        if not src_sub.is_dir():
            continue
        for item in src_sub.iterdir():
            if item.is_dir() and not item.is_symlink():
                continue
            rel = f"{subj_id}/{subdir}/{item.name}"
            classified = _classify_wg_pct_relpath(rel, surf_files, label_files)
            if classified is None:
                continue
            resolved = item.resolve()
            if resolved.is_file():
                _link_or_copy(resolved, dst_subj / classified[1])

    _verify_staged_subject(dst_subj, required_rel)
    return subj_id


def detect_execution_backend(
    sif_image: Path,
    fs_license: Path,
    prefer_backend: str = "auto",
) -> tuple[str, str | None]:
    """Detect native FreeSurfer `mri_segstats` vs Singularity/Apptainer container."""
    fs_home = Path(os.environ.get("FREESURFER_HOME", "/opt/freesurfer"))
    has_native = (fs_home / "bin" / "mri_segstats").is_file() or (
        shutil.which("mri_segstats") is not None
    )

    if prefer_backend in ("auto", "native") and has_native:
        return "native", None
    if prefer_backend == "native":
        raise RuntimeError(
            "Requested --backend=native, but FreeSurfer `mri_segstats` was not found in "
            "$FREESURFER_HOME/bin or PATH."
        )

    container_bin = shutil.which("singularity") or shutil.which("apptainer")
    if container_bin is None:
        raise RuntimeError(
            "Neither native `mri_segstats` nor `singularity`/`apptainer` were found on PATH. "
            "On Matilda HPC, run `module load Singularity/3.9.6` before launching."
        )
    if not sif_image.is_file():
        raise FileNotFoundError(f"Singularity image not found: {sif_image}")
    if not fs_license.is_file():
        raise FileNotFoundError(f"FreeSurfer license file not found: {fs_license}")

    return "container", container_bin


def run_wg_pct_segstats(
    subjects_dir: Path,
    subj_id: str,
    annot_name: str,
    tmp_out_dir: Path,
    backend: str,
    container_bin: str | None,
    sif_image: Path,
    fs_license: Path,
) -> None:
    """Run `mri_segstats --in ?h.w-g.pct.mgh --annot <subjid> <hemi> <annot_name> --snr`."""
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "native":
        fs_home = os.environ.get("FREESURFER_HOME")
        if not fs_home:
            segstats_bin = shutil.which("mri_segstats")
            fs_home = str(Path(segstats_bin).resolve().parents[1]) if segstats_bin else "/opt/freesurfer"
        env = os.environ.copy()
        env.update(
            {
                "FREESURFER_HOME": fs_home,
                "FS_LICENSE_PATH": str(fs_license.resolve()),
                "FS_SUBJECTS_DIR": str(subjects_dir.resolve()),
                "FS_SUBJ_ID": subj_id,
                "FS_ANNOT_NAME": annot_name,
                "FS_OUT_DIR": str(tmp_out_dir.resolve()),
            }
        )
        cmd = ["bash", "-c", WG_PCT_SEGSTATS_SHELL]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    else:
        assert container_bin is not None
        cmd = [
            container_bin,
            "exec",
            "--cleanenv",
            "-B",
            f"{subjects_dir.resolve()}:/fsdir",
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
            f"FS_ANNOT_NAME={annot_name}",
            "--env",
            "FS_OUT_DIR=/out",
            str(sif_image.resolve()),
            "bash",
            "-c",
            WG_PCT_SEGSTATS_SHELL,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"mri_segstats exited with {proc.returncode}: {err_tail}")


def count_wg_pct_parcels(stats_path: Path) -> int:
    """Count non-ignored cortical parcel rows in a `.w-g.pct.stats` file."""
    if not stats_path.is_file() or stats_path.stat().st_size == 0:
        return 0
    count = 0
    with stats_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 11 and cols[4].lower() not in IGNORED_PARCEL_NAMES:
                count += 1
    return count


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _init_worker(
    annot_name: str,
    freesurfer_dir: str,
    stats_backup_dir: str | None,
    output_dir: str,
    sif_image: str,
    fs_license: str,
    backend: str,
    container_bin: str | None,
    tmp_dir: str | None,
    sync_backup_stats: bool,
    overwrite: bool,
) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "annot_name": annot_name,
            "freesurfer_dir": Path(freesurfer_dir),
            "stats_backup_dir": Path(stats_backup_dir) if stats_backup_dir else None,
            "output_dir": Path(output_dir),
            "sif_image": Path(sif_image),
            "fs_license": Path(fs_license),
            "backend": backend,
            "container_bin": container_bin,
            "tmp_dir": Path(tmp_dir) if tmp_dir else None,
            "sync_backup_stats": sync_backup_stats,
            "overwrite": overwrite,
        }
    )


def process_single_session(sub_id: str, ses_id: str) -> dict[str, Any]:
    """Extract `{lh,rh}.{annot_name}.w-g.pct.stats` for one session."""
    t0 = time.monotonic()
    session_tag = f"sub-{sub_id}_ses-{ses_id}"
    annot_name: str = _WORKER_STATE["annot_name"]
    output_dir: Path = _WORKER_STATE["output_dir"]
    stats_backup_dir: Path | None = _WORKER_STATE["stats_backup_dir"]
    out_session_dir = output_dir / "stats" / session_tag

    lh_out = out_session_dir / f"lh.{annot_name}.w-g.pct.stats"
    rh_out = out_session_dir / f"rh.{annot_name}.w-g.pct.stats"

    rec: dict[str, Any] = {
        "session_tag": session_tag,
        "BIDS_Subject": f"sub-{sub_id}",
        "BIDS_Session": f"ses-{ses_id}",
        "status": "error",
        "annot_name": annot_name,
        "n_parcels_lh": 0,
        "n_parcels_rh": 0,
        "elapsed_s": 0.0,
        "lh_stats_path": str(lh_out),
        "rh_stats_path": str(rh_out),
        "error_message": "",
    }

    if not _WORKER_STATE["overwrite"] and lh_out.is_file() and rh_out.is_file():
        n_lh = count_wg_pct_parcels(lh_out)
        n_rh = count_wg_pct_parcels(rh_out)
        if n_lh >= MIN_PARCELS_PER_HEMI and n_rh >= MIN_PARCELS_PER_HEMI:
            rec.update(
                status="skipped_existing",
                n_parcels_lh=n_lh,
                n_parcels_rh=n_rh,
                elapsed_s=round(time.monotonic() - t0, 2),
            )
            return rec

    fs_dir: Path = _WORKER_STATE["freesurfer_dir"]
    tar_path = fs_dir / f"{session_tag}.tar"
    raw_dir = fs_dir / session_tag
    if not tar_path.is_file() and not raw_dir.is_dir():
        rec["status"] = "missing_freesurfer"
        rec["error_message"] = (
            f"Neither {tar_path.name} nor directory {session_tag}/ found in {fs_dir}."
        )
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
        return rec

    tmp_base: Path | None = _WORKER_STATE["tmp_dir"]
    if tmp_base is None:
        for env_tmp in ("SLURM_TMPDIR", "TMPDIR"):
            env_val = os.environ.get(env_tmp)
            if env_val and Path(env_val).is_dir():
                tmp_base = Path(env_val)
                break

    tmp_root: Path | None = None
    try:
        if tmp_base is not None:
            tmp_base.mkdir(parents=True, exist_ok=True)
        tmp_root = Path(
            tempfile.mkdtemp(
                prefix=f"fs_wgpct_{session_tag}_",
                dir=str(tmp_base) if tmp_base is not None else None,
            )
        )
        subjects_dir = tmp_root / "subjects"
        tmp_out_dir = tmp_root / "out_stats"

        if tar_path.is_file():
            subj_id = stage_subject_from_tar(tar_path, subjects_dir, annot_name)
        else:
            subj_id = stage_subject_from_dir(raw_dir, subjects_dir, annot_name)

        run_wg_pct_segstats(
            subjects_dir=subjects_dir,
            subj_id=subj_id,
            annot_name=annot_name,
            tmp_out_dir=tmp_out_dir,
            backend=_WORKER_STATE["backend"],
            container_bin=_WORKER_STATE["container_bin"],
            sif_image=_WORKER_STATE["sif_image"],
            fs_license=_WORKER_STATE["fs_license"],
        )

        gen_lh = tmp_out_dir / f"lh.{annot_name}.w-g.pct.stats"
        gen_rh = tmp_out_dir / f"rh.{annot_name}.w-g.pct.stats"
        n_lh = count_wg_pct_parcels(gen_lh)
        n_rh = count_wg_pct_parcels(gen_rh)
        if n_lh < MIN_PARCELS_PER_HEMI or n_rh < MIN_PARCELS_PER_HEMI:
            raise RuntimeError(
                f"Too few cortical parcels in generated w-g.pct stats: lh={n_lh}, rh={n_rh}"
            )

        _copy_atomic(gen_lh, lh_out)
        _copy_atomic(gen_rh, rh_out)

        if _WORKER_STATE["sync_backup_stats"] and stats_backup_dir is not None:
            backup_stats_subdir = stats_backup_dir / session_tag / "stats"
            if backup_stats_subdir.is_dir():
                for src_file, dest_name in ((gen_lh, lh_out.name), (gen_rh, rh_out.name)):
                    dest_backup = backup_stats_subdir / dest_name
                    if dest_backup.resolve() != (out_session_dir / dest_name).resolve():
                        _copy_atomic(src_file, dest_backup)

        rec.update(
            status="completed",
            n_parcels_lh=n_lh,
            n_parcels_rh=n_rh,
            elapsed_s=round(time.monotonic() - t0, 2),
        )
    except Exception as exc:
        rec["status"] = "failed"
        rec["error_message"] = f"{type(exc).__name__}: {exc}"
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)

    return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Destrieux (aparc.a2009s) gray-white contrast stats via mri_segstats --snr."
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
        help="Optional backup stats root (`sub-*_ses-*/stats/`) to sync generated files into.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (defaults to <repo>/extracted_freesurfer_stats/destrieux148_subcort14).",
    )
    parser.add_argument(
        "--annot-name",
        type=str,
        default="aparc.a2009s",
        help="FreeSurfer surface annotation stem in `label/` (default: aparc.a2009s).",
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
        help="Execution backend for `mri_segstats` (default: auto).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Temporary directory for staged files (defaults to $SLURM_TMPDIR, $TMPDIR, or /tmp).",
    )
    parser.add_argument(
        "--no-sync-backup-stats",
        action="store_true",
        help="Do not copy generated `.w-g.pct.stats` files into --stats-backup-dir/<session>/stats/.",
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
        help="Re-run `mri_segstats` and overwrite existing `.w-g.pct.stats` files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend, container_bin = detect_execution_backend(
        sif_image=args.sif_image,
        fs_license=args.fs_license,
        prefer_backend=args.backend,
    )
    print(f"FreeSurfer execution backend: {backend} | annot={args.annot_name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = load_sessions_list(args.sessions_list)
    shard_sessions = select_shard(all_sessions, args.shard_index, args.num_shards)
    if args.limit is not None:
        shard_sessions = shard_sessions[: args.limit]

    print(
        f"Processing {len(shard_sessions)} sessions "
        f"(shard {args.shard_index}/{args.num_shards}, workers={args.workers}) -> {args.output_dir}"
    )

    init_args = (
        args.annot_name,
        str(args.freesurfer_dir),
        str(args.stats_backup_dir) if args.stats_backup_dir else None,
        str(args.output_dir),
        str(args.sif_image),
        str(args.fs_license),
        backend,
        container_bin,
        str(args.tmp_dir) if args.tmp_dir else None,
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
        f"destrieux_wg_pct_manifest_shard_{args.shard_index:04d}.csv"
        if args.num_shards > 1
        else "destrieux_wg_pct_manifest.csv"
    )
    manifest_path = args.output_dir / manifest_name
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
