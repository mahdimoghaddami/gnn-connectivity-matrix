#!/usr/bin/env python3
"""
Extract per-visit FreeSurfer cortical morphometry for the Schaefer 2018
200-parcel / 7-network atlas from the archived fMRIPrep `recon-all` outputs.

For every `<fs-tar-dir>/sub-*_ses-*.tar`:
  1. Extract only the files needed (surf/?h.{white,pial,thickness,curv,sphere.reg},
     label/?h.cortex.label, mri/orig.mgz) into a temporary directory.
  2. Inside the fMRIPrep Singularity image (FreeSurfer 7.3.2), map the CBIG
     fsaverage Schaefer200 `.annot` onto the visit's native surface with
     `mri_surf2surf`.
  3. Run `mris_anatomical_stats -th3 -mgz -b` to write `lh.stats` / `rh.stats`
     with the same 9 columns as aparc.a2009s stats:
     NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd.

Output: <output-dir>/<session_tag>/{lh,rh}.Schaefer2018_200Parcels_7Networks.stats
        <output-dir>/schaefer200_fs_stats_manifest.csv

Usage on the Matilda login node (stdlib only; Python 3.6 compatible):
    module load Singularity/3.9.6
    python3 src/gnn_connectivity_matrix/09b-extract-schaefer-freesurfer-stats.py --prepare-only
    python3 src/gnn_connectivity_matrix/09b-extract-schaefer-freesurfer-stats.py --limit 3
    nohup python3 src/gnn_connectivity_matrix/09b-extract-schaefer-freesurfer-stats.py --workers 4 > slurm_logs/schaefer_fs.log 2>&1 &
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FS_TAR_DIR = Path("/scratch/users/moghaddami/fmriprep-output/freesurfer")
DEFAULT_SIF = Path("/scratch/users/moghaddami/fmriprep-output/fmriprep-25.2.7.sif")
DEFAULT_LICENSE = REPO_DIR / "freesurfer_license.txt"
DEFAULT_OUTPUT_DIR = REPO_DIR / "extracted_freesurfer_schaefer200"
DEFAULT_ANNOT_DIR = REPO_DIR / "data" / "atlases" / "schaefer200_fsaverage_annot"

ATLAS_STEM = "Schaefer2018_200Parcels_7Networks"
ANNOT_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/"
    "fsaverage/label/{hemi}.Schaefer2018_200Parcels_7Networks_order.annot"
)

TAR_PATTERNS = [
    "*/surf/?h.white",
    "*/surf/?h.pial",
    "*/surf/?h.thickness",
    "*/surf/?h.curv",
    "*/surf/?h.sphere.reg",
    "*/label/?h.cortex.label",
    "*/mri/orig.mgz",
]

# Runs inside the container. $1 = subject dir name, $2 = output tag.
CONTAINER_SCRIPT = r"""
set -e
export FREESURFER_HOME="${FREESURFER_HOME:-/opt/freesurfer}"
export FS_LICENSE=/opt/freesurfer/license.txt
export SUBJECTS_DIR=/subjects
SUB="$1"
[ -e /subjects/fsaverage ] || ln -s "${FREESURFER_HOME}/subjects/fsaverage" /subjects/fsaverage
for H in lh rh; do
  mri_surf2surf --srcsubject fsaverage --trgsubject "$SUB" --hemi $H \
    --sval-annot /annot/$H.%(stem)s_order.annot \
    --tval /subjects/$SUB/label/$H.%(stem)s.annot > /dev/null
  mris_anatomical_stats -th3 -mgz -noglobal \
    -l /subjects/$SUB/label/$H.cortex.label \
    -f /out/$H.%(stem)s.stats -b \
    -a /subjects/$SUB/label/$H.%(stem)s.annot \
    "$SUB" $H white > /dev/null
done
""" % {"stem": ATLAS_STEM}


def prepare_annots(annot_dir):
    annot_dir.mkdir(parents=True, exist_ok=True)
    for hemi in ("lh", "rh"):
        dest = annot_dir / ("%s.%s_order.annot" % (hemi, ATLAS_STEM))
        if not dest.exists() or dest.stat().st_size == 0:
            url = ANNOT_URL.format(hemi=hemi)
            print("Downloading %s" % url)
            urllib.request.urlretrieve(url, str(dest))
    return annot_dir


def count_rows(stats_path):
    n = 0
    with open(str(stats_path)) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                n += 1
    return n


def process_session(tar_path, args):
    t0 = time.time()
    tag = tar_path.stem
    out_dir = args.output_dir / tag
    rec = {"session_tag": tag, "status": "failed", "n_rois_lh": 0, "n_rois_rh": 0,
           "elapsed_s": 0.0, "error_message": ""}
    lh_out = out_dir / ("lh.%s.stats" % ATLAS_STEM)
    rh_out = out_dir / ("rh.%s.stats" % ATLAS_STEM)

    if not args.overwrite and lh_out.exists() and rh_out.exists() and \
            lh_out.stat().st_size > 0 and rh_out.stat().st_size > 0:
        rec.update(status="skipped_existing", n_rois_lh=count_rows(lh_out),
                   n_rois_rh=count_rows(rh_out), elapsed_s=round(time.time() - t0, 2))
        return rec

    tmp_root = tempfile.mkdtemp(prefix="schaefer_fs_", dir=str(args.tmp_dir) if args.tmp_dir else None)
    try:
        cmd = ["tar", "-xf", str(tar_path), "-C", tmp_root, "--wildcards",
               "--exclude=*fsaverage*"] + TAR_PATTERNS
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        tag_dir = os.path.join(tmp_root, tag)
        candidates = [d for d in sorted(glob.glob(os.path.join(tag_dir, "*")))
                      if os.path.isdir(os.path.join(d, "surf")) and "fsaverage" not in d]
        if not candidates:
            raise RuntimeError("No subject directory with surf/ found inside %s" % tar_path.name)
        subj_dir = candidates[0]
        subj = os.path.basename(subj_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        sing = [
            "singularity", "exec", "--cleanenv",
            "-B", "%s:/subjects" % tag_dir,
            "-B", "%s:/annot:ro" % args.annot_dir,
            "-B", "%s:/out" % out_dir,
            "-B", "%s:/opt/freesurfer/license.txt:ro" % args.license,
            str(args.sif), "bash", "-c", CONTAINER_SCRIPT, "bash", subj,
        ]
        proc = subprocess.run(sing, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        if proc.returncode != 0:
            raise RuntimeError("container failed: %s" % proc.stderr.strip()[-500:])

        n_lh, n_rh = count_rows(lh_out), count_rows(rh_out)
        if n_lh < 200 or n_rh < 200:
            raise RuntimeError("Too few parcels in stats: lh=%d rh=%d" % (n_lh, n_rh))
        rec.update(status="completed", n_rois_lh=n_lh, n_rois_rh=n_rh)
    except Exception as exc:
        rec["error_message"] = "%s: %s" % (type(exc).__name__, str(exc).replace("\n", " ")[:600])
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            rec["error_message"] += " | " + exc.stderr.decode("utf-8", "replace").strip()[-300:]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fs-tar-dir", type=Path, default=DEFAULT_FS_TAR_DIR)
    p.add_argument("--sif", type=Path, default=DEFAULT_SIF)
    p.add_argument("--license", type=Path, default=DEFAULT_LICENSE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--annot-dir", type=Path, default=DEFAULT_ANNOT_DIR)
    p.add_argument("--tmp-dir", type=Path, default=None, help="Temp root (default: $TMPDIR or /tmp).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Only download the fsaverage annots.")
    args = p.parse_args()

    prepare_annots(args.annot_dir)
    if args.prepare_only:
        print("Annotations ready in %s" % args.annot_dir)
        return

    if shutil.which("singularity") is None:
        raise SystemExit("`singularity` not found. Run: module load Singularity/3.9.6")
    for path in (args.sif, args.license, args.fs_tar_dir):
        if not path.exists():
            raise SystemExit("Missing path: %s" % path)

    tars = sorted(args.fs_tar_dir.glob("sub-*_ses-*.tar"))
    if args.limit:
        tars = tars[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Processing %d sessions with %d workers -> %s" % (len(tars), args.workers, args.output_dir))

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(process_session, t, args) for t in tars]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            if i % 25 == 0 or i == len(tars) or rec["status"] == "failed":
                print("[%d/%d] %s -> %s (%.1fs) %s" % (
                    i, len(tars), rec["session_tag"], rec["status"], rec["elapsed_s"],
                    rec["error_message"][:200]), flush=True)

    results.sort(key=lambda r: r["session_tag"])
    manifest = args.output_dir / "schaefer200_fs_stats_manifest.csv"
    with open(str(manifest), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["session_tag"])
        w.writeheader()
        w.writerows(results)

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("Done in %.1fs | %s | Manifest: %s" % (time.time() - t_start, counts, manifest))


if __name__ == "__main__":
    main()
