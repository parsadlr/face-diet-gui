"""
Pipeline helpers for Face-Diet GUI: stage subprocess runners and session/annotation helpers.

Used by GUI tabs to run stage scripts via subprocess and to query session/review status.
"""

import json
import re
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from face_diet_gui.core.settings_manager import ReviewerRegistry


# Project root (face_diet_gui/core/pipeline_helpers.py -> parent.parent.parent)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ProcessingStopped(Exception):
    """Raised when the user stops processing via the Stop button."""


def _discard_annotations_for_session(project_dir: Path, participant_name: str, session_name: str) -> None:
    """
    Remove reviewer annotations that depend on face detection for this session.
    After re-running Stage 1, the following are no longer valid. Deletes for all reviewers:
    - That session's is_face.csv (face/non-face review)
    - That participant's merges.csv (manual merges and media flags)
    Face ID clustering output (face_ids.csv in participant folder) is not removed.
    """
    try:
        registry = ReviewerRegistry(project_dir)
        for reviewer_id in registry.get_reviewer_ids():
            is_face_path = registry.get_is_face_annotation_path(reviewer_id, participant_name, session_name)
            if is_face_path.exists():
                is_face_path.unlink()
            merges_path = registry.get_merges_path(reviewer_id, participant_name)
            if merges_path.exists():
                merges_path.unlink()
    except Exception as e:
        print(f"Warning: could not discard some annotations: {e}")


def _load_review_status_for_session(registry: ReviewerRegistry, reviewer_id: str, participant: str, session: str) -> Dict:
    """Load {reviewed: bool} for a session from reviewer's review_status.json. Used for session list."""
    ann_path = registry.get_is_face_annotation_path(reviewer_id, participant, session)
    path = ann_path.parent / "review_status.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"reviewed": bool(data.get("reviewed", False))}
        except Exception:
            pass
    return {"reviewed": False}


def _load_mismatches_resolved_flag(registry: ReviewerRegistry, participant: str, session: str) -> bool:
    """Load global 'mismatches resolved' flag for a session (Tab 3), from _annotations/consensus/."""
    path = registry.get_mismatches_resolved_path(participant, session)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("resolved", False))
        except Exception:
            pass
    return False


def _get_sessions_with_review_status(project_dir: Path) -> List[Dict]:
    """
    For each session (participant/session with face_detections.csv), return:
    - reviewers_with_tab2_count: number of reviewers who have submitted AND marked session as fully reviewed
    - mismatch_count: instances where those reviewers disagree (vs consensus if saved, else pairwise)
    - resolved: 2+ such reviewers AND mismatch_count == 0
    """
    registry = ReviewerRegistry(project_dir)
    reviewer_ids = registry.get_reviewer_ids()
    result = []
    for participant_dir in sorted(project_dir.iterdir()):
        if not participant_dir.is_dir() or participant_dir.name.startswith(("_", ".")):
            continue
        participant = participant_dir.name
        for session_dir in sorted(participant_dir.iterdir()):
            if not session_dir.is_dir() or (session_dir / "face_detections.csv").exists() is False:
                continue
            session = session_dir.name
            if session.startswith(("_", ".")):
                continue
            with_tab2 = [rid for rid in reviewer_ids if registry.get_is_face_annotation_path(rid, participant, session).exists()]
            reviewers_with_tab2 = [rid for rid in with_tab2 if _load_review_status_for_session(registry, rid, participant, session).get("reviewed", False)]
            mismatch_count = 0
            if len(reviewers_with_tab2) >= 2:
                try:
                    df = pd.read_csv(session_dir / "face_detections.csv")
                    if "confidence" in df.columns:
                        df = df.sort_values("confidence", ascending=True).reset_index(drop=True)
                    indices = list(df.index)
                    per_reviewer = {}
                    for rid in reviewers_with_tab2:
                        ann_path = registry.get_is_face_annotation_path(rid, participant, session)
                        ann_df = pd.read_csv(ann_path)
                        per_reviewer[rid] = dict(zip(ann_df["instance_index"].astype(int), ann_df["is_face"].astype(bool)))
                    consensus_path = registry.get_consensus_annotation_path(participant, session)
                    if consensus_path.exists():
                        try:
                            consensus_mtime = consensus_path.stat().st_mtime
                            cons_df = pd.read_csv(consensus_path)
                            consensus = dict(zip(cons_df["instance_index"].astype(int), cons_df["is_face"].astype(bool)))
                            post_consensus = [rid for rid in reviewers_with_tab2
                                              if registry.get_is_face_annotation_path(rid, participant, session).stat().st_mtime > consensus_mtime]
                            for idx in indices:
                                cons_val = consensus.get(int(idx), True)
                                for rid in post_consensus:
                                    if per_reviewer[rid].get(int(idx), True) != cons_val:
                                        mismatch_count += 1
                                        break
                        except Exception:
                            pass
                    else:
                        for idx in indices:
                            vals = [per_reviewer[rid].get(int(idx), True) for rid in reviewers_with_tab2]
                            if len(set(vals)) > 1:
                                mismatch_count += 1
                except Exception:
                    pass
            resolved = (len(reviewers_with_tab2) >= 2 and mismatch_count == 0)
            result.append({
                "participant": participant,
                "session": session,
                "session_dir": session_dir,
                "reviewers_with_tab2_count": len(reviewers_with_tab2),
                "mismatch_count": mismatch_count,
                "resolved": resolved,
            })
    return result


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _run_stage1_via_subprocess(session_dir: str, sampling_rate: int, use_gpu: bool,
                               min_confidence: float, reporter, debug_mode: bool = False,
                               settings=None,
                               process_holder: Optional[List] = None,
                               stop_check: Optional[Callable[[], bool]] = None):
    """Run detect_faces via subprocess using python -m face_diet_gui.stages.detect_faces."""
    import cv2

    processing_python = Path(sys.executable)
    end_time = None
    if debug_mode:
        session_path = Path(session_dir)
        video_files = list(session_path.glob("scenevideo.*"))
        if video_files:
            try:
                cap = cv2.VideoCapture(str(video_files[0]))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = total_frames / fps if fps > 0 else 0.0
                    end_time = video_duration * 0.05
                    cap.release()
                    reporter.log(f"DEBUG MODE: Processing only first 5% of video ({end_time:.1f}s)")
            except Exception as e:
                reporter.log(f"Warning: Could not calculate video duration for debug mode: {e}")

    cmd = [
        str(processing_python),
        "-u",
        "-m",
        "face_diet_gui.stages.detect_faces",
        session_dir,
        "--sampling-rate", str(sampling_rate),
        "--min-confidence", str(min_confidence),
    ]
    if use_gpu:
        cmd.append("--gpu")
    if end_time is not None:
        cmd.extend(["--end-time", str(end_time)])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        cwd=str(_PROJECT_ROOT),
    )
    if process_holder is not None:
        process_holder[0] = process

    progress_pattern = re.compile(r'\[\s*(\d+)%\].*?(\d+)/(\d+)\s+frames')
    last_percent = 0
    last_processed = 0
    last_total = 0
    step_start_time = time.time()
    error_lines = []

    for line in process.stdout:
        reporter.log(line.rstrip())
        match = progress_pattern.search(line)
        if match:
            percent = int(match.group(1))
            processed = int(match.group(2))
            total = int(match.group(3))
            reporter.update_progress(percent / 100.0, f"{percent}%")
            if processed > 0 and total > 0:
                elapsed = time.time() - step_start_time
                frames_remaining = total - processed
                if processed > 0:
                    avg_time_per_frame = elapsed / processed
                    estimated_remaining = avg_time_per_frame * frames_remaining
                    elapsed_str = _format_time(elapsed)
                    remaining_str = _format_time(estimated_remaining)
                    reporter.update_time_estimate(elapsed_str, remaining_str)
            last_percent = percent
            last_processed = processed
            last_total = total

    def read_stderr():
        for line in process.stderr:
            error_lines.append(line.rstrip())
            reporter.log(f"ERROR: {line.rstrip()}")

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    return_code = process.wait()
    stderr_thread.join(timeout=1.0)
    if process_holder is not None:
        process_holder[0] = None

    if stop_check and stop_check():
        raise ProcessingStopped()
    if return_code != 0:
        error_msg = f"Face detection failed with return code {return_code}"
        if error_lines:
            error_msg += f"\n\nError output:\n" + "\n".join(error_lines[-10:])
        raise RuntimeError(error_msg)
    return True


def _run_stage2_via_subprocess(session_dir: str, batch_size: int, reporter,
                               debug_mode: bool = False, settings=None,
                               process_holder: Optional[List] = None,
                               stop_check: Optional[Callable[[], bool]] = None):
    """Run extract_attributes via subprocess using python -m face_diet_gui.stages.extract_attributes."""
    limit = None
    if debug_mode:
        session_path = Path(session_dir)
        csv_path = session_path / "face_detections.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                total_faces = len(df)
                limit = max(1, int(total_faces * 0.05))
                reporter.log(f"DEBUG MODE: Processing only first {limit} faces (5% of {total_faces} total)")
            except Exception as e:
                reporter.log(f"Warning: Could not calculate face limit for debug mode: {e}")

    processing_python = Path(sys.executable)
    cmd = [
        str(processing_python),
        "-u",
        "-m",
        "face_diet_gui.stages.extract_attributes",
        session_dir,
        "--batch-size", str(batch_size),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        cwd=str(_PROJECT_ROOT),
    )
    if process_holder is not None:
        process_holder[0] = process

    progress_pattern = re.compile(r'\[\s*(\d+)%\].*?(\d+)/(\d+)\s+faces')
    step_start_time = time.time()
    error_lines = []

    for line in process.stdout:
        reporter.log(line.rstrip())
        match = progress_pattern.search(line)
        if match:
            percent = int(match.group(1))
            processed = int(match.group(2))
            total = int(match.group(3))
            reporter.update_progress(percent / 100.0, f"{percent}%")
            if processed > 0 and total > 0:
                elapsed = time.time() - step_start_time
                faces_remaining = total - processed
                if processed > 0:
                    avg_time_per_face = elapsed / processed
                    estimated_remaining = avg_time_per_face * faces_remaining
                    elapsed_str = _format_time(elapsed)
                    remaining_str = _format_time(estimated_remaining)
                    reporter.update_time_estimate(elapsed_str, remaining_str)

    def read_stderr():
        for line in process.stderr:
            error_lines.append(line.rstrip())
            reporter.log(f"ERROR: {line.rstrip()}")

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    return_code = process.wait()
    stderr_thread.join(timeout=1.0)
    if process_holder is not None:
        process_holder[0] = None

    if stop_check and stop_check():
        raise ProcessingStopped()
    if return_code != 0:
        error_msg = f"Attribute extraction failed with return code {return_code}"
        if error_lines:
            error_msg += f"\n\nError output:\n" + "\n".join(error_lines[-10:])
        raise RuntimeError(error_msg)
    return True


def _run_stage3_via_subprocess(
    participant_dir: str,
    annotations_dir: str,
    output_dir: str,
    similarity_threshold: float,
    k_neighbors: int,
    min_confidence: float,
    algorithm: str,
    enable_refinement: bool,
    min_cluster_size: int,
    k_voting: int,
    min_votes: int,
    reporter,
    process_holder: Optional[List] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    consensus_dir: str = None,
):
    """Run cluster_face_ids via subprocess using python -m face_diet_gui.stages.cluster_face_ids."""
    processing_python = Path(sys.executable)
    cmd = [
        str(processing_python),
        "-u",
        "-m",
        "face_diet_gui.stages.cluster_face_ids",
        participant_dir,
        "--threshold", str(similarity_threshold),
        "--k-neighbors", str(k_neighbors),
        "--min-confidence", str(min_confidence),
        "--algorithm", algorithm,
        "--annotations_dir", annotations_dir,
        "--output_dir", output_dir,
    ]
    if consensus_dir:
        cmd.extend(["--consensus_dir", consensus_dir])
    if not enable_refinement:
        cmd.append("--no-refine")
    else:
        cmd.extend(["--min-cluster-size", str(min_cluster_size)])
        cmd.extend(["--k-voting", str(k_voting)])
        cmd.extend(["--min-votes", str(min_votes)])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        cwd=str(_PROJECT_ROOT),
    )
    if process_holder is not None:
        process_holder[0] = process

    progress_pattern = re.compile(r'\[\s*(\d+)%\].*?Processed\s+[\d,]+/?([\d,]+)\s+.*?faces')
    step_start_time = time.time()

    for line in process.stdout:
        reporter.log(line.rstrip())
        match = progress_pattern.search(line)
        if match:
            percent = int(match.group(1))
            reporter.update_progress(percent / 100.0, f"{percent}%")
            elapsed = time.time() - step_start_time
            if percent > 0:
                estimated_total = elapsed / (percent / 100.0)
                estimated_remaining = estimated_total - elapsed
                elapsed_str = _format_time(elapsed)
                remaining_str = _format_time(estimated_remaining)
                reporter.update_time_estimate(elapsed_str, remaining_str)

    return_code = process.wait()
    if process_holder is not None:
        process_holder[0] = None

    if stop_check and stop_check():
        raise ProcessingStopped()
    if return_code != 0:
        raise RuntimeError(f"Face ID clustering failed with return code {return_code}")
    return {
        'total_faces': 0,
        'unique_global_ids': 0,
    }
