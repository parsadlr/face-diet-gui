"""
Process all participants and their sessions through the complete pipeline.

Hierarchy:
  data/
  ├── pilot-1/
  │   ├── session1/
  │   └── session2/
  └── pilot-2/
      ├── session1/
      └── session2/

Runs:
1. Stage 1 for each session (face detection)
2. Stage 2 for each session (attributes)
3. Stage 3 for each participant (global IDs)
"""

import argparse
import sys
from pathlib import Path

# Import stage functions
from stage1_detect_faces import stage1_detect_faces
from stage2_extract_attributes import stage2_extract_attributes
from stage3_graph_clustering import stage3_graph_clustering


def find_all_sessions(data_dir: str):
    """
    Find all participant folders and their sessions.
    
    Returns
    -------
    Dict[str, List[str]]
        Mapping from participant_dir to list of session_dirs
    """
    data_path = Path(data_dir).resolve()
    
    participants = {}
    
    # Find all participant directories
    for participant_dir in sorted(data_path.iterdir()):
        if not participant_dir.is_dir():
            continue
        
        # Find sessions within participant
        sessions = []
        for session_dir in sorted(participant_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            
            # Check if it has scenevideo
            video_files = list(session_dir.glob("scenevideo.*"))
            if video_files:
                sessions.append(str(session_dir))
        
        if sessions:
            participants[str(participant_dir)] = sessions
    
    return participants


def run_stage(stage_func, description, **kwargs):
    """Run a stage function and report status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    try:
        result = stage_func(**kwargs)
        print(f"\n✓ SUCCESS: {description}")
        return True, result
    except Exception as e:
        print(f"\n❌ FAILED: {description}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def process_all_participants(
    data_dir: str,
    sampling_rate: int = 1,
    use_gpu: bool = False,
    batch_size: int = 32,
    similarity_threshold: float = 0.6,
    min_confidence: float = 0.65,
    k_neighbors: int = 50,
    min_cluster_size: int = 12,
    k_voting: int = 10,
    min_votes: int = 5,
    reassign_threshold: float = 0.55,
    skip_participants: list = None,
    only_participants: list = None,
):
    """
    Process all participants and sessions.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory containing participant folders
    sampling_rate : int
        Sampling rate for Stage 1
    use_gpu : bool
        Use GPU for Stages 1
    batch_size : int
        Batch size for Stage 2
    similarity_threshold : float
        Similarity threshold for Stage 3
    min_confidence : float
        Minimum confidence for Stage 3
    k_neighbors : int
        k-neighbors for Stage 3
    min_cluster_size : int
        Min cluster size for Stage 3 refinement
    k_voting : int
        k-voting for Stage 3 refinement
    min_votes : int
        Min votes for Stage 3 refinement
    reassign_threshold : float
        Reassignment threshold for Stage 3 refinement
    skip_participants : list, optional
        List of participant names to skip (e.g., ['pilot-1'])
    only_participants : list, optional
        Only process these participants (e.g., ['pilot-2', 'pilot-3'])
    """
    print("=" * 80)
    print("PROCESSING ALL PARTICIPANTS")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print()
    
    # Find all participants and sessions
    all_participants = find_all_sessions(data_dir)
    
    if not all_participants:
        print("❌ No participants found!")
        return
    
    # Filter participants
    participants = {}
    for participant_dir, sessions in all_participants.items():
        participant_name = Path(participant_dir).name
        
        # Apply filters
        if skip_participants and participant_name in skip_participants:
            print(f"  Skipping {participant_name} (in skip list)")
            continue
        
        if only_participants and participant_name not in only_participants:
            print(f"  Skipping {participant_name} (not in only list)")
            continue
        
        participants[participant_dir] = sessions
    
    if not participants:
        print("❌ No participants remaining after filtering!")
        return
    
    print(f"Processing {len(participants)} participant(s):")
    for participant_dir, sessions in participants.items():
        participant_name = Path(participant_dir).name
        print(f"  {participant_name}: {len(sessions)} session(s)")
    print()
    
    # STAGE 1: Face Detection
    print("\n" + "=" * 80)
    print("STAGE 1: FACE DETECTION (All Sessions)")
    print("=" * 80)
    
    all_sessions = [session for sessions in participants.values() for session in sessions]
    
    for i, session_dir in enumerate(all_sessions, 1):
        session_name = Path(session_dir).name
        participant_name = Path(session_dir).parent.name
        
        success, result = run_stage(
            stage1_detect_faces,
            f"Stage 1: {participant_name}/{session_name} ({i}/{len(all_sessions)})",
            session_dir=session_dir,
            sampling_rate=sampling_rate,
            start_time=None,
            end_time=None,
            use_gpu=use_gpu,
        )
        
        if not success:
            print(f"⚠️  Continuing with next session...")
    
    # STAGE 2: Attribute Extraction
    print("\n" + "=" * 80)
    print("STAGE 2: ATTRIBUTE EXTRACTION (All Sessions)")
    print("=" * 80)
    
    for i, session_dir in enumerate(all_sessions, 1):
        session_name = Path(session_dir).name
        participant_name = Path(session_dir).parent.name
        
        success, result = run_stage(
            stage2_extract_attributes,
            f"Stage 2: {participant_name}/{session_name} ({i}/{len(all_sessions)})",
            session_dir=session_dir,
            batch_size=batch_size,
            limit=None,
        )
        
        if not success:
            print(f"⚠️  Continuing with next session...")
    
    # STAGE 3: Global ID Assignment
    print("\n" + "=" * 80)
    print("STAGE 3: GLOBAL ID ASSIGNMENT (All Participants)")
    print("=" * 80)
    
    for i, (participant_dir, sessions) in enumerate(participants.items(), 1):
        participant_name = Path(participant_dir).name
        
        success, result = run_stage(
            stage3_graph_clustering,
            f"Stage 3: {participant_name} ({i}/{len(participants)})",
            participant_dir=participant_dir,
            similarity_threshold=similarity_threshold,
            k_neighbors=k_neighbors,
            output_name='faces_combined.csv',
            min_confidence=min_confidence,
            algorithm='leiden',
            enable_refinement=True,
            min_cluster_size=min_cluster_size,
            k_voting=k_voting,
            min_votes=min_votes,
            reassign_threshold=reassign_threshold,
        )
        
        if not success:
            print(f"⚠️  Continuing with next participant...")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ ALL PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Processed {len(participants)} participant(s)")
    print(f"Processed {len(all_sessions)} session(s)")
    print("\nOutputs:")
    for participant_dir in participants.keys():
        participant_name = Path(participant_dir).name
        print(f"  {participant_name}/faces_combined.csv")
        print(f"  {participant_name}/faces_combined.stats.txt")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process all participants and sessions through complete pipeline"
    )
    
    parser.add_argument(
        'data_dir',
        help='Path to data directory containing participant folders'
    )
    
    # Stage 1 parameters
    parser.add_argument(
        '-s', '--sampling-rate',
        type=int,
        default=1,
        help='Sampling rate for Stage 1 (default: 1 = every frame)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for Stage 1'
    )
    
    # Stage 2 parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for Stage 2 (default: 32)'
    )
    
    # Stage 3 parameters
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.6,
        help='Similarity threshold for Stage 3 (default: 0.6)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.65,
        help='Minimum confidence for Stage 3 (default: 0.65)'
    )
    parser.add_argument(
        '-k', '--k-neighbors',
        type=int,
        default=50,
        help='k-neighbors for Stage 3 (default: 50)'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=12,
        help='Min cluster size for Stage 3 refinement (default: 12)'
    )
    parser.add_argument(
        '--k-voting',
        type=int,
        default=10,
        help='k-voting for Stage 3 refinement (default: 10)'
    )
    parser.add_argument(
        '--min-votes',
        type=int,
        default=5,
        help='Min votes for Stage 3 refinement (default: 5)'
    )
    parser.add_argument(
        '--reassign-threshold',
        type=float,
        default=0.55,
        help='Reassignment threshold for Stage 3 refinement (default: 0.55)'
    )
    
    # Filtering options
    parser.add_argument(
        '--skip',
        nargs='+',
        help='Skip these participants (e.g., --skip pilot-1 pilot-3)'
    )
    parser.add_argument(
        '--only',
        nargs='+',
        help='Only process these participants (e.g., --only pilot-2)'
    )
    
    args = parser.parse_args()
    
    try:
        process_all_participants(
            data_dir=args.data_dir,
            sampling_rate=args.sampling_rate,
            use_gpu=args.gpu,
            batch_size=args.batch_size,
            similarity_threshold=args.threshold,
            min_confidence=args.min_confidence,
            k_neighbors=args.k_neighbors,
            min_cluster_size=args.min_cluster_size,
            k_voting=args.k_voting,
            min_votes=args.min_votes,
            reassign_threshold=args.reassign_threshold,
            skip_participants=args.skip,
            only_participants=args.only,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

