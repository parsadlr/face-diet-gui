"""
Stage 3: Combine Sessions and Assign Global Face IDs

Self-contained script that combines all sessions and assigns consistent face IDs
using FAISS-based two-stage clustering.

Input: <session_dir>/stage2_attributes.csv (for each session)
Output: <participant_dir>/faces_combined.csv
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================================
# FAISS-BASED CLUSTERING FUNCTIONS
# ============================================================================

def create_faiss_index(dim: int, use_gpu: bool = False):
    """Create FAISS index for inner product (cosine similarity)."""
    index = faiss.IndexFlatIP(dim)
    
    if use_gpu:
        try:
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print(f"      FAISS: Using GPU")
            else:
                print(f"      FAISS: Using CPU")
        except (AttributeError, RuntimeError):
            print(f"      FAISS: Using CPU")
    else:
        print(f"      FAISS: Using CPU")
    
    return index


def load_session_data(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load session CSV and parse embeddings."""
    df = pd.read_csv(csv_path)
    
    embeddings = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        embedding_str = row.get('embedding', '')
        if embedding_str and not pd.isna(embedding_str):
            try:
                embedding = np.array(json.loads(embedding_str), dtype=np.float32)
                embeddings.append(embedding)
                valid_indices.append(idx)
            except (json.JSONDecodeError, ValueError):
                continue
    
    if not embeddings:
        return df.iloc[valid_indices], np.array([])
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    return df_valid, embeddings_array


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def local_temporal_clustering(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    similarity_threshold: float,
    temporal_window_seconds: float,
    use_gpu: bool = False,
) -> Tuple[List[int], Dict, Dict]:
    """Local temporal clustering within a session."""
    n = len(df)
    print(f"  Local temporal clustering: {n:,} faces")
    
    if n == 0:
        return [], {}, {}
    
    embeddings_norm = normalize_embeddings(embeddings)
    
    # Build FAISS index
    dim = embeddings_norm.shape[1]
    index = create_faiss_index(dim, use_gpu=use_gpu)
    index.add(embeddings_norm)
    
    # Sort by time
    df_sorted = df.copy().sort_values('time_seconds').reset_index(drop=True)
    times = df_sorted['time_seconds'].values
    
    # Split into windows
    min_time, max_time = times[0], times[-1]
    window_boundaries = np.arange(min_time, max_time + temporal_window_seconds, temporal_window_seconds)
    
    local_ids = [-1] * n
    next_local_id = 0
    cluster_representatives = {}
    
    num_windows = len(window_boundaries) - 1
    print(f"    Processing {num_windows} windows...")
    
    k_neighbors = 5
    
    for window_idx in range(num_windows):
        window_start = window_boundaries[window_idx]
        window_end = window_boundaries[window_idx + 1]
        
        mask = (times >= window_start) & (times < window_end)
        window_indices = np.where(mask)[0]
        
        for idx in window_indices:
            if local_ids[idx] >= 0:
                continue
            
            query_emb = embeddings_norm[idx:idx+1]
            distances, neighbor_indices = index.search(query_emb, k_neighbors + 1)
            
            similarities = distances[0]
            neighbors = neighbor_indices[0]
            
            merged = False
            frame_num = df_sorted.iloc[idx]['frame_number']
            
            for neighbor_idx, sim in zip(neighbors, similarities):
                if neighbor_idx == idx:
                    continue
                
                if local_ids[neighbor_idx] >= 0 and sim >= similarity_threshold:
                    neighbor_frame = df_sorted.iloc[neighbor_idx]['frame_number']
                    if frame_num != neighbor_frame:
                        local_ids[idx] = local_ids[neighbor_idx]
                        merged = True
                        break
            
            if not merged:
                local_ids[idx] = next_local_id
                next_local_id += 1
        
        if (window_idx + 1) % 50 == 0 or window_idx == num_windows - 1:
            percent = int(100 * (window_idx + 1) / num_windows)
            print(f"      [{percent:3d}%] Processed {window_idx + 1}/{num_windows} windows, {next_local_id} clusters")
    
    # Compute centroids
    for cluster_id in range(next_local_id):
        cluster_mask = np.array(local_ids) == cluster_id
        cluster_embeddings = embeddings_norm[cluster_mask]
        cluster_frames = df_sorted[cluster_mask]['frame_number'].values
        
        if len(cluster_embeddings) > 0:
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            cluster_representatives[cluster_id] = (centroid, cluster_frames)
    
    stats = {'num_faces': n, 'num_local_clusters': next_local_id}
    return local_ids, cluster_representatives, stats


def merge_clusters_centroid(sessions_data, similarity_threshold, use_gpu):
    """Merge using centroids (faster)."""
    print("\n  Global merging (centroid)...")
    
    all_centroids = []
    centroid_to_session_cluster = []
    
    for session_idx, session in enumerate(sessions_data):
        reps = session['cluster_representatives']
        for local_id, (centroid, frames) in reps.items():
            all_centroids.append(centroid)
            centroid_to_session_cluster.append((session_idx, local_id))
    
    num_centroids = len(all_centroids)
    print(f"    Merging {num_centroids} clusters...")
    
    if num_centroids == 0:
        return {}, {}
    
    centroids_array = np.array(all_centroids, dtype=np.float32)
    dim = centroids_array.shape[1]
    
    index = create_faiss_index(dim, use_gpu)
    index.add(centroids_array)
    
    global_ids = [-1] * num_centroids
    next_global_id = 0
    
    for i in range(num_centroids):
        if global_ids[i] >= 0:
            continue
        
        global_ids[i] = next_global_id
        
        query = centroids_array[i:i+1]
        distances, neighbors = index.search(query, min(100, num_centroids))
        
        for neighbor_idx, sim in zip(neighbors[0], distances[0]):
            if neighbor_idx != i and global_ids[neighbor_idx] < 0 and sim >= similarity_threshold:
                global_ids[neighbor_idx] = next_global_id
        
        next_global_id += 1
        
        if (i + 1) % 100 == 0 or i == num_centroids - 1:
            print(f"    [{100*(i+1)//num_centroids:3d}%] {i+1}/{num_centroids}, {next_global_id} IDs")
    
    print(f"    ✓ Complete: {next_global_id} global IDs")
    
    session_local_to_global = {}
    for centroid_idx, (session_idx, local_id) in enumerate(centroid_to_session_cluster):
        session_local_to_global[(session_idx, local_id)] = global_ids[centroid_idx]
    
    return session_local_to_global, {'num_global_ids': next_global_id}


def merge_clusters_min_distance(sessions_data, similarity_threshold, use_gpu):
    """Merge using minimum distance (single-linkage)."""
    print("\n  Global merging (min-distance)...")
    
    all_faces = []
    face_to_session_cluster = []
    
    for session_idx, session in enumerate(sessions_data):
        embeddings = session['embeddings']
        local_ids = session['local_cluster_ids']
        
        for face_idx, local_id in enumerate(local_ids):
            all_faces.append(embeddings[face_idx])
            face_to_session_cluster.append((session_idx, local_id))
    
    num_faces = len(all_faces)
    print(f"    Analyzing {num_faces:,} faces...")
    
    if num_faces == 0:
        return {}, {}
    
    faces_array = np.array(all_faces, dtype=np.float32)
    faces_norm = normalize_embeddings(faces_array)
    dim = faces_norm.shape[1]
    
    index = create_faiss_index(dim, use_gpu)
    index.add(faces_norm)
    
    # Find connections
    local_cluster_connections = {}
    
    for i, (session_idx, local_id) in enumerate(face_to_session_cluster):
        query = faces_norm[i:i+1]
        distances, neighbors = index.search(query, 11)
        
        key = (session_idx, local_id)
        if key not in local_cluster_connections:
            local_cluster_connections[key] = set()
        
        for neighbor_idx, sim in zip(neighbors[0], distances[0]):
            if neighbor_idx != i and sim >= similarity_threshold:
                neighbor_key = face_to_session_cluster[neighbor_idx]
                if neighbor_key != key:
                    local_cluster_connections[key].add(neighbor_key)
        
        if (i + 1) % 10000 == 0 or i == num_faces - 1:
            print(f"      [{100*(i+1)//num_faces:3d}%] {i+1:,}/{num_faces:,}")
    
    # Union-Find
    all_keys = list(set(face_to_session_cluster))
    parent = {key: key for key in all_keys}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for key, connections in local_cluster_connections.items():
        for neighbor_key in connections:
            union(key, neighbor_key)
    
    unique_roots = list(set(find(key) for key in all_keys))
    root_to_global = {root: i for i, root in enumerate(unique_roots)}
    
    session_local_to_global = {key: root_to_global[find(key)] for key in all_keys}
    
    print(f"    ✓ Complete: {len(unique_roots)} global IDs")
    
    return session_local_to_global, {'num_global_ids': len(unique_roots)}


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def stage3_combine_sessions(
    participant_dir: str,
    similarity_threshold: float = 0.6,
    temporal_window: float = 10.0,
    output_name: str = 'faces_combined.csv',
    use_gpu: bool = False,
    min_confidence: float = 0.0,
    merge_method: str = 'centroid',
):
    """
    Stage 3: Combine sessions and assign global face IDs.
    
    Parameters
    ----------
    participant_dir : str
        Path to participant directory
    similarity_threshold : float
        Cosine similarity threshold (0-1)
    temporal_window : float
        Temporal window size in seconds
    output_name : str
        Output filename
    use_gpu : bool
        Use GPU for FAISS
    min_confidence : float
        Minimum detection confidence to include
    merge_method : str
        'centroid' (faster) or 'min_distance' (more sensitive)
    """
    if not FAISS_AVAILABLE:
        print("❌ ERROR: FAISS not installed")
        print("   Install with: pip install faiss-cpu")
        sys.exit(1)
    
    participant_path = Path(participant_dir).resolve()
    output_csv = participant_path / output_name
    
    print("=" * 80)
    print("STAGE 3: GLOBAL FACE ID ASSIGNMENT")
    print("=" * 80)
    print(f"Participant: {participant_path.name}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Min confidence: {min_confidence}")
    print(f"Merge method: {merge_method}")
    print(f"Temporal window: {temporal_window}s")
    print()
    
    # Find sessions
    session_csvs = []
    for session_dir in sorted(participant_path.iterdir()):
        if not session_dir.is_dir():
            continue
        stage2_csv = session_dir / "stage2_attributes.csv"
        if stage2_csv.exists():
            session_csvs.append({
                'session_name': session_dir.name,
                'session_dir': session_dir,
                'csv_path': stage2_csv,
            })
    
    if not session_csvs:
        raise FileNotFoundError("No stage2_attributes.csv files found")
    
    print(f"Found {len(session_csvs)} sessions")
    
    # LOCAL CLUSTERING
    print("\n" + "=" * 80)
    print("LOCAL TEMPORAL CLUSTERING")
    print("=" * 80)
    
    sessions_data = []
    
    for session_idx, session in enumerate(session_csvs):
        print(f"\nSession {session_idx + 1}: {session['session_name']}")
        
        df, embeddings = load_session_data(session['csv_path'])
        
        if len(df) == 0:
            continue
        
        print(f"  Loaded {len(df):,} faces")
        
        # Filter by confidence
        if min_confidence > 0:
            mask = df['confidence'] >= min_confidence
            df = df[mask].reset_index(drop=True)
            embeddings = embeddings[mask]
            print(f"  Filtered: {len(df):,} faces (conf≥{min_confidence})")
        
        if len(df) == 0:
            continue
        
        local_ids, reps, stats = local_temporal_clustering(
            df, embeddings, similarity_threshold, temporal_window, use_gpu
        )
        
        print(f"  → {stats['num_local_clusters']} clusters")
        
        sessions_data.append({
            'session_idx': session_idx,
            'session_name': session['session_name'],
            'session_dir': session['session_dir'],
            'df': df,
            'embeddings': embeddings,
            'local_cluster_ids': local_ids,
            'cluster_representatives': reps,
            'stats': stats,
        })
    
    # GLOBAL MERGING
    print("\n" + "=" * 80)
    print(f"GLOBAL MERGING ({merge_method})")
    print("=" * 80)
    
    if merge_method == 'min_distance':
        local_to_global_map, global_stats = merge_clusters_min_distance(
            sessions_data, similarity_threshold, use_gpu
        )
    else:
        local_to_global_map, global_stats = merge_clusters_centroid(
            sessions_data, similarity_threshold, use_gpu
        )
    
    print(f"\n✓ {global_stats['num_global_ids']} global IDs")
    
    # PROPAGATE IDs
    print("\n" + "=" * 80)
    print("PROPAGATING IDs")
    print("=" * 80)
    
    all_dfs = []
    
    for session in sessions_data:
        df = session['df'].copy()
        local_ids = session['local_cluster_ids']
        
        global_face_ids = []
        for local_id in local_ids:
            key = (session['session_idx'], local_id)
            global_id = local_to_global_map.get(key, -1)
            global_face_ids.append(f"FACE_{global_id:05d}" if global_id >= 0 else "UNKNOWN")
        
        df['face_id'] = global_face_ids
        df['session_name'] = session['session_name']
        all_dfs.append(df)
        
        print(f"  ✓ {session['session_name']}: {len(df):,} faces")
    
    # Combine
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Reorder columns
    cols = list(combined_df.columns)
    cols.remove('face_id')
    cols.remove('session_name')
    if 'embedding' in cols:
        cols.remove('embedding')
        cols = ['face_id', 'session_name'] + cols + ['embedding']
    else:
        cols = ['face_id', 'session_name'] + cols
    
    combined_df = combined_df[cols]
    combined_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Output: {output_csv}")
    
    # Create stats file
    stats_file = participant_path / "faces_combined.stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Global Face ID Assignment Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total faces: {len(combined_df):,}\n")
        f.write(f"Unique global IDs: {global_stats['num_global_ids']}\n")
        f.write(f"Sessions: {len(sessions_data)}\n\n")
        
        f.write("Per-session breakdown:\n\n")
        for session in sessions_data:
            session_df = combined_df[combined_df['session_name'] == session['session_name']]
            unique_ids = session_df['face_id'].nunique()
            attended = int(session_df['attended'].sum()) if 'attended' in session_df.columns else 0
            attended_pct = 100 * attended / len(session_df) if len(session_df) > 0 else 0
            
            f.write(f"{session['session_name']}:\n")
            f.write(f"  Total faces: {len(session_df):,}\n")
            f.write(f"  Local clusters: {session['stats']['num_local_clusters']}\n")
            f.write(f"  Unique global IDs: {unique_ids}\n")
            f.write(f"  Attended faces: {attended:,} ({attended_pct:.1f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Global ID Distribution (sorted by instance count):\n\n")
        
        # Get counts and sort by instance count (descending)
        id_counts = combined_df['face_id'].value_counts()  # Already sorted by count
        
        for face_id in id_counts.index:
            if face_id == 'UNKNOWN':
                continue
            count = id_counts[face_id]
            faces = combined_df[combined_df['face_id'] == face_id]
            sessions_present = faces['session_name'].unique()
            attended_count = int(faces['attended'].sum()) if 'attended' in faces.columns else 0
            
            f.write(f"{face_id}: {count} instances across {len(sessions_present)} session(s) ({attended_count} attended)\n")
    
    print(f"  ✓ Stats file: {stats_file}")
    print(f"  {len(combined_df):,} faces, {global_stats['num_global_ids']} global IDs")
    print("=" * 80)
    
    return {
        'total_faces': len(combined_df),
        'unique_global_ids': global_stats['num_global_ids'],
        'output_csv': str(output_csv),
        'stats_file': str(stats_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Combine sessions and assign global face IDs"
    )
    
    parser.add_argument('participant_dir', help='Path to participant directory')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Similarity threshold (default: 0.6)')
    parser.add_argument('--temporal-window', type=float, default=10.0, help='Temporal window size in seconds (default: 10.0)')
    parser.add_argument('-o', '--output', default='faces_combined.csv', help='Output filename')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for FAISS')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence (default: 0.0)')
    parser.add_argument('--merge-method', choices=['centroid', 'min_distance'], default='centroid', 
                        help='Merge method: centroid (faster) or min_distance (more sensitive)')
    
    args = parser.parse_args()
    
    try:
        stage3_combine_sessions(
            participant_dir=args.participant_dir,
            similarity_threshold=args.threshold,
            temporal_window=args.temporal_window,
            output_name=args.output,
            use_gpu=args.gpu,
            min_confidence=args.min_confidence,
            merge_method=args.merge_method,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
