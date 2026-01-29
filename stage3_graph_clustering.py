"""
Stage 3: Graph-Based Community Detection for Global Face IDs

Uses FAISS k-NN + Louvain/Leiden community detection for global clustering.
No temporal windowing - all faces clustered globally at once.

Algorithm:
1. Load all embeddings from all sessions
2. Build k-NN similarity graph using FAISS
3. Enforce same-frame constraint (no edges between same-frame faces)
4. Run community detection (Louvain or Leiden)
5. Assign global face IDs based on communities
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    import faiss
except ImportError:
    print("ERROR: FAISS not installed. Install with: pip install faiss-cpu")
    sys.exit(1)

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

if not LEIDEN_AVAILABLE and not NETWORKX_AVAILABLE:
    print("ERROR: No community detection library available")
    print("Install one of:")
    print("  - igraph + leidenalg: pip install python-igraph leidenalg")
    print("  - networkx: pip install networkx")
    sys.exit(1)


def load_all_sessions(participant_dir: str, min_confidence: float = 0.0) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Load all session data into one combined dataset.
    
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, Dict]
        (combined_df, embeddings_array, metadata)
    """
    participant_path = Path(participant_dir).resolve()
    
    # Find all session CSVs
    session_csvs = []
    for session_dir in sorted(participant_path.iterdir()):
        if not session_dir.is_dir():
            continue
        face_csv = session_dir / "face_detections.csv"
        if face_csv.exists():
            session_csvs.append({
                'session_name': session_dir.name,
                'session_dir': session_dir,
                'csv_path': face_csv,
            })
    
    if not session_csvs:
        raise FileNotFoundError("No face_detections.csv files found")
    
    print(f"Found {len(session_csvs)} sessions")
    
    # Load all sessions
    all_dfs = []
    all_embeddings = []
    
    for session_idx, session in enumerate(session_csvs):
        print(f"  Loading {session['session_name']}...")
        
        df = pd.read_csv(session['csv_path'])
        
        # Parse embeddings
        embeddings = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            embedding_str = row.get('embedding', '')
            if embedding_str and not pd.isna(embedding_str):
                try:
                    embedding = np.array(json.loads(embedding_str), dtype=np.float32)
                    embeddings.append(embedding)
                    valid_indices.append(idx)
                except:
                    continue
        
        df = df.iloc[valid_indices].reset_index(drop=True)
        
        # Filter by confidence
        if min_confidence > 0 and 'confidence' in df.columns:
            mask = df['confidence'] >= min_confidence
            df = df[mask].reset_index(drop=True)
            embeddings = [embeddings[i] for i, m in enumerate(mask) if m]
        
        if len(df) == 0:
            print(f"    No valid faces, skipping")
            continue
        
        df['session_name'] = session['session_name']
        df['session_index'] = session_idx
        df['original_row_index'] = df.index
        
        all_dfs.append(df)
        all_embeddings.extend(embeddings)
        
        print(f"    Loaded {len(df):,} faces")
    
    if not all_dfs:
        raise ValueError("No valid faces found in any session")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    
    metadata = {
        'num_sessions': len(session_csvs),
        'session_names': [s['session_name'] for s in session_csvs],
    }
    
    return combined_df, embeddings_array, metadata


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def build_knn_graph(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    k: int = 50,
    similarity_threshold: float = 0.6,
) -> Tuple[List, List, List]:
    """
    Build k-NN similarity graph with same-frame constraint.
    
    Returns
    -------
    Tuple[List, List, List]
        (edge_sources, edge_targets, edge_weights)
    """
    n = len(embeddings)
    print(f"\nBuilding k-NN graph:")
    print(f"  {n:,} faces, k={k}, threshold={similarity_threshold}")
    
    # Normalize embeddings
    embeddings_norm = normalize_embeddings(embeddings)
    
    # Build FAISS index
    dim = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    
    print(f"  Searching for {k} nearest neighbors...")
    
    # Find k nearest neighbors for all faces
    distances, neighbors = index.search(embeddings_norm, k + 1)
    
    # Build edge list
    edge_sources = []
    edge_targets = []
    edge_weights = []
    
    frame_numbers = df['frame_number'].values
    
    print(f"  Building edges (enforcing same-frame constraint)...")
    
    for i in range(n):
        frame_i = frame_numbers[i]
        
        for j_idx in range(k + 1):
            j = neighbors[i, j_idx]
            sim = distances[i, j_idx]
            
            if i == j:
                continue  # Skip self
            
            if sim < similarity_threshold:
                continue  # Below threshold
            
            # Same-frame constraint
            frame_j = frame_numbers[j]
            if frame_i == frame_j:
                continue  # Skip same-frame connections
            
            # Add edge (undirected, so only add i < j to avoid duplicates)
            if i < j:
                edge_sources.append(i)
                edge_targets.append(j)
                edge_weights.append(float(sim))
        
        if (i + 1) % 10000 == 0:
            print(f"    [{100*(i+1)//n:3d}%] Processed {i+1:,}/{n:,} faces, {len(edge_sources):,} edges")
    
    print(f"  [OK] Graph built: {len(edge_sources):,} edges")
    
    return edge_sources, edge_targets, edge_weights


def detect_communities_leiden(edge_sources, edge_targets, edge_weights, n_nodes):
    """Run Leiden community detection using igraph."""
    print(f"\nRunning Leiden community detection...")
    print(f"  Building igraph with {n_nodes:,} nodes, {len(edge_sources):,} edges...")
    
    # Create igraph
    g = ig.Graph()
    g.add_vertices(n_nodes)
    edges = list(zip(edge_sources, edge_targets))
    g.add_edges(edges)
    g.es['weight'] = edge_weights
    
    print(f"  Running Leiden algorithm...")
    
    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights='weight',
        n_iterations=-1,  # Run until convergence
    )
    
    communities = partition.membership
    num_communities = len(set(communities))
    
    print(f"  [OK] Found {num_communities} communities")
    print(f"  Modularity: {partition.modularity:.4f}")
    
    return communities


def detect_communities_louvain(edge_sources, edge_targets, edge_weights, n_nodes):
    """Run Louvain community detection using networkx."""
    print(f"\nRunning Louvain community detection...")
    print(f"  Building networkx graph with {n_nodes:,} nodes, {len(edge_sources):,} edges...")
    
    # Create networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    for src, tgt, weight in zip(edge_sources, edge_targets, edge_weights):
        G.add_edge(src, tgt, weight=weight)
    
    print(f"  Running Louvain algorithm...")
    
    # Run Louvain
    communities_dict = nx_community.louvain_communities(G, weight='weight', resolution=1.0)
    
    # Convert to list format
    communities = [-1] * n_nodes
    for community_id, community_set in enumerate(communities_dict):
        for node in community_set:
            communities[node] = community_id
    
    num_communities = len(communities_dict)
    
    print(f"  [OK] Found {num_communities} communities")
    
    return communities


def refine_small_clusters(
    combined_df: pd.DataFrame,
    embeddings: np.ndarray,
    min_cluster_size: int = 12,
    k_voting: int = 10,
    min_votes: int = 5,
    reassign_threshold: float = 0.58,
) -> pd.DataFrame:
    """
    Post-processing: Reassign faces from small clusters using k-NN voting.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        DataFrame with initial face_id assignments
    embeddings : np.ndarray
        Face embeddings (normalized)
    min_cluster_size : int
        Clusters with ≤ this many faces are candidates for reassignment
    k_voting : int
        Number of neighbors to check for voting
    min_votes : int
        Minimum votes needed to reassign
    reassign_threshold : float
        Minimum similarity to accept reassignment
    
    Returns
    -------
    pd.DataFrame
        DataFrame with refined face_id assignments
    """
    print(f"\nPost-processing: Refining small clusters")
    print(f"  Min cluster size: {min_cluster_size}")
    print(f"  k-voting: {k_voting}, min votes: {min_votes}")
    print(f"  Reassignment threshold: {reassign_threshold}")
    
    # Identify small clusters
    cluster_sizes = combined_df['face_id'].value_counts()
    small_clusters = cluster_sizes[cluster_sizes <= min_cluster_size].index.tolist()
    
    if not small_clusters:
        print(f"  No small clusters found (all clusters > {min_cluster_size})")
        return combined_df
    
    print(f"  Found {len(small_clusters)} small clusters")
    print(f"  Total faces in small clusters: {cluster_sizes[small_clusters].sum()}")
    
    # Build FAISS index on all embeddings
    embeddings_norm = normalize_embeddings(embeddings)
    dim = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    
    # Get indices of faces in small clusters
    small_cluster_mask = combined_df['face_id'].isin(small_clusters)
    small_cluster_indices = combined_df[small_cluster_mask].index.tolist()
    num_to_process = len(small_cluster_indices)
    
    print(f"  Processing {num_to_process:,} faces in small clusters...")
    
    # Process faces in small clusters
    reassigned_count = 0
    face_ids = combined_df['face_id'].values.copy()
    processed = 0
    
    for idx in small_cluster_indices:
        row = combined_df.iloc[idx]
        current_id = row['face_id']
        
        # Get k-NN for voting
        query = embeddings_norm[idx:idx+1]
        distances, neighbors = index.search(query, k_voting + 1)
        
        similarities = distances[0]
        neighbor_indices = neighbors[0]
        
        # Count votes from neighbors
        votes = {}
        best_similarity = {}
        
        for neighbor_idx, sim in zip(neighbor_indices, similarities):
            if neighbor_idx == idx:
                continue  # Skip self
            
            neighbor_id = face_ids[neighbor_idx]
            
            # Only vote for LARGE clusters
            if neighbor_id in small_clusters:
                continue
            
            # Record vote
            if neighbor_id not in votes:
                votes[neighbor_id] = 0
                best_similarity[neighbor_id] = 0.0
            
            votes[neighbor_id] += 1
            best_similarity[neighbor_id] = max(best_similarity[neighbor_id], sim)
        
        if not votes:
            continue  # No votes from large clusters
        
        # Find winner
        winner_id = max(votes, key=votes.get)
        winner_votes = votes[winner_id]
        winner_sim = best_similarity[winner_id]
        
        # Reassign if meets criteria
        if winner_votes >= min_votes and winner_sim >= reassign_threshold:
            face_ids[idx] = winner_id
            reassigned_count += 1
        
        processed += 1
        
        # Progress update
        if processed % 100 == 0 or processed == num_to_process:
            percent = int(100 * processed / num_to_process)
            print(f"    [{percent:3d}%] Processed {processed:,}/{num_to_process:,}, reassigned {reassigned_count} so far")
    
    # Update dataframe
    combined_df['face_id'] = face_ids
    
    # Report results
    final_cluster_sizes = pd.Series(face_ids).value_counts()
    final_small_clusters = final_cluster_sizes[final_cluster_sizes <= min_cluster_size]
    
    print(f"  [OK] Reassigned {reassigned_count} faces")
    print(f"  Small clusters remaining: {len(final_small_clusters)} (was {len(small_clusters)})")
    print(f"  Unique IDs after refinement: {len(final_cluster_sizes)}")
    
    return combined_df


def stage3_graph_clustering(
    participant_dir: str,
    similarity_threshold: float = 0.6,
    k_neighbors: int = 50,
    output_name: str = 'faces_combined.csv',
    min_confidence: float = 0.0,
    algorithm: str = 'leiden',
    enable_refinement: bool = True,
    min_cluster_size: int = 5,
    k_voting: int = 10,
    min_votes: int = 5,
    reassign_threshold: float = None,
):
    """
    Stage 3: Graph-based community detection for global face IDs.
    
    Parameters
    ----------
    participant_dir : str
        Path to participant directory
    similarity_threshold : float
        Cosine similarity threshold for edges
    k_neighbors : int
        Number of nearest neighbors to consider
    output_name : str
        Output filename
    min_confidence : float
        Minimum detection confidence
    algorithm : str
        'leiden' or 'louvain'
    enable_refinement : bool
        Whether to refine small clusters using k-NN voting (default: True)
    min_cluster_size : int
        Clusters with ≤ this many faces are candidates for reassignment (default: 5)
    k_voting : int
        Number of neighbors to check for voting in refinement (default: 10)
    min_votes : int
        Minimum votes needed to reassign a face (default: 5)
    reassign_threshold : float
        Similarity threshold for reassignment (default: similarity_threshold + 0.05)
    """
    participant_path = Path(participant_dir).resolve()
    output_csv = participant_path / output_name
    
    print("=" * 80)
    print("STAGE 3: GRAPH-BASED COMMUNITY DETECTION")
    print("=" * 80)
    print(f"Participant: {participant_path.name}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"k-neighbors: {k_neighbors}")
    print(f"Min confidence: {min_confidence}")
    print()
    
    # STEP 1: Load all data
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    combined_df, embeddings, metadata = load_all_sessions(participant_dir, min_confidence)
    
    print(f"\n[OK] Loaded {len(combined_df):,} faces from {metadata['num_sessions']} sessions")
    
    # STEP 2: Build k-NN graph
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING k-NN GRAPH")
    print("=" * 80)
    
    edge_sources, edge_targets, edge_weights = build_knn_graph(
        embeddings, combined_df, k_neighbors, similarity_threshold
    )
    
    # STEP 3: Community detection
    print("\n" + "=" * 80)
    print("STEP 3: COMMUNITY DETECTION")
    print("=" * 80)
    
    if algorithm == 'leiden' and LEIDEN_AVAILABLE:
        communities = detect_communities_leiden(
            edge_sources, edge_targets, edge_weights, len(combined_df)
        )
    elif algorithm == 'louvain' and NETWORKX_AVAILABLE:
        communities = detect_communities_louvain(
            edge_sources, edge_targets, edge_weights, len(combined_df)
        )
    elif algorithm == 'leiden' and not LEIDEN_AVAILABLE:
        print("ERROR: Leiden requested but not available")
        print("Install with: pip install python-igraph leidenalg")
        sys.exit(1)
    elif algorithm == 'louvain' and not NETWORKX_AVAILABLE:
        print("ERROR: Louvain requested but not available")
        print("Install with: pip install networkx")
        sys.exit(1)
    else:
        print(f"ERROR: Unknown algorithm: {algorithm}")
        sys.exit(1)
    
    # STEP 4: Assign global face IDs
    print("\n" + "=" * 80)
    print("STEP 4: ASSIGNING GLOBAL FACE IDs")
    print("=" * 80)
    
    num_communities = len(set(communities))
    print(f"  Assigning IDs to {num_communities} communities...")
    
    # Create 5-digit face IDs
    face_ids = [f"FACE_{community_id:05d}" for community_id in communities]
    combined_df['face_id'] = face_ids
    
    initial_num_ids = len(set(face_ids))
    
    # Post-processing: Refine small clusters
    if enable_refinement:
        print("\n" + "=" * 80)
        print("STEP 5: REFINING SMALL CLUSTERS")
        print("=" * 80)
        
        # Use provided reassign_threshold or default to similarity_threshold + 0.05
        if reassign_threshold is None:
            reassign_threshold = similarity_threshold + 0.05
        
        combined_df = refine_small_clusters(
            combined_df=combined_df,
            embeddings=embeddings,
            min_cluster_size=min_cluster_size,
            k_voting=k_voting,
            min_votes=min_votes,
            reassign_threshold=reassign_threshold,
        )
        
        final_num_ids = combined_df['face_id'].nunique()
        print(f"  IDs: {initial_num_ids} → {final_num_ids} (reduced by {initial_num_ids - final_num_ids})")
    
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
    
    # Save combined CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"  [OK] Saved: {output_csv}")
    
    # Save per-session CSVs
    for session_name in combined_df['session_name'].unique():
        session_df = combined_df[combined_df['session_name'] == session_name]
        session_dirs = [d for d in participant_path.iterdir() if d.name == session_name]
        if session_dirs:
            session_output = session_dirs[0] / "stage3_global_ids.csv"
            session_df.to_csv(session_output, index=False)
            print(f"  [OK] {session_name}: {len(session_df):,} faces")
    
    # Get final unique ID count after refinement
    final_num_unique_ids = combined_df['face_id'].nunique()
    
    # STEP 6: Generate stats
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING STATISTICS")
    print("=" * 80)
    
    stats_file = participant_path / "faces_combined.stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Global Face ID Assignment Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Algorithm: {algorithm.upper()}\n")
        f.write(f"Similarity threshold: {similarity_threshold}\n")
        f.write(f"k-neighbors: {k_neighbors}\n")
        if min_confidence > 0:
            f.write(f"Min confidence: {min_confidence}\n")
        f.write(f"\nTotal faces: {len(combined_df):,}\n")
        f.write(f"Unique global IDs: {final_num_unique_ids}\n")
        f.write(f"Sessions: {metadata['num_sessions']}\n")
        f.write(f"Graph edges: {len(edge_sources):,}\n\n")
        
        f.write("Per-session breakdown:\n\n")
        for session_name in metadata['session_names']:
            session_df = combined_df[combined_df['session_name'] == session_name]
            if len(session_df) == 0:
                continue
            unique_ids = session_df['face_id'].nunique()
            attended = int(session_df['attended'].sum()) if 'attended' in session_df.columns else 0
            attended_pct = 100 * attended / len(session_df) if len(session_df) > 0 else 0
            
            f.write(f"{session_name}:\n")
            f.write(f"  Total faces: {len(session_df):,}\n")
            f.write(f"  Unique global IDs: {unique_ids}\n")
            f.write(f"  Attended faces: {attended:,} ({attended_pct:.1f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Global ID Distribution (sorted by instance count):\n\n")
        
        id_counts = combined_df['face_id'].value_counts()
        
        for face_id in id_counts.index:
            count = id_counts[face_id]
            faces = combined_df[combined_df['face_id'] == face_id]
            sessions_present = faces['session_name'].unique()
            attended_count = int(faces['attended'].sum()) if 'attended' in faces.columns else 0
            
            f.write(f"{face_id}: {count} instances across {len(sessions_present)} session(s) ({attended_count} attended)\n")
    
    print(f"  [OK] Stats saved: {stats_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("[OK] STAGE 3 COMPLETE")
    print("=" * 80)
    print(f"Total faces: {len(combined_df):,}")
    print(f"Unique global IDs: {final_num_unique_ids}")
    print(f"Graph edges: {len(edge_sources):,}")
    print(f"Output: {output_csv}")
    print("=" * 80)
    
    return {
        'total_faces': len(combined_df),
        'unique_global_ids': final_num_unique_ids,
        'num_edges': len(edge_sources),
        'output_csv': str(output_csv),
        'stats_file': str(stats_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Graph-based community detection for global face IDs"
    )
    
    parser.add_argument('participant_dir', help='Path to participant directory')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, 
                        help='Similarity threshold (default: 0.6)')
    parser.add_argument('-k', '--k-neighbors', type=int, default=50,
                        help='Number of nearest neighbors (default: 50)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence (default: 0.0)')
    parser.add_argument('-a', '--algorithm', choices=['leiden', 'louvain'], default='leiden',
                        help='Community detection algorithm (default: leiden)')
    parser.add_argument('-o', '--output', default='faces_combined.csv',
                        help='Output filename (default: faces_combined.csv)')
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable small cluster refinement')
    parser.add_argument('--min-cluster-size', type=int, default=5,
                        help='Clusters with ≤ N faces are refined (default: 5)')
    parser.add_argument('--k-voting', type=int, default=10,
                        help='Neighbors to check for voting in refinement (default: 10)')
    parser.add_argument('--min-votes', type=int, default=5,
                        help='Minimum votes needed to reassign a face (default: 5)')
    parser.add_argument('--reassign-threshold', type=float, default=None,
                        help='Similarity threshold for reassignment (default: threshold + 0.05)')
    
    args = parser.parse_args()
    
    try:
        stage3_graph_clustering(
            participant_dir=args.participant_dir,
            similarity_threshold=args.threshold,
            k_neighbors=args.k_neighbors,
            output_name=args.output,
            min_confidence=args.min_confidence,
            algorithm=args.algorithm,
            enable_refinement=not args.no_refine,
            min_cluster_size=args.min_cluster_size,
            k_voting=args.k_voting,
            min_votes=args.min_votes,
            reassign_threshold=args.reassign_threshold,
        )
    except Exception as e:
        print(f"\n[ERROR] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

