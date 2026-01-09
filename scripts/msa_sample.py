import os
import sys
import glob
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import Sequence, Iterable, Optional, Dict, Tuple, List


# ==========================================
# SECTION 1: Utility Functions (Shared)
# ==========================================

def read_a3m_sequences(a3m_path: str) -> Dict[str, str]:
    """
    Reads sequences from an A3M/FASTA file, keeping only uppercase residues and gaps (-).
    Returns: Dictionary {seq_id: sequence_string}
    """
    sequences = {}
    current_id = None
    current_sequence = ""

    try:
        with open(a3m_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        # Process previous sequence
                        processed_seq = ''.join([r for r in current_sequence if r.isupper() or r == "-"])
                        if processed_seq:
                            sequences[current_id] = processed_seq
                    # Parse new ID (remove '>' and description)
                    current_id = line[1:].strip().split(maxsplit=1)[0]
                    current_sequence = ""
                else:
                    current_sequence += line

            # Process the last sequence
            if current_id:
                processed_seq = ''.join([r for r in current_sequence if r.isupper() or r == "-"])
                if processed_seq:
                    sequences[current_id] = processed_seq
    except Exception as e:
        print(f"[Error] Reading {a3m_path}: {e}", file=sys.stderr)
        return {}
    return sequences


def get_reference_query(protein_folder_path: str) -> Optional[Tuple[str, str]]:
    """
    Determines the reference query sequence from the 'msas' folder.
    Priority: bfd > uniref30 > uniref90 > any .a3m > any .sto
    """
    msas_dir = os.path.join(protein_folder_path, "msas")
    if not os.path.exists(msas_dir):
        return None

    potential_files = ["bfd_hits.a3m", "uniref30_hits.a3m", "uniref90_hits.a3m"]
    target_path = None

    # Check specific files
    for fname in potential_files:
        p = os.path.join(msas_dir, fname)
        if os.path.exists(p):
            target_path = p
            break

    # Fallback to any a3m
    if not target_path:
        all_a3m = glob.glob(os.path.join(msas_dir, "*.a3m"))
        if all_a3m:
            target_path = all_a3m[0]

    if not target_path:
        return None

    # Get first sequence
    seqs = read_a3m_sequences(target_path)
    if seqs:
        return list(seqs.items())[0]
    return None


# ==========================================
# SECTION 2: Logic from Script 1 (Masking)
# ==========================================

def process_npz(npz_path: str, threshold: float):
    """Loads NPZ, binarizes deviation matrix, calculates column sums."""
    try:
        data = np.load(npz_path)
        if 'deviation' not in data:
            return None, None
        matrix = data['deviation']
        if matrix.ndim == 1:
            matrix = np.expand_dims(matrix, axis=1)
        matrix_binary = (matrix > threshold).astype(int)
        prob_sum = np.sum(matrix_binary, axis=1)
        return matrix_binary, prob_sum
    except Exception as e:
        print(f"[Error] Processing NPZ {npz_path}: {e}", file=sys.stderr)
        return None, None


def get_merged_fragments(prob_sum, window_size: int, top_percent: float):
    """Identifies flexible fragments using sliding window."""
    L = len(prob_sum)
    if L < window_size:
        return []

    # Sliding window
    window_sums = [(i, np.sum(prob_sum[i:i + window_size])) for i in range(L - window_size + 1)]
    window_sums.sort(key=lambda x: x[1], reverse=True)

    num_top = max(1, int(len(window_sums) * top_percent))
    top_windows = window_sums[:num_top]

    # Merge windows
    if not top_windows:
        return []

    top_windows.sort(key=lambda x: x[0])
    merged = []

    curr_start, curr_end = top_windows[0][0], top_windows[0][0] + window_size - 1

    for i in range(1, len(top_windows)):
        start, _ = top_windows[i]
        end = start + window_size - 1
        if start <= curr_end + 1:
            curr_end = max(curr_end, end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = start, end
    merged.append((curr_start, curr_end))
    return merged


def firetrain_mask_generator(start: int, end: int, step_size: int):
    """Generates mask indices for 'firetrain' strategy."""
    masks = []
    left, right = start, end

    # While fragment is long enough to split
    while (left + 2) < (right - 2):
        l_mask = list(range(left, left + 3))
        r_mask = list(range(right - 2, right + 1))
        masks.append(l_mask + r_mask)
        left += step_size
        right -= step_size

    if right - left + 1 >= 3:
        masks.append(list(range(left, right + 1)))
    elif fragment_length := (end - start + 1) < 6:
        # Small fragment fallback
        masks.append(list(range(start, end + 1)))

    return masks


def step_masking(protein_id: str, base_dir: str, args):
    """
    Executes the masking step.
    Reads flex.npz -> Detects Fragments -> Masks 'msas_cluster_search' -> Saves to 'msas_cluster_masked'.
    """
    protein_dir = os.path.join(base_dir, protein_id)
    npz_path = os.path.join(protein_dir, args.npz_name)
    input_msa_dir = os.path.join(protein_dir, "msas_cluster_search")
    output_msa_dir = os.path.join(protein_dir, "msas_cluster_masked")

    if not os.path.exists(npz_path) or not os.path.exists(input_msa_dir):
        return  # Skip silently if inputs missing

    # 1. Analyze NPZ
    _, prob_sum = process_npz(npz_path, args.threshold)
    if prob_sum is None: return

    fragments = get_merged_fragments(prob_sum, args.window_size, args.top_percent)
    if not fragments: return

    # 2. Process MSAs
    os.makedirs(output_msa_dir, exist_ok=True)
    msa_files = [f for f in os.listdir(input_msa_dir) if f.endswith(".a3m")]

    for msa_file in msa_files:
        full_msa_path = os.path.join(input_msa_dir, msa_file)
        seqs_dict = read_a3m_sequences(full_msa_path)
        if not seqs_dict: continue

        base_name = os.path.splitext(msa_file)[0]

        for frag_idx, (start, end) in enumerate(fragments):
            mask_sets = firetrain_mask_generator(start, end, args.step_size)

            for mask_idx, mask_pos in enumerate(mask_sets):
                # Apply mask
                masked_seqs = {}
                for i, (sid, seq) in enumerate(seqs_dict.items()):
                    if i == 0:  # Query is never masked
                        masked_seqs[sid] = seq
                        continue

                    seq_list = list(seq)
                    for p in mask_pos:
                        if 0 <= p < len(seq_list) and seq_list[p] != "-":
                            seq_list[p] = "X"
                    masked_seqs[sid] = "".join(seq_list)

                # Save
                out_name = f"{base_name}_{frag_idx:02d}_{mask_idx:02d}.a3m"
                out_path = os.path.join(output_msa_dir, out_name)
                with open(out_path, 'w') as f:
                    for sid, s in masked_seqs.items():
                        f.write(f">{sid}\n{s}\n")


# ==========================================
# SECTION 3: Logic from Script 2 (Sampling)
# ==========================================

DEPTH_SPLIT_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def consolidate_files(protein_dir: str, target_folder_name: str, protein_id: str) -> int:
    """Consolidates MSAs from source folders into sub_msas."""
    sources = ["msas", "msas_cluster", "msas_cluster_search", "msas_cluster_masked"]
    target_dir = os.path.join(protein_dir, target_folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # Check if already populated (optimization)
    existing = glob.glob(os.path.join(target_dir, "*.a3m"))
    if len(existing) > 100:
        # Assuming existing index is roughly count
        return len(existing) + 1

    idx = 1
    for src in sources:
        src_dir = os.path.join(protein_dir, src)
        if os.path.exists(src_dir):
            files = glob.glob(os.path.join(src_dir, "*.a3m"))
            for fp in files:
                dest = os.path.join(target_dir, f"{protein_id}_{idx:04d}.a3m")
                try:
                    shutil.copy2(fp, dest)
                    idx += 1
                except:
                    pass
    return idx


def step_sampling(protein_id: str, base_dir: str, args):
    """
    Executes the sampling step.
    Consolidates MSAs -> Randomly samples from 'msas_cluster_masked' until target count.
    """
    protein_dir = os.path.join(base_dir, protein_id)
    target_dir_name = "sub_msas"
    sampling_source_dir = os.path.join(protein_dir, "msas_cluster_masked")
    output_dir = os.path.join(protein_dir, target_dir_name)

    # 1. Consolidate existing
    current_count = consolidate_files(protein_dir, target_dir_name, protein_id) - 1

    if current_count >= args.target_total_count:
        return  # Target reached

    # 2. Prepare for random sampling
    if not os.path.exists(sampling_source_dir):
        return

    candidates = glob.glob(os.path.join(sampling_source_dir, "*.a3m"))
    if not candidates:
        return

    fixed_query = get_reference_query(protein_dir)
    if not fixed_query:
        print(f"[Warn] No reference query for {protein_id}, skipping sampling.", file=sys.stderr)
        return

    q_id, q_seq = fixed_query

    # 3. Random Sample Loop
    # We loop until target count is met or we run out of iterations (safety break)
    safety_iter = 0
    max_iter = (args.target_total_count - current_count) * 10

    while current_count < args.target_total_count and safety_iter < max_iter:
        safety_iter += 1
        src_path = random.choice(candidates)

        try:
            seqs = read_a3m_sequences(src_path)
            # Remove query from pool to sample
            non_query = [(k, v) for k, v in seqs.items() if k != q_id]
            total_avail = len(non_query)

            # Try depths
            for depth in DEPTH_SPLIT_SIZES:
                if current_count >= args.target_total_count: break

                sample_size = min(depth - 1, total_avail)
                if sample_size <= 0: continue

                sampled = random.sample(non_query, sample_size)

                # Write file
                idx_label = f"{current_count + 1}"
                out_path = os.path.join(output_dir, f"{protein_id}_{idx_label}.a3m")

                with open(out_path, 'w') as f:
                    f.write(f">{q_id}\n{q_seq}\n")
                    for s_id, s_seq in sampled:
                        f.write(f">{s_id}\n{s_seq}\n")

                current_count += 1
        except:
            continue


# ==========================================
# SECTION 4: Logic from Script 3 (Checking/Fixing)
# ==========================================

def step_fix_query(protein_id: str, base_dir: str):
    """
    Executes the fix step.
    Ensures all files in 'sub_msas' start with the correct reference query.
    """
    protein_dir = os.path.join(base_dir, protein_id)
    sub_msas_dir = os.path.join(protein_dir, "sub_msas")

    if not os.path.exists(sub_msas_dir): return

    ref_query = get_reference_query(protein_dir)
    if not ref_query:
        print(f"[Error] No reference query found for {protein_id}. Cannot fix sub_msas.", file=sys.stderr)
        return

    ref_id, ref_seq = ref_query
    ref_block = f">{ref_id}\n{ref_seq}\n"

    files = glob.glob(os.path.join(sub_msas_dir, "*.a3m"))

    for fp in files:
        try:
            # Check first sequence
            first_seq = None
            sequences = read_a3m_sequences(fp)
            if sequences:
                first_seq = list(sequences.items())[0]

            if first_seq == ref_query:
                continue  # Match, no fix needed

            # Mismatch found, rewrite file
            with open(fp, 'r') as f:
                lines = f.readlines()

            # Find boundaries of the first sequence in the raw file
            start_idx = -1
            next_start_idx = len(lines)

            for i, line in enumerate(lines):
                if line.strip().startswith(">"):
                    if start_idx == -1:
                        start_idx = i
                    else:
                        next_start_idx = i
                        break

            if start_idx == -1: continue  # Empty or invalid file

            # Reconstruct content
            new_content = [ref_block] + lines[next_start_idx:]

            with open(fp, 'w') as f:
                f.writelines(new_content)

        except Exception as e:
            print(f"[Error] Fixing {os.path.basename(fp)}: {e}", file=sys.stderr)


# ==========================================
# SECTION 5: Cleanup Logic (New)
# ==========================================

def step_cleanup(protein_id: str, base_dir: str):
    """
    Deletes the intermediate directories:
    - msas_cluster
    - msas_cluster_masked
    - msas_cluster_search
    to save disk space after processing.
    """
    protein_dir = os.path.join(base_dir, protein_id)
    folders_to_remove = ["msas_cluster", "msas_cluster_masked", "msas_cluster_search"]

    for folder in folders_to_remove:
        folder_path = os.path.join(protein_dir, folder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"[Warn] Failed to delete {folder_path}: {e}", file=sys.stderr)


# ==========================================
# MAIN WORKFLOW
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Integrated Sasmple: Masking -> Sampling -> Fixing -> Cleanup")

    # Common Args
    parser.add_argument("--base_dir", required=True, help="Base directory containing protein folders")
    parser.add_argument("--protein_list_file", required=True, help="TXT file with protein IDs")

    # Masking Args
    parser.add_argument("--npz_name", default="flex.npz", help="Name of NPZ file (default: flex.npz)")
    parser.add_argument("--threshold", type=float, default=0.3, help="NPZ threshold (default: 0.3)")
    parser.add_argument("--window_size", type=int, default=3, help="Sliding window size (default: 3)")
    parser.add_argument("--top_percent", type=float, default=0.2, help="Top fragment percent (default: 0.2)")
    parser.add_argument("--step_size", type=int, default=2, help="Firetrain step size (default: 2)")

    # Sampling Args
    parser.add_argument("--target_total_count", type=int, default=1500, help="Target MSA count (default: 1500)")

    args = parser.parse_args()

    # Load Protein List
    if not os.path.exists(args.protein_list_file):
        print(f"Error: List file {args.protein_list_file} not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.protein_list_file, 'r') as f:
        proteins = [line.strip() for line in f if line.strip()]

    print(f"Starting Integrated Sample for {len(proteins)} proteins...")

    # Main Loop
    for protein_id in tqdm(proteins, desc="Processing Sample"):
        protein_path = os.path.join(args.base_dir, protein_id)
        if not os.path.isdir(protein_path):
            continue

        # Step 1: Masking
        step_masking(protein_id, args.base_dir, args)

        # Step 2: Sampling (Consolidates files including masked ones)
        step_sampling(protein_id, args.base_dir, args)

        # Step 3: Check & Fix
        step_fix_query(protein_id, args.base_dir)

        # Step 4: Cleanup (Delete intermediate folders)
        step_cleanup(protein_id, args.base_dir)

    print("Ssample completed successfully.")


if __name__ == "__main__":
    main()