import argparse
import multiprocessing
import os
import numpy as np
import subprocess
import shutil
import sys
from typing import List, Optional, Iterable

# --- Helper Functions for Stockholm to A3M Conversion ---

def _convert_sto_seq_to_a3m(query_non_gaps: List[bool], sto_seq: str) -> str:
    """Converts a single Stockholm sequence to A3M format based on query gaps."""
    return ''.join(sequence_res.lower() if not is_query_res_non_gap and sequence_res != '-'
                   else sequence_res
                   for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq)
                   if is_query_res_non_gap or sequence_res != '-')

def convert_stockholm_to_a3m(stockholm_format: str,
                             max_sequences: Optional[int] = None,
                             remove_first_row_gaps: bool = True) -> str:
    """Converts MSA in Stockholm format to the A3M format."""
    descriptions = {}
    sequences = {}
    seq_lines = []
    desc_lines = []

    for line in stockholm_format.splitlines():
        if line.strip() and not line.startswith(('//')):
            if line.startswith('#=GS'):
                desc_lines.append(line)
            elif not line.startswith(('#')):
                seq_lines.append(line)

    for line in seq_lines:
        seqname, aligned_seq = line.split(maxsplit=1)
        if seqname not in sequences:
            if max_sequences is not None and len(sequences) >= max_sequences:
                continue
            sequences[seqname] = ''
        sequences[seqname] += aligned_seq.strip()

    for line in desc_lines:
        columns = line.split(maxsplit=3)
        seqname, feature = columns[1:3]
        value = columns[3] if len(columns) == 4 else ''
        if feature == 'DE':
            if max_sequences is not None and seqname not in sequences:
                continue
            descriptions[seqname] = value.strip()

    query_non_gaps = []
    if remove_first_row_gaps and sequences:
        first_seqname = next(iter(sequences))
        query_sequence_sto = sequences[first_seqname]
        query_non_gaps = [res != '-' for res in query_sequence_sto]

    a3m_parts = []
    for seqname, sto_sequence in sequences.items():
        out_sequence = sto_sequence.replace('.', '')
        if remove_first_row_gaps:
            out_sequence_processed = _convert_sto_seq_to_a3m(query_non_gaps, out_sequence)
        else:
            out_sequence_processed = out_sequence
        a3m_parts.append(f">{seqname} {descriptions.get(seqname, '')}\n{out_sequence_processed}")

    return '\n'.join(a3m_parts) + '\n'

# --- MSA Merging and Processing Functions ---

def merge_a3m_files(output_merged_a3m_path: str, *a3m_paths: str) -> bool:
    """
    Merges multiple A3M files by simple concatenation.
    *a3m_paths should be the list of paths to the A3M files to merge.
    """
    try:
        with open(output_merged_a3m_path, 'w') as outfile:
            for f_path in a3m_paths:
                if f_path and os.path.exists(f_path): # Check for None and existence
                    with open(f_path, 'r') as infile:
                        outfile.write(infile.read())
                    outfile.write('\n')
                elif f_path: # Only print warning if path was provided but didn't exist
                    print(f"Warning: Input file not found for merging: {f_path}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Error merging files to {output_merged_a3m_path}: {e}", file=sys.stderr)
        return False

def process_uniref90_sto_to_a3m(sto_file_path: str, output_a3m_path: str) -> bool:
    """Reads a Stockholm file, converts it to A3M, and saves it."""
    try:
        with open(sto_file_path, 'r') as f:
            sto_str = f.read()
        a3m_str = convert_stockholm_to_a3m(sto_str)

        os.makedirs(os.path.dirname(output_a3m_path), exist_ok=True) # Ensure output dir exists
        with open(output_a3m_path, 'w') as f:
            f.write(a3m_str)
        return True
    except Exception as e:
        print(f"Error converting {sto_file_path} to A3M: {e}", file=sys.stderr)
        return False

def deal(combined_a3m_path: str, protein_embedding_output_dir: str, pname: str) -> None:
    """
    Processes the combined A3M file by running external MSA processing scripts.
    protein_embedding_output_dir is the target /msas_embeddings/ subfolder for outputs.
    """
    # Outputs are now expected to be in protein_embedding_output_dir
    a3m_random_path = os.path.join(protein_embedding_output_dir, pname + "_random.a3m")
    a3m_process_path = os.path.join(protein_embedding_output_dir, pname + "_process.a3m") # Temporary output of process_msa.py
    a3m_msatransformer_npz_path = os.path.join(protein_embedding_output_dir, pname + "_random.npz") # Final output

    print(f"Processing {pname} embeddings...")

    # Step 1: Generate _random.a3m from the combined A3M
    if not os.path.exists(a3m_random_path):
        command1 = [
            "python", "./movement_residue/MSA_embedding/process_msa.py",
            combined_a3m_path,
            a3m_process_path,
            a3m_random_path
        ]
        try:
            subprocess.run(command1, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running process_msa.py for {pname}: {e.stderr}", file=sys.stderr)
            with open("./movement_residue/MSA_embedding/msa_trans_error.txt", "a") as f:
                f.write(f"{pname} (process_msa.py error)\n")
            return
    else:
        print(f"Skipping process_msa.py for {pname}, output {a3m_random_path} already exists.")

    # Step 2: Generate _random.npz from _random.a3m
    if os.path.exists(a3m_random_path) and not os.path.exists(a3m_msatransformer_npz_path):
        command2 = [
            "python", "./movement_residue/MSA_embedding/MSA_transformer_process.py",
            a3m_random_path,
            a3m_msatransformer_npz_path
        ]
        try:
            subprocess.run(command2, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running MSA_transformer_process.py for {pname}: {e.stderr}", file=sys.stderr)
            with open("./movement_residue/MSA_embedding/error.txt", "a") as f:
                f.write(f"{pname} (MSA_transformer_process.py error)\n")
            return
    else:
        print(f"Skipping MSA_transformer_process.py for {pname}.")
        if not os.path.exists(a3m_random_path):
            print(f"  Reason: Input A3M file {a3m_random_path} does not exist.", file=sys.stderr)

    if os.path.exists(a3m_msatransformer_npz_path):
        if os.path.exists(combined_a3m_path):
            try:
                os.remove(combined_a3m_path)
            except OSError as e:
                print(f"Error removing temporary combined MSA {combined_a3m_path}: {e}", file=sys.stderr)
    else:
        with open("./movement_residue/MSA_embedding/error.txt", "a") as f:
            f.write(f"{pname} (final npz missing)\n")
        print(f"{pname} processing FAILED: Final NPZ missing.", file=sys.stderr)

parser = argparse.ArgumentParser(description="Process MSA and generate embeddings.")
parser.add_argument("--input_base_dir", required=True, help="Input directory containing subfolders with MSAs.")
parser.add_argument("--output_base_dir", required=True, help="Output directory to store the MSA embeddings.")
parser.add_argument("--num_threads", type=int, default=None,
                    help="Number of parallel threads to use. Defaults to number of CPU cores.")
args = parser.parse_args()

input_base_dir = args.input_base_dir
output_base_dir = args.output_base_dir


if __name__ == '__main__':
    num_threads = args.num_threads or multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_threads)

    for root, dirs, files in os.walk(input_base_dir):
        relative_path = os.path.relpath(root, input_base_dir)

        if relative_path == "." or os.sep in relative_path:
            continue

        pname = os.path.basename(root)
        protein_msa_input_dir = os.path.join(root, "msas")
        uniref90_sto_path = os.path.join(protein_msa_input_dir, "uniref90_hits.sto")
        uniref90_a3m_path = os.path.join(protein_msa_input_dir, "uniref90_hits.a3m")
        uniref30_a3m_path = os.path.join(protein_msa_input_dir, "uniref30_hits.a3m")

        protein_output_dir = os.path.join(output_base_dir, pname, "msas_embeddings")
        os.makedirs(protein_output_dir, exist_ok=True)
        final_npz_output_path = os.path.join(protein_output_dir, pname + "_random.npz")

        if os.path.exists(final_npz_output_path):
            print(f"Final output already exists for {pname}, skipping.")
            continue

        # --- uniref90.sto → uniref90.a3m ---
        if os.path.isfile(uniref90_sto_path):
            if not os.path.exists(uniref90_a3m_path):
                if not process_uniref90_sto_to_a3m(uniref90_sto_path, uniref90_a3m_path):
                    print(f"Skipping {pname} due to UniRef90 .sto to .a3m conversion error.", file=sys.stderr)
            else:
                print(f"UniRef90.a3m already exists for {pname}, skipping conversion.")
        else:
            print(f"Warning: UniRef90.sto not found for {pname}. Skipping conversion.", file=sys.stderr)

        if not os.path.exists(uniref30_a3m_path):
            print(f"Missing UniRef30 A3M for {pname}, skipping.", file=sys.stderr)
            continue

        pool.apply_async(deal, (uniref30_a3m_path, protein_output_dir, pname))

    pool.close()
    pool.join()
    print("All MSA embeddings processing complete!")

