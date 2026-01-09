import os
import re
import shutil
import argparse
import subprocess
import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm # For progress bar

# --- Constants & Utility Functions (from original scripts) ---

AA_DICT = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def three_to_one(residue_name):
    return AA_DICT.get(residue_name, "X")

def extract_sequences_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    sequences = []
    for model in structure:
        for chain in model:
            seq = ''.join(three_to_one(res.resname) for res in chain if is_aa(res, standard=True))
            if seq:
                sequences.append((chain.id, seq))
    return sequences

def extract_residues_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    return [res for model in structure for chain in model for res in chain if is_aa(res, standard=True)]

def get_ca_coordinate(residue):
    if "CA" in residue:
        return residue["CA"].get_coord()
    elif "C" in residue:
        return residue["C"].get_coord()
    return None


# --- Stage 1: copy_pdb.py functionality ---
def find_and_copy_pdb(pdb_id, root, template_dir, output_dir, base_input_dir):
    pdb_file_path = os.path.join(template_dir, f'{pdb_id}.pdb')
    if os.path.exists(pdb_file_path):
        relative_path = os.path.relpath(root, base_input_dir)
        output_pdb_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_pdb_dir, exist_ok=True)
        shutil.copy(pdb_file_path, os.path.join(output_pdb_dir, f'{pdb_id}.pdb'))
        return None # Return None for success
    else:
        return f'Not found: {pdb_id}.pdb in template library for {os.path.basename(root)}.'


def stage1_process_m8_file(m8_file_path, root, template_dir, output_dir, base_input_dir):
    results = []
    try:
        with open(m8_file_path, 'r') as m8_file:
            lines = [line.strip() for line in m8_file if line.strip()]
            if not lines:
                return [f'Empty m8 file: {m8_file_path}']

            target_id = lines[0].split()[0]
            query_ids = set()

            for line in lines:
                parts = line.split()
                if len(parts) > 1:
                    query_id = parts[1]
                    if query_id != target_id:
                        query_ids.add(query_id)

            if not query_ids:
                return [f'No valid queries (templates) found in {m8_file_path} for target {target_id}.']

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(find_and_copy_pdb, qid, root, template_dir, output_dir, base_input_dir) for qid in query_ids]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
    except Exception as e:
        results.append(f"Error processing {m8_file_path}: {e}")
    return results

def run_stage1(input_dir, template_dir, output_dir, global_pbar=None):
    all_results = []
    m8_files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.m8'):
                m8_files_to_process.append((os.path.join(root, file), root))

    if global_pbar:
        global_pbar.set_description("Stage 1: Copying PDBs")
        global_pbar.refresh() # Ensures description updates

    for m8_path, root in m8_files_to_process:
        all_results.extend(stage1_process_m8_file(m8_path, root, template_dir, output_dir, input_dir))
        if global_pbar:
            global_pbar.update(1) # Increment for each m8 file processed


# --- Stage 2: structure_profile2.py functionality (TM-align) ---
def stage2_process_subfolder(query_name, args, tmalign_binary_path, align_dir_stage2, target_dir_stage1):
    query_pdb_path = os.path.join(args.query_pdb_dir, query_name, f"{query_name}.pdb")
    if not os.path.exists(query_pdb_path):
        print(f"Skipping {query_name}: No matching PDB file found in {args.query_pdb_dir}.")
        return

    target_subdir = os.path.join(target_dir_stage1, query_name)
    if not os.path.exists(target_subdir):
        print(
            f"Skipping {query_name}: No templates found or copied in stage 1 for this query. Target subfolder '{target_subdir}' not found.")
        return

    output_subdir_stage2 = os.path.join(align_dir_stage2, query_name)
    os.makedirs(output_subdir_stage2, exist_ok=True)

    output_txt_file = os.path.join(output_subdir_stage2, f"{query_name}.txt")
    with open(output_txt_file, "w") as f:
        f.write(f"{query_name}\n")

    query_sequences = extract_sequences_from_pdb(query_pdb_path)
    query_sequence = query_sequences[0][1] if query_sequences else ''
    query_sequence_written = False
    query_residues = extract_residues_from_pdb(query_pdb_path)

    for target_pdb in os.listdir(target_subdir):
        if not target_pdb.endswith(".pdb"):
            continue
        target_pdb_path = os.path.join(target_subdir, target_pdb)
        target_pdb_id = target_pdb[:-4]

        cmd = [tmalign_binary_path, query_pdb_path, target_pdb_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.split("\n")

        except subprocess.CalledProcessError as e:
            print(f"Error running TMalign for {query_name} vs {target_pdb_id}: {e.stderr}")
            continue
        except FileNotFoundError:
            print(f"Error: TMalign binary not found at {tmalign_binary_path}. Please check --tmalign_path.")
            return
        except Exception as e:
            print(f"An unexpected error occurred during TMalign for {query_name} vs {target_pdb_id}: {e}")
            continue

        sequences, alignment_line = [], ""
        aligned_target_seq = ""
        aligned_template_seq = ""

        found_alignment_block = False
        for i in range(len(lines) - 2):
            line1 = re.sub(r'[\t\n\r\v\f]', '', lines[i])
            line2 = re.sub(r'[\t\n\r\v\f]', '', lines[i + 1])
            line3 = re.sub(r'[\t\n\r\v\f]', '', lines[i + 2])

            if (line1 and line2 and line3 and
                    len(set(line2) - {':', '.', ' '}) == 0 and
                    len(line1) == len(line2) and len(line2) == len(line3)):

                if any(c.isalpha() for c in line1) and any(c.isalpha() for c in line3):
                    aligned_target_seq = line1
                    alignment_line = line2
                    aligned_template_seq = line3
                    sequences = [aligned_target_seq, aligned_template_seq]
                    found_alignment_block = True
                    break

        if not found_alignment_block:
            print(f"DEBUG: Failed to parse TMalign output for {query_name} vs {target_pdb_id}. "
                  f"Could not find a valid alignment block. Full output (truncated to first 20 lines):\n"
                  + '\n'.join(lines[:20]) + "\n--- End Truncated Output ---")
            continue

        aligned_target, aligned_template = sequences
        matching_residues = []
        target_counter = -1
        template_counter = -1

        for i in range(len(alignment_line)):
            t_res, tmpl_res, sym = aligned_target[i], aligned_template[i], alignment_line[i]
            if t_res != '-': target_counter += 1
            if tmpl_res != '-': template_counter += 1

            if sym in {':', '.'} and t_res != '-' and tmpl_res != '-':
                matching_residues.append({
                    'target_idx': target_counter,
                    'template_idx': template_counter,
                    'query_residue_char': t_res,
                    'template_residue_char': tmpl_res,
                    'align_symbol': sym
                })

        target_structure_residues = extract_residues_from_pdb(target_pdb_path)
        L = len(query_sequence)
        result_tensor = np.full((L, 1), -1.0, dtype=float)

        for match in matching_residues:
            q_idx = match['target_idx']
            t_idx = match['template_idx']

            if q_idx < len(query_residues) and t_idx < len(target_structure_residues):
                coord1 = get_ca_coordinate(query_residues[q_idx])
                coord2 = get_ca_coordinate(target_structure_residues[t_idx])
                distance = np.linalg.norm(coord1 - coord2) if coord1 is not None and coord2 is not None else -1.0
                result_tensor[q_idx, 0] = distance
            else:
                pass  # Removed detailed printing

        with open(output_txt_file, "a") as f:
            if not query_sequence_written:
                f.write(f"{query_sequence}\n")
                query_sequence_written = True
            f.write(f"{target_pdb_id}\n")
            for match in matching_residues:
                f.write(f"Query {match['query_residue_char']} (Pos {match['target_idx'] + 1}) <-> "
                        f"Template {match['template_residue_char']} (Pos {match['template_idx'] + 1}), "
                        f"Symbol: {match['align_symbol']}\n")

        np.savez(os.path.join(output_subdir_stage2, f"{target_pdb_id}.npz"), data=result_tensor)

def run_stage2(args, query_dir_stage2, target_dir_stage1, align_dir_stage2, global_pbar=None):
    os.makedirs(align_dir_stage2, exist_ok=True)

    tmalign_binary_path = os.path.join(args.tmalign_path, "TMalign")
    if not os.path.exists(tmalign_binary_path):
        raise FileNotFoundError(f"TMalign binary not found at: {tmalign_binary_path}. Please check --tmalign_path.")

    all_subfolders = [d for d in os.listdir(query_dir_stage2) if os.path.isdir(os.path.join(query_dir_stage2, d))]
    subfolders_to_process = []
    if args.filter_list and os.path.isfile(args.filter_list):
        with open(args.filter_list, 'r') as f:
            filtered = set(line.strip() for line in f if line.strip())
        subfolders_to_process = [d for d in all_subfolders if d in filtered]
    else:
        subfolders_to_process = all_subfolders

    if global_pbar:
        global_pbar.set_description("Stage 2: Running TMalign")
        global_pbar.refresh()


    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(stage2_process_subfolder, query_name, args, tmalign_binary_path, align_dir_stage2, target_dir_stage1)
                   for query_name in subfolders_to_process]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a query subfolder in Stage 2: {e}")
            if global_pbar:
                global_pbar.update(1)


# --- Stage 3: structure_profile3.py functionality ---
def run_stage3(input_dir_stage3, output_dir_final, temp_dirs_to_remove, global_pbar=None):
    os.makedirs(output_dir_final, exist_ok=True)

    subfolders_to_process_npz = []
    for root, _, files in os.walk(input_dir_stage3):
        if root == input_dir_stage3:
            continue
        npz_files_in_subfolder = [f for f in files if f.endswith(".npz")]
        if npz_files_in_subfolder:
            subfolders_to_process_npz.append(root)

    if global_pbar:
        global_pbar.set_description("Stage 3: Aggregating NPZ")
        global_pbar.refresh()

    for root in subfolders_to_process_npz:
        subfolder_name = os.path.basename(root)
        all_arrays = []
        npz_files_in_subfolder = [f for f in os.listdir(root) if f.endswith(".npz")]

        for file in npz_files_in_subfolder:
            file_path = os.path.join(root, file)
            try:
                data = np.load(file_path)
                for array_name in data.files:
                    array = data[array_name]
                    if array.ndim == 2 and array.shape[1] == 1:
                        all_arrays.append(array)
                    else:
                        print(f"Warning: Unexpected array shape in {file_path}. Expected (N, 1), got {array.shape}.")
                data.close()
            except Exception as e:
                print(f"Error loading or processing NPZ file {file_path}: {e}")
                continue

        if not all_arrays:
            print(f"No valid arrays found after processing NPZ files in {root}. Skipping aggregation for this subfolder.")
            if global_pbar:
                global_pbar.update(1)
            continue

        try:
            stacked_arrays = np.hstack(all_arrays)
        except ValueError as e:
            print(f"Error stacking arrays for {subfolder_name}: {e}. Skipping aggregation for this subfolder.")
            if global_pbar:
                global_pbar.update(1)
            continue

        result_array = np.full((stacked_arrays.shape[0], 1), -1.0, dtype=float)

        for i in range(stacked_arrays.shape[0]):
            row = stacked_arrays[i, :]
            valid_values = row[row > 2]
            if valid_values.size > 0:
                result_array[i, 0] = valid_values.mean()
            else:
                result_array[i, 0] = -1.0

        output_subfolder = os.path.join(output_dir_final, subfolder_name)
        os.makedirs(output_subfolder, exist_ok=True)

        output_path = os.path.join(output_subfolder, "structure_profile.npz")
        np.savez(output_path, result=result_array)

        if global_pbar:
            global_pbar.update(1) # Increment for each subfolder's NPZ aggregation

    for remp in temp_dirs_to_remove:
        if os.path.exists(remp):
            shutil.rmtree(remp)
        else:
            pass


# --- Main script execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a multi-stage structure profile generation pipeline.")

    # Stage 1 arguments
    parser.add_argument('--input_m8_dir', required=True,
                        help='Input directory containing .m8 files (for Stage 1). This will be structure_profile_temp1.')
    parser.add_argument('--template_pdb_dir', required=True,
                        help='Directory containing all PDB templates (for Stage 1).')
    parser.add_argument('--stage1_output_dir', default=None,
                        help='Output directory for PDB templates copied in Stage 1. Defaults to temp2. (structure_profile_temp2)')

    # Stage 2 arguments
    parser.add_argument("--tmalign_path", required=True,
                        help="Path to TM-align binary directory.")
    parser.add_argument("--query_pdb_dir", required=True,
                        help="Directory containing query subfolders with PDBs (for Stage 2).")
    parser.add_argument("--stage2_output_dir", default=None,
                        help="Output directory for TM-align results and .npz files. Defaults to temp3. (structure_profile_temp3)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Maximum number of threads for parallel processing in Stage 2 (and Stage 1).")
    parser.add_argument("--filter_list", default=None,
                        help="Optional: file listing query names to process in Stage 2.")

    # Stage 3 arguments
    parser.add_argument('--final_output_dir', required=True,
                        help="Final output directory for structure profiles. (out)")

    args = parser.parse_args()

    # Define temporary directories based on final output directory
    temp1_dir = args.input_m8_dir
    temp2_dir = args.stage1_output_dir if args.stage1_output_dir else os.path.join(args.final_output_dir, "structure_profile_temp2")
    temp3_dir = args.stage2_output_dir if args.stage2_output_dir else os.path.join(args.final_output_dir, "structure_profile_temp3")

    # Create necessary output directories before starting
    os.makedirs(temp2_dir, exist_ok=True)
    os.makedirs(temp3_dir, exist_ok=True)
    os.makedirs(args.final_output_dir, exist_ok=True)

    # --- Determine total tasks for the global progress bar ---
    total_tasks = 0

    # Stage 1: Count .m8 files
    for _, _, files in os.walk(temp1_dir):
        total_tasks += sum(1 for f in files if f.endswith('.m8'))

    # Stage 2: Count query PDB subfolders
    all_query_subfolders = [d for d in os.listdir(args.query_pdb_dir) if os.path.isdir(os.path.join(args.query_pdb_dir, d))]
    if args.filter_list and os.path.isfile(args.filter_list):
        with open(args.filter_list, 'r') as f:
            filtered_queries = set(line.strip() for line in f if line.strip())
        total_tasks += sum(1 for d in all_query_subfolders if d in filtered_queries)
    else:
        total_tasks += len(all_query_subfolders)

    # Stage 3: Count subfolders with NPZ files (potential inputs)
    # Note: This is an estimate, actual processing depends on successful Stage 2 outputs.
    # For now, we'll just add the count of subfolders in temp3_dir.
    # This count must be accurate for the progress bar to show correct total.
    # If Stage 2 produces fewer subfolders than predicted here, the total will be off.
    # However, if Stage 2 completes, we know how many actual subfolders are created.
    # A more robust way would be to dynamically update total in Stage 3, but that makes total calculation more complex.
    stage3_initial_subfolder_count = len([d for d in os.listdir(temp3_dir) if os.path.isdir(os.path.join(temp3_dir, d))])
    total_tasks += stage3_initial_subfolder_count


    with tqdm(total=total_tasks, dynamic_ncols=True, position=0, leave=True, desc="Overall Progress") as global_pbar:
        # Run Stage 1
        global_pbar.set_description("Stage 1: Initializing...")
        run_stage1(
            input_dir=args.input_m8_dir,
            template_dir=args.template_pdb_dir,
            output_dir=temp2_dir,
            global_pbar=global_pbar
        )

        # Run Stage 2
        global_pbar.set_description("Stage 2: Initializing...")
        run_stage2(
            args=args,
            query_dir_stage2=args.query_pdb_dir,
            target_dir_stage1=temp2_dir,
            align_dir_stage2=temp3_dir,
            global_pbar=global_pbar
        )

        # Run Stage 3
        global_pbar.set_description("Stage 3: Initializing...")
        # For Stage 3, we want its completion to signal overall completion.
        # So we ensure its progress updates contribute to the global_pbar.
        run_stage3(
            input_dir_stage3=temp3_dir,
            output_dir_final=args.final_output_dir,
            temp_dirs_to_remove=[temp1_dir, temp2_dir, temp3_dir],
            global_pbar=global_pbar
        )
