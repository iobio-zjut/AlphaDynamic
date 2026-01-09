import sys
import functools
import os

print = functools.partial(print, flush=True)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from glob import glob
import time
import threading
import queue

# ================= Configuration =================
parser = argparse.ArgumentParser()
parser.add_argument("input_msa", help="Directory or glob pattern containing .a3m files")
parser.add_argument("--recycles", type=int, default=1, help="Number of recycles")
parser.add_argument("--model_num", type=int, default=1, help="Which AF2 model to use")
parser.add_argument("--seed", type=int, default=0, help="RNG Seed")
parser.add_argument("--af2_dir", default="./alphafold-2.2.0", help="AlphaFold directory")
parser.add_argument("--output_dir", default="./af-output", help="Output directory")
parser.add_argument("--use_gpu_relax", action='store_true', help="Enable GPU relaxation")

args = parser.parse_args()

sys.path.append(args.af2_dir)

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data
from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.relax import relax

# ================= Constants =================
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

# ================= Functions =================

def make_model_runner(name, recycles):
    cfg = config.model_config(name)
    cfg.data.common.num_recycle = recycles
    cfg.model.num_recycle = recycles
    cfg.data.eval.num_ensemble = 1
    params = data.get_model_haiku_params(name, args.af2_dir)
    return model.RunModel(cfg, params)


def get_sequence_length(a3m_path):
    try:
        with open(a3m_path, 'r') as f:
            f.readline()  # 跳过 header
            return len(f.readline().strip())
    except:
        return 0

def make_processed_feature_dict(runner, a3m_file, name="test", seed=0):
    with open(a3m_file, 'r') as f: content = f.read()
    sequence = content.splitlines()[1].strip()
    feature_dict = pipeline.make_sequence_features(sequence, name, len(sequence))
    feature_dict.update(pipeline.make_msa_features([parsers.parse_a3m(content)]))

    feature_dict.update({
        'template_aatype': np.zeros((0, len(sequence), 22), dtype=np.float32),
        'template_all_atom_masks': np.zeros((0, len(sequence), 37), dtype=np.float32),
        'template_all_atom_positions': np.zeros((0, len(sequence), 37, 3), dtype=np.float32),
        'template_domain_names': np.zeros([0], dtype=object),
        'template_sequence': np.zeros([0], dtype=object),
        'template_sum_probs': np.zeros([0], dtype=np.float32),
    })
    return runner.process_features(feature_dict, random_seed=seed)

def parse_results(prediction_result, processed_feature_dict):
    b_factors = prediction_result['plddt'][:, None] * prediction_result['structure_module']['final_atom_mask']
    unrelaxed_prot = protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors)
    return {
        "unrelaxed_protein": unrelaxed_prot,
        "pLDDT": prediction_result['plddt'].mean(),
        "pTMscore": prediction_result['ptm'],
        "prediction_result": prediction_result
    }


def write_results(result, pdb_path, pkl_path, relaxed_pdb_str=None):
    plddt = float(result['pLDDT'])
    ptm = float(result["pTMscore"])

    final_pdb_content = relaxed_pdb_str if relaxed_pdb_str else protein.to_pdb(result["unrelaxed_protein"])
    with open(pdb_path, 'w') as f: f.write(final_pdb_content)

    def _to_np(x): return np.array(x) if isinstance(x, jnp.ndarray) else x

    output_data = {k: _to_np(result['prediction_result'][k]) for k in
                   ['plddt', 'ptm', 'ranking_confidence', 'predicted_aligned_error']}
    with open(pkl_path, 'wb') as f: pickle.dump(output_data, f)

    status = "RELAXED" if relaxed_pdb_str else "UNRELAXED"
    print(f'   -> [Saved] {status} PDB | pLDDT: {plddt:.3f}, pTM: {ptm:.3f}')

# ================= Worker Logic =================
def preprocess_worker(file_queue, data_queue, runner, seed):
    while True:
        try:
            msa_path = file_queue.get(timeout=2)
        except queue.Empty:
            break
        try:
            name = os.path.basename(msa_path).replace('.a3m', '')
            features = make_processed_feature_dict(runner, msa_path, name=name, seed=seed)
            seq_len = features['aatype'].shape[0]
            data_queue.put((name, features, seq_len))
        except Exception as e:
            print(f"[CPU Error] processing {msa_path}: {e}")
            data_queue.put(("ERROR", None, None))
        file_queue.task_done()

def main():
    if os.path.isdir(args.input_msa):
        all_input_files = glob(os.path.join(args.input_msa, "*.a3m"))
    else:
        all_input_files = sorted(glob(args.input_msa))

    if not all_input_files:
        print("No files found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = []
    print(f"[Check] Checking {len(all_input_files)} files for existing results...")

    for f in all_input_files:
        name = os.path.basename(f).replace('.a3m', '')
        expected_pdb = os.path.join(args.output_dir, f"{name}.pdb")

        if os.path.exists(expected_pdb):
            pass
        else:
            input_files.append(f)

    if len(input_files) == 0:
        print("[All Done] All files in this folder are already processed.")
        return

    print(f"[Check] {len(all_input_files) - len(input_files)} skipped. {len(input_files)} to process.")
    print(f"[Pre-scan] Sorting {len(input_files)} files by sequence length to leverage JAX cache...")
    input_files.sort(key=lambda f: get_sequence_length(f))
    print("[Init] Loading AlphaFold model...")
    runner = make_model_runner(f"model_{args.model_num}_ptm", args.recycles)
    print("[Init] Initializing Amber Relaxer (GPU)...")
    try:
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=RELAX_MAX_ITERATIONS, tolerance=RELAX_ENERGY_TOLERANCE,
            stiffness=RELAX_STIFFNESS, exclude_residues=RELAX_EXCLUDE_RESIDUES,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS, use_gpu=True)
        print("   -> Relaxer Ready.")
    except Exception as e:
        print(f"   -> Relaxer Init FAILED: {e}")
        print("   -> WARNING: Will run in UNRELAXED mode.")
        amber_relaxer = None

    file_queue = queue.Queue()
    data_queue = queue.Queue(maxsize=2)
    for f in input_files: file_queue.put(f)
    threading.Thread(target=preprocess_worker, args=(file_queue, data_queue, runner, args.seed), daemon=True).start()

    start_time = time.time()
    processed = 0
    last_len = -1

    while processed < len(input_files):
        print(f"[{processed + 1}/{len(input_files)}] Waiting for CPU data...")
        name, features, seq_len = data_queue.get()

        if name == "ERROR":
            print(f"   -> [SKIP] Skipping corrupted file due to CPU error.")
            processed += 1
            continue

        if last_len != -1 and seq_len != last_len:
            print(f"   -> [Cache MISS] Length changed ({last_len} -> {seq_len}). JAX will RECOMPILE.")
        last_len = seq_len

        print(f"   -> [GPU] Predicting structure for {name} ({seq_len} aa)...")
        t_total_start = time.time()

        prediction_result = runner.predict(features, random_seed=args.seed)
        t_predict = time.time() - t_total_start

        result = parse_results(prediction_result, features)
        relaxed_pdb_str = None
        if amber_relaxer:
            try:
                t_relax_start = time.time()
                relaxed_pdb_str, _, _ = amber_relaxer.process(prot=result["unrelaxed_protein"])
                print(f"   -> [Relax] Done in {time.time() - t_relax_start:.1f}s")
            except Exception as e:
                print(f"   -> [Relax Failed] {e}")

        total_duration = time.time() - t_total_start

        print(f"   -> [GPU] Prediction done in {total_duration:.1f}s")

        write_results(result, os.path.join(args.output_dir, f"{name}.pdb"),
                      os.path.join(args.output_dir, f"result_{name}.pkl"), relaxed_pdb_str)
        processed += 1

    print(f"Done. Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
