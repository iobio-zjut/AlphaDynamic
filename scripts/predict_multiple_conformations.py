#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def process_protein_folder(protein_folder_name, input_base_dir, output_base_dir, af2_root_dir):
    input_dir = Path(input_base_dir) / protein_folder_name / "sub_msas"
    output_pdb_subdir = Path(output_base_dir) / protein_folder_name / "pdb"
    output_pdb_subdir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        log(f"[WARN] Missing folder: {input_dir}")
        return False

    if not list(input_dir.glob("*.a3m")):
        log(f"[SKIP] No .a3m files in {input_dir}")
        return True

    log(f"[START BATCH] Processing folder: {protein_folder_name}")

    command = [
        "python",
        "-u",
        "./scripts/RunAF2.py",
        str(input_dir),
        "--af2_dir", af2_root_dir,
        "--output_dir", str(output_pdb_subdir),
        "--model_num", "1",
        "--recycles", "1",
        "--seed", "0"
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
        ) as proc:

            current_file = "Unknown"

            for line in iter(proc.stdout.readline, ''):
                if not line: break
                line = line.strip()
                if "Predicting structure for" in line:
                    try:
                        parts = line.split("for ")
                        if len(parts) > 1:
                            current_file = parts[1].split("(")[0].strip()
                    except:
                        pass

                elif "Prediction done in" in line:
                    try:
                        time_str = line.split("in")[-1].strip()
                        log(f"[COMPLETED] {protein_folder_name}/{current_file} | Time: {time_str}")
                    except:
                        pass

                elif "JAX will RECOMPILE" in line:
                    log(f"[WARN] JAX Recompiling for {current_file}")

                elif "Error" in line or "Exception" in line or "Traceback" in line:
                    print(f"    [SUB-LOG ERROR] {line}", flush=True)

        return_code = proc.wait()

        if return_code == 0:
            log(f"[DONE BATCH] Finished folder: {protein_folder_name}")
            return True
        else:
            log(f"[ERROR] Process failed for folder: {protein_folder_name} with return code {return_code}")
            return False

    except Exception as e:
        log(f"[CRITICAL ERROR] Failed to launch subprocess for {protein_folder_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch run AlphaFold2 per FOLDER with Real-time Logging")
    parser.add_argument("--input_base_dir", required=True, help="Base input directory.")
    parser.add_argument("--output_base_dir", required=True, help="Base output directory.")
    parser.add_argument("--folders_file", required=True, help="List of protein folder names.")
    parser.add_argument("--num_threads", type=int, default=1, help="Concurrent python processes (Keep low for GPU).")
    parser.add_argument("--af2_dir", required=True, help="Path to AlphaFold2 installation.")

    args = parser.parse_args()

    log("Job started (Real-time Folder-Batch Mode).")

    if not os.path.isfile(args.folders_file):
        log(f"Error: Cannot find folder list file: {args.folders_file}")
        sys.exit(1)

    with open(args.folders_file) as f:
        folder_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    tasks = []

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for protein_folder in folder_list:
            future = executor.submit(
                process_protein_folder,
                protein_folder,
                args.input_base_dir,
                args.output_base_dir,
                args.af2_dir
            )
            tasks.append(future)

        for future in as_completed(tasks):
            future.result()

    log("Job finished.")


if __name__ == "__main__":
    main()