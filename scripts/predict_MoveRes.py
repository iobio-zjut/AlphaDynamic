#!/software/conda/envs/tensorflow/bin/python
import sys
import argparse
import os
from os.path import isfile, isdir, join
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def min_max_normalize(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    if max_val == min_val:
        return np.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)


def main():
    parser = argparse.ArgumentParser(description="Error predictor network", epilog="v1.0.0")
    parser.add_argument("--input", "-i", type=str, default="list.txt",
                        help="Path to input txt file (each line contains a sample name)")
    parser.add_argument("--output", "-o", type=str, default="predict",
                        help="Path to output (folder path, npz, or csv)")
    parser.add_argument("--model", "-m", type=str, default="model",
                        help="Path to model folder")
    parser.add_argument("--msa_folder", "-msa", type=str, default="predict",
                        help="Path to msa folder (subfolders: <query_name>/msas_embeddings/*.npz)")
    parser.add_argument("--pdb_folder", "-pdbs", type=str, default="pdb",
                        help="Path to pdb database folder")
    parser.add_argument("--template_folder", "-tem", type=str, default="predict",
                        help="Path to template database folder (<query_name>/structure_profile.npz)")
    parser.add_argument("--use_bfactor", "-b", action="store_true", default=False)
    parser.add_argument("--pdb", "-pdb", action="store_true", default=False,
                        help="Run on a single pdb file")
    parser.add_argument("--leaveTempFile", "-lt", action="store_true", default=False)
    parser.add_argument("--process", "-p", type=int, default=8,
                        help="Number of CPUs for featurization")
    parser.add_argument("--featurize", "-f", action="store_true", default=False,
                        help="Only run featurization")
    parser.add_argument("--reprocess", "-r", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.pdb:
        if not isfile(args.input):
            print("Input file does not exist.", file=sys.stderr)
            return -1

    if not isdir(args.output):
        if args.verbose:
            print("Creating output folder:", args.output)
        os.makedirs(args.output)

    script_dir = os.path.dirname(__file__)
    modelpath = os.path.join(script_dir, args.model)

    if not isdir(modelpath):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1

    sys.path.insert(0, script_dir)
    import movement_residue as ax

    num_process = max(args.process, 1)

    pdb_path = args.pdb_folder
    if not args.pdb:
        if args.input.endswith('.npy'):
            samples = np.load(args.input).tolist()
        else:
            with open(args.input, 'r') as f:
                samples = [line.strip() for line in f if line.strip()]
    else:
        samples = [os.path.basename(args.input).split('.')[0]]

    if args.verbose:
        print("# samples:", len(samples))

    msa_folder = args.msa_folder
    template_folder = args.template_folder

    if not args.pdb:
        inputs = [join(pdb_path, s) + ".pdb" for s in samples]
    else:
        inputs = [args.input]

    tmpoutputs = [join(args.output, s) + ".features.npz" for s in samples]

    if not args.reprocess:
        arguments = [(inputs[i], tmpoutputs[i], args.verbose)
                     for i in range(len(inputs)) if not isfile(tmpoutputs[i])]
        already_processed = [(inputs[i], tmpoutputs[i], args.verbose)
                             for i in range(len(inputs)) if isfile(tmpoutputs[i])]
    else:
        arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
        already_processed = [(inputs[i], tmpoutputs[i], args.verbose)
                             for i in range(len(inputs)) if isfile(tmpoutputs[i])]

    if args.verbose:
        print("Featurizing", len(arguments), "samples.", len(already_processed), "already processed.")

    try:
        if num_process == 1:
            for a in tqdm(arguments, desc="Featurizing"):
                ax.process(a)
        else:
            with multiprocessing.Pool(num_process) as pool:
                for _ in tqdm(pool.imap_unordered(ax.process, arguments), total=len(arguments), desc="Featurizing"):
                    pass
    except Exception as e:
        print(f"Feature processing error: {e}", file=sys.stderr)

    if args.featurize:
        return 0

    if args.verbose:
        print("Using model path:", modelpath)

    samples = [s for s in samples if isfile(join(args.output, s + ".features.npz"))]
    modelnames = ["model.pkl"]
    # modelnames = ["best.pkl"]

    for modelname in modelnames:
        model = ax.Model_final(use_bfactor=args.use_bfactor)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(join(modelpath, modelname), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        for s in tqdm(samples, desc="Predicting"):
            with torch.no_grad():
                try:
                    if args.verbose:
                        print(f"Predicting sample: {s}")

                    filename = join(args.output, s + ".features.npz")
                    msa_base_path = join(msa_folder, s, "msas_embeddings")
                    msa_path_random = join(msa_base_path, f"{s}_random.npz")
                    msa_path_default = join(msa_base_path, f"{s}.npz")

                    if os.path.exists(msa_path_random):
                        msa_path = msa_path_random
                    elif os.path.exists(msa_path_default):
                        msa_path = msa_path_default
                    else:
                        raise FileNotFoundError(f"MSA file not found for {s} in {msa_base_path}")

                    template_path = join(template_folder, s, "structure_profile.npz")
                    if not os.path.exists(template_path):
                        raise FileNotFoundError(f"Template file not found for {s} at {template_path}")
                    template_data = np.load(template_path, allow_pickle=True)

                    if "result" not in template_data:
                        raise KeyError(f"Template file {template_path} missing key 'result'")
                    tem = min_max_normalize(template_data["result"].astype(np.float32))

                    (idx, val), (f1d, _), f2d, dmy, msa = ax.getData(
                        filename, msa_path,
                        use_msa_transformer=True,
                        use_bfactor=args.use_bfactor,
                    )

                    f1d = torch.Tensor(np.concatenate([f1d, tem], axis=-1)).to(device)
                    f2d = torch.Tensor(np.expand_dims(f2d.transpose(2, 0, 1), 0)).to(device)
                    idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
                    val = torch.Tensor(val).to(device)
                    msa = msa if isinstance(msa, dict) else msa[0].to(device)

                    if isinstance(msa, dict):
                        msa = {k: torch.Tensor(v).to(device) for k, v in msa.items()}

                    esto_pred, esto_logits = model(idx, val, f1d, f2d, msa)

                    out_dir = join(args.output, s)
                    os.makedirs(out_dir, exist_ok=True)
                    np.savez_compressed(join(out_dir, "profile.npz"),
                                        deviation=esto_pred.cpu().detach().numpy().astype(np.float16))
                except Exception as e:
                    print(f"[{s}] Prediction error: {e}", file=sys.stderr)

    if not args.leaveTempFile:
        ax.clean(samples, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
