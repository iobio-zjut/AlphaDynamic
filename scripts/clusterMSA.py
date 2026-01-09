#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import glob
import csv
import multiprocessing as mp
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Optional, Sequence, Iterable

try:
    from polyleven import levenshtein
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import pdist, squareform
except ImportError as e:
    sys.exit(f"Be sure you have polyleven, scikit-learn, scipy。")

# 可选依赖：sklearn-extra
try:
    from sklearn_extra.cluster import KMedoids

    SKLEARN_EXTRA_AVAILABLE = True
except Exception:
    SKLEARN_EXTRA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:
    plt = None
    sns = None
    PCA = None
    TSNE = None

try:
    from Bio import SeqIO
except ImportError:
    sys.exit("please 'pip install biopython'")

try:
    from utils import load_fasta, encode_seqs, consensusVoting, lprint, write_fasta
except Exception:
    def lprint(msg, f=None):
        if f:
            f.write(msg + '\n')
            f.flush()
        print(msg)

    def load_fasta(path: str):
        if not os.path.exists(path):
            return [], []
        IDs = []
        seqs = []
        cur_id = None
        cur_seq = ''
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                if line.startswith('>'):
                    if cur_id is not None:
                        IDs.append(cur_id)
                        seqs.append(cur_seq)
                    cur_id = line[1:].split()[0]
                    cur_seq = ''
                else:
                    cur_seq += line.strip()
        if cur_id is not None:
            IDs.append(cur_id)
            seqs.append(cur_seq)
        clean_seqs = [''.join([ch for ch in s if ch.isupper() or ch == '-']) for s in seqs]
        return IDs, clean_seqs

    def encode_seqs(seqs: List[str], max_len: Optional[int] = None):
        if not seqs:
            return np.array([])
        L = max_len or max(len(s) for s in seqs)
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                    '-']
        mapping = {c: i for i, c in enumerate(alphabet)}
        D = len(alphabet)
        out = np.zeros((len(seqs), L * D), dtype=float)
        for ii, s in enumerate(seqs):
            for i, ch in enumerate(s):
                if i >= L: break
                idx = mapping.get(ch.upper(), mapping['-'])
                out[ii, i * D + idx] = 1.0
        return out

    def consensusVoting(seqs: List[str]):
        if not seqs:
            return ""
        L = max(len(s) for s in seqs)
        res = []
        for i in range(L):
            cnt = Counter()
            for s in seqs:
                if i < len(s):
                    cnt[s[i]] += 1
            if cnt:
                res.append(cnt.most_common(1)[0][0])
            else:
                res.append('-')
        return ''.join(res)

    def write_fasta(names: List[str], seqs: List[str], outfile: str):
        with open(outfile, 'w') as fo:
            for n, s in zip(names, seqs):
                fo.write(f">{n}\n{s}\n")

def _convert_sto_seq_to_a3m(query_non_gaps: Sequence[bool], sto_seq: str) -> Iterable[str]:
    for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
        if is_query_res_non_gap:
            yield sequence_res
        elif sequence_res != '-':
            yield sequence_res.lower()


def convert_stockholm_to_a3m(stockholm_format: str, max_sequences: Optional[int] = None,
                             remove_first_row_gaps: bool = True) -> str:
    descriptions = {}
    sequences = {}

    for line in stockholm_format.splitlines():
        if line.strip() and not line.startswith(('#', '//')):
            parts = line.split(maxsplit=1)
            if len(parts) < 2: continue
            seqname, aligned_seq = parts
            if max_sequences is not None and len(sequences) >= max_sequences and seqname not in sequences:
                continue
            sequences.setdefault(seqname, '')
            sequences[seqname] += aligned_seq.replace('.', '')

    for line in stockholm_format.splitlines():
        if line.startswith('#=GS'):
            columns = line.split(maxsplit=3)
            if len(columns) >= 3:
                seqname, feature = columns[1:3]
                value = columns[3].strip() if len(columns) == 4 else ''
                if feature == 'DE':
                    if max_sequences is not None and seqname not in sequences:
                        continue
                    descriptions[seqname] = value

    a3m_sequences = {}
    query_non_gaps = []

    if sequences:
        first_seq_id = next(iter(sequences))
        query_sequence_raw = sequences[first_seq_id]
        if remove_first_row_gaps:
            query_non_gaps = [res != '-' for res in query_sequence_raw]
        else:
            query_non_gaps = [True] * len(query_sequence_raw)

    for seqname, sto_sequence_raw in sequences.items():
        if remove_first_row_gaps:
            out_sequence = ''.join(_convert_sto_seq_to_a3m(query_non_gaps, sto_sequence_raw))
        else:
            out_sequence = sto_sequence_raw
        a3m_sequences[seqname] = out_sequence

    fasta_chunks = (f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}" for k in a3m_sequences)
    return '\n'.join(fasta_chunks) + '\n'

def process_uniref90_sto_to_a3m(sto_file_path: str, output_a3m_path: str) -> bool:
    try:
        with open(sto_file_path, 'r') as f:
            sto_str = f.read()
        a3m_str = convert_stockholm_to_a3m(sto_str, remove_first_row_gaps=True)
        os.makedirs(os.path.dirname(output_a3m_path), exist_ok=True)
        with open(output_a3m_path, 'w') as fo:
            fo.write(a3m_str)
        return True
    except Exception as e:
        lprint(f"Error converting {sto_file_path} to A3M: {e}", sys.stderr)
        return False

def get_first_sequence(msa_path: str) -> Optional[Tuple[str, str]]:
    IDs, seqs = load_fasta(msa_path)
    if IDs and seqs:
        return IDs[0], seqs[0]
    return None

def merge_a3m_files(file_paths: List[str], output_path: str) -> bool:
    if not file_paths:
        return False
    try:
        first_seq_info = get_first_sequence(file_paths[0])
        if not first_seq_info:
            lprint(f"Error: Could not extract query from {file_paths[0]} for merging.", sys.stderr)
            return False
        query_id, query_seq = first_seq_info
        all_sequences = {query_id: query_seq}
        for path in file_paths:
            IDs, seqs = load_fasta(path)
            for seq_id, seq in zip(IDs, seqs):
                if seq_id not in all_sequences:
                    all_sequences[seq_id] = seq
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f">{query_id}\n{query_seq}\n")
            for seq_id, seq in all_sequences.items():
                if seq_id != query_id:
                    f.write(f">{seq_id}\n{seq}\n")
        return True
    except Exception as e:
        lprint(f"Error merging A3M files to {output_path}: {e}", sys.stderr)
        return False

def pad_seqs_to_max(seqs: List[str]):
    max_len = max((len(s) for s in seqs), default=0)
    padded = [s + '-' * (max_len - len(s)) if len(s) < max_len else s for s in seqs]
    return padded, max_len

def cluster_center_by_similarity(seqs: List[str]):
    if len(seqs) == 0:
        return None
    if len(seqs) == 1:
        return seqs[0]
    best_seq = None
    best_score = -np.inf
    for i, s in enumerate(seqs):
        score_sum = 0.0
        for j, t in enumerate(seqs):
            if i == j:
                continue
            maxlen = max(len(s), len(t))
            if maxlen == 0:
                sim = 0.0
            else:
                sim = 1.0 - levenshtein(s, t) / float(maxlen)
            score_sum += sim
        divisor = len(seqs) - 1
        avg_score = score_sum / divisor if divisor > 0 else -np.inf
        if avg_score > best_score:
            best_score = avg_score
            best_seq = s
    return best_seq


def cluster_medoids_by_kmedoids(medoid_seqs: List[str], n_clusters: int, n_jobs: int, max_iter=300, random_state=0,
                                log_f=None):
    if len(medoid_seqs) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    m = len(medoid_seqs)
    if n_clusters >= m:
        labels = np.arange(m)
        medoid_indices = np.arange(m)
        return labels, medoid_indices
    medoid_padded, max_len = pad_seqs_to_max(medoid_seqs)
    medoid_ohe = encode_seqs(medoid_padded, max_len=max_len)
    if log_f:
        lprint(f"Computing pairwise distances among medoids (n={m}) using {n_jobs} cores...", log_f)
    dist_matrix = pairwise_distances(medoid_ohe, metric="euclidean", n_jobs=n_jobs)
    np.fill_diagonal(dist_matrix, 0.0)
    if SKLEARN_EXTRA_AVAILABLE:
        if log_f:
            lprint(f"Running sklearn_extra.cluster.KMedoids (precomputed) with K={n_clusters}...", log_f)
        kmed = KMedoids(n_clusters=n_clusters, metric="precomputed", init='k-medoids++', max_iter=max_iter,
                        random_state=random_state)
        kmed.fit(dist_matrix)
        labels = kmed.labels_
        medoid_indices = np.array(kmed.medoid_indices_, dtype=int)
        return labels, medoid_indices
    else:
        from scipy.cluster.hierarchy import linkage, fcluster
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method='average')
        labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
        n_actual = np.max(labels) + 1 if len(labels) > 0 else 0
        if log_f:
            lprint(f"Hierarchical fallback produced {n_actual} clusters (target {n_clusters})", log_f)
        medoid_indices = []
        for lab in range(n_actual):
            idxs = np.where(labels == lab)[0]
            if len(idxs) == 1:
                medoid_indices.append(idxs[0])
            else:
                submat = dist_matrix[np.ix_(idxs, idxs)]
                sums = submat.sum(axis=1)
                medoid_indices.append(idxs[int(np.argmin(sums))])
        medoid_indices = np.array(medoid_indices, dtype=int)
        return labels, medoid_indices


def run_dbscan_core(keyword: str, input_path: str, outdir: str, args, log_f):
    IDs, seqs = load_fasta(input_path)
    seqs = [''.join([x for x in seq if x.isupper() or x == '-']) for seq in seqs]
    df = pd.DataFrame({'SequenceName': IDs, 'sequence': seqs})

    query_ = df.iloc[:1]
    df = df.iloc[1:]

    if args.resample:
        df = df.sample(frac=1)

    if len(df) == 0:
        lprint(f"No sequences to cluster in {input_path}.", log_f)
        return None, None, None

    L = len(df.sequence.iloc[0])
    df['frac_gaps'] = [x.count('-') / L for x in df['sequence']]

    former_len = len(df)
    df = df.loc[df.frac_gaps < args.gap_cutoff]
    new_len = len(df)

    lprint(keyword, log_f)
    lprint("%d seqs removed for containing more than %d%% gaps, %d remaining" % (
    former_len - new_len, int(args.gap_cutoff * 100), new_len), log_f)
    ohe_seqs = encode_seqs(df.sequence.tolist(), max_len=L)

    n_clusters = []
    eps_test_vals = np.arange(float(args.min_eps), float(args.max_eps) + float(args.eps_step), float(args.eps_step))

    if args.eps_val is None:
        lprint('eps\tn_clusters\tn_not_clustered', log_f)
        for eps in eps_test_vals:
            testset = encode_seqs(df.sample(frac=0.25).sequence.tolist(), max_len=L)
            clustering = DBSCAN(eps=eps, min_samples=int(args.min_samples)).fit(testset)
            n_clust = len(set(clustering.labels_))
            n_not_clustered = len(clustering.labels_[np.where(clustering.labels_ == -1)])
            lprint('%.2f\t%d\t%d' % (eps, n_clust, n_not_clustered), log_f)
            n_clusters.append(n_clust)
            if eps > 20 and n_clust == 1:
                break
        eps_to_select = eps_test_vals[np.argmax(n_clusters)]
    else:
        eps_to_select = float(args.eps_val)

    clustering = DBSCAN(eps=eps_to_select, min_samples=int(args.min_samples)).fit(ohe_seqs)

    lprint('Selected eps=%.2f' % eps_to_select, log_f)
    lprint("%d total seqs" % len(df), log_f)

    df['dbscan_label'] = clustering.labels_

    clusters = [x for x in df.dbscan_label.unique() if x >= 0]
    unclustered = len(df.loc[df.dbscan_label == -1])

    lprint('%d clusters, %d of %d not clustered (%.2f)' % (len(clusters), unclustered, len(df), unclustered / len(df)),
           log_f)

    cluster_metadata = []
    for clust in clusters:
        tmp = df.loc[df.dbscan_label == clust]
        cs = consensusVoting(tmp.sequence.tolist())
        avg_dist_to_cs = np.mean([1 - levenshtein(x, cs) / L for x in tmp.sequence.tolist()]) if len(tmp) > 0 else 0.0
        avg_dist_to_query_c = np.mean(
            [1 - levenshtein(x, query_['sequence'].iloc[0]) / L for x in tmp.sequence.tolist()]) if len(
            tmp) > 0 else 0.0

        tmp_write = pd.concat([query_, tmp], axis=0)
        cluster_metadata.append({'cluster_ind': clust, 'consensusSeq': cs, 'avg_lev_dist': '%.3f' % avg_dist_to_cs,
                                 'avg_dist_to_query': '%.3f' % avg_dist_to_query_c, 'size': len(tmp_write)})
        if args.save_dbscan:
            write_fasta(tmp_write.SequenceName.tolist(), tmp_write.sequence.tolist(),
                        outfile=os.path.join(outdir, f"{keyword}_" + "%03d" % clust + '.a3m'))

    if args.save_controls:
        try:
            for i in range(int(args.n_controls)):
                tmp = df.sample(n=10) if len(df) >= 10 else df
                tmp_write = pd.concat([query_, tmp], axis=0)
                write_fasta(tmp_write.SequenceName.tolist(), tmp_write.sequence.tolist(),
                            outfile=os.path.join(outdir, f"{keyword}_U10-" + "%03d" % i + '.a3m'))
            if len(df) > 100:
                for i in range(int(args.n_controls)):
                    tmp = df.sample(n=100)
                    tmp_write = pd.concat([query_, tmp], axis=0)
                    write_fasta(tmp_write.SequenceName.tolist(), tmp_write.sequence.tolist(),
                                outfile=os.path.join(outdir, f"{keyword}_U100-" + "%03d" % i + '.a3m'))
        except Exception as e:
            lprint(f"Error writing control clusters: {e}", log_f)

    outfile = os.path.join(outdir, f"{keyword}_clustering_assignments.tsv")
    df.to_csv(outfile, index=False, sep='\t')
    metad_outfile = os.path.join(outdir, f"{keyword}_cluster_metadata.tsv")
    pd.DataFrame.from_records(cluster_metadata).to_csv(metad_outfile, index=False, sep='\t')

    return df, query_, L, cluster_metadata


def medoid_level_clustering(df: pd.DataFrame, query_, L: int, outdir: str, keyword: str, args, log_f):
    clusters_stage1 = [x for x in df.dbscan_label.unique() if x >= 0]
    if not clusters_stage1:
        lprint("No Stage1 tight clusters for medoid-level clustering.", log_f)
        return
    initial_clusters = {lab: df.loc[df.dbscan_label == lab, 'sequence'].tolist() for lab in clusters_stage1}

    medoid_list = []
    medoid_stage1_label = []
    medoid_cluster_sizes = {}
    for lab, seqs_in in initial_clusters.items():
        med = cluster_center_by_similarity(seqs_in)
        medoid_list.append(med)
        medoid_stage1_label.append(lab)
        medoid_cluster_sizes[lab] = len(seqs_in)

    if len(medoid_list) == 0:
        return

    target_k = int(args.max_final_clusters)
    n_jobs_dist = mp.cpu_count() if args.n_jobs == -1 else int(args.n_jobs)

    labels_medoid, medoid_indices = cluster_medoids_by_kmedoids(medoid_list, target_k, n_jobs=n_jobs_dist, log_f=log_f)

    final_label_to_medoid_stage1 = {}
    for final_lab in np.unique(labels_medoid):
        member_idxs = np.where(labels_medoid == final_lab)[0]
        member_stage1 = [medoid_stage1_label[i] for i in member_idxs]
        chosen = max(member_stage1, key=lambda x: medoid_cluster_sizes.get(x, 0))
        final_label_to_medoid_stage1[final_lab] = chosen

    def map_stage1_to_final(x):
        if x == -1: return -1
        try:
            idx = medoid_stage1_label.index(x)
            return labels_medoid[idx]
        except ValueError:
            return -1

    df['dbscan_label_medoid'] = df['dbscan_label'].apply(map_stage1_to_final)
    final_clusters = [lab for lab in np.unique(df['dbscan_label_medoid']) if lab != -1]

    medoid_meta = []
    cluster_idx = 0
    query_seq = query_['sequence'].iloc[0]
    query_name = query_['SequenceName'].iloc[0]

    for flab in final_clusters:
        tmp = df.loc[df['dbscan_label_medoid'] == flab].copy()
        if tmp.empty: continue
        chosen_stage1 = final_label_to_medoid_stage1.get(flab)
        idx_med = medoid_stage1_label.index(chosen_stage1)
        medoid_seq = medoid_list[idx_med]
        rows = df.loc[df['dbscan_label'] == chosen_stage1]
        medoid_name = rows.iloc[0]['SequenceName'] if not rows.empty else f"{keyword}_M{cluster_idx:03d}_medoid"

        out_names = [query_name, medoid_name]
        out_seqs = [query_seq, medoid_seq]
        other_rows = tmp.loc[tmp['sequence'] != medoid_seq]
        out_names.extend(other_rows['SequenceName'].tolist())
        out_seqs.extend(other_rows['sequence'].tolist())

        out_fname = os.path.join(outdir, f"{keyword}_M{cluster_idx:03d}.a3m")
        write_fasta(out_names, out_seqs, out_fname)

        cs = consensusVoting(tmp['sequence'].tolist())
        try:
            avg_dist_to_cs = np.mean([1.0 - levenshtein(s, cs) / float(L) for s in tmp['sequence'].tolist()])
            avg_dist_to_query = np.mean([1.0 - levenshtein(s, query_seq) / float(L) for s in tmp['sequence'].tolist()])
        except Exception:
            avg_dist_to_cs = 0.0
            avg_dist_to_query = 0.0

        medoid_meta.append(
            {'cluster_ind': cluster_idx, 'final_label': flab, 'medoid_stage1': chosen_stage1, 'size': len(tmp) + 1,
             'consensusSeq': cs, 'avg_lev_dist': f'{avg_dist_to_cs:.3f}',
             'avg_dist_to_query': f'{avg_dist_to_query:.3f}'})
        cluster_idx += 1

    meta_out = os.path.join(outdir, f"{keyword}_medoid_cluster_metadata.tsv")
    pd.DataFrame.from_records(medoid_meta).to_csv(meta_out, index=False, sep='\t')
    lprint(f"Wrote medoid metadata to {meta_out}", log_f)


def process_single_protein_workflow(protein_dir: str, args):
    protein_name = os.path.basename(protein_dir.rstrip(os.sep))
    msas_dir = os.path.join(protein_dir, "msas")
    if not os.path.isdir(msas_dir):
        lprint(f"Skipping {protein_name}: 'msas' directory not found.", sys.stderr)
        return

    merged_a3m_path = os.path.join(msas_dir, f"{protein_name}.a3m")
    needs_merge = True
    if os.path.exists(merged_a3m_path) and not args.force_merge:
        needs_merge = False

    if needs_merge:
        sto_path = os.path.join(msas_dir, "uniref90_hits.sto")
        uniref90_a3m_path = os.path.join(msas_dir, "uniref90_hits.a3m")
        if os.path.exists(sto_path):
            lprint(f"Converting {protein_name}/uniref90_hits.sto...", sys.stderr)
            if not process_uniref90_sto_to_a3m(sto_path, uniref90_a3m_path):
                return
        a3m_files_to_merge = []
        for name in ["bfd_hits.a3m", "uniref30_hits.a3m"]:
            path = os.path.join(msas_dir, name)
            if os.path.exists(path): a3m_files_to_merge.append(path)
        if os.path.exists(uniref90_a3m_path): a3m_files_to_merge.append(uniref90_a3m_path)

        if not a3m_files_to_merge: return
        if not merge_a3m_files(a3m_files_to_merge, merged_a3m_path): return

    final_output_dir = os.path.join(protein_dir, "msas_cluster")
    os.makedirs(final_output_dir, exist_ok=True)

    if glob.glob(os.path.join(final_output_dir, f"{protein_name}_*.a3m")) and not args.force_cluster:
        lprint(f"Skipping clustering for {protein_name}: already done.", sys.stderr)
        return

    log_fpath = os.path.join(final_output_dir, f"{protein_name}.log")
    with open(log_fpath, "w") as f:
        lprint(f"Starting clustering for {protein_name}", f)
        try:
            df, query_, L, cluster_metadata = run_dbscan_core(protein_name, merged_a3m_path, final_output_dir, args, f)
            if df is not None:
                medoid_level_clustering(df, query_, L, final_output_dir, protein_name, args, f)
        except Exception as e:
            lprint(f"Error processing {protein_name}: {e}", f)

def batch_a3m_to_csv_second_seq(base_dir, protein_list_file):

    protein_names = []
    try:
        with open(protein_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    protein_names.append(name)
    except FileNotFoundError:
        print(f"can not find {protein_list_file}")
        return

    if not protein_names:
        print("list is empty")
        return

    for protein_name in protein_names:
        a3m_dir = os.path.join(base_dir, protein_name, 'msas_cluster')
        output_csv = os.path.join(a3m_dir, 'msas_cluster.csv')

        if not os.path.isdir(a3m_dir):
            print(f"can not find {protein_name}")
            continue

        processed_count = 0
        file_count = 0

        try:
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['name', 'seqres']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()

                for filename in os.listdir(a3m_dir):
                    if filename.endswith(".a3m"):
                        file_count += 1
                        file_path = os.path.join(a3m_dir, filename)
                        name = os.path.splitext(filename)[0]
                        second_sequence = None
                        try:
                            # 使用 SeqIO 解析
                            records = list(SeqIO.parse(file_path, "fasta"))
                            if len(records) >= 2:
                                record = records[1]
                                seq = str(record.seq).strip()
                                second_sequence = ''.join([c for c in seq if c.isupper() or c == '-'])
                                if not second_sequence: continue
                                second_sequence = "'" + second_sequence
                                writer.writerow({'name': name, 'seqres': second_sequence})
                                processed_count += 1
                        except Exception:
                            continue

        except Exception as e:
            print(f" process {protein_name} falid: {e}")

def main():
    p = argparse.ArgumentParser(description="Integrated Workflow: Cluster MSA -> Extract CSV")

    p.add_argument("--input_dir", required=True, help="base_dir")
    p.add_argument("--protein_list_file", required=False,
                   help="skip mask CSV 。")

    p.add_argument("--n_controls", type=int, default=20, help="Number of control msas (Default 20)")
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--scan', action='store_true', help='Scan eps value')
    p.add_argument('--eps_val', type=float, default=None)
    p.add_argument('--resample', action='store_true')
    p.add_argument("--gap_cutoff", type=float, default=0.25)
    p.add_argument('--min_eps', type=float, default=3)
    p.add_argument('--max_eps', type=float, default=20)
    p.add_argument('--eps_step', type=float, default=.5)
    p.add_argument('--min_samples', type=int, default=3)
    p.add_argument('--run_PCA', action='store_true')
    p.add_argument('--run_TSNE', action='store_true')
    p.add_argument("--max_final_clusters", type=int, default=30)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--force_merge", action='store_true')
    p.add_argument("--force_cluster", action='store_true')
    p.add_argument("--save_dbscan", action="store_true", default=False)
    p.add_argument("--save_controls", action="store_true", default=False)

    args = p.parse_args()

    if args.n_jobs == -1: args.n_jobs = -1

    if not os.path.isdir(args.input_dir):
        lprint(f"Error: Input directory {args.input_dir} not found.", sys.stderr)
        return

    protein_dirs = sorted([os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if
                           os.path.isdir(os.path.join(args.input_dir, d))])

    if not protein_dirs:
        lprint(f"No protein subdirectories found in {args.input_dir}.", sys.stderr)
    else:
        for pd in protein_dirs:
            process_single_protein_workflow(pd, args)

    if args.protein_list_file:
        batch_a3m_to_csv_second_seq(args.input_dir, args.protein_list_file)
    else:
        print("\ncan not find --protein_list_file")

    lprint("\ncluster processing done.", sys.stderr)


if __name__ == "__main__":
    main()