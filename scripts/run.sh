#!/bin/bash
source "$(dirname "$0")/config.sh"
#----------------------*** Search MSA ***----------------------
python scripts/search_MSA.py \
  --pdb_dir "$pdb_dir" \
  --fasta_dir "$fasta_dir" \
  --msa_out_dir "$msa_out_dir" \
  --flagfile "$flagfile" \
  --num_threads "$num_threads"

#-------------------*** Structure Profile ***-------------------
bash scripts/structure_profile.sh \
  --input_base_dir "$pdb_dir" \
  --target_db "$target_db" \
  --output_dir_base "$msa_out_dir/structure_profile_temp1" \
  --tmp_dir_base "$msa_out_dir/structure_profile_temp1" \
  --filter_list "$filter_list" \
  --req_count 200 \
  --max_jobs 10

python scripts/sp.py \
  --input_m8_dir "$msa_out_dir/structure_profile_temp1" \
  --template_pdb_dir "$template_dir" \
  --query_pdb_dir "$pdb_dir" \
  --tmalign_path "$BASE_DIR/scripts" \
  --final_output_dir "$msa_out_dir" \
  --filter_list "$filter_list" \
  --max_workers 1

#-------------------*** MSA embedding ***-------------------
python "$BASE_DIR"/movement_residue/MSA_embedding/run_MSA_embeddings.py \
  --input_base_dir "$msa_out_dir" \
  --output_base_dir "$msa_out_dir"

#-------------------*** Predict MoveRes ***-------------------
python scripts/predict_MoveRes.py \
  --input "$filter_list" \
  --output "$msa_out_dir" \
  --model "$BASE_DIR/movement_residue" \
  --msa_folder "$msa_out_dir" \
  --pdb_folder "$pdb_dir" \
  --template_folder "$msa_out_dir" \
  --process 1 \

#-------------------*** MSA sample ***-------------------
python scripts/clusterMSA.py \
  --input_dir "$msa_out_dir" \
  --protein_list_file "$filter_list" \
  --scan \
  --n_jobs -1

python -m scripts.mmseqs_search_helper \
  --base_dir "$msa_out_dir" \
  --protein_list_file "$filter_list" \
  --db_dir "$msa_database_dir"

python scripts/msa_sample.py \
  --base_dir "$msa_out_dir" \
  --protein_list_file "$filter_list" \
  --npz_name profile.npz \
  --threshold 0.3 \
  --window_size 3 \
  --top_percent 0.2 \
  --step_size 2 \
  --target_total_count "$num_comformations"

#-------------------*** predicted conformation ***-------------------
python scripts/predict_multiple_conformations.py \
  --input_base_dir "$msa_out_dir" \
  --output_base_dir "$msa_out_dir" \
  --folders_file "$filter_list" \
  --num_threads 1 \
  --af2_dir "$BASE_DIR/af_multiple_conformation"

