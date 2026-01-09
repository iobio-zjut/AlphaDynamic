#!/bin/bash

usage() {
  echo "Usage: $0 --input_base_dir DIR --target_db DB --output_dir_base DIR --tmp_dir_base DIR --filter_list FILE --req_count INT --max_jobs INT"
  exit 1
}

max_jobs=10
req_count=200


while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --input_base_dir)
      input_base_dir="$2"
      shift 2
      ;;
    --target_db)
      target_db="$2"
      shift 2
      ;;
    --output_dir_base)
      output_dir_base="$2"
      shift 2
      ;;
    --tmp_dir_base)
      tmp_dir_base="$2"
      shift 2
      ;;
    --filter_list)
      filter_list="$2"
      shift 2
      ;;
    --req_count)
      req_count="$2"
      shift 2
      ;;
    --max_jobs)
      max_jobs="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      usage
      ;;
  esac
done

[[ -z "$input_base_dir" || -z "$target_db" || -z "$output_dir_base" || -z "$tmp_dir_base" || -z "$filter_list" ]] && usage


for pdb_file in "$input_base_dir"/*.pdb; do
    [[ -e "$pdb_file" ]] || continue
    filename=$(basename "$pdb_file")
    base=${filename%.pdb}
    target_folder="$input_base_dir/$base"

    mkdir -p "$target_folder"
    cp "$pdb_file" "$target_folder/"
done


if [[ -f "$filter_list" ]]; then
    echo "Loading target list from: $filter_list"
    mapfile -t target_folders < "$filter_list"
else
    echo "Filter list not found or is a directory, scanning input dir..."
    target_folders=($(ls -d "$input_base_dir"/*/ 2>/dev/null | xargs -n 1 basename))
fi

if [ ${#target_folders[@]} -eq 0 ]; then
    echo "Error: No targets found. Please check input directory or filter list."
    exit 1
fi


for folder in "${target_folders[@]}"; do
    model_file="$input_base_dir/$folder/$folder.pdb"

    if [[ -f "$model_file" ]]; then
        (
            output_dir="$output_dir_base/$folder"
            tmp_dir="$tmp_dir_base/$folder"
            mkdir -p "$output_dir" "$tmp_dir"

            result_file="$output_dir/${folder}_results.m8"
            foldseek easy-search "$model_file" "$target_db" "$result_file" "$tmp_dir" -c 0.3 > /dev/null 2>&1

            grep -v "^#" "$result_file" | \
            awk '{if ($3 > 0.3) print $0}' | \
            sort -k3,3nr | head -n "$req_count" > "$result_file.tmp" && mv "$result_file.tmp" "$result_file"

        ) &

        while (( $(jobs -r | wc -l) >= max_jobs )); do
            sleep 1
        done
    else
        echo "Warring: can not finf $model_file,skipping $folder"
    fi
done

wait
