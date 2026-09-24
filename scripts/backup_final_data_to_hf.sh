#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable backup of the CONUS404 final data to Hugging Face.
# Override HF_REPO, HF_REPO_TYPE, or FINAL_DATA_DIR when needed.
HF_REPO="${HF_REPO:-mudhil/diffusion-downscaling-model}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
FINAL_DATA_DIR="${FINAL_DATA_DIR:-/gpfsm/dnb33/hpmille1/final_data}"
UPLOAD_DIR="${UPLOAD_DIR:-$PWD/.hf_final_data_upload}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: the Hugging Face 'hf' command is not installed." >&2
  exit 127
fi

if [[ ! -d "$FINAL_DATA_DIR" ]]; then
  echo "Error: source directory does not exist: $FINAL_DATA_DIR" >&2
  exit 1
fi

file_count=$(find "$FINAL_DATA_DIR" -maxdepth 1 -type f -name 'conus404_yearly_*.nc' | wc -l)
if [[ "$file_count" -eq 0 ]]; then
  echo "Error: no conus404_yearly_*.nc files found in $FINAL_DATA_DIR" >&2
  exit 1
fi

echo "Uploading $file_count yearly NetCDF files from $FINAL_DATA_DIR"
echo "Destination: $HF_REPO (type: $HF_REPO_TYPE)"
echo "The Hugging Face large-folder uploader is resumable; rerun this script after interruptions."

# final_data is read-only on the shared filesystem. Build a tiny writable
# staging directory of symlinks so the uploader can store its .cache metadata
# without copying the multi-terabyte source files.
mkdir -p "$UPLOAD_DIR"
while IFS= read -r source_file; do
  ln -sfn "$source_file" "$UPLOAD_DIR/$(basename "$source_file")"
done < <(find "$FINAL_DATA_DIR" -maxdepth 1 -type f -name 'conus404_yearly_*.nc' -print)

exec hf upload-large-folder "$HF_REPO" "$UPLOAD_DIR" \
  --type "$HF_REPO_TYPE" \
  --include 'conus404_yearly_*.nc' \
  --num-workers "$NUM_WORKERS"
