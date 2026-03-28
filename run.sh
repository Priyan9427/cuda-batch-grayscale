#!/usr/bin/env bash
# run.sh — Build and execute the CUDA Batch Grayscale Converter
# Usage: ./run.sh [input_dir] [output_dir]

set -euo pipefail

INPUT_DIR="${1:-data/images}"
OUTPUT_DIR="${2:-output}"
LOG_FILE="${OUTPUT_DIR}/run.log"

echo "=== Building project ==="
make

echo ""
echo "=== Creating output directory ==="
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "=== Running grayscale converter ==="
echo "Input  : ${INPUT_DIR}"
echo "Output : ${OUTPUT_DIR}"
echo "Log    : ${LOG_FILE}"
echo ""

./grayscale_converter \
  --input  "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}" \
  --log    "${LOG_FILE}"

echo ""
echo "=== Done! Check '${OUTPUT_DIR}' for results and '${LOG_FILE}' for the log ==="
