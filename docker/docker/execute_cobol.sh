#!/bin/bash
# LegacyCodeBench COBOL Execution Script
# Compiles and executes a COBOL program, capturing output

set -e

SOURCE_FILE=$1
INPUT_DATA=$2
OUTPUT_FILE=${3:-/workspace/output/result.txt}

if [ -z "$SOURCE_FILE" ]; then
    echo "Usage: execute_cobol.sh <source_file> [input_data] [output_file]"
    exit 1
fi

# Compile the COBOL program
echo "Compiling: $SOURCE_FILE"
cobc -x -o /workspace/output/program "$SOURCE_FILE" 2>&1

# Execute with optional input
echo "Executing..."
if [ -n "$INPUT_DATA" ] && [ -f "$INPUT_DATA" ]; then
    /workspace/output/program < "$INPUT_DATA" > "$OUTPUT_FILE" 2>&1
else
    /workspace/output/program > "$OUTPUT_FILE" 2>&1
fi

echo "Execution complete. Output saved to: $OUTPUT_FILE"
cat "$OUTPUT_FILE"

