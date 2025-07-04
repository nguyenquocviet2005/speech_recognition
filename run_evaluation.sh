#!/bin/bash

# Vietnamese Whisper Model Evaluation Runner
# This script activates the environment and runs the model evaluation

echo "=== Vietnamese Whisper Model Evaluation Runner ==="
echo "Activating environment..."

# Activate the environment
source "$HOME/myenv/bin/activate"

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Environment activated successfully!"
else
    echo "Error: Failed to activate environment at $HOME/myenv/bin/activate"
    echo "Please check if the environment path is correct."
    exit 1
fi

# Check if required files exist
if [ ! -f "evaluate_model.py" ]; then
    echo "Error: evaluate_model.py not found in current directory"
    exit 1
fi

if [ ! -f "fpt_test.csv" ]; then
    echo "Error: fpt_test.csv not found in current directory"
    exit 1
fi

if [ ! -d "whisper-vi-finetuned/checkpoint-7000" ]; then
    echo "Error: Model checkpoint not found at whisper-vi-finetuned/checkpoint-7000"
    exit 1
fi

echo "All required files found. Starting evaluation..."
echo "=========================================="

# Run the evaluation script
python evaluate_model.py

echo "=========================================="
echo "Evaluation completed!"

# Check if results file was created
if [ -f "evaluation_results.txt" ]; then
    echo "Results saved to: evaluation_results.txt"
    echo ""
    echo "=== Quick Results Summary ==="
    head -n 10 evaluation_results.txt
else
    echo "Warning: Results file not created"
fi 