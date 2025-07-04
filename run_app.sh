#!/bin/bash

echo "=== Vietnamese Whisper Speech Recognition App Runner ==="

# --- Activate Environment ---
echo "Activating Python environment..."
if [ -f "$HOME/myenv/bin/activate" ]; then
    source "$HOME/myenv/bin/activate"
    echo "Environment activated."
else
    echo "Error: Environment not found at $HOME/myenv/bin/activate."
    echo "Please ensure your virtual environment is set up correctly."
    exit 1
fi

# --- Install/Update Dependencies ---
echo "Checking and installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet --disable-pip-version-check
echo "Dependencies are up to date."

# --- Launch the App ---
echo ""
echo "Launching the Gradio application..."
echo "Please wait for the model to load and the interface to start."
echo "Once it's running, you will see a local URL (like http://127.0.0.1:7860) and a public URL."
echo "Open one of these URLs in your web browser."
echo ""

python app.py 