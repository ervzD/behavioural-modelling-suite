#!/bin/bash

# Move to the project folder regardless of where the script is called from
cd "$(dirname "$0")"

echo ""
echo "Behavioural Modelling Suite"
echo "==========================="

# Install UV if not already present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing package manager (this only happens once)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python and all dependencies
echo ""
echo "Checking dependencies..."
uv sync

echo ""
echo "Starting — your browser will open automatically."
echo "To stop the app when you are done, press Ctrl+C."
echo ""

uv run streamlit run app.py
