#!/bin/bash

echo "Setting up Python virtual environment and creating folders..."
echo "Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "WARNING: Python 3 not found."
    echo "Closing in 5 seconds..."
    sleep 5
    exit 1
fi

echo "Python 3 found."
echo "Creating virtual environment..."
python3 -m venv .venv
echo "Virtual environment successfully created."
source .venv/bin/activate
echo "Installing requirements..."
pip install -r requirements.txt
echo "Requirements successfully installed."
deactivate
echo "Creating folders..."
mkdir models .roi_tiles .preds
echo "Folders successfully created."

echo 
echo "Setup completed successfully. Closing in 5 seconds..."
sleep 5
exit 0
