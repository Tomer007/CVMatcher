#!/bin/bash

# CV Matcher Run Script

echo "Starting CV Matcher App..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create cv_data directory if it doesn't exist
mkdir -p cv_data

# Run the application
echo "Starting Flask application..."
echo "Open your browser and go to: http://localhost:5000"
python ui.py
