#!/usr/bin/env python3
"""
CV Matcher Run Script
macOS/Linux Python script to start the CV Matcher application
"""

import os
import sys
import subprocess

def run_command(command, shell=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("Starting CV Matcher App...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create cv_data directory if it doesn't exist
    os.makedirs("cv_data", exist_ok=True)
    
    # Check if virtual environment exists
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        if not run_command([sys.executable, "-m", "venv", venv_path], shell=False):
            print("Failed to create virtual environment")
            sys.exit(1)
    
    # Set paths for macOS/Linux
    activate_script = os.path.join(venv_path, "bin", "activate")
    pip_command = os.path.join(venv_path, "bin", "pip")
    python_command = os.path.join(venv_path, "bin", "python")
    
    # Install dependencies
    print("Installing dependencies...")
    if not run_command([pip_command, "install", "-r", "requirements.txt", "--upgrade"], shell=False):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Fix huggingface_hub version compatibility
    print("Fixing dependency compatibility...")
    run_command([pip_command, "install", "huggingface_hub==0.16.4", "--force-reinstall"], shell=False)
    
    # Run the application
    print("Starting Flask application...")
    print("🔄 Vectors will be regenerated on every startup")
    print("Open your browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([python_command, "ui.py"], check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
