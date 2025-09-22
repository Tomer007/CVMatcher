# CV Matcher App

A simple Python application that matches job descriptions against a collection of CVs using semantic similarity.

## Features

- Web-based UI for easy interaction
- Semantic matching using sentence transformers
- Configurable similarity threshold
- Support for both text and PDF CVs
- Cached embeddings for faster subsequent searches

## Setup

### Quick Start (Recommended)

**Option 1: Using the Python run script (macOS/Linux)**
```bash
python3 run.py
```

**Option 2: Using shell script**
```bash
./run.sh
```

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your CV files in the `cv_data/` directory (supports .txt and .pdf files)

3. Run the application:
```bash
python ui.py
```

4. Open your browser and go to `http://localhost:5001`

## Usage

1. Enter a job title and description
2. Adjust the similarity threshold (default: 25%)
3. Click "Match CVs" to find matching CVs
4. View results sorted by similarity score

## Testing

Run the test script to verify everything is working:
```bash
python3 test.py
```

## Project Structure

**Core Application:**
- `ui.py` - Flask web application (main entry point)
- `backend.py` - Core matching logic
- `vectorizer.py` - Text vectorization and similarity calculation
- `templates/index.html` - Web UI

**Setup & Scripts:**
- `run.py` - Python run script (macOS/Linux)
- `run.sh` - Shell script (macOS/Linux)
- `requirements.txt` - Python dependencies

**Testing:**
- `test.py` - Test script with sample data

**Data:**
- `cv_data/` - Directory for CV files (supports .txt, .pdf, .doc, .docx)
- `cv_embeddings.pkl` - Cached CV embeddings (auto-generated, always regenerated)
