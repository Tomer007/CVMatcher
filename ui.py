from flask import Flask, render_template, request, jsonify
import logging
import os
import time
from typing import Dict, Any, List
from backend import CVMatcherBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
backend = CVMatcherBackend()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_cvs() -> tuple:
    """Match CVs against job description.
    
    Returns:
        tuple: JSON response and HTTP status code.
    """
    try:
        start_time = time.time()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        job_title = data.get('job_title', '').strip()
        job_description = data.get('job_description', '').strip()
        threshold = data.get('threshold', 25)
        
        # Validate input
        if not job_title:
            return jsonify({'error': 'Job title is required'}), 400
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 100:
            return jsonify({'error': 'Threshold must be between 0 and 100'}), 400
            
        logger.info(f"Processing match request for: {job_title}")
        matches = backend.match_cvs(job_title, job_description, int(threshold))
        
        end_time = time.time()
        latency = round((end_time - start_time) * 1000, 2)
        
        return jsonify({
            'matches': matches,
            'count': len(matches),
            'latency_ms': latency
        })
        
    except Exception as e:
        logger.error(f"Error matching CVs: {str(e)}")
        return jsonify({'error': 'Internal server error occurred during matching'}), 500

@app.route('/positions', methods=['GET'])
def get_positions() -> tuple:
    """Get available positions.
    
    Returns:
        tuple: JSON response and HTTP status code.
    """
    try:
        positions = backend.get_available_positions()
        return jsonify({'positions': positions})
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return jsonify({'error': 'Failed to retrieve positions'}), 500

@app.route('/position/<path:filename>', methods=['GET'])
def get_position_description(filename: str) -> tuple:
    """Get job description for a specific position.
    
    Args:
        filename (str): Position filename.
        
    Returns:
        tuple: JSON response and HTTP status code.
    """
    try:
        positions = backend.get_available_positions()
        position = next((p for p in positions if p['filename'] == filename), None)
        
        if not position:
            return jsonify({'error': 'Position not found'}), 404
            
        description = backend.get_position_description(position['path'])
        return jsonify({
            'title': position['title'],
            'description': description
        })
    except Exception as e:
        logger.error(f"Error getting position description: {str(e)}")
        return jsonify({'error': 'Failed to retrieve position description'}), 500

@app.route('/view_cv/<path:filename>')
def view_cv(filename: str) -> tuple:
    """View CV content by filename.
    
    Args:
        filename (str): CV filename to view.
        
    Returns:
        tuple: HTML content and HTTP status code.
    """
    try:
        # Get source parameter to determine back navigation
        source = request.args.get('source', 'home')
        # Find the full path of the CV file
        cv_path = None
        for cv_file in backend.vectorizer.cv_files:
            if os.path.basename(cv_file) == filename:
                cv_path = cv_file
                break
        
        if not cv_path or not os.path.exists(cv_path):
            return f"CV file '{filename}' not found", 404
        
        # Read the CV content based on file type
        if cv_path.lower().endswith('.pdf'):
            # For PDF files, we need to extract text
            try:
                import PyPDF2
                with open(cv_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
            except Exception as e:
                content = f"Error reading PDF: {str(e)}"
        elif cv_path.lower().endswith(('.doc', '.docx')):
            # For Word documents
            try:
                from docx import Document
                doc = Document(cv_path)
                content = ""
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\n"
            except ImportError:
                content = "Error: python-docx not installed. Cannot read Word documents."
            except Exception as e:
                content = f"Error reading Word document: {str(e)}"
        else:
            # For text files - try different encodings
            content = ""
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(cv_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    content = f"Error reading file: {str(e)}"
                    break
            
            if not content:
                content = "Error: Could not decode file with any supported encoding."
        
        # Return as HTML with basic styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CV: {filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .content {{ white-space: pre-wrap; }}
                .back-btn {{ 
                    background: #007bff; 
                    color: white; 
                    padding: 10px 20px; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    display: inline-block;
                    margin-right: 10px;
                }}
                .back-btn:hover {{
                    background: #0056b3;
                    color: white;
                    text-decoration: none;
                }}
                .nav-buttons {{
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>CV: {filename}</h2>
                <div class="nav-buttons">
                    <a href="javascript:history.back()" class="back-btn" id="backBtn">← Back to Results</a>
                    <a href="/" class="back-btn">🏠 Home</a>
                    <span style="color: #666; font-size: 14px; margin-left: 15px;">
                        💡 Press <kbd>ESC</kbd> to go back
                    </span>
                </div>
            </div>
            <div class="content">{content}</div>
            
            <script>
                // Enhanced back button functionality
                document.getElementById('backBtn').addEventListener('click', function(e) {{
                    e.preventDefault();
                    
                    // Check if we came from results page
                    const urlParams = new URLSearchParams(window.location.search);
                    const source = urlParams.get('source');
                    
                    if (source === 'results' && window.opener) {{
                        // Close this tab and focus on the opener
                        window.close();
                    }} else if (window.history.length > 1) {{
                        // Try browser back
                        window.history.back();
                    }} else {{
                        // Fallback to home page
                        window.location.href = '/';
                    }}
                }});
                
                // Also support browser back button
                window.addEventListener('popstate', function(e) {{
                    // This will be triggered by browser back button
                }});
                
                // Handle keyboard shortcuts
                document.addEventListener('keydown', function(e) {{
                    if (e.key === 'Escape') {{
                        // ESC key to go back
                        document.getElementById('backBtn').click();
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return html_content
        
    except Exception as e:
        logger.error(f"Error viewing CV {filename}: {str(e)}")
        return f"Error loading CV: {str(e)}", 500

if __name__ == '__main__':
    # Initialize backend with force regeneration
    logger.info("🔄 Regenerating CV vectors on startup...")
    if not backend.initialize():
        logger.error("Failed to initialize backend")
        exit(1)
    
    # Print vector contents
    logger.info("📊 Vector Contents Summary:")
    logger.info(f"   • Total CVs processed: {len(backend.vectorizer.cv_files)}")
    logger.info(f"   • Embedding dimensions: {backend.vectorizer.embeddings.shape[1] if backend.vectorizer.embeddings is not None else 'N/A'}")
    logger.info(f"   • Memory usage: {backend.vectorizer.embeddings.nbytes if backend.vectorizer.embeddings is not None else 'N/A':,} bytes")
    
    # List all CV files
    logger.info("📄 CV Files loaded:")
    for i, cv_file in enumerate(backend.vectorizer.cv_files, 1):
        filename = os.path.basename(cv_file)
        logger.info(f"   {i:2d}. {filename}")
    
    logger.info("🚀 Starting Flask application on http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)
