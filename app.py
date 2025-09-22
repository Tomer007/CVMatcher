#!/usr/bin/env python3
"""
CV Matcher Production Application
Optimized for Render deployment
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
import time
from backend import CVMatcherBackend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
backend = CVMatcherBackend()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/positions', methods=['GET'])
def get_positions():
    """Get available positions"""
    try:
        positions = backend.get_available_positions()
        return jsonify({'positions': positions})
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/position/<path:filename>', methods=['GET'])
def get_position_description(filename):
    """Get job description for a specific position"""
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
        return jsonify({'error': str(e)}), 500

@app.route('/match', methods=['POST'])
def match_cvs():
    try:
        start_time = time.time()
        data = request.get_json()
        job_title = data.get('job_title', '')
        job_description = data.get('job_description', '')
        threshold = data.get('threshold', 25)
        
        if not job_title or not job_description:
            return jsonify({'error': 'Job title and description are required'}), 400
            
        matches = backend.match_cvs(job_title, job_description, threshold)
        end_time = time.time()
        latency = round((end_time - start_time) * 1000, 2)
        
        return jsonify({
            'matches': matches,
            'count': len(matches),
            'latency_ms': latency
        })
        
    except Exception as e:
        logger.error(f"Error matching CVs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_cv/<path:filename>')
def view_cv(filename):
    """View CV content by filename"""
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
        
        # Read CV content
        cv_content = backend.vectorizer._read_cv(cv_path)
        
        # Create HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CV: {filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .back-btn {{ background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
                .cv-content {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 20px; white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <h1>CV: {filename}</h1>
            <a href="/" class="back-btn">← Back to CV Matcher</a>
            <div class="cv-content">{cv_content}</div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error viewing CV {filename}: {str(e)}")
        return f"Error loading CV: {str(e)}", 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'service': 'cv-matcher',
        'version': '1.0.0'
    })

def initialize_app():
    """Initialize the application with CV data"""
    logger.info("🚀 Initializing CV Matcher Application...")
    
    # Initialize backend
    if not backend.initialize():
        logger.error("❌ Failed to initialize backend")
        return False
    
    logger.info("✅ CV Matcher Application initialized successfully")
    return True

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        # Get port from environment variable (Render sets this)
        port = int(os.environ.get('PORT', 5001))
        
        logger.info(f"🌐 Starting CV Matcher on port {port}")
        logger.info(f"📊 CV Database: {len(backend.vectorizer.cv_files)} CVs loaded")
        logger.info(f"🎯 Positions: {len(backend.get_available_positions())} positions available")
        
        # Run the application
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("❌ Failed to start application")
        exit(1)

