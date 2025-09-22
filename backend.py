import logging
import os
from typing import List, Dict, Optional, Any
from vectorizer import CVVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVMatcherBackend:
    """CV Matching Backend - Main interface for CV matching operations."""
    
    def __init__(self) -> None:
        """Initialize the CV Matcher Backend."""
        self.vectorizer = CVVectorizer()
        
    def initialize(self) -> bool:
        """Initialize the backend by loading CVs.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            logger.info("Initializing CV Matcher Backend...")
            success = self.vectorizer.load_cvs('data/cv')
            if success:
                logger.info("Backend initialization completed successfully")
            else:
                logger.error("Backend initialization failed")
            return success
        except Exception as e:
            logger.error(f"Error during backend initialization: {e}")
            return False
        
    def get_available_positions(self) -> List[Dict[str, str]]:
        """Get list of available positions from data/positions directory.
        
        Returns:
            List[Dict[str, str]]: List of position dictionaries with title, filename, and path.
        """
        positions_dir = 'data/positions'
        positions = []
        
        try:
            if not os.path.exists(positions_dir):
                logger.warning(f"Positions directory {positions_dir} not found")
                return positions
                
            for filename in os.listdir(positions_dir):
                if filename.endswith(('.txt', '.docx', '.doc')):
                    # Extract job title from filename (remove extension and position_ prefix)
                    job_title = os.path.splitext(filename)[0]
                    if job_title.startswith('position_'):
                        job_title = job_title[9:]  # Remove 'position_' prefix
                    
                    positions.append({
                        'title': job_title,
                        'filename': filename,
                        'path': os.path.join(positions_dir, filename)
                    })
            
            logger.info(f"Found {len(positions)} available positions")
            return sorted(positions, key=lambda x: x['title'])
            
        except Exception as e:
            logger.error(f"Error getting available positions: {e}")
            return []
    
    def get_position_description(self, position_path: str) -> str:
        """Get job description from position file.
        
        Args:
            position_path (str): Path to the position file.
            
        Returns:
            str: Job description content or empty string if error.
        """
        try:
            if not os.path.exists(position_path):
                logger.error(f"Position file {position_path} not found")
                return ""
                
            if position_path.endswith('.txt'):
                with open(position_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.debug(f"Successfully read position file: {position_path}")
                    return content
            elif position_path.endswith(('.docx', '.doc')):
                logger.warning(f"Word document {position_path} - text extraction not implemented")
                return "Job description from Word document (text extraction needed)"
            else:
                logger.warning(f"Unsupported file format: {position_path}")
                return ""
                
        except Exception as e:
            logger.error(f"Error reading position file {position_path}: {e}")
            return ""
        
    def match_cvs(self, job_title: str, job_description: str, threshold: int = 25) -> List[Dict[str, Any]]:
        """Match CVs against job description.
        
        Args:
            job_title (str): The job title to match against.
            job_description (str): The job description to match against.
            threshold (int): Similarity threshold percentage (0-100).
            
        Returns:
            List[Dict[str, Any]]: List of matching CVs with scores and metadata.
        """
        try:
            logger.info(f"Matching CVs for job: {job_title} (threshold: {threshold}%)")
            
            if not job_title.strip() or not job_description.strip():
                logger.warning("Empty job title or description provided")
                return []
            
            # Combine title and description for better matching
            full_description = f"{job_title}\n\n{job_description}"
            
            # Find matches
            matches = self.vectorizer.find_matches(full_description, threshold/100)
            
            logger.info(f"Found {len(matches)} matches above {threshold}% threshold")
            return matches
            
        except Exception as e:
            logger.error(f"Error during CV matching: {e}")
            return []
