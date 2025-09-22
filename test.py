import os
import logging
from backend import CVMatcherBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cv_matcher():
    """Test the CV Matcher functionality"""
    logger.info("Starting CV Matcher test...")
    
    # Create test CV files
    test_cvs = [
        ("cv1.txt", "Software Engineer with 5 years of experience in Python and web development"),
        ("cv2.txt", "Data Scientist specializing in machine learning and statistical analysis"),
        ("cv3.txt", "Frontend Developer with expertise in React and JavaScript"),
        ("cv4.txt", "DevOps Engineer with experience in AWS and Docker"),
        ("cv5.txt", "Full-stack Developer with Python and JavaScript skills")
    ]
    
    # Create CV files
    for filename, content in test_cvs:
        with open(f"data/cv/{filename}", "w") as f:
            f.write(content)
    
    # Initialize backend
    backend = CVMatcherBackend()
    if not backend.initialize():
        logger.error("Failed to initialize backend")
        return False
    
    # Test job description
    job_title = "Senior Python Developer"
    job_description = """
    We are looking for a Senior Python Developer with strong experience in web development,
    API design, and database management. The ideal candidate should have experience with
    frameworks like Django or Flask, and be comfortable working in a fast-paced environment.
    """
    
    # Test matching
    matches = backend.match_cvs(job_title, job_description, threshold=25)
    
    logger.info(f"Found {len(matches)} matches:")
    for match in matches:
        logger.info(f"  - {match['file']}: {match['similarity']:.1f}%")
    
    # Clean up test files
    for filename, _ in test_cvs:
        os.remove(f"data/cv/{filename}")
    
    logger.info("Test completed successfully!")
    return True

if __name__ == "__main__":
    test_cv_matcher()
