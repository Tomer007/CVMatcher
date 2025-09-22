import os
import pickle
import logging
import re
from typing import List, Dict, Set, Optional, Any, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InMemoryCVDatabase:
    """Lightweight in-memory database for fast CV queries."""
    
    def __init__(self) -> None:
        """Initialize the in-memory CV database."""
        self.cvs: Dict[int, Dict[str, Any]] = {}  # {cv_id: cv_data}
        self.skill_index: Dict[str, Set[int]] = {}  # {skill: set(cv_ids)}
        self.level_index: Dict[str, Set[int]] = {}  # {level: set(cv_ids)}
        self.embeddings: Dict[int, np.ndarray] = {}  # {cv_id: embedding}
        self.next_id: int = 1
    
    def add_cv(self, cv_data: Dict[str, Any]) -> int:
        """Add CV to in-memory database.
        
        Args:
            cv_data (Dict[str, Any]): CV data dictionary.
            
        Returns:
            int: The assigned CV ID.
        """
        cv_id = self.next_id
        self.next_id += 1
        
        # Store CV data
        self.cvs[cv_id] = cv_data
        
        # Index by skills
        for skill in cv_data.get('skills', []):
            if skill not in self.skill_index:
                self.skill_index[skill] = set()
            self.skill_index[skill].add(cv_id)
        
        # Index by level
        level = cv_data.get('level', 'unknown')
        if level not in self.level_index:
            self.level_index[level] = set()
        self.level_index[level].add(cv_id)
        
        # Store embedding
        self.embeddings[cv_id] = cv_data['skill_embedding']
        
        return cv_id
    
    def query_by_skills(self, skills: Set[str]) -> Set[int]:
        """Query CVs by skills - FAST.
        
        Args:
            skills (Set[str]): Set of skills to search for.
            
        Returns:
            Set[int]: Set of CV IDs that match the skills.
        """
        if not skills:
            return set(self.cvs.keys())
        
        # Find CVs that have any of the required skills
        matching_cvs = set()
        for skill in skills:
            if skill in self.skill_index:
                matching_cvs.update(self.skill_index[skill])
        
        return matching_cvs
    
    def query_by_level(self, level: str) -> Set[int]:
        """Query CVs by experience level - FAST.
        
        Args:
            level (str): Experience level to search for.
            
        Returns:
            Set[int]: Set of CV IDs that match the level.
        """
        if level == 'unknown':
            return set(self.cvs.keys())
        
        return self.level_index.get(level, set())
    
    def get_cv_data(self, cv_id: int) -> Optional[Dict[str, Any]]:
        """Get CV data by ID.
        
        Args:
            cv_id (int): CV ID to retrieve.
            
        Returns:
            Optional[Dict[str, Any]]: CV data or None if not found.
        """
        return self.cvs.get(cv_id)
    
    def get_embedding(self, cv_id: int) -> Optional[np.ndarray]:
        """Get CV embedding by ID.
        
        Args:
            cv_id (int): CV ID to retrieve.
            
        Returns:
            Optional[np.ndarray]: CV embedding or None if not found.
        """
        return self.embeddings.get(cv_id)
    
    def get_all_cv_ids(self) -> List[int]:
        """Get all CV IDs.
        
        Returns:
            List[int]: List of all CV IDs.
        """
        return list(self.cvs.keys())

class CVVectorizer:
    """CV Vectorizer for semantic matching using Sentence Transformers."""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> None:
        """Initialize the CV Vectorizer.
        
        Args:
            model_name (str): Name of the Sentence Transformer model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded Sentence Transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        self.embeddings: Optional[np.ndarray] = None
        self.cv_files: List[str] = []
        self.embeddings_file = 'cv_embeddings.pkl'
        self.db = InMemoryCVDatabase()  # In-memory database
        self.is_preprocessed = False
        
    def load_cvs(self, cv_dir: str = 'cv_data') -> bool:
        """Load and preprocess CVs into in-memory database.
        
        Args:
            cv_dir (str): Directory containing CV files.
            
        Returns:
            bool: True if loading successful, False otherwise.
        """
        try:
            logger.info("Loading CVs into in-memory database...")
            
            if not os.path.exists(cv_dir):
                logger.error(f"CV directory {cv_dir} not found")
                return False
                
            # Find all CV files
            self.cv_files = self._find_cv_files(cv_dir)
            logger.info(f"Found {len(self.cv_files)} CV files")
            
            if not self.cv_files:
                logger.warning("No CV files found in directory")
                return False
            
            # Remove existing embeddings for regeneration
            if os.path.exists(self.embeddings_file):
                logger.info("Removing existing embeddings for regeneration...")
                os.remove(self.embeddings_file)
            
            # Process CVs in batches
            batch_size = 50
            total_batches = (len(self.cv_files) + batch_size - 1) // batch_size
            
            for i in range(0, len(self.cv_files), batch_size):
                batch = self.cv_files[i:i + batch_size]
                self._process_cv_batch(batch)
                logger.info(f"Processed batch {i//batch_size + 1}/{total_batches}")
            
            # Create embeddings array for compatibility
            self._create_embeddings_array()
            
            self.is_preprocessed = True
            logger.info(f"Successfully loaded {len(self.cv_files)} CVs into in-memory database")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CVs: {e}")
            return False
    
    def _find_cv_files(self, cv_dir: str) -> List[str]:
        """Find all CV files recursively, excluding position files.
        
        Args:
            cv_dir (str): Directory to search for CV files.
            
        Returns:
            List[str]: List of CV file paths.
        """
        cv_files = []
        try:
            for root, dirs, files in os.walk(cv_dir):
                for file in files:
                    if file.lower().endswith(('.txt', '.pdf', '.doc', '.docx')):
                        # Exclude files with position_ prefix
                        if not file.startswith('position_'):
                            cv_files.append(os.path.join(root, file))
        except Exception as e:
            logger.error(f"Error finding CV files in {cv_dir}: {e}")
        
        return cv_files
    
    def _process_cv_batch(self, cv_files: List[str]) -> None:
        """Process a batch of CVs.
        
        Args:
            cv_files (List[str]): List of CV file paths to process.
        """
        for cv_file in cv_files:
            try:
                # Extract text
                text = self._read_cv(cv_file)
                if not text.strip():
                    logger.warning(f"Empty text extracted from {cv_file}")
                    continue
                
                # Extract skills using AI
                skills = self._extract_skills_ai(text)
                
                # Extract experience level
                level = self._extract_experience_level(text)
                
                # Create skill embedding
                skill_embedding = self.model.encode([text])
                
                # Extract sections
                sections = self._extract_cv_sections(text)
                
                # Add to database
                cv_data = {
                    'file_path': cv_file,
                    'text': text,
                    'skills': list(skills),
                    'level': level,
                    'skill_embedding': skill_embedding[0],
                    'sections': sections,
                    'processed_at': datetime.now()
                }
                
                self.db.add_cv(cv_data)
                logger.debug(f"Successfully processed CV: {os.path.basename(cv_file)}")
                
            except Exception as e:
                logger.error(f"Error processing {cv_file}: {e}")
    
    def _extract_skills_ai(self, text):
        """Extract skills using AI semantic matching"""
        # Use existing skill patterns but with better normalization
        skills = self._extract_skills(text)
        
        # Add semantic skill extraction
        semantic_skills = self._extract_semantic_skills(text)
        
        # Combine and normalize
        all_skills = skills.union(semantic_skills)
        return self._normalize_skills(all_skills)
    
    def _extract_semantic_skills(self, text):
        """Extract skills using semantic similarity"""
        # Common skill keywords
        skill_keywords = [
            'programming', 'coding', 'development', 'software', 'web', 'mobile',
            'database', 'cloud', 'devops', 'frontend', 'backend', 'fullstack',
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
            'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure', 'docker'
        ]
        
        # Find semantically similar words
        semantic_skills = set()
        for keyword in skill_keywords:
            if self._is_semantically_similar(text, keyword):
                semantic_skills.add(keyword.upper())
        
        return semantic_skills
    
    def _is_semantically_similar(self, text, keyword):
        """Check if text contains semantically similar words to keyword"""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Check for exact match or common variations
        if keyword_lower in text_lower:
            return True
        
        # Check for common variations
        variations = {
            'programming': ['coding', 'development', 'software'],
            'javascript': ['js', 'ecmascript'],
            'python': ['py', 'python3'],
            'database': ['db', 'data storage'],
            'cloud': ['aws', 'azure', 'gcp']
        }
        
        if keyword_lower in variations:
            for variation in variations[keyword_lower]:
                if variation in text_lower:
                    return True
        
        return False
    
    def _normalize_skills(self, skills):
        """Normalize skills for better matching"""
        normalized = set()
        
        for skill in skills:
            skill_upper = skill.upper()
            
            # Normalize common variations
            if skill_upper in ['HTML5', 'HTML']:
                normalized.add('HTML')
            elif skill_upper in ['JAVASCRIPT', 'JS']:
                normalized.add('JAVASCRIPT')
            elif skill_upper in ['TYPESCRIPT', 'TS']:
                normalized.add('TYPESCRIPT')
            elif skill_upper in ['SQL', 'MYSQL', 'POSTGRESQL']:
                normalized.add('SQL')
            elif skill_upper in ['CSS3', 'CSS']:
                normalized.add('CSS')
            else:
                normalized.add(skill_upper)
        
        return normalized
    
    def _create_embeddings_array(self):
        """Create embeddings array for compatibility"""
        if not self.db.cvs:
            return
        
        embeddings_list = []
        for cv_id in sorted(self.db.cvs.keys()):
            embeddings_list.append(self.db.get_embedding(cv_id))
        
        self.embeddings = np.array(embeddings_list)
        
    def _read_cv(self, file_path):
        """Read CV content from file"""
        try:
            if file_path.lower().endswith('.pdf'):
                import PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            elif file_path.lower().endswith(('.doc', '.docx')):
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.warning("python-docx not installed, skipping Word document")
                    return ""
            else:
                # For text files - try different encodings
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            return file.read()
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error reading {file_path} with {encoding}: {str(e)}")
                        continue
                
                logger.error(f"Could not decode {file_path} with any supported encoding")
                return ""
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return ""
            
    def find_matches(self, job_description: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Find CV matches using in-memory database - FAST.
        
        Args:
            job_description (str): Job description to match against.
            threshold (float): Similarity threshold (0.0 to 1.0).
            
        Returns:
            List[Dict[str, Any]]: List of matching CVs with scores and metadata.
        """
        if not self.is_preprocessed:
            logger.error("CVs not preprocessed yet")
            return []
        
        try:
            # Extract job requirements
            job_skills = self._extract_skills_ai(job_description)
            job_level = self._extract_experience_level(job_description)
            
            # Query database for skill matches
            skill_matches = self.db.query_by_skills(job_skills)
            
            # Filter by level if specified
            if job_level != 'unknown':
                level_matches = self.db.query_by_level(job_level)
                skill_matches = skill_matches.intersection(level_matches)
            
            # Calculate semantic similarity for filtered CVs
            matches = []
            job_embedding = self.model.encode([job_description])
            
            for cv_id in skill_matches:
                cv_data = self.db.get_cv_data(cv_id)
                cv_embedding = self.db.get_embedding(cv_id)
                
                if cv_data is None or cv_embedding is None:
                    continue
                
                # Calculate similarity
                similarity = cosine_similarity(job_embedding, [cv_embedding])[0][0]
                
                if similarity > threshold:
                    # Calculate skill match
                    cv_skills = set(cv_data.get('skills', []))
                    skill_score = self._calculate_skill_match(job_skills, cv_skills)
                    
                    # Calculate level match
                    level_score = self._calculate_level_match(job_level, cv_data.get('level', 'unknown'))
                    
                    # Combined score
                    combined_score = (similarity * 0.6) + (skill_score * 0.4) + (level_score * 0.1)
                    
                    # Only include matches with skill score >= 5% and combined score > threshold
                    if combined_score > threshold and skill_score >= 0.05:  # 5% skill threshold
                        matches.append({
                            'file': os.path.basename(cv_data['file_path']),
                            'similarity': float(combined_score * 100),
                            'semantic_score': float(similarity * 100),
                            'skill_score': float(skill_score * 100),
                            'level_score': float(level_score * 100),
                            'matched_skills': list(job_skills.intersection(cv_skills)),
                            'cv_id': cv_id
                        })
            
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top 5 matches only
            top_matches = matches[:5]
            
            logger.info(f"Found {len(matches)} matches above threshold, returning top {len(top_matches)}")
            return top_matches
            
        except Exception as e:
            logger.error(f"Error during CV matching: {e}")
            return []
    
    def _extract_skills(self, text):
        """Extract technical skills from text with improved matching"""
        import re
        
        # Common technical skills and technologies
        skill_patterns = [
            r'\b(?:HTML5?|HTML|CSS3?|CSS|JavaScript|JS|TypeScript|TS)\b',
            r'\b(?:Python|Java|C#|C\+\+|PHP|Ruby|Go|Rust|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue\.?js|Node\.?js|Express|Django|Flask|Laravel|Spring|ASP\.NET)\b',
            r'\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Oracle|SQLite)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|GitHub|GitLab)\b',
            r'\b(?:Bootstrap|jQuery|SASS|SCSS|Webpack|Babel|NPM|Yarn)\b',
            r'\b(?:\.NET|\.NET Core|WebAPI|REST|GraphQL|Microservices)\b',
            r'\b(?:Machine Learning|ML|AI|Data Science|Analytics|TensorFlow|PyTorch)\b',
            r'\b(?:DevOps|CI/CD|Agile|Scrum|TDD|BDD)\b'
        ]
        
        skills = set()
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            skills.update([match.upper() for match in matches])
        
        # Add normalized skills for better matching
        normalized_skills = set()
        for skill in skills:
            # Normalize HTML variations
            if skill in ['HTML', 'HTML5']:
                normalized_skills.add('HTML')
            # Normalize JavaScript variations
            elif skill in ['JAVASCRIPT', 'JS']:
                normalized_skills.add('JAVASCRIPT')
            # Normalize SQL variations
            elif skill in ['SQL', 'MYSQL']:
                normalized_skills.add('SQL')
            # Keep other skills as is
            else:
                normalized_skills.add(skill)
        
        return normalized_skills
    
    def _extract_experience_level(self, text):
        """Extract experience level from text"""
        text_lower = text.lower()
        
        # Junior indicators
        if any(word in text_lower for word in ['junior', 'entry', 'graduate', 'intern', 'trainee', 'beginner', 'new']):
            return 'junior'
        
        # Senior indicators
        if any(word in text_lower for word in ['senior', 'lead', 'principal', 'architect', 'expert', 'advanced', '5+', '10+']):
            return 'senior'
        
        # Mid-level indicators
        if any(word in text_lower for word in ['mid', 'intermediate', '3+', '4+', '5+']):
            return 'mid'
        
        return 'unknown'
    
    def _calculate_skill_match(self, job_skills, cv_skills):
        """Calculate skill matching score"""
        if not job_skills:
            return 0.5  # Neutral score if no skills specified
        
        if not cv_skills:
            return 0.0  # No skills in CV
        
        # Calculate Jaccard similarity
        intersection = len(job_skills.intersection(cv_skills))
        union = len(job_skills.union(cv_skills))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_level_match(self, job_level, cv_level):
        """Calculate experience level matching score"""
        if job_level == 'unknown' or cv_level == 'unknown':
            return 0.5  # Neutral score
        
        if job_level == cv_level:
            return 1.0  # Perfect match
        
        # Some flexibility in level matching
        if (job_level == 'junior' and cv_level == 'mid') or \
           (job_level == 'mid' and cv_level in ['junior', 'senior']) or \
           (job_level == 'senior' and cv_level == 'mid'):
            return 0.7  # Good match
        
        return 0.3  # Poor match
    
    def _extract_job_sections(self, job_description):
        """Extract structured sections from job description"""
        import re
        
        text = job_description.lower()
        sections = {
            'requirements': '',
            'skills': '',
            'education': '',
            'experience': '',
            'responsibilities': '',
            'overview': ''
        }
        
        # Extract requirements section
        req_patterns = [
            r'what do you need to succeed[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'requirements[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'qualifications[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'you need[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in req_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections['requirements'] = match.group(1).strip()
                break
        
        # Extract skills from bullet points
        skill_pattern = r'[●•\-\*]\s*([^●•\-\*\n]+(?:html5|c#|javascript|sql|mysql|bootstrap|webapi|\.net)[^●•\-\*\n]*)'
        skill_matches = re.findall(skill_pattern, text, re.IGNORECASE)
        if skill_matches:
            sections['skills'] = ' '.join(skill_matches)
        
        # Extract education requirements
        edu_patterns = [
            r'(?:bachelor|degree|graduate|diploma|certificate)[^●•\-\*\n]*',
            r'(?:computer science|computer engineering|full stack)[^●•\-\*\n]*'
        ]
        edu_matches = []
        for pattern in edu_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            edu_matches.extend(matches)
        if edu_matches:
            sections['education'] = ' '.join(edu_matches)
        
        # Extract experience level
        exp_patterns = [
            r'(?:junior|entry|graduate|intern|trainee|beginner|new)[^●•\-\*\n]*',
            r'(?:senior|lead|principal|architect|expert|advanced)[^●•\-\*\n]*',
            r'(?:years? of experience|experience level)[^●•\-\*\n]*'
        ]
        exp_matches = []
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            exp_matches.extend(matches)
        if exp_matches:
            sections['experience'] = ' '.join(exp_matches)
        
        # Extract responsibilities
        resp_patterns = [
            r'(?:responsibilities|duties|what you will do)[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'(?:you will|you\'ll)[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in resp_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections['responsibilities'] = match.group(1).strip()
                break
        
        # Overview is the general description
        sections['overview'] = job_description
        
        return sections
    
    def _extract_cv_sections(self, cv_content):
        """Extract structured sections from CV content"""
        import re
        
        text = cv_content.lower()
        sections = {
            'skills': '',
            'education': '',
            'experience': '',
            'summary': '',
            'overview': ''
        }
        
        # Extract skills section
        skill_patterns = [
            r'(?:skills|technologies|programming languages|tools)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]|$)',
            r'(?:technical skills|core competencies)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]|$)'
        ]
        
        for pattern in skill_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections['skills'] = match.group(1).strip()
                break
        
        # Extract education section
        edu_patterns = [
            r'(?:education|academic background|qualifications)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]|$)',
            r'(?:university|college|degree|bachelor|master|phd)[^●•\-\*\n]*'
        ]
        
        for pattern in edu_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                if match.groups():
                    sections['education'] = match.group(1).strip()
                else:
                    sections['education'] = match.group(0).strip()
                break
        
        # Extract experience section
        exp_patterns = [
            r'(?:experience|work history|employment|professional experience)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]|$)',
            r'(?:years? of experience|experience in)[^●•\-\*\n]*'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                if match.groups():
                    sections['experience'] = match.group(1).strip()
                else:
                    sections['experience'] = match.group(0).strip()
                break
        
        # Extract summary/objective
        summary_patterns = [
            r'(?:summary|objective|profile|about)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]|$)',
            r'(?:professional summary|career objective)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]|$)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                if match.groups():
                    sections['summary'] = match.group(1).strip()
                else:
                    sections['summary'] = match.group(0).strip()
                break
        
        # Overview is the entire CV content
        sections['overview'] = cv_content
        
        return sections
    
    def _create_job_embeddings(self, job_sections):
        """Create embeddings for each job section"""
        embeddings = {}
        
        for section_name, section_content in job_sections.items():
            if section_content.strip():
                embeddings[section_name] = self.model.encode([section_content])
            else:
                embeddings[section_name] = None
        
        return embeddings
    
    def _calculate_section_scores(self, job_embeddings, cv_sections):
        """Calculate semantic similarity scores for each section"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        scores = {}
        section_weights = {
            'requirements': 0.25,
            'skills': 0.25,
            'education': 0.15,
            'experience': 0.20,
            'responsibilities': 0.10,
            'overview': 0.05
        }
        
        weighted_score = 0
        total_weight = 0
        
        for section_name, job_embedding in job_embeddings.items():
            if job_embedding is not None and section_name in cv_sections:
                cv_content = cv_sections[section_name]
                if cv_content.strip():
                    # Create embedding for CV section
                    cv_embedding = self.model.encode([cv_content])
                    
                    # Calculate similarity
                    similarity = cosine_similarity(job_embedding, cv_embedding)[0][0]
                    scores[section_name] = similarity
                    
                    # Add to weighted score
                    weight = section_weights.get(section_name, 0.05)
                    weighted_score += similarity * weight
                    total_weight += weight
                else:
                    scores[section_name] = 0.0
            else:
                scores[section_name] = 0.0
        
        # Calculate overall weighted score
        scores['overall'] = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return scores
    
    def _calculate_combined_score(self, section_scores, skill_score, level_score):
        """Calculate final combined score with weights"""
        # Weights: 50% semantic sections, 30% skills, 20% level
        semantic_weight = 0.5
        skill_weight = 0.3
        level_weight = 0.2
        
        combined = (
            section_scores['overall'] * semantic_weight +
            skill_score * skill_weight +
            level_score * level_weight
        )
        
        return combined
