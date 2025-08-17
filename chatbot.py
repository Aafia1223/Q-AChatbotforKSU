import json
import nltk
import spacy
import re
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')

class UniversityChatbot():
    def __init__(self, json_file_path):
        """Initialize the chatbot with university data"""
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize Sentence Transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"⚠️  Error loading sentence transformer: {e}")
            try:
                self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
            except:
                try:
                    self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                except:
                    print("❌ Could not load any sentence transformer model")
                    self.sentence_model = None
        
        # Initialize local LLM for better response generation (optional)
        try:
            self.generator = pipeline("text-generation", 
                                    model="distilgpt2", 
                                    tokenizer="distilgpt2",
                                    max_length=100,
                                    temperature=0.7,
                                    do_sample=True,
                                    pad_token_id=50256)
        except Exception as e:
            print(f"⚠️  Could not load language model: {e}")
            self.generator = None
        
        # Load university data
        self.data = self.load_data(json_file_path)
        
        # Prepare semantic corpus
        self.prepare_semantic_corpus()
        
        # Enhanced intent patterns with semantic variations
        self.intent_patterns = {
            'admission_requirements': [
                'admission requirements', 'how to apply', 'application process', 'entrance requirements',
                'eligibility criteria', 'prerequisites for admission', 'admission criteria',
                'what do I need to apply', 'application requirements', 'how can I get admitted',
                'admission guidelines', 'entry requirements', 'application procedure'
            ],
            'academic_calendar': [
                'academic calendar', 'semester schedule', 'academic year dates', 'term dates',
                'when does semester start', 'exam schedule', 'registration dates',
                'academic timeline', 'semester timeline', 'course schedule', 'class schedule',
                'important dates', 'academic deadlines', 'holiday schedule'
            ],
            'degree_programs': [
                'degree programs', 'available majors', 'courses offered', 'study programs',
                'what can I study', 'academic programs', 'curriculum information',
                'undergraduate programs', 'graduate programs', 'fields of study',
                'departments and colleges', 'course catalog', 'program information', "majors", 
                'bachelors program', 'masters program', 'phd programs'
            ],
            'faculty': [
                'faculty directory', 'professor information', 'faculty members', 'teaching staff',
                'instructor details', 'faculty contacts', 'who teaches what',
                'professor contacts', 'faculty profiles', 'academic staff',
                'department faculty', 'faculty list'
            ],
            'contact_info': [
                'contact information', 'phone numbers', 'email addresses', 'office locations',
                'how to reach', 'contact details', 'department contacts',
                'office hours', 'where to find', 'contact directory'
            ],
            'housing': [
                'housing information', 'dormitory details', 'campus accommodation', 'residence halls',
                'where to live', 'student housing', 'residential facilities',
                'accommodation options', 'campus living', 'housing services'
            ],
            'library': [
                'library information', 'library services', 'study resources', 'library hours',
                'book collection', 'research materials', 'library facilities',
                'study spaces', 'library locations', 'library catalog'
            ],
            'grading': [
                'grading system', 'grade scale', 'how grades work', 'GPA calculation',
                'academic evaluation', 'marking scheme', 'grade distribution',
                'transcript information', 'academic performance'
            ],
            'plagiarism': [
                'plagiarism policy', 'academic integrity', 'cheating policy', 'academic dishonesty',
                'citation requirements', 'academic misconduct', 'integrity guidelines'
            ],
            'attendance': [
                'attendance policy', 'class attendance', 'attendance requirements',
                'absence policy', 'attendance rules', 'missing classes'
            ],
            'research': [
                'research opportunities', 'research labs', 'research facilities', 'research centers',
                'laboratory information', 'research programs', 'research projects'
            ],
            'it_support': [
                'IT support', 'technical help', 'computer problems', 'system issues',
                'help desk', 'technology support', 'login problems', 'password issues',
                'wifi problems', 'software help', 'hardware issues'
            ],
            'fees_tuition': [
                'tuition fees', 'cost of education', 'fee structure', 'payment information',
                'how much does it cost', 'financial information', 'fee payment'
            ],
            'scholarships': [
                'scholarship information', 'financial aid', 'funding opportunities',
                'scholarship applications', 'financial assistance', 'grants available'
            ]
        }
        
        # Create intent embeddings for semantic matching
        if self.sentence_model:
            self.create_intent_embeddings()
        
        # User context to track conversation state
        self.user_context = {'last_intent': None, 'entities': {}}
        self.conversation_state = None
        self.user_type = None
        
    def load_data(self, json_file_path):
        """Load the university data from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"✅ Loaded university data from {json_file_path}")
                return data
        except FileNotFoundError:
            print(f"❌ Error: Could not find {json_file_path}")
            print("Creating sample data structure...")
            return self.create_sample_data()
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in {json_file_path}")
            return {}
    
    def create_sample_data(self): # needs updating
        """Create sample data structure if JSON file is not found"""
        return {
            "admission_requirements": {
                "undergraduate": {
                    "requirements": "High school diploma, SAT/ACT scores, transcripts, essay",
                    "deadline": "January 15th for fall semester",
                    "link": "https://university.edu/admissions/undergraduate"
                },
                "graduate": {
                    "requirements": "Bachelor's degree, GRE scores, recommendation letters",
                    "deadline": "December 1st for fall semester",
                    "link": "https://university.edu/admissions/graduate"
                }
            },
            "academic_calendar": {
                "fall_2024": {
                    "start_date": "August 28, 2024",
                    "end_date": "December 15, 2024",
                    "registration": "August 15-25, 2024"
                }
            },
            "departments": {
                "engineering": {
                    "contact": "engineering@university.edu",
                    "programs": ["Computer Science", "Electrical Engineering", "Mechanical Engineering"]
                }
            },
            "faqs": [
                {
                    "question": "What are the admission requirements?",
                    "answer": "Requirements vary by program. Please check specific program pages."
                }
            ]
        }
    
    def prepare_semantic_corpus(self):
        """Prepare text corpus with semantic embeddings"""
        self.corpus = []
        self.corpus_metadata = []
        self.corpus_embeddings = None
        
        def extract_text_recursive(data, path="", parent_key="", parent_obj=None):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}/{key}" if path else key
                    extract_text_recursive(value, current_path, key, data)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]"
                    extract_text_recursive(item, current_path, parent_key, data)
            elif isinstance(data, str) and len(data.strip()) > 10:
                additional_fields = {}
                if parent_obj and isinstance(parent_obj, dict):
                    for field in ['info', 'url', 'link', 'source', 'reference']:
                        if field in parent_obj and parent_obj[field]:
                            additional_fields[field] = parent_obj[field]
                
                self.corpus.append(data.strip())
                self.corpus_metadata.append({
                    'path': path,
                    'parent_key': parent_key,
                    'content': data.strip(),
                    **additional_fields 
                })
        
        extract_text_recursive(self.data)
        
        if self.corpus and self.sentence_model:
            print(f"🔍 Creating embeddings for {len(self.corpus)} text segments...")
            self.corpus_embeddings = self.sentence_model.encode(self.corpus, convert_to_tensor=True)
            print("✅ Semantic corpus ready")
    
    def create_intent_embeddings(self):
        """Create embeddings for intent patterns for better semantic matching"""
        self.intent_embeddings = {}
        
        for intent, patterns in self.intent_patterns.items():
            # Create embeddings for all patterns of this intent
            pattern_embeddings = self.sentence_model.encode(patterns, convert_to_tensor=True)
            # Use mean embedding as the intent representation
            self.intent_embeddings[intent] = torch.mean(pattern_embeddings, dim=0)
    
    def identify_intent_semantic(self, user_input):
        """Identify user intent using semantic similarity"""
        if not self.sentence_model or not hasattr(self, 'intent_embeddings'):
            return self.identify_intent_fallback(user_input)
        
        user_embedding = self.sentence_model.encode([user_input], convert_to_tensor=True)
        
        best_intent = 'general'
        best_score = 0.0
        
        for intent, intent_embedding in self.intent_embeddings.items():
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(user_embedding, intent_embedding.unsqueeze(0))
            score = similarity.item()
            
            if score > best_score and score > 0.3:  # Threshold for intent confidence
                best_score = score
                best_intent = intent
        
        return best_intent, best_score
    
    def identify_intent_fallback(self, user_input):
        """Fallback intent identification using keyword matching"""
        user_input_lower = user_input.lower()
        intent_scores = defaultdict(float)
        
        # Keyword matching
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.lower() in user_input_lower:
                    intent_scores[intent] += 1
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            best_score = intent_scores[best_intent] / len(self.intent_patterns[best_intent])
            return best_intent, best_score
        
        return 'general', 0.0
    
    def extract_entities_enhanced(self, user_input):
        """Enhanced entity extraction with better NLP"""
        entities = {
            'college': None,
            'department': None,
            'level': None,
            'user_type': None,
            'specific_info': []
        }
        
        user_input_lower = user_input.lower()
        
        # Enhanced education level detection
        level_patterns = {
            'undergraduate': ['undergraduate', 'undergrad', 'bachelor', 'bachelors', 'bsc', 'ba', 'first degree'],
            'masters': ['master', 'masters', 'graduate', 'msc', 'ma', 'postgraduate', 'masters degree'],
            'phd': ['phd', 'doctorate', 'doctoral', 'ph.d', 'doctor of philosophy']
        }
        
        for level, patterns in level_patterns.items():
            if any(pattern in user_input_lower for pattern in patterns):
                entities['level'] = level
                break
        
        # Enhanced user type detection
        user_type_patterns = {
            'student': ['student', 'students', 'pupil', 'learner', 'studying'],
            'staff': ['staff', 'employee', 'worker', 'administration'],
            'faculty': ['faculty', 'professor', 'teacher', 'instructor', 'lecturer', 'academic staff']
        }
        
        for user_type, patterns in user_type_patterns.items():
            if any(pattern in user_input_lower for pattern in patterns):
                entities['user_type'] = user_type
                break
        
        # Use spaCy for named entity recognition
        if self.nlp:
            doc = self.nlp(user_input)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities['college'] = ent.text
                elif ent.label_ in ["PERSON", "GPE"]:
                    entities['specific_info'].append(ent.text)
        
        return entities
    
    def find_relevant_content_semantic(self, query, top_k=5, threshold=0.25):
        """Find most relevant content using semantic similarity"""
        if not self.sentence_model or self.corpus_embeddings is None or len(self.corpus) == 0:
            return self.find_relevant_content_fallback(query, top_k)
        
        query_embedding = self.sentence_model.encode([query], convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = torch.cosine_similarity(query_embedding, self.corpus_embeddings)
        
        # Get top-k most similar documents
        top_indices = torch.topk(similarities, min(top_k, len(similarities))).indices
        
        results = []
        for idx in top_indices:
            score = similarities[idx].item()
            if score > threshold:
                results.append({
                    'content': self.corpus[idx],
                    'metadata': self.corpus_metadata[idx],
                    'similarity': score
                })
        
        return results
    
    def find_relevant_content_fallback(self, query, top_k=5):
        """Fallback content search using simple text matching"""
        query_lower = query.lower()
        results = []
        
        for i, content in enumerate(self.corpus):
            content_lower = content.lower()
            # Simple relevance scoring based on keyword overlap
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words.intersection(content_words))
            
            if overlap > 0:
                score = overlap / len(query_words)
                results.append({
                    'content': content,
                    'metadata': self.corpus_metadata[i],
                    'similarity': score
                })
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def extract_links_from_content(self, content):
        """Extract URLs from content"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = re.findall(url_pattern, str(content))
        return links
    
    def format_table_from_html(self, html_content):
        """Convert HTML table to readable format"""
        if '<table' in str(html_content).lower():
            return f"📊 **Table Data:**\n{str(html_content)}\n"
        return str(html_content)
    
    def get_colleges_and_departments(self):
        """Extract list of colleges and departments from data"""
        colleges = {}
        
        def search_recursive(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}/{key}" if path else key
                    
                    if any(term in key.lower() for term in ['college', 'school', 'faculty']):
                        colleges[key] = []
                        
                        if isinstance(value, dict):
                            for sub_key in value.keys():
                                if any(term in sub_key.lower() for term in ['department', 'dept', 'program']):
                                    colleges[key].append(sub_key)
                    
                    search_recursive(value, current_path)
        
        search_recursive(self.data)
        
        # If no colleges found, try semantic search
        if not colleges and self.sentence_model:
            college_query = "college school department faculty academic division"
            relevant_content = self.find_relevant_content_semantic(college_query, top_k=10)
            
            for content in relevant_content:
                path = content['metadata']['path']
                if any(term in path.lower() for term in ['college', 'school', 'department']):
                    parts = path.split('/')
                    for part in parts:
                        if any(term in part.lower() for term in ['college', 'school']):
                            if part not in colleges:
                                colleges[part] = []
        
        return colleges

    def handle_admission_requirements(self, entities, user_input):
        """Admission requirements handler with JSON navigation and follow-up support"""

        if not hasattr(self, 'conversation_state'):
            self.conversation_state = None

        # Handle follow-up input if waiting for level
        if self.conversation_state == "awaiting_admission_level":
            # Re-extract entities from the follow-up input
            follow_up_entities = self.extract_entities_enhanced(user_input)
            if follow_up_entities['level']:
                entities['level'] = follow_up_entities['level']
            
            self.conversation_state = None

        # If still no level detected, prompt user
        if not entities['level']:
            self.conversation_state = "awaiting_admission_level"
            return (
                "Hi! I'd be happy to help you with admission requirements. 😊\n\n"
                "Could you please specify which level you're interested in?\n"
                "🎓 **Undergraduate** (Bachelor's degree)\n"
                "📚 **Masters** (Graduate degree)\n"
                "🔬 **PhD** (Doctoral degree)\n\n"
                "Just reply with one of those!"
            )

        self.conversation_state = None  

        # Handle admission requirements for undergraduate
        if entities['level'] == 'undergraduate':
            # Look for FAQs section
            for section in self.data:
                if section.get("title", "").lower() == "faqs":
                    # Look for the specific FAQ about high school certificate requirements
                    for faq in section.get("children", []):
                        title = faq.get("title", "").lower()
                        if "are students who obtained their high school certificate" in title:
                            answer = faq.get("answer", "No answer provided.")
                            url = section.get("url", "")
                            
                            return (
                                f"{answer}\n\n"
                                f"🔗 For more detailed information, visit: {url}"
                            )
                    
                    # If specific FAQ not found, provide general undergraduate info from FAQs
                    url = section.get("url", "")
                    return (
                        "Undergraduate Admission Requirements:\n"
                        "Based on the available information, undergraduate admission requirements include:\n\n"
                        "• High school certificate (from within or outside the Kingdom)\n"
                        "• General Aptitude Test scores (Qudrat scores)\n"
                        "• Academic Achievement Test scores (Tahsili scores)\n"
                        "• Each track and major requires different scores for each of these tests.\n\n"
                        f"For complete details and FAQs, visit: {url}"
                    )

        # Handle admission requirements for masters
        elif entities['level'] == 'masters':
            # Look for Admission Requirements section
            for section in self.data:
                if section.get("title", "").lower() == "admission requirements":
                    # Look for Master's Programs in children
                    for child in section.get("children", []):
                        if child.get("title", "").lower() in ["master’s programs", "master's programs"]:
                            url = child.get("url", section.get("url", ""))
                            content = child.get("content", "")
                            
                            # Drill down one more level into nested children
                            sub_items = child.get("children", [])
                            if sub_items:
                                content += "\n\nAdditional Details:\n"
                                for item in sub_items:
                                    title = item.get("title", "")
                                    body = item.get("content", "No details available.")
                                    content += f"\n• **{title}**: {body}"
                            
                            return (
                                f"{content}\n\n"
                                f"For complete information, visit: {url}"
                            )
                    
                    # If Master's Programs not found in children, provide general info
                    url = section.get("url", "")
                    content = section.get("content", "")
                    return (
                        "Master's Programs Admission Requirements: \n"
                        f"{content}\n\n"
                        "Please check the admission requirements section for specific master's program criteria.\n\n"
                        f"🔗 For complete information, visit: {url}"
                    )

        # Handle PHD requests
        elif entities['level'] == 'phd':
            # Look for Admission Requirements section
            for section in self.data:
                if section.get("title", "").lower() == "admission requirements":
                    # Look for PhD Programs in children
                    for child in section.get("children", []):
                        if child.get("title", "").lower() in ["phd programs"]:
                            url = child.get("url", section.get("url", ""))
                            content = child.get("content", "")
                            
                            # Drill down one more level into nested children
                            sub_items = child.get("children", [])
                            if sub_items:
                                content += "\n\nAdditional Details:\n"
                                for item in sub_items:
                                    title = item.get("title", "")
                                    body = item.get("content", "No details available.")
                                    content += f"\n• **{title}**: {body}"
                            
                            return (
                                f"{content}\n\n"
                                f"🔗 For complete information, visit: {url}"
                            )
                    
                    # If PhD Programs not found in children, provide general info
                    url = section.get("url", "")
                    content = section.get("content", "")
                    return (
                        "PhD Programs Admission Requirements: \n"
                        f"{content}\n\n"
                        "Please check the admission requirements section for specific PhD program criteria.\n\n"
                        f"🔗 For complete information, visit: {url}"
                    )

        # Fallback if level is recognized but no specific info found
        return (
            f"I couldn't find specific {entities['level']} admission requirements in our database. "
            "Please try contacting the admissions office directly or visit the university website for the most accurate and up-to-date information."
        )

    def handle_academic_calendar(self):
        """Return the academic calendar from structured JSON with table formatting"""

        # Look for the academic calendar item in self.data
        calendar_item = next(
            (item for item in self.data if item.get("title", "").lower() == "academic calendar"),
            None
        )

        if not calendar_item:
            return (
                "Academic Calendar:"
                "I couldn't find the academic calendar in our current data. "
                "Please visit the registrar’s official site for updated info.\n\n"
                "https://dar.ksu.edu.sa/en/CurrentCalendar"
            )

        url = calendar_item.get("url", "")
        table = calendar_item.get("table", {})

        headers = table.get("headers", [])
        rows = table.get("rows", [])

        # Clean up any duplicate header row in the rows
        if rows and headers and rows[0] == headers:
            rows = rows[1:]

        # Start formatting
        response = "Academic Calendar: \n"
        response += f"Here is the calendar: ({url})\n\n"

        # Format headers
        if headers and all(isinstance(h, str) for h in headers):
            response += "| " + " | ".join(headers) + " |\n"
            response += "|" + " --- |" * len(headers) + "\n"

        # Format rows
        for row in rows:
            # Ensure each row has exactly 4 items
            padded_row = row + [""] * (len(headers) - len(row))
            response += "| " + " | ".join(padded_row[:len(headers)]) + " |\n"

        return response + "\n Need specific dates?** You can ask about registration deadlines, exam schedules, or semester start dates!"
      
    def handle_degree_programs(self, entities, user_input):
        """
        Simple method to navigate college hierarchy
        Handles: Colleges -> College Categories -> Individual Colleges -> Academic Departments
        """
        
        def find_node(target_title, json_data):
            """Find node by title with fuzzy matching"""
            target_lower = target_title.lower().strip()
            best_match = None
            best_score = 0
            
            # Check if this is a department search
            is_department_search = any(word in target_lower for word in ['department', 'dept'])
            
            # Debug: track all nodes we're checking
            checked_nodes = []
            
            def search_recursive(node, path="root"):
                nonlocal best_match, best_score
                
                if isinstance(node, dict) and 'title' in node:
                    title_lower = node['title'].lower().strip()
                    checked_nodes.append((path, node['title']))
                    
                    # Calculate match score
                    if title_lower == target_lower:
                        score = 100
                    elif target_lower in title_lower:
                        score = 90
                    elif title_lower in target_lower:
                        score = 85
                    else:
                        # Word overlap
                        target_words = set(target_lower.split())
                        title_words = set(title_lower.split())
                        if target_words and title_words:
                            overlap = len(target_words.intersection(title_words))
                            union = len(target_words.union(title_words))
                            score = (overlap / union) * 80
                        else:
                            score = 0
                    
                    if score > best_score:
                        best_score = score
                        best_match = node
                    
                    # Special handling for department searches
                    if is_department_search and 'children' in node:
                        # Look for Academic Departments section
                        for child in node.get('children', []):
                            if isinstance(child, dict) and child.get('section') == 'Academic Departments':
                                # Search within the departments
                                for dept in child.get('children', []):
                                    if isinstance(dept, dict) and 'title' in dept:
                                        dept_title_lower = dept['title'].lower().strip()
                                        checked_nodes.append((f"{path} -> Academic Departments -> {dept['title']}", dept['title']))
                                        
                                        # Calculate department match score
                                        if dept_title_lower == target_lower:
                                            dept_score = 100
                                        elif target_lower in dept_title_lower:
                                            dept_score = 90
                                        elif dept_title_lower in target_lower:
                                            dept_score = 85
                                        else:
                                            # Word overlap for departments
                                            dept_target_words = set(target_lower.split())
                                            dept_title_words = set(dept_title_lower.split())
                                            if dept_target_words and dept_title_words:
                                                dept_overlap = len(dept_target_words.intersection(dept_title_words))
                                                dept_union = len(dept_target_words.union(dept_title_words))
                                                dept_score = (dept_overlap / dept_union) * 80
                                            else:
                                                dept_score = 0
                                        
                                        if dept_score > best_score:
                                            best_score = dept_score
                                            best_match = dept
                    
                    # Search children normally
                    if 'children' in node:
                        for i, child in enumerate(node['children']):
                            search_recursive(child, f"{path} -> children[{i}]")
                
                elif isinstance(node, list):
                    for i, item in enumerate(node):
                        search_recursive(item, f"{path}[{i}]")
            
            search_recursive(json_data)
                
            # Look specifically for department-like nodes
            dept_nodes = [(path, title) for path, title in checked_nodes if 'department' in title.lower()]
            for i, (path, title) in enumerate(dept_nodes[:10]):
                print(f"  {i+1}. {title} (at {path})")
            
            return best_match if best_score > 30 else None
        
        def get_node_type(node):
            """Determine what type of node this is"""
            if not isinstance(node, dict) or 'title' not in node:
                return 'unknown'
            
            title = node['title'].lower()
            
            # Main "Colleges" node
            if title == 'colleges' and 'children' in node:
                return 'colleges_root'
            
            # College categories (contains "colleges" in title)
            if 'colleges' in title and 'children' in node:
                return 'category'
            
            # Individual college (has "About College" section)
            if 'children' in node:
                for child in node.get('children', []):
                    if isinstance(child, dict) and child.get('section') == 'About College':
                        return 'college'
            
            # Academic department (has contact_info, faculty_links, faculty_staff_links, or detailed content)
            if any(key in node for key in ['contact_info', 'faculty_links', 'faculty_staff_links']) or \
            (node.get('content') and len(node.get('content', '')) > 100):
                return 'department'
            
            return 'unknown'
        
        # Find the target node
        target_node = find_node(user_input.strip(), self.data)
        
        if not target_node:
            return ("I apologize. I could not find that. Try:\n"
                    "• 'Colleges' to see all categories\n"
                    "• 'Science Colleges' or any college categories to see colleges in category\n"
                    "• 'College of Engineering' or any colleges to check out specific college\n"
                    "• 'Computer Science Department' or any Academic departments in a college to get department info")
        
        node_type = get_node_type(target_node)
        title = target_node.get('title', '')
        
        # Handle based on node type
        if node_type == 'colleges_root':
            # Show college categories only
            response = "King Saud University - Colleges:\n"
            response += "College Categories:\n"
            
            total_colleges = 0
            for category in target_node.get('children', []):
                if isinstance(category, dict) and 'title' in category:
                    college_count = len(category.get('children', []))
                    total_colleges += college_count
                    response += f"• **{category['title']}** ({college_count} colleges)\n"
            
            response += f"\n Total: {total_colleges} colleges\n\n"
            response += "Ask about any category above for details!"
            return response
        
        elif node_type == 'category':
            # Show colleges in this category
            response = f"{title}\n\n"
            colleges = target_node.get('children', [])
            
            if colleges:
                response += f"**Colleges ({len(colleges)} total):**\n\n"
                for i, college in enumerate(colleges, 1):
                    if isinstance(college, dict) and 'title' in college:
                        response += f"{i}. {college['title']}\n"
                        if college.get('url'):
                            response += f"    {college['url']}\n"
                        response += "\n"
            
            response += "Ask about any college above for details!"
            return response
        
        elif node_type == 'college':
            # Show college info + list departments (titles only)
            response = f"{title}\n\n"
            
            if target_node.get('url'):
                response += f"Website: {target_node['url']}\n\n"
            
            # Find About College section
            about_content = None
            departments = []
            
            for child in target_node.get('children', []):
                if isinstance(child, dict):
                    if child.get('section') == 'About College':
                        about_content = child.get('content', '')
                    elif child.get('section') == 'Academic Departments':
                        departments = child.get('children', [])
            
            if about_content:
                response += f"About the College:\n{about_content}\n\n"
            
            if departments:
                response += f"Academic Departments ({len(departments)} total):\n\n"
                for i, dept in enumerate(departments, 1):
                    if isinstance(dept, dict) and 'title' in dept:
                        response += f"{i}. {dept['title']}\n"
                
                response += "\n Ask about any department for detailed info!"
            
            return response
        
        elif node_type == 'department':
            # Show full department details
            response = f"{title}\n\n"
            
            if target_node.get('url'):
                response += f"Website: {target_node['url']}\n\n"
            
            if target_node.get('content'):
                content = target_node['content'].strip()
                if content:
                    response += f"About the Department:\n{content}\n\n"
            
            if target_node.get('contact_info'):
                contact = target_node['contact_info'].strip()
                if contact:
                    response += f"Contact Information:\n{contact}\n\n"
            
            # Faculty links
            faculty_links = target_node.get('faculty_links', []) + target_node.get('faculty_staff_links', [])
            if faculty_links:
                response += "Faculty & Staff:\n"
                for link in faculty_links:
                    if isinstance(link, dict):
                        title_text = link.get('title', 'Faculty Link')
                        url = link.get('url', '#')
                        response += f"• [{title_text}]({url})\n"
                response += "\n"
            
            return response
        
        else:
            return f"Found '{title}' but couldn't determine its type. Please be more specific."
       
    def handle_faculty_directory(self, entities, user_input):
        """Handle faculty directory by parsing hierarchical structure"""
        
        if not entities['college'] and not entities['department']:
            # Show available colleges and departments
            response = "👥 **Faculty Directories**\n\nPlease specify which department or college:\n\n"
            response += "**Available Colleges:**\n"
            response += "• College of Business Administration\n"
            response += "• College of Architecture and Planning\n"
            response += "• College of Computer and Information Sciences\n"
            response += "• College of Food and Agricultural Sciences\n"
            response += "• College of Engineering\n"
            response += "• College of Applied Studies\n"
            response += "• Health Colleges\n\n"
            response += "Example: 'Show faculty for Computer Science department' or 'Faculty directory for College of Engineering'"
            return response

        # Parse the hierarchical data structure to find faculty links
        faculty_data = self.parse_faculty_structure()
        
        response = "Faculty Directory\n\n"
        faculty_found = False
        
        college_query = entities.get('college', '').lower()
        department_query = entities.get('department', '').lower()
        
        for college_name, departments in faculty_data.items():
            # Check if this college matches the query
            if college_query and college_query not in college_name.lower():
                continue
                
            college_has_matches = False
            college_response = f"{college_name}\n"
            
            for dept_name, faculty_links in departments.items():
                # Check if this department matches the query
                if department_query and department_query not in dept_name.lower():
                    continue
                    
                if faculty_links:
                    faculty_found = True
                    college_has_matches = True
                    college_response += f"\n{dept_name}\n"
                    
                    for faculty_link in faculty_links:
                        title = faculty_link.get('title', 'Faculty Directory')
                        url = faculty_link.get('url', '#')
                        college_response += f"[{title}]({url})\n"
                    
                    college_response += "\n"
            
            if college_has_matches:
                response += college_response
        
        # If no specific query, show all faculty links
        if not college_query and not department_query:
            for college_name, departments in faculty_data.items():
                response += f"{college_name}\n"
                for dept_name, faculty_links in departments.items():
                    if faculty_links:
                        response += f"\n{dept_name}\n"
                        for faculty_link in faculty_links[:5]:  # Limit to 2 links per dept
                            title = faculty_link.get('title', 'Faculty Directory')
                            url = faculty_link.get('url', '#')
                            response += f"[{title}]({url})\n"
                        response += "\n"
                response += "\n"
            faculty_found = True
        
        if not faculty_found:
            response += "No faculty directory links found for the specified criteria.\n"
            response += "Try searching for a specific college or department name."
        
        return response

    def handle_housing(self, entities, user_input):
        """Handle housing queries using the correct Housing section"""
        
        # Find the Housing section that actually has data (not the empty one)
        housing_section = self.find_housing_with_data()
        
        if not housing_section:
            return "Housing information not found."
        
        # If no user type specified, ask for clarification
        if not entities.get('user_type'):
            housing_types = self.get_housing_types(housing_section)
            response = "I'd be happy to help with housing information! Please specify:\n\n"
            for housing_type in housing_types:
                response += f"{housing_type}\n"
            return response + "\nWhich type of housing are you interested in?"
        
        # Handle faculty housing
        if entities['user_type'].lower() in ['faculty', 'staff', 'employee']:
            return self.handle_faculty_housing(housing_section)
        
        # Handle student housing
        elif entities['user_type'].lower() in ['student']:
            return self.handle_student_housing(housing_section)
        
        return "Please specify faculty or student housing."

    def find_housing_with_data(self):
        """Find the Housing section that contains actual data (not empty)"""
        def search_recursive(data):
            if isinstance(data, dict):
                title = data.get('title', '')
                if title.lower() == 'housing' and data.get('children'):
                    # Found Housing section with children - this is the one we want
                    return data
                
                # Search in children
                for child in data.get('children', []):
                    result = search_recursive(child)
                    if result:
                        return result
            elif isinstance(data, list):
                for item in data:
                    result = search_recursive(item)
                    if result:
                        return result
            return None
        
        return search_recursive(self.data)

    def get_housing_types(self, housing_section):
        """Extract housing types from the housing section"""
        housing_types = []
        for child in housing_section.get('children', []):
            title = child.get('title', '')
            if title and 'housing' in title.lower():
                housing_types.append(title)
        return housing_types

    def handle_faculty_housing(self, housing_section):
        """Handle faculty housing navigation"""
        # Find Faculty Housing section
        faculty_housing = None
        for child in housing_section.get('children', []):
            if child.get('title', '').lower() == 'faculty housing':
                faculty_housing = child
                break
        
        if not faculty_housing:
            available_children = [child.get('title', 'No title') for child in housing_section.get('children', [])]
            return f"Faculty housing not found. Available options: {', '.join(available_children)}"
        
        response = "Faculty Housing\n\n"
        
        # Display direct children of Faculty Housing
        for child in faculty_housing.get('children', []):
            title = child.get('title', '')
            url = child.get('url', '')
            
            response += f"{title}\n"
            
            if url:
                response += f"[Access {title}]({url})\n"
            
            # If this child has its own children (like "Related Links"), display them
            if child.get('children'):
                response += self.display_child_links(child)
            
            response += "\n"
        
        return response

    def handle_student_housing(self, housing_section):
        """Handle student housing navigation"""
        student_housing = None
        for child in housing_section.get('children', []):
            if child.get('title', '').lower() == 'student housing':
                student_housing = child
                break
        
        if not student_housing:
            available_children = [child.get('title', 'No title') for child in housing_section.get('children', [])]
            return f"Student housing not found. Available options: {', '.join(available_children)}"
        
        response = "Student Housing:\n\n"
        
        for child in student_housing.get('children', []):
            title = child.get('title', '')
            url = child.get('url', '')
            content = child.get('content', '')
            
            response += f"{title}\n"
            
            # Show content if available (like the procedural guide)
            if content:
                # Format the content nicely
                formatted_content = self.format_housing_content(content)
                response += f"{formatted_content}\n"
             
            if url:
                response += f"🔗 [Access {title}]({url})\n"
            
            # Show child links if any (like for Female Student Housing)
            if child.get('children'):
                response += self.display_child_links(child)
            
            response += "\n"
        
        return response

    def display_child_links(self, parent_section):
        """Display children of a section (like Related Links)"""
        if not parent_section.get('children'):
            return ""
        
        response = f"\nAvailable options:\n"
        
        for child in parent_section.get('children', []):
            child_title = child.get('title', '')
            child_url = child.get('url', '')
            
            if child_title:
                response += f"• {child_title}"
                if child_url:
                    response += f" - [Access here]({child_url})"
                response += "\n"
        
        return response

    def format_housing_content(self, content):
        """Format housing content for better readability"""
        if not content:
            return ""
        
        # Split into lines and format
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Make steps and important headings bold
                if any(keyword in line.lower() for keyword in ['step ', 'condition', 'how to apply']):
                    formatted_lines.append(f"**{line}**")
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def handle_library(self, entities, user_input):
        user_input_lower = user_input.lower()

        def contains_title(item, text):
            return item.get("title", "").lower() in text

        def find_matching_node(data, text):
            for node in data:
                if contains_title(node, text):
                    return node
            return None

        def find_matching_child(parent, text):
            for child in parent.get("children", []):
                if contains_title(child, text):
                    return child
            return None

        def format_contact_info(section):
            lines = []
            for table in section.get("tables", []):
                for row in table.get("rows", []):
                    lines.append(" | ".join(row))
            return "\n".join(lines)

        # STEP 1: Start from top-level "Libraries"
        libraries_root = find_matching_node(self.data, "libraries")
        if not libraries_root:
            return "❌ Could not find 'Libraries' section in data."

        # STEP 2: If user specifies a category (e.g., Shared libraries)
        selected_category = find_matching_child(libraries_root, user_input_lower)
        if selected_category:
            # STEP 2a: If user already specified a library
            matched_library = find_matching_child(selected_category, user_input_lower)
            if matched_library:
                info, contact, location = "", "", ""

                for section in matched_library.get("children", []):
                    title = section.get("title", "").lower()
                    if "information" in title:
                        info = section.get("content", "").strip()
                    elif "contact" in title:
                        contact = format_contact_info(section)
                    elif "location" in title:
                        location = section.get("url", "")

                response = f"{matched_library['title']}\n\n"
                if info:
                    response += f"Information:\n{info}\n"
                if contact:
                    response += f"Contact Info:\n{contact}\n"
                if location:
                    response += f"Location: {location}"
                return response.strip()

            # STEP 2b: User just selected the category, list children
            library_titles = [child["title"] for child in selected_category.get("children", [])]
            return f"Here are the libraries under {selected_category['title']}:\n" + "\n".join(f"- {title}" for title in library_titles)

        # STEP 3: User mentioned a library directly without saying category
        for category in libraries_root.get("children", []):
            matched_library = find_matching_child(category, user_input_lower)
            if matched_library:
                info, contact, location = "", "", ""
                for section in matched_library.get("children", []):
                    title = section.get("title", "").lower()
                    if "information" in title:
                        info = section.get("content", "").strip()
                    elif "contact" in title:
                        contact = format_contact_info(section)
                    elif "location" in title:
                        location = section.get("url", "")

                response = f"{matched_library['title']}\n\n"
                if info:
                    response += f"Information:\n{info}\n"
                if contact:
                    response += f"Contact Info:\n{contact}\n"
                if location:
                    response += f"Location: {location}"
                return response.strip()

        # STEP 4: User only said "libraries" → ask to choose category
        category_titles = [cat["title"] for cat in libraries_root.get("children", [])]
        return "Would you like to know about one of the following library categories?\n" + "\n".join(f"- {title}" for title in category_titles)
    
    def handle_grading_system(self):
        """Handle grading system with semantic search"""
        query = "grading system grade scale GPA evaluation assessment"
        relevant_content = self.find_relevant_content_semantic(query, top_k=5)
        
        response = "Grading System\n"
        
        for content in relevant_content[:3]:
            content_text = content['content']
            response += content_text + "\n\n"
            
            # Extract links from content text
            links = self.extract_links_from_content(content_text)
            if links:
                response += f"🔗 [Grading policy PDF]({links[0]})\n\n"
            
            # Check metadata for additional fields like 'info'
            metadata = content.get('metadata', {})
            if 'info' in metadata and metadata['info']:
                response += f"Here are complete Study and Examinations Regulations PDF:({metadata['info']})\n\n"
            elif 'url' in metadata and metadata['url']:
                response += f"Additional information:({metadata['url']})\n\n"
        
        if not relevant_content:
            response += "Grading system information is not available in our current database. Please check the student handbook or contact academic affairs.\n\n"
        
        return response
    
    def handle_plagiarism(self):
        """Handle plagiarism queries with semantic search"""
        query = "plagiarism academic integrity policy cheating misconduct"
        relevant_content = self.find_relevant_content_semantic(query, top_k=5)
        
        response = "Academic Integrity & Plagiarism Policy\n"
        
        for content in relevant_content[:3]:
            response += content['content'] + "\n\n"
            links = self.extract_links_from_content(content['content'])
            if links:
                response += f"Full policy: ({links[0]})\n\n"
        
        if not relevant_content:
            response += "Plagiarism policy information is not available in our current database. Please check the student handbook or contact academic affairs.\n\n"
        
        return response
    
    def handle_attendance(self):
        """Handle attendance queries"""
        query = "attendance policy class attendance requirements"
        relevant_content = self.find_relevant_content_semantic(query, top_k=3)
        
        response = "Attendance Policy: \n"
        
        if relevant_content:
            for content in relevant_content:
                response += content['content'] + "\n\n"
                links = self.extract_links_from_content(content['content'])
                if links:
                    response += f"Attendance policy: ({links[0]})\n\n"
        else:
            response += "Please refer to the grading system document for detailed attendance requirements.\n\n"
        
        # Always check for grading system info field as it contains attendance details
        grading_query = "grading system grade scale GPA evaluation assessment"
        grading_content = self.find_relevant_content_semantic(grading_query, top_k=5)
        
        for content in grading_content:
            metadata = content.get('metadata', {})
            # Check if this is grading system content and has info field
            if 'info' in metadata and metadata['info']:
                response += f"📖 Please refer to the Study and Examinations Regulations for detailed attendance policy:\n"
                response += f"Attendance found here: ({metadata['info']})\n\n"
                break
        
        return response
    
    def handle_research_labs(self):
        """Handle research labs and facilities - find Research node with Labs child"""
        
        response = "Research Labs & Facilities: \n"
        
        # This assumes you have access to the original JSON data
        # You'll need to replace 'self.json_data' with however you access your JSON
        
        def find_research_with_labs(data):
            """Recursively find Research node that has Labs as a child"""
            if isinstance(data, dict):
                # Check if this is a Research node
                if data.get('title') == 'Research':
                    # Check if it has children
                    if 'children' in data:
                        for child in data['children']:
                            if isinstance(child, dict) and child.get('title') == 'Labs':
                                # Found Research->Labs! Return the Labs children
                                return child.get('children', [])
                
                # Recursively search in children
                if 'children' in data:
                    for child in data['children']:
                        result = find_research_with_labs(child)
                        if result:
                            return result
            
            elif isinstance(data, list):
                # If data is a list, search each item
                for item in data:
                    result = find_research_with_labs(item)
                    if result:
                        return result
            
            return None
        
        try:
            labs_children = find_research_with_labs(self.data)  # You'll need to adjust this
            
            if labs_children:
                response += "Here are the research labs and facilities:\n\n"
                for i, lab in enumerate(labs_children, 1):
                    title = lab.get('title', 'Unknown Lab')
                    url = lab.get('url', '')
                    
                    # Normalize URL
                    if url and url.startswith('/'):
                        url = f"https://ksu.edu.sa{url}"
                    
                    response += f"{i}. **{title}**\n"
                    if url:
                        response += f"   🔗 {url}\n"
                    response += "\n"
            else:
                response += "Could not find Research node with Labs child.\n\n"
                
        except AttributeError:
            # Fallback if we don't have direct access to JSON data
            response += "Unable to access JSON data directly. Please ensure the JSON data is available.\n\n"
        
        return response

    def handle_it_support(self, entities, user_input):
        from difflib import get_close_matches

        # Find IT Helpdesk node
        it_helpdesk = next((item for item in self.data if item.get("title", "").lower() == "it helpdesk"), None)
        if not it_helpdesk:
            return "⚠️ Sorry, I couldn't find the IT Helpdesk information."

        user_input_lower = user_input.lower()
        user_type = None

        if "student" in user_input_lower:
            user_type = "Student"
        elif "staff" in user_input_lower or "faculty" in user_input_lower or "professor" in user_input_lower:
            user_type = "Staff"

        if not user_type:
            return "❓ Are you a student or staff/faculty? Please specify so I can assist you."

        # Navigate to the relevant section
        section = next((child for child in it_helpdesk.get("children", []) if child.get("title", "").lower() == user_type.lower()), None)
        if not section:
            return f"⚠️ Sorry, I couldn’t find IT support info for {user_type}."

        # Extract all issues and sub-issues
        def extract_issues_with_hierarchy(node, parent_title=None):
            results = []
            title = node.get("title")
            if title:
                full_title = f"{parent_title} → {title}" if parent_title else title
                results.append((full_title, title))
            for child in node.get("children", []):
                results.extend(extract_issues_with_hierarchy(child, title))
            return results

        all_issues = extract_issues_with_hierarchy(section)
        plain_titles = [t[1].lower() for t in all_issues]

        # Try to match the user input to a known issue
        matched = get_close_matches(user_input.lower(), plain_titles, n=1, cutoff=0.4)

        if matched:
            matched_title = next(full for full, plain in all_issues if plain.lower() == matched[0])
            ksu_code = "KSU1" if user_type == "Staff" else "KSU2"
            return (
                f"🛠️ It looks like you're facing: **{matched_title}**.\n"
                f"Please visit the [IT Helpdesk]({it_helpdesk['url']}), select **{ksu_code}**, and click **'Report an Issue'**."
            )

        # If no match found, show all options (parents and their children)
        options_text = f"📋 I couldn't find an exact match. Here are support topics for {user_type}:\n\n"
        grouped = {}
        for full_title, child_title in all_issues:
            parent = full_title.split("→")[0].strip()
            grouped.setdefault(parent, []).append(child_title)

        for parent, children in grouped.items():
            options_text += f"🔹 **{parent}**\n"
            for c in children:
                options_text += f"  • {c}\n"

        options_text += "\n💬 Please choose one of the topics above or rephrase your issue."

        return options_text
    
    def handle_contact_info(self, entities, user_input):
        """Handle contact information queries with semantic search"""
        query_parts = ["contact information"]
        
        if entities['department']:
            query_parts.append(entities['department'])
        if entities['college']:
            query_parts.append(entities['college'])
        
        # Add any specific departments mentioned in the query
        if self.nlp:
            doc = self.nlp(user_input)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    query_parts.append(ent.text)
        
        query = " ".join(query_parts) + " phone email address office"
        relevant_content = self.find_relevant_content_semantic(query, top_k=5)
        
        response = "📞 **Contact Information**\n\n"
        
        for content in relevant_content[:4]:
            response += content['content'] + "\n\n"
            links = self.extract_links_from_content(content['content'])
            if links:
                response += f"🔗 [Contact directory]({links[0]})\n\n"
        
        if not relevant_content:
            response += ("Contact information is not available in our current database. "
                        "Please check the university directory or contact the main office.\n\n")
        
        return response
    
    def handle_fees_tuition(self, user_input):
        """Handle tuition and fees queries"""
        query = "tuition fees cost payment financial charges"
        relevant_content = self.find_relevant_content_semantic(query, top_k=5)
        
        response = "💰 **Tuition & Fees Information**\n\n"
        
        for content in relevant_content[:3]:
            response += content['content'] + "\n\n"
            links = self.extract_links_from_content(content['content'])
            if links:
                response += f"🔗 [Fee structure]({links[0]})\n\n"
        
        if not relevant_content:
            response += ("Tuition and fee information is not available in our current database. "
                        "Please contact the finance office or check the student portal for current rates.\n\n")
        
        return response
    
    def handle_scholarships(self, user_input):
        """Handle scholarship and financial aid queries"""
        query = "scholarship financial aid funding grants assistance"
        relevant_content = self.find_relevant_content_semantic(query, top_k=5)
        
        response = "🎓 **Scholarships & Financial Aid**\n\n"
        
        for content in relevant_content[:3]:
            response += content['content'] + "\n\n"
            links = self.extract_links_from_content(content['content'])
            if links:
                response += f"🔗 [Scholarship portal]({links[0]})\n\n"
        
        if not relevant_content:
            response += ("Scholarship information is not available in our current database. "
                        "Please contact the financial aid office for information about available scholarships and grants.\n\n")
        
        return response
    
    def handle_general_query_enhanced(self, user_input):
        """Enhanced general query handling with better semantic understanding"""
        # First, try to find relevant content
        relevant_content = self.find_relevant_content_semantic(user_input, top_k=5, threshold=0.2)
        
        if relevant_content:
            response = "💡 **Here's what I found for you:**\n\n"
            
            for i, content in enumerate(relevant_content[:3], 1):
                text = content['content']
                
                # Intelligent summarization
                if len(text) > 250:
                    sentences = text.split('.')
                    summary = sentences[0]
                    if len(summary) < 200 and len(sentences) > 1:
                        summary += '. ' + sentences[1]
                    response += f"**{i}.** {summary}...\n\n"
                else:
                    response += f"**{i}.** {text}\n\n"
                
                # Add links
                links = self.extract_links_from_content(content['content'])
                if links:
                    response += f"   🔗 [More information]({links[0]})\n\n"
            
            # Add contextual follow-up suggestions
            response += "❓ **Want to know more?** You can ask me about:\n"
            response += "• Admission requirements and application process\n"
            response += "• Academic programs and course information\n"
            response += "• Campus facilities and student services\n"
            response += "• Contact information for departments\n"
            
        else:
            response = ("🤔 I couldn't find specific information about that in our database. "
                       "However, I can help you with:\n\n"
                       "🎓 **Academics:** Admission requirements, degree programs, academic calendar\n"
                       "🏠 **Campus Life:** Housing, libraries, dining, recreation\n"
                       "💼 **Services:** IT support, financial aid, career services\n"
                       "📞 **Contact:** Department information, faculty directories\n"
                       "📋 **Policies:** Grading, attendance, academic integrity\n"
                       "🔬 **Research:** Labs, facilities, opportunities\n\n"
                       "Could you try rephrasing your question or ask about one of these topics?")
        
        return response
    
    def chat(self, user_input):
        """Enhanced main chat function with better semantic understanding"""
        # Identify intent using semantic similarity
        intent, confidence = self.identify_intent_semantic(user_input)
        entities = self.extract_entities_enhanced(user_input)
        
        # Update user context
        self.user_context['last_intent'] = intent
        self.user_context['entities'].update(entities)
        
        # Route to appropriate handler based on intent
        if intent == 'admission_requirements':
            return self.handle_admission_requirements(entities, user_input)
        elif intent == 'academic_calendar':
            return self.handle_academic_calendar()
        elif intent == 'degree_programs':
            return self.handle_degree_programs(entities, user_input)
        elif intent == 'faculty':
            return self.handle_faculty_directory(entities, user_input)
        elif intent == 'housing':
            return self.handle_housing(entities, user_input)
        elif intent == 'library':
            return self.handle_library(entities, user_input)
        elif intent == 'grading':
            return self.handle_grading_system()
        elif intent == 'plagiarism':
            return self.handle_plagiarism()
        elif intent == 'attendance':
            return self.handle_attendance()
        elif intent == 'research':
            return self.handle_research_labs()
        elif intent == 'it_support':
            return self.handle_it_support(entities, user_input)
        elif intent == 'contact_info':
            return self.handle_contact_info(entities, user_input)
        elif intent == 'fees_tuition':
            return self.handle_fees_tuition(user_input)
        elif intent == 'scholarships':
            return self.handle_scholarships(user_input)
        else:
            return self.handle_general_query_enhanced(user_input)
    
    def run_interactive_chat(self):
        """Run interactive chat session with enhanced experience"""
        print("🎓 Welcome! I can help with admissions, academics, libraries, housing, faculty, fees, research, and more. Type your question or 'help' for examples.")
        print("💡 Tip: Ask natural questions like 'How do I apply for undergraduate admission?' or 'What are the library hours?'")
        print("Type 'help' for examples, or 'exit' to quit.\n")
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input("🙋 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("🤖 Assistant: Thank you for using the University AI Assistant! Have a wonderful day! 👋")
                    break
                
                if user_input.lower() == 'help':
                    print("🤖 Assistant: Here are some example questions you can ask:")
                    print("• 'What are the admission requirements for undergraduate programs?'")
                    print("• 'Show me the academic calendar'")
                    print("• 'I need information about computer science department'")
                    print("• 'Where can I find student housing?'")
                    print("• 'What's the grading system?'")
                    print("• 'I'm having trouble with my login'")
                    print("• 'Tell me about research opportunities'")
                    print("• 'How much does tuition cost?'")
                    print("• 'What scholarships are available?'\n")
                    continue
                
                if not user_input:
                    print("🤖 Assistant: I'm here to help! Please ask me something about the university. 😊\n")
                    continue
                
                print("🤖 Assistant: ", end="")
                response = self.chat(user_input)
                print(f"{response}\n")
                
                conversation_count += 1
                
                # Provide helpful suggestions every few interactions
                if conversation_count % 5 == 0:
                    print("💡 **Quick tip:** You can ask follow-up questions or request more specific information anytime!\n")
                
            except KeyboardInterrupt:
                print("\n🤖 Assistant: Goodbye! Thanks for using the University AI Assistant! 👋")
                break
            except Exception as e:
                print(f"🤖 Assistant: I encountered an error processing your request. Please try again! 🔧")
                print(f"(Technical details: {str(e)})\n")

def main():
    """Main function to run the enhanced chatbot"""
    # Initialize chatbot with your JSON file
    json_file_path = "C:\\Nawal\\IT Department\\Practical Training\\Final Chatbot\\data_backups\\menu_hierarchy.json"  # Replace with your actual JSON file path
    
    print("🚀 Starting University AI Assistant...")
    
    try:
        chatbot = UniversityChatbot(json_file_path)
        chatbot.run_interactive_chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {str(e)}")
        print("\n🔧 **Setup Requirements:**")
        print("1. Install required packages:")
        print("   pip install sentence-transformers transformers torch nltk spacy scikit-learn")
        print("2. Download spaCy model:")
        print("   python -m spacy download en_core_web_sm")
        print("3. Ensure your JSON file path is correct")
        print("4. Make sure you have sufficient disk space for model downloads")
        print("\n💡 **Note:** The chatbot will work with fallback methods if some models fail to load.")

if __name__ == "__main__":
    main()


