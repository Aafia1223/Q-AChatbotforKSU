import json
import nltk
import spacy
import re
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')
import requests
import fitz
from io import BytesIO
import nest_asyncio
import asyncio
import aiohttp
import pickle
import os
import hashlib 

def compute_md5(text):
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()

async def initialize_chatbot(json_file_path):
    """Initialize chatbot with cached embeddings and merged PDFs."""
    chatbot = UniversityChatbot(json_file_path)

    # --- JSON embeddings caching ---
    json_cache_file = "json_embeddings.pkl"
    json_hash_file = "json_hash.pkl"

    with open(json_file_path, "r", encoding="utf-8") as f:
        json_text = f.read()
    json_hash = compute_md5(json_text)

    json_cache_valid = os.path.exists(json_cache_file) and os.path.exists(json_hash_file)
    if json_cache_valid:
        with open(json_hash_file, "rb") as f:
            cached_hash = pickle.load(f)
        json_cache_valid = (cached_hash == json_hash)

    if json_cache_valid:
        with open(json_cache_file, "rb") as f:
            chatbot.corpus_embeddings = pickle.load(f)
        with open(json_hash_file, "rb") as f:
            pickle.load(f)
        print("Loaded JSON embeddings from cache")
    else:
        chatbot.prepare_semantic_corpus()
        with open(json_cache_file, "wb") as f:
            pickle.dump(chatbot.corpus_embeddings, f)
        with open(json_hash_file, "wb") as f:
            pickle.dump(json_hash, f)
        print("Created and cached JSON embeddings")

    # --- PDF embeddings caching ---
    pdf_cache_file = "pdf_embeddings.pkl"
    if os.path.exists(pdf_cache_file):
        with open(pdf_cache_file, "rb") as f:
            chatbot.pdf_docs = pickle.load(f)
        print("Loaded PDF embeddings from cache")
    else:
        # If async is desired, wrap this in asyncio.run(chatbot.load_all_pdfs()) outside
        await chatbot.load_all_pdfs()
        with open(pdf_cache_file, "wb") as f:
            pickle.dump(chatbot.pdf_docs, f)
        print("Loaded and cached PDFs")

    # --- Merge PDFs into main corpus ---
    for doc_name, doc_info in chatbot.pdf_docs.items():
        if doc_info.get("text") and (doc_info.get("chunks") is None or doc_info.get("embeddings") is None):
            chunks = [doc_info["text"][i:i+1000] for i in range(0, len(doc_info["text"]), 1000)]
            doc_info["chunks"] = chunks
            doc_info["embeddings"] = chatbot.sentence_model.encode(chunks, convert_to_tensor=True)

        for i, chunk in enumerate(doc_info.get("chunks", [])):
            chatbot.corpus.append(chunk)
            chatbot.corpus_metadata.append({
                "path": f"PDF:{doc_name}_chunk_{i}",
                "parent_key": doc_name,
                "content": chunk,
                "source": doc_info.get("url")
            })
        if doc_info.get("embeddings") is not None:
            if chatbot.corpus_embeddings is not None:
                chatbot.corpus_embeddings = torch.cat([chatbot.corpus_embeddings, doc_info["embeddings"]])
            else:
                chatbot.corpus_embeddings = doc_info["embeddings"]

    print(f"Total corpus segments: {len(chatbot.corpus)}")
    pdf_count = sum(1 for m in chatbot.corpus_metadata if "PDF:" in m["path"])
    print(f"Number of PDF segments in corpus: {pdf_count}")
    print("Chatbot initialized with JSON + PDF embeddings")

    return chatbot

class UniversityChatbot():

    # Initializing the university chatbot with the data and intent
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
            self.nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            print("⚠️  spaCy Multilingual model not found. Install with: python -m spacy download xx_ent_wiki_sm")
            self.nlp = None
        
        # Initialize Sentence Transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"⚠️ Error loading sentence transformer: {e}")
            self.sentence_model = None

        # Initialize local LLM for Arabic response generation
        try:
            self.generator = pipeline(
                "text-generation", 
                model="aubmindlab/aragpt2-base", 
                tokenizer="aubmindlab/aragpt2-base",
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
        except Exception as e:
            print(f"⚠️ Could not load Arabic language model: {e}")
            self.generator = None
        
        # Load university data
        self.json_file_path = json_file_path
        self.data = self.load_data(json_file_path)
        
        
        # Enhanced intent patterns with semantic variations
        self.intent_patterns = {
            'متطلبات القبول في جامعة الملك سعود': [
                'شروط القبول', 'كيفية التقديم', 'عملية التقديم', 'متطلبات القبول',
                'معايير القبول', 'شروط الالتحاق', 'الأهلية للتقديم',
                'ما الذي أحتاجه للتقديم', 'إجراءات التقديم', 'كيفية الالتحاق بالجامعة'
            ],
            'التقويم الأكاديمي': [
                'التقويم الأكاديمي', 'جدول الفصل الدراسي', 'تواريخ العام الدراسي', 'تواريخ الفصل',
                'متى يبدأ الفصل الدراسي', 'جدول الامتحانات', 'تواريخ التسجيل',
                'الجدول الزمني الأكاديمي', 'الجدول الزمني للفصل الدراسي', 'جدول المقررات', 'جدول الحصص',
                'التواريخ المهمة', 'المواعيد النهائية الأكاديمية', 'جدول العطل'
            ],
            'الكليات': [
                'البرامج الدراسية', 'التخصصات المتاحة', 'المقررات الدراسية', 'البرامج الأكاديمية',
                'ما الذي يمكنني دراسته', 'الكليات والتخصصات', 'الخطة الأكاديمية',
                'برامج البكالوريوس', 'برامج الماجستير', 'برامج الدكتوراه', 'الأقسام العلمية'
            ],
            'هيئة التدريس': [
                'هيئة التدريس', 'أعضاء هيئة التدريس', 'قائمة الأساتذة', 'الكادر التعليمي',
                'معلومات الأساتذة', 'جهات الاتصال بالهيئة التدريسية', 'طاقم التدريس'
            ],
            'contact_info': [
                'معلومات الاتصال', 'أرقام الهواتف', 'عناوين البريد الإلكتروني', 'مواقع المكاتب',
                'كيفية التواصل', 'تفاصيل الاتصال', 'أوقات الدوام', 'دليل الاتصال'
            ],
            'اسكان جامعة': [
                'السكن الجامعي', 'مساكن الطلاب', 'الإقامة في الجامعة', 'الطلاب',
                'الخدمات السكنية', 'أماكن السكن', 'خيارات الإقامة'
            ],
            'المكتبات': [
                'مكتبة الجامعة', 'خدمات المكتبة', 'موارد الدراسة', 'ساعات المكتبة',
                'المراجع البحثية', 'أماكن الدراسة', 'كتالوج المكتبة'
            ],
            'نظام الدرجات': [
                'نظام الدرجات', 'مقياس التقدير', 'كيفية حساب المعدل', 'التقييم الأكاديمي',
                'خطة الدرجات', 'توزيع الدرجات', 'كشف الدرجات'
            ],
            'الانتحال العلمي': [
                'سياسة الانتحال', 'الأمانة الأكاديمية', 'سياسة الغش', 'مكافحة الانتحال',
                'الأخلاقيات الأكاديمية', 'مخالفات أكاديمية', 'قواعد النزاهة'
            ],
            'attendance': [
                'سياسة الحضور', 'متطلبات الحضور', 'الغياب', 'قواعد الحضور', 'التغيب عن المحاضرات'
            ],
            'بحث': [
                'البحث العلمي', 'فرص البحث', 'المختبرات البحثية', 'مراكز الأبحاث',
                'برامج البحث', 'المشاريع البحثية'
            ],
            'it_support': [
                'الدعم الفني', 'مساعدة تقنية', 'مشاكل الكمبيوتر', 'مشاكل النظام',
                'مكتب المساعدة', 'مشكلة تسجيل الدخول', 'مشكلة كلمة المرور',
                'مشكلة الواي فاي', 'مشكلة البرامج', 'مشكلة الأجهزة'
            ],
            'الأسئلة الشائعة': [
                'الأسئلة الشائعة', 'المساعدة', 'الاستفسارات', 'الإجابات',
                'شروط القبول', 'طريقة التقديم', 'الأهلية', 'الأولوية في القبول',
                'القبول المباشر', 'القبول العام', 'قواعد التحويل', 'التخرج', 'التقديم أثناء الدراسة'
            ]
        }

        
        # Create intent embeddings for semantic matching
        self.intent_embeddings = None
        if self.sentence_model:
            self.create_intent_embeddings()
        
        # Conversation context
        self.user_context = {'last_intent': None, 'entities': {}}
        self.conversation_state = None
        self.user_type = None

        # PDFs
        self.pdf_docs = {
            "Student Rights Protection Unit": {"url": "...", "text": "", "embeddings": None, "chunks": None},
            "Engineering Regulations": {"url": "...", "text": "", "embeddings": None, "chunks": None}
        }
        
        # Corpus placeholders (JSON + PDFs)
        self.corpus = []
        self.corpus_metadata = []
        self.corpus_embeddings = None
        
    async def load_pdf_async(self, doc_name, doc_info):
        """Download PDF, extract text with fitz, split into chunks, compute embeddings asynchronously.
        Skip if embeddings already exist."""
        try:
            # Skip if embeddings are already present
            if doc_info.get("embeddings") is not None and doc_info.get("text"):
                print(f"Skipping {doc_name}, embeddings already loaded.")
                return

            async with aiohttp.ClientSession() as session:
                async with session.get(doc_info["url"]) as response:
                    pdf_bytes = await response.read()
            
            pdf_file = BytesIO(pdf_bytes)
            text = ""
            with fitz.open(stream=pdf_file, filetype="pdf") as pdf:
                for page in pdf:
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n"
            
            self.pdf_docs[doc_name]["text"] = text
            # Split text into chunks
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            self.pdf_docs[doc_name]["chunks"] = chunks

            # Compute embeddings asynchronously only if sentence_model is available
            if self.sentence_model:
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, lambda: self.sentence_model.encode(
                        chunks, show_progress_bar=False, convert_to_tensor=True
                    )
                )
                self.pdf_docs[doc_name]["embeddings"] = embeddings
                print(f"Loaded and embedded PDF: {doc_name}")
            else:
                print(f"Sentence model not loaded, skipping embeddings for {doc_name}")

        except Exception as e:
            print(f"Error loading {doc_name}: {e}")

    async def load_all_pdfs(self):
        """Load all PDFs concurrently and update corpus embeddings"""
        tasks = [
            self.load_pdf_async(doc_name, doc_info)
            for doc_name, doc_info in self.pdf_docs.items()
        ]
        await asyncio.gather(*tasks)

        # Recompute embeddings for the entire corpus (JSON + PDFs)
        if self.corpus and self.sentence_model:
            print(f"Creating embeddings for {len(self.corpus)} text segments (JSON + PDFs)...")
            self.corpus_embeddings = self.sentence_model.encode(self.corpus, convert_to_tensor=True)
            print("Semantic corpus ready")
        
    # Loading the data    
    def load_data(self, json_file_path):
        """Load JSON safely."""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Loaded university data from {json_file_path}")
                return data
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return {}
    
    # Add the semantic embeddings onto the text corpus
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
                # Extract additional fields from parent object
                additional_fields = {}
                if parent_obj and isinstance(parent_obj, dict):
                    # Look for common additional fields like 'info', 'url', 'link', etc.
                    for field in ['info', 'url', 'link', 'source', 'reference']:
                        if field in parent_obj and parent_obj[field]:
                            additional_fields[field] = parent_obj[field]
                
                self.corpus.append(data.strip())
                self.corpus_metadata.append({
                    'path': path,
                    'parent_key': parent_key,
                    'content': data.strip(),
                    **additional_fields  # Include additional fields from parent object
                })
        
        extract_text_recursive(self.data)
        
        if self.corpus and self.sentence_model:
            print(f"Creating embeddings for {len(self.corpus)} text segments...")
            self.corpus_embeddings = self.sentence_model.encode(self.corpus, convert_to_tensor=True)
            print("Semantic corpus ready")
    
    # Creating and adding embeddings for intent patterns and for better semantic matching
    def create_intent_embeddings(self):
        """Create embeddings for intent patterns for better semantic matching"""
        self.intent_embeddings = {}
        
        for intent, patterns in self.intent_patterns.items():
            # Create embeddings for all patterns of this intent
            pattern_embeddings = self.sentence_model.encode(patterns, convert_to_tensor=True, batch_size=16)
            # Use mean embedding as the intent representation
            self.intent_embeddings[intent] = torch.mean(pattern_embeddings, dim=0)
    
    # Identifying user intent using semantic similarity
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
    
    # Falling back using keyword matching 
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
    
    # Extradting entities and adding similar names to recognize them 
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
            'arabic_degree_keywords': {
                'undergraduate': ['بكالوريوس', 'درجة البكالوريوس', 'المرحلة الجامعية الأولى', 'الليسانس', 'البكالوريوس'],
                'masters': ['ماجستير', 'درجة الماجستير', 'الدراسات العليا', 'المستوى الجامعي الثاني', 'ماستر', 'درجة الماجستير المهنية'],
                'phd': ['دكتوراه', 'درجة الدكتوراه', 'الدكتوراه الفلسفية', 'دكتور', 'برامج الدكتوراه']
            }
        }
        
        for level, patterns in level_patterns.items():
            if any(pattern in user_input_lower for pattern in patterns):
                entities['level'] = level
                break
        
        # Enhanced user type detection
        user_type_patterns = {
            'student': ['طالب', 'طلاب', 'تلميذ', 'متعلمين', 'دارس', 'دارسة', 'الطالب', 'الطالبة', 'الطلاب', 'الدارسون'],
            'staff': ['موظف', 'الموظف', 'موظفين', 'عمال', 'العاملين', 'إداري', 'إداريين', 'العامل', 'الموظفين', 'إدارة', 'إداري/إدارية'],
            'faculty': ['أستاذ', 'أستاذة', 'الأساتذة', 'أعضاء هيئة التدريس', 'هيئة التدريس', 'مدرس', 'معلم', 'محاضر', 'محاضرة', 'أكاديمي', 'الطاقم الأكاديمي']
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
    
    # Find most relevant content using semantic similarity which includes fallback check, encoding query, comparing similarities, best matching and filtering based on the similarity threshold
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
    
    # Fallback method for the previous one 
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
    
    # Extracting links from content
    def extract_links_from_content(self, content):
        """Extract URLs from content"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = re.findall(url_pattern, str(content))
        return links
    
    def format_table_from_html(self, html_content):
        """Convert HTML table to readable format"""
        if '<table' in str(html_content).lower():
            return f"Table Data:\n{str(html_content)}\n"
        return str(html_content)
    
    # Help getter function to search for departments and colleges
    def get_colleges_and_departments(self):
        """Extract list of colleges and departments from data"""
        colleges = {}
        
        def search_recursive(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}/{key}" if path else key
                    
                    # Check if this looks like a college or department
                    if any(term in key for term in ['الكليات', 'قسم', 'أعضاء هيئة التدريس ']):
                        colleges[key] = []
                        
                        # Look for departments within this college
                        if isinstance(value, dict):
                            for sub_key in value.keys():
                                if any(term in sub_key.lower() for term in ['قسم', 'أقسام', 'شعبة', 'برنامج', 'برامج', 'الخطة الدراسية', 'المسار الأكاديمي', 'البرنامج الأكاديمي']):
                                    colleges[key].append(sub_key)
                    
                    search_recursive(value, current_path)
        
        search_recursive(self.data)
        
        # If no colleges found, try semantic search
        if not colleges and self.sentence_model:
            college_query = "كلية مدرسة قسم شعبة برنامج هيئة التدريس أكاديمي الأكاديمية"
            relevant_content = self.find_relevant_content_semantic(college_query, top_k=10)
            
            for content in relevant_content:
                path = content['metadata']['path']
                if any(term in path.lower() for term in ['كلية', 'مدرسة', 'قسم']):
                    parts = path.split('/')
                    for part in parts:
                        if any(term in part.lower() for term in ['كلية', 'مدرسة']):
                            if part not in colleges:
                                colleges[part] = []
        
        return colleges

    # This function helps get admission requirements of undergraduate (found in FAQs), masters and PhD from links found in Admission Requirements section from the json file
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
                "مرحبًا! يسعدني مساعدتك في متطلبات القبول. هل يمكنك تحديد المستوى الذي تهتم به؟\n"
                "بكالوريوس (الدرجة الجامعية الأولى)\n"
                "ماجستير (الدرجة الجامعية العليا)\n"
                "دكتوراه (الدرجة العلمية)\n\n"
                "فقط أجب بأحد الخيارات أعلاه!"
            )

        self.conversation_state = None  # Clear state

        # Handle UNDERGRADUATE requests
        if entities['level'] == 'undergraduate':
            # Look for FAQs section
            for section in self.data:
                if section.get("title", "") == "الأسئلة الشائعة":
                    # Look for the specific FAQ about high school certificate requirements
                    for faq in section.get("children", []):
                        title = faq.get("title", "").lower()
                        if "الحاصلين على شهادة الثانوية العامة من داخل المملكة أو خارجها مطالبين بأداء اختبارات المركز الوطني" in title:
                            answer = faq.get("answer", "لم يتم تقديم أي إجابة")
                            url = section.get("url", "")
                            
                            return (
                                "متطلبات القبول في مرحلة البكالوريوس: \n\n"
                                f"{answer}\n\n"
                                f"لمزيد من المعلومات التفصيلية، قم بزيارة: {url}"
                            )
                    
                    # If specific FAQ not found, provide general undergraduate info from FAQs
                    url = section.get("url", "")
                    return (
                        "متطلبات القبول لدرجة البكالوريوس: \n\n"
                        "استنادًا إلى المعلومات المتاحة، تشمل متطلبات القبول لدرجة البكالوريوس:\n\n"
                        "• شهادة الثانوية العامة (من داخل المملكة أو خارجها)\n"
                        "• نتائج اختبار القدرات العامة\n"
                        "• نتائج اختبار التحصيل الدراسي\n"
                        "• استيفاء متطلبات البرنامج المحددة\n\n"
                        f"للاطلاع على التفاصيل الكاملة والأسئلة الشائعة، يرجى زيارة: {url}"
                    )

        # Handle MASTERS requests
        elif entities['level'] == 'masters':
            # Look for Admission Requirements section
            for section in self.data:
                if section.get("title", "") == "متطلبات القبول في جامعة الملك سعود":
                    # Look for Master's Programs in children
                    for child in section.get("children", []):
                        if child.get("title", "").lower() in ["master’s programs", "master's programs"]:
                            url = child.get("url", section.get("url", ""))
                            content = child.get("content", "")
                            
                            # Drill down one more level into nested children
                            sub_items = child.get("children", [])
                            if sub_items:
                                content += "\n\nتفاصيل إضافية:\n"
                                for item in sub_items:
                                    title = item.get("title", "")
                                    body = item.get("content", "لا توجد تفاصيل متاحة.")
                                    content += f"\n {title}: {body}• "
                            
                            return (
                                "متطلبات القبول لبرامج الماجستير:\n\n"
                                f"{content}\n\n"
                                f"{url} : للحصول على المعلومات الكاملة، يرجى زيارة"
                            )

                    
                    # If Master's Programs not found in children, provide general info
                    url = section.get("url", "")
                    content = section.get("content", "")
                    return (
                        "متطلبات القبول لبرامج الماجستير\n\n"
                        f"{content}\n\n"
                        "يرجى التحقق من قسم متطلبات القبول لمعايير برنامج الماجستير المحدد.\n\n"
                        f"{url} : للحصول على المعلومات الكاملة، يرجى زيارة"
                    )

        # Handle PHD requests
        elif entities['level'] == 'phd':
            # Look for Admission Requirements section
            for section in self.data:
                if section.get("title", "").lower() == "متطلبات القبول في جامعة الملك سعود:":
                    # Look for PhD Programs in children
                    for child in section.get("children", []):
                        if child.get("title", "").lower() in ["phd programs"]:
                            url = child.get("url", section.get("url", ""))
                            content = child.get("content", "")
                            
                            # Drill down one more level into nested children
                            sub_items = child.get("children", [])
                            if sub_items:
                                content += "\n\nتفاصيل إضافية:\n"
                                for item in sub_items:
                                    title = item.get("title", "")
                                    body = item.get("content", "لا توجد تفاصيل متاحة.")
                                    content += f"\n• **{title}**: {body}"
                            
                            return (
                                "متطلبات القبول في برامج الدكتوراه:\n\n"
                                f"{content}\n\n"
                                f"للحصول على معلومات كاملة، قم بزيارة: {url}"
                            )
                    
                    # If PhD Programs not found in children, provide general info
                    url = section.get("url", "")
                    content = section.get("content", "")
                    return (
                        "متطلبات القبول في برامج الدكتوراه:\n\n"
                        f"{content}\n\n"
                        "يرجى مراجعة قسم متطلبات القبول لمعرفة معايير برنامج الدكتوراه المحددة.\n\n"
                        f"للحصول على معلومات كاملة، قم بزيارة:{url}"
                    )

        # Fallback if level is recognized but no specific info found
        return (
            f"لم أتمكن من العثور على متطلبات القبول المحددة لـ {entities['level']} في قاعدة بياناتنا. "
            "يرجى محاولة التواصل مع مكتب القبول مباشرة أو زيارة موقع الجامعة للحصول على المعلومات الأكثر دقة وحداثة."
        )

    # This function helps get academic calendar from the json file and format it into a table by putting each row on top of another (since it is not in HTML format)
    def handle_academic_calendar(self, query: str = None):
        """Return academic calendar or specific date/event if query is given"""

        # Look for the academic calendar item in self.data
        calendar_item = next(
            (item for item in self.data if item.get("title", "") == 'التقويم الأكاديمي'),
            None
        )

        if not calendar_item:
            return (
                "التقويم الأكاديمي"
                "لم أجد التقويم الأكاديمي في بياناتنا."
                "يرجى زيارة الموقع الرسمي لعمادة القبول والتسجيل للاطلاع على أحدث المعلومات:"
                "https://dar.ksu.edu.sa/ar/CurrentCalendar"
                )

        url = calendar_item.get("url", "")
        table = calendar_item.get("table", {})
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        # Clean duplicate header row if present
        if rows and headers and rows[0] == headers:
            rows = rows[1:]

        # If user asked for a specific event/date
        if query:
            query_lower = query.lower()

            for row in rows:
                row_text = " ".join(row).lower()

                # If query matches part of an event OR part of a date
                if query_lower in row_text:
                    gregorian_date = row[0] if len(row) > 0 else "N/A"
                    hijri_date = row[1] if len(row) > 1 else "N/A"
                    day = row[2] if len(row) > 2 else "N/A"
                    event = row[3] if len(row) > 3 else "N/A"

                    return (
                        f"{event}\n\n"
                        f"- ميلادي: {gregorian_date}\n"
                        f"- هجري: {hijri_date}\n"
                        f"- اليوم: {day}\n\n"
                        f"التقويم الكامل: {url}"
                    )

            return f"عذرًا، لم أتمكن من العثور على أي معلومات عن '{query}'. يرجى التحقق من التقويم الكامل: {url}"

        # Otherwise, return full table in a neat markdown format
        response = "التقويم الأكاديمي\n\n"
        response += f"{url}: عرض التقويم الكامل\n\n"

        if headers:
            response += "| " + " | ".join(headers) + " |\n"
            response += "|" + " --- |" * len(headers) + "\n"

        for row in rows:
            padded_row = row + [""] * (len(headers) - len(row))
            response += "| " + " | ".join(padded_row[:len(headers)]) + " |\n"

        response += (
            "\n💡 نصائح: يمكنك أيضًا سؤالي عن أشياء مثل:\n"
            "'متى تبدأ الامتحانات النهائية؟' -\n"
            "'ما هو الموعد النهائي للتسجيل؟' -\n"
            "'متى عطلة اليوم الوطني؟' -"
        )
        return response

    # This function gets the college categories, colleges, departments, their about, contact info and faculty directories
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
            is_department_search = any(word in target_lower for word in ['قسم', 'الأقسام'])
            
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
                            if isinstance(child, dict) and child.get('section') == 'الأقسام الأكاديمية':
                                # Search within the departments
                                for dept in child.get('children', []):
                                    if isinstance(dept, dict) and 'title' in dept:
                                        dept_title_lower = dept['title'].lower().strip()
                                        checked_nodes.append((f"{path} -> الأقسام الأكاديمية -> {dept['title']}", dept['title']))
                                        
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
            
            return best_match if best_score > 30 else None
        
        def get_node_type(node):
            """Determine what type of node this is"""
            if not isinstance(node, dict) or 'title' not in node:
                return 'unknown'
            
            title = node['title'].lower()
            
            # Main "Colleges" node
            if title == 'الكليات' and 'children' in node:
                return 'colleges_root'
            
            # College categories (contains "colleges" in title)
            if 'الكليات' in title and 'children' in node:
                return 'category'
            
            # Individual college (has "About College" section)
            if 'children' in node:
                for child in node.get('children', []):
                    if isinstance(child, dict) and child.get('section') == 'عن الكلية':
                        return 'college'
            
            # Academic department (has contact_info, faculty_links, faculty_staff_links, or detailed content)
            if any(key in node for key in ['contact_info', 'faculty_links', 'faculty_staff_links']) or \
            (node.get('content') and len(node.get('content', '')) > 100):
                return 'department'
            
            return 'unknown'
        
        # Find the target node
        target_node = find_node(user_input.strip(), self.data)
        
        if not target_node:
            return (" :لم أستطع العثور على ذلك. حاول\n"
                    "'الكليات' - انظر جميع الفئات •\n"
                    "'الكليات العلمية' - انظر الكليات في الفئة •\n"
                    "'كلية الهندسة' - كلية محددة •\n"
                    "'قسم علوم الحاسب' - معلومات القسم •")
        
        node_type = get_node_type(target_node)
        title = target_node.get('title', '')
        
        # Handle based on node type
        if node_type == 'colleges_root':
            # Show college categories only
            response = "جامعة الملك سعود - الكليات\n\n"
            response += ":فئات الكليات\n\n"
            
            total_colleges = 0
            for category in target_node.get('children', []):
                if isinstance(category, dict) and 'title' in category:
                    college_count = len(category.get('children', []))
                    total_colleges += college_count
                    response += f" (كليات {college_count} {category['title']}• )\n"
            
            response += f"\n كليات {total_colleges}:المجموع \n\n"
            response += "اسأل عن أي فئة أعلاه للحصول على التفاصيل"
            return response
        
        elif node_type == 'category':
            # Show colleges in this category
            response = f"{title}\n\n"
            colleges = target_node.get('children', [])
            
            if colleges:
                response += f":الكليات ({len(colleges)} المجموع):\n\n"
                for i, college in enumerate(colleges, 1):
                    if isinstance(college, dict) and 'title' in college:
                        response += f"{i}. {college['title']}\n"
                        if college.get('url'):
                            response += f"   {college['url']}\n"
                        response += "\n"
            
            response += "اسأل عن أي كلية أعلاه للحصول على التفاصيل!"
            return response
        
        elif node_type == 'college':
            # Show college info + list departments (titles only)
            response = f"{title}\n\n"
            
            if target_node.get('url'):
                response += f"{target_node['url']}: موقع إلكتروني\n\n"
            
            # Find About College section
            about_content = None
            departments = []
            
            for child in target_node.get('children', []):
                if isinstance(child, dict):
                    if child.get('section') == 'عن الكلية':
                        about_content = child.get('content', '')
                    elif child.get('section') == 'الأقسام الأكاديمية':
                        departments = child.get('children', [])
            
            if about_content:
                response += f"عن الكلية:\n{about_content}\n\n"
            
            if departments:
                response += f"الأقسام الأكاديمية ({len(departments)} قسم):\n\n"
                for i, dept in enumerate(departments, 1):
                    if isinstance(dept, dict) and 'title' in dept:
                        response += f"{i}. {dept['title']}\n"
                
                response += "\nيمكنك السؤال عن أي قسم للحصول على مزيد من المعلومات!"
            
            return response
        
        elif node_type == 'department':
            # Show full department details
            response = f"{title}\n\n"
            
            if target_node.get('url'):
                response += f"الموقع الإلكتروني: {target_node['url']}\n\n"
            
            if target_node.get('content'):
                content = target_node['content'].strip()
                if content:
                    response += f"عن القسم:\n{content}\n\n"
            
            if target_node.get('contact_info'):
                contact = target_node['contact_info'].strip()
                if contact:
                    response += f"معلومات التواصل:\n{contact}\n\n"
            
            # Faculty links
            faculty_links = target_node.get('faculty_links', []) + target_node.get('faculty_staff_links', [])
            if faculty_links:
                response += "هيئة التدريس والموظفون:\n"
                for link in faculty_links:
                    if isinstance(link, dict):
                        title_text = link.get('title', 'رابط')
                        url = link.get('url', '#')
                        response += f"• [{title_text}]({url})\n"
                response += "\n"
            
            return response
        
        else:
            return f"تم العثور على '{title}' لكن لا يمكن تحديد نوعه. يُرجى التحديد بشكل أدق."
       
    # This function finds the housing information (things that I thought were important)
    def handle_housing(self, entities, user_input):
        """Handle housing queries using the correct Housing section"""
        
        # Find the Housing section that actually has data (not the empty one)
        housing_section = self.find_housing_with_data()
        
        if not housing_section:
            return ".لم يتم العثور على معلومات السكن"
        
        # If no user type specified, ask for clarification
        if not entities.get('user_type'):
            housing_types = self.get_housing_types(housing_section)
            response = ":يسعدني مساعدتك في معلومات السكن! يرجى التوضيح.\n\n"
            for housing_type in housing_types:
                emoji = "🎓" if "الطلاب" in housing_type else "👨‍🏫"
                response += f"{emoji} {housing_type}\n"
            return response + "\nما هو نوع السكن الذي أنت مهتم به؟"
        
        # Handle faculty housing
        if entities['user_type'].lower() in [
            'عضو هيئة تدريس', 'عضوة هيئة تدريس', 'أعضاء هيئة التدريس', 'عضوات هيئة التدريس',  
            'الموظف', 'الموظفة', 'الموظفين', 'الموظفات', 'هيئة التدريس']:
            return self.handle_faculty_housing(housing_section)
        
        # Handle student housing
        elif entities['user_type'] in ['طالب', 'طالبة', 'طلاب', 'طالبات']:
            return self.handle_student_housing(housing_section)
        
        return ".يرجى تحديد سكن أعضاء هيئة التدريس أو الطلاب."

    # This addition function gets the housing information
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

    # This additional function gets housing type whether it is student or faculty housing
    def get_housing_types(self, housing_section):
        """Extract housing types from the housing section"""
        housing_types = []
        for child in housing_section.get('children', []):
            title = child.get('title', '')
            if title and 'اسكان جامعة الملك سعود' in title:
                housing_types.append(title)
        return housing_types

    # This additional function helps navigating Faculty housing sections that I thought were important
    def handle_faculty_housing(self, housing_section):
        """Handle faculty housing navigation"""
        # Find Faculty Housing section
        faculty_housing = None
        for child in housing_section.get('children', []):
            if child.get('title', '') == 'اسكان أعضاء هيئة التدريس':
                faculty_housing = child
                break
        
        if not faculty_housing:
            available_children = [child.get('title', 'No title') for child in housing_section.get('children', [])]
            return f"{', '.join(available_children)}: لم يتم العثور على سكن أعضاء هيئة التدريس. الخيارات المتاحة"
        
        response = "اسكان أعضاء هيئة التدريس\n\n"
        
        # Display direct children of Faculty Housing
        for child in faculty_housing.get('children', []):
            title = child.get('title', '')
            url = child.get('url', '')
            
            response += f"{title}\n"
            
            if url:
                response += f"({url})[{title}وصول]\n"
            
            # If this child has its own children (like "Related Links"), display them
            if child.get('children'):
                response += self.display_child_links(child)
            
            response += "\n"
        
        return response

    # This additional function helps navigating Student housing sections that I thought were important
    def handle_student_housing(self, housing_section):
        """Handle student housing navigation"""
        student_housing = None
        for child in housing_section.get('children', []):
            if child.get('title', '') == 'اسكان الطلاب':
                student_housing = child
                break
        
        if not student_housing:
            available_children = [child.get('title', 'No title') for child in housing_section.get('children', [])]
            return f"{', '.join(available_children)}: لم يتم العثور على سكن طلابي. الخيارات المتاحة"
        
        response = "اسكان الطلاب\n\n"
        
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
                response += f"({url})[{title}وصول]\n"
            
            # Show child links if any (like for Female Student Housing)
            if child.get('children'):
                response += self.display_child_links(child)
            
            response += "\n"
        
        return response

    # Gets child links 
    def display_child_links(self, parent_section):
        """Display children of a section (like Related Links)"""
        if not parent_section.get('children'):
            return ""
        
        response = f"\n:الخيارات المتاحة\n"
        
        for child in parent_section.get('children', []):
            child_title = child.get('title', '')
            child_url = child.get('url', '')
            
            if child_title:
                response += f"**{child_title}** •"
                if child_url:
                    response += f"({child_url})[الوصول هنا]-"
                response += "\n"
        
        return response

    # Editing the housing information content for better readability
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
                if any(keyword in line.lower() for keyword in ['الخطوة','الشرط','كيفية التقديم']):
                    formatted_lines.append(f"**{line}**")
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    # This is to handle library sections, their libraries and content of the libraries along with Location links
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
        libraries_root = find_matching_node(self.data, "المكتبات")
        if not libraries_root:
            return "لم أتمكن من العثور على قسم 'المكتبات' في البيانات."

        # STEP 2: If user specifies a category (e.g., Shared libraries)
        selected_category = find_matching_child(libraries_root, user_input_lower)
        if selected_category:
            # STEP 2a: If user already specified a library
            matched_library = find_matching_child(selected_category, user_input_lower)
            if matched_library:
                info, contact, location = "", "", ""

                for section in matched_library.get("children", []):
                    title = section.get("title", "").lower()
                    if "معلومات" in title:
                        info = section.get("content", "").strip()
                    elif "معلومات الاتصال" in title:
                        contact = format_contact_info(section)
                    elif "الموقع" in title:
                        location = section.get("url", "")

                response = f"{matched_library['title']}\n\n"
                if info:
                    response += f"\n{info}:معلومات\n\n"
                if contact:
                    response += f"\n{contact}:معلومات الاتصال\n\n"
                if location:
                    response += f"{location}: موقع"
                return response.strip()

            # STEP 2b: User just selected the category, list children
            library_titles = [child["title"] for child in selected_category.get("children", [])]
            return f"\n".join(f"- {title}" for title in library_titles) + "**{selected_category['title']}**: وهنا المكتبات الموجودة تحت\n"

        # STEP 3: User mentioned a library directly without saying category
        for category in libraries_root.get("children", []):
            matched_library = find_matching_child(category, user_input_lower)
            if matched_library:
                info, contact, location = "", "", ""
                for section in matched_library.get("children", []):
                    title = section.get("title", "").lower()
                    if "معلومات" in title:
                        info = section.get("content", "").strip()
                    elif "معلومات الاتصال" in title:
                        contact = format_contact_info(section)
                    elif "الموقع" in title:
                        location = section.get("url", "")

                response = f"{matched_library['title']}\n\n"
                if info:
                    response += f"\n{info}:معلومات\n\n"
                if contact:
                    response += f"\n{contact}:معلومات الاتصال\n\n"
                if location:
                    response += f"{location}: موقع"
                return response.strip()

        # STEP 4: User only said "libraries" → ask to choose category
        category_titles = [cat["title"] for cat in libraries_root.get("children", [])]
        return "\n".join(f"- {title}" for title in category_titles) + "هل ترغب في معرفة المزيد عن إحدى فئات المكتبة التالية؟\n" 
    
    # This function is to scrape grading system scale and add the PDF
    def handle_grading_system(self):
        """Handle grading system with semantic search"""
        query = "نظام التقدير مقياس الدرجات التقييم المعدل التراكمي تقييم الدرجات العلامات الأداء الأكاديمي معايير التقدير درجات المقررات"
        relevant_content = self.find_relevant_content_semantic(query, top_k=5)
        response = ":نظام الدرجات\n\n"
        
        for content in relevant_content[:3]:
            content_text = content['content']
            response += content_text + "\n\n"
            
            # Extract links from content text
            links = self.extract_links_from_content(content_text)
            if links:
                response += f"({links[0]})[PDF سياسة التصنيف]\n\n"
            
            # Check metadata for additional fields like 'info'
            metadata = content.get('metadata', {})
            if 'info' in metadata and metadata['info']:
                response += f"({metadata['info']})[PDF لائحة الدراسة والامتحانات]\n\n"
            elif 'url' in metadata and metadata['url']:
                response += f"({metadata['url']})[معلومات إضافية]\n\n"
        
        if not relevant_content:
            response += "معلومات نظام التقييم غير متوفرة في قاعدة بياناتنا الحالية. يُرجى مراجعة دليل الطالب أو التواصل مع الشؤون الأكاديمية.\n\n"
        
        return response
    
    # This function is to handle the plagiarism with the PDF
    def handle_plagiarism(self):
        """Handle plagiarism queries with semantic search"""
        query = (
            "السرقة الأدبية النزاهة الأكاديمية سياسة الغش المخالفات الأكاديمية الانتحال "
            "الاقتباس غير الموثق النسخ الاحتيال في الواجبات امتحانات غير نزيهة "
            "تزييف النتائج التحايل على التقييمات المخالفات العلمية الأخلاقيات الأكاديمية"
        )
        relevant_content = self.find_relevant_content_semantic(query, top_k=5)
        
        response = "سياسة النزاهة الأكاديمية والانتحال\n\n"
        
        for content in relevant_content[:3]:
            response += content['content'] + "\n\n"
            links = self.extract_links_from_content(content['content'])
            if links:
                response += f"({links[0]})[السياسة الكاملة]\n\n"
        
        if not relevant_content:
            response += "لا تتوفر معلومات حول سياسة الانتحال في قاعدة بياناتنا الحالية. يُرجى مراجعة دليل الطالب أو التواصل مع الشؤون الأكاديمية.\n\n"
        
        return response
    
    def handle_faq(self, entities, user_input):
        """Handle user queries that fall under the 'faq' intent."""
        faqs_section = next((item for item in self.data if item.get('title') == 'الأسئلة الشائعة'), None)
        if not faqs_section:
            return " عذراً، لم أتمكن من العثور على أي أسئلة شائعة في الوقت الحالي."

        faqs = faqs_section.get('children', [])
        if not faqs:
            return "لا تتوفر إدخالات الأسئلة الشائعة."

        # If SentenceTransformer is available, use semantic search
        if self.sentence_model:
            faq_titles = [faq.get('title', '') for faq in faqs if faq.get('title')]
            if not faq_titles:
                return "لم يتم العثور على عناوين الأسئلة الشائعة للبحث الدلالي."

            user_emb = self.sentence_model.encode([user_input], convert_to_tensor=True)
            faq_embs = self.sentence_model.encode(faq_titles, convert_to_tensor=True)
            
            import torch
            cos_scores = torch.nn.functional.cosine_similarity(user_emb, faq_embs)
            if len(cos_scores) == 0:
                return " لم أتمكن من حساب التشابه مع عناوين الأسئلة الشائعة."

            best_idx = cos_scores.argmax().item()
            best_score = cos_scores[best_idx].item()

            # Threshold for confident match
            if best_score > 0.65:
                return faqs[best_idx].get('answer', "لا يوجد إجابة متاحة لهذا السؤال الشائع.")
            else:
                return "وجدتُ بعض مواضيع الأسئلة الشائعة، لكنني لستُ متأكدًا أيّها تقصد. هل يمكنك إعادة صياغة سؤالك؟"

        # Fallback to simple keyword match
        for faq in faqs:
            title = faq.get('title', '')
            if any(word.lower() in user_input.lower() for word in title.split()):
                return faq.get('answer', "لا يوجد إجابة متاحة لهذا السؤال الشائع.")

        return "وجدتُ بعض مواضيع الأسئلة الشائعة، لكنني لستُ متأكدًا أيّها تقصد. هل يمكنك إعادة صياغة سؤالك؟"

    # This function is to handle the attendance rules with the PDF
    def handle_attendance(self):
        """Handle attendance queries"""
        query = (
            "سياسة الحضور متطلبات الحضور حضور الصفوف الغياب التأخير الانصراف "
            "ساعات الحضور الالتزام الأكاديمي التعليم الحضوري متابعة الطلاب "
            "تسجيل الحضور سجلات الحضور نظام الحضور تقييم الحضور"
        )
        relevant_content = self.find_relevant_content_semantic(query, top_k=3)
        
        response = "سياسة الحضور\n\n"
        
        if relevant_content:
            for content in relevant_content:
                response += content['content'] + "\n\n"
                links = self.extract_links_from_content(content['content'])
                if links:
                    response += f"({links[0]})[سياسة الحضور]\n\n"
        else:
            response += "يرجى الرجوع إلى وثيقة نظام الدرجات للحصول على متطلبات الحضور التفصيلية.\n\n"
        
        # Always check for grading system info field as it contains attendance details
        grading_query = (
            "نظام التقييم الدرجات مقياس الدرجات المعدل التراكمي التقييم "
            "تقدير الطلاب الأداء الأكاديمي نتائج الامتحانات معايير التقييم "
            "التقديرات النهائية الدرجات الفصلية النقاط الأكاديمية تقييم الطلاب "
            "تقدير الكفاءة التحصيل الدراسي"
        )
        grading_content = self.find_relevant_content_semantic(grading_query, top_k=5)
        
        for content in grading_content:
            metadata = content.get('metadata', {})
            # Check if this is grading system content and has info field
            if 'info' in metadata and metadata['info']:
                response += f"يرجى الرجوع إلى لوائح الدراسة والامتحانات لمعرفة سياسة الحضور التفصيلية:\n"
                response += f"({metadata['info']})[:الحضور موجود هنا]\n\n"
                break
        
        return response
    
    # This is to scrape the research labs links
    def handle_research_labs(self):
        """Handle research labs and facilities - find Research node with Labs child"""
        
        response = "مختبرات ومرافق الأبحاث\n\n"
        
        def find_research_with_labs(data):
            """Recursively find Research node that has Labs as a child"""
            if isinstance(data, dict):
                # Check if this is a Research node
                if data.get('title') == 'بحث':
                    # Check if it has children
                    if 'children' in data:
                        for child in data['children']:
                            if isinstance(child, dict) and child.get('title') == 'المختبرات':
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
                response += ":وهنا مختبرات ومرافق الأبحاث\n\n"
                for i, lab in enumerate(labs_children, 1):
                    title = lab.get('title', 'مختبر غير معروف')
                    url = lab.get('url', '')
                    
                    # Normalize URL
                    if url and url.startswith('/'):
                        url = f"https://ksu.edu.sa{url}"
                    
                    response += f"{i}. **{title}**\n"
                    if url:
                        response += f"   {url}\n"
                    response += "\n"
            else:
                response += "لم يتم العثور على عقدة البحث مع مختبرات الطفل.\n\n"
                
        except AttributeError:
            # Fallback if we don't have direct access to JSON data
            response += "تعذر الوصول إلى بيانات JSON مباشرةً. يُرجى التأكد من توفر بيانات JSON.\n\n"
        
        return response

    # This function adds all the IT helpdesk for student and staff and paths inside. 
    def handle_it_support(self, entities, user_input):
        from difflib import get_close_matches

        # Find IT Helpdesk node
        it_helpdesk = next((item for item in self.data if item.get("title", "").lower() == "it helpdesk"), None)
        if not it_helpdesk:
            return "Sorry, I couldn't find the IT Helpdesk information."

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
            return f"Sorry, I couldn’t find IT support info for {user_type}."

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
        options_text = f"I couldn't find an exact match. Here are support topics for {user_type}:\n\n"
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
    
    #---------------- THESE THINGS HAVE NOT BEEN SCRAPED YET ---------------------------------------#
    # def handle_fees_tuition(self, user_input):
        """Handle tuition and fees queries"""
        query = "tuition fees cost payment charges"
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
    
    # def handle_scholarships(self, user_input):
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
    
    #-----------------------------------------------------------------------------------------#
    
    # General queries
    def handle_general_query_enhanced(self, user_input):
        """Enhanced general query handling with better semantic understanding"""
        # First, try to find relevant content
        relevant_content = self.find_relevant_content_semantic(user_input, top_k=5, threshold=0.2)
        
        if relevant_content:
            response = "💡 **:هذا ما وجدته لك**\n\n"
            
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
                    response += f"({links[0]})[مزيد من المعلومات]\n\n"
            
            # Add contextual follow-up suggestions
            response += ":❓ **هل تريد أن تعرف المزيد؟** يمكنك أن تسألني عن\n"
            response += "متطلبات القبول وعملية التقديم •\n"
            response += "البرامج الأكاديمية ومعلومات الدورة•\n"
            response += "مرافق الحرم الجامعي والخدمات الطلابية •\n"
            response += "معلومات الاتصال للأقسام •\n"
            
        else:
            response = ("🤔 لم أتمكن من العثور على معلومات محددة حول هذا الموضوع في قاعدة البيانات الخاصة بنا. "
                        "ومع ذلك، يمكنني مساعدتك في:\n\n"
                        "🎓 **الأمور الأكاديمية:** متطلبات القبول، برامج الشهادات، التقويم الأكاديمي\n"
                        "🏠 **الحياة الجامعية:** السكن، المكتبات، المطاعم، الترفيه\n"
                        "💼 **الخدمات:** دعم تكنولوجيا المعلومات، المساعدات المالية، خدمات التوظيف\n"
                        "📞 **للتواصل:** معلومات القسم، دليل أعضاء هيئة التدريس\n"
                        "📋 **السياسات:** التقييم، الحضور، النزاهة الأكاديمية\n"
                        "🔬 **البحث:** المختبرات، المرافق، الفرص\n\n"
                        "هل يمكنك إعادة صياغة سؤالك أو الاستفسار عن أحد هذه المواضيع؟")
        return response
    
    # This is all the chats handler functions inside
    def chat(self, user_input):
        """Enhanced main chat function with better semantic understanding"""
        # Identify intent using semantic similarity
        intent, confidence = self.identify_intent_semantic(user_input)
        entities = self.extract_entities_enhanced(user_input)
        
        # Update user context
        self.user_context['last_intent'] = intent
        self.user_context['entities'].update(entities)
        
        # Route to appropriate handler based on intent
        if intent == 'متطلبات القبول في جامعة الملك سعود':
            return self.handle_admission_requirements(entities, user_input)
        elif intent == 'التقويم الأكاديمي':
            return self.handle_academic_calendar()
        elif intent == 'الكليات':
            return self.handle_degree_programs(entities, user_input)
        #elif intent == 'faculty':
        #    return self.handle_faculty_directory(entities, user_input)
        elif intent == 'اسكان جامعة':
            return self.handle_housing(entities, user_input)
        elif intent == 'المكتبات':
            return self.handle_library(entities, user_input)
        elif intent == 'نظام الدرجات':
            return self.handle_grading_system()
        elif intent == 'الانتحال العلمي':
            return self.handle_plagiarism()
        elif intent == 'attendance':
            return self.handle_attendance()
        elif intent == 'بحث':
            return self.handle_research_labs()
        elif intent == 'it_support':
            return self.handle_it_support(entities, user_input)
        elif intent == 'contact_info':
            return self.handle_contact_info(entities, user_input)
        elif intent == 'الأسئلة الشائعة':
            return self.handle_faq(entities, user_input)
        # elif intent == 'fees_tuition':
            return self.handle_fees_tuition(user_input)
        # elif intent == 'scholarships':
            return self.handle_scholarships(user_input)
        else:
            return self.handle_general_query_enhanced(user_input)
    
    # Interactive chats
    def run_interactive_chat(self):
        """Run interactive chat session with enhanced experience"""
        print("🎓 أهلاً وسهلاً! يمكنني مساعدتك في أمور القبول، والدراسات الأكاديمية، والمكتبات، والسكن، وأعضاء هيئة التدريس، والرسوم، والبحث، وغيرها. اكتب سؤالك أو 'ساعدني' للحصول على أمثلة.")
        print("💡 نصيحة: اطرح أسئلة طبيعية مثل 'كيف يمكنني التقدم بطلب القبول في المرحلة الجامعية؟' أو 'ما هي ساعات عمل المكتبة؟'")
        print("اكتب 'ساعدني' للحصول على أمثلة، أو 'خروج' للخروج.\n")
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input("أنت🙋 :").strip()

                if user_input.lower() in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']:
                    print("🤖 المساعد: مرحبًا! كيف يمكنني مساعدتك اليوم؟ 😊\n")
                    continue

                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("🤖 المساعد: شكرًا لاستخدامك مساعد الجامعة الذكي! نتمنى لك يومًا رائعًا! 👋")
                    break

                if user_input.lower() == 'help':
                    print("🤖 المساعد: إليك بعض الأمثلة على الأسئلة التي يمكنك طرحها:")
                    print("• 'ما هي متطلبات القبول لبرامج البكالوريوس؟'")
                    print("• 'أرني التقويم الأكاديمي'")
                    print("• 'أحتاج معلومات عن قسم علوم الحاسب'")
                    print("• 'أين أجد سكن الطلاب؟'")
                    print("• 'ما هو نظام التقييم؟'")
                    print("• 'أواجه مشكلة في تسجيل الدخول'")
                    print("• 'أخبرني عن فرص البحث العلمي'")
                    print("• 'كم تبلغ تكلفة الرسوم الدراسية؟'")
                    print("• 'ما هي المنح الدراسية المتاحة؟'\n")
                    continue

                if not user_input:
                    print("🤖 المساعد: أنا هنا لمساعدتك! يرجى طرح أي سؤال عن الجامعة. 😊\n")
                    continue
                
                print("🤖 المساعد: ", end="")
                response = self.chat(user_input)
                print(f"{response}\n")
                
                conversation_count += 1
                
                # Provide helpful suggestions every few interactions
                if conversation_count % 5 == 0:
                    print("💡 **نصيحة سريعة:** يمكنك طرح أسئلة متابعة أو طلب معلومات أكثر تحديدًا في أي وقت!\n")
                
            except KeyboardInterrupt:
                print("\n🤖 المساعد: مع السلامة! شكرًا لاستخدامك مساعد الذكاء الاصطناعي الجامعي! 👋")
                break
            except Exception as e:
                print(f"🤖 المساعد: واجهتُ خطأً أثناء معالجة طلبك. يُرجى المحاولة مجددًا! 🔧")
                print(f"(التفاصيل الفنية: {str(e)})\n")

# Main function to run all the functions
def main():
    """Main function to run the enhanced chatbot"""
    # Initialize chatbot with your JSON file
    json_file_path = "C:\\Nawal\\IT Department\\Practical Training\\Final Chatbot\\university_data_ar.json"  # Replace with your actual JSON file path
    
    print("🚀 Starting University AI Assistant...")
    
    try:
        #chatbot = UniversityChatbot(json_file_path)
        chatbot = asyncio.run(initialize_chatbot(json_file_path))
        chatbot.run_interactive_chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {str(e)}")
        print("\nSetup Requirements:")
        print("1. Install required packages:")
        print("   pip install sentence-transformers transformers torch nltk spacy scikit-learn")
        print("2. Download spaCy model:")
        print("   python -m spacy download en_core_web_sm")
        print("3. Ensure your JSON file path is correct")
        print("4. Make sure you have sufficient disk space for model downloads")
        print("\n💡 Note: The chatbot will work with fallback methods if some models fail to load.")

if __name__ == "__main__":
    main()