# ENHANCED SMART STUDENT DATA SYSTEM
# Universal data extraction with smart hierarchical organization

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import warnings
import pandas as pd
import os
import fitz # PyMuPDF
import re
from datetime import datetime
from chromadb.utils import embedding_functions # Import for consistent embedding function
import requests # 🆕 ADD THIS IMPORT
import json # 🆕 ADD THIS IMPORT

warnings.filterwarnings("ignore", category=FutureWarning)

class SmartStudentDataSystem:
    
    
    """A smart student data system that uses AI to analyze queries and retrieve relevant information."""
    def __init__(self):
        
        self.client = chromadb.PersistentClient(path="./chroma_store")
        # Initialize the SentenceTransformer model for 384-dimensional embeddings
        self.model = SentenceTransformer("all-MiniLM-L6-v2") 
        # Define the embedding function for ChromaDB to use, ensuring consistency
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collections = {}
        self.data_loaded = False
        
        # 🆕 --- START OF NEW CODE ---
        # System Configuration Options
        self.debug_mode = False  # Set to False for clean, user-facing output
        self.api_mode = 'online' # Options: 'online' or 'offline'
        
        # API Credentials and Endpoints
        self.mistral_api_key = "fcbJyUY4pHwpCNOTB7Wq3IZaivGdzz01"
        self.mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
        self.ollama_api_url = "http://localhost:11434/api/chat" # Standard Ollama API endpoint
        # 🆕 --- END OF NEW CODE ---
        
     
    # ======================== HELPER FUNCTIONS ========================   
    def debug(self, message):
        """A helper function to print messages only when debug mode is ON."""
        if self.debug_mode:
            print(message)
            
            
            
    def determine_query_type(self, query: str) -> str:
        """
        Uses an AI to determine the optimal search strategy: list retrieval or enrichment.
        """
        self.debug("🤖 Using AI to determine query type...")
        system_prompt = """
        You are a query classification expert. Your task is to analyze a user's query and determine the best way to handle it.
        You must classify the query into one of two types:

        1.  `list_retrieval`: Choose this if the query asks for a list or group of multiple entities based on shared criteria.
            Examples: "3rd year bscs students", "list all faculty", "who are the students in section A"

        2.  `enrichment_search`: Choose this if the query is about a specific, named entity. This type of search benefits from finding the main entity and then finding all documents related to it.
            Examples: "who is lee pace", "what is the schedule for jane doe", "tell me about professor smith"

        You MUST respond with a single, valid JSON object and nothing else, in the format: {"query_type": "list_retrieval"} or {"query_type": "enrichment_search"}.
        """
        user_prompt = f"Classify this query: \"{query}\""

        try:
            headers = {"Authorization": f"Bearer {self.mistral_api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "mistral-small-latest",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"}
            }
            response = requests.post(self.mistral_api_url, headers=headers, data=json.dumps(payload), timeout=20)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            query_type = json.loads(content).get('query_type', 'enrichment_search')
            self.debug(f"🤖 AI classified query as: {query_type}")
            return query_type
        except Exception as e:
            self.debug(f"🤖 AI query type classification failed: {e}. Defaulting to enrichment_search.")
            return 'enrichment_search'
        
    # =====================NAG AI ANALYZE NG INTENT NG QUERY TAPOS ISESEND SA FILTERING LOGIC========================
    
    def summarize_enriched_results_with_llm(self, query, documents):
        """
        A new, specialized summarizer with a "Connective Synthesis" prompt,
        used ONLY for the 'Search and Enrich' workflow.
        """
        self.debug("🧠 Using CONNECTIVE SYNTHESIS prompt for the final answer.")
        context = "\n\n---\n\n".join([doc['content'] for doc in documents])

        system_prompt = """
        You are PDMAI, an expert data analyst for Pambayang Dalubhasaan ng Marilao.
        You have been provided with a set of related documents for a specific entity. Your task is to synthesize this information into a single, direct answer.

        **Core Instructions:**
        1.  **Assume Connection:** The documents provided are related, even if they don't explicitly reference each other. Your primary job is to find and state that connection.
        2.  **Synthesize, Don't Just List:** Do not describe the documents separately. Use the information from one document (e.g., a student's profile) to interpret another (e.g., a general class schedule).
        3.  **Answer Directly:** Form a direct, confident answer to the user's original query using all pieces of information. For example, if you have a student's profile and their class schedule, synthesize them to state "Here is the schedule for [Student Name]..."
        """
        user_prompt = f"Synthesize the provided documents to form a direct answer to my query.\n\nQuery: {query}\n\nDocuments:\n{context}\n\nYour Synthesized Answer:"

        # --- This API call logic is identical to your original function ---
        headers = {}
        payload = {}
        api_url = ""

        if self.api_mode == 'online':
            headers = {"Authorization": f"Bearer {self.mistral_api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "mistral-small-latest",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            }
            api_url = self.mistral_api_url
        elif self.api_mode == 'offline':
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "mistral:instruct",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "stream": False
            }
            api_url = self.ollama_api_url

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status()
            response_json = response.json()
            if 'choices' in response_json and response_json['choices']:
                return response_json['choices'][0]['message']['content'].strip()
            elif 'message' in response_json and 'content' in response_json['message']:
                return response_json['message']['content'].strip()
            else:
                return "Error: Could not extract a valid response from the LLM."
        except requests.exceptions.RequestException as e:
            error_message = f"API Error: Could not connect to the LLM. Please check your connection and API settings.\nDetails: {e}"
            if self.debug_mode:
                print(f"❌ {error_message}")
            return error_message
        
    def run_search_and_enrich_workflow(self, query, max_results=100):
        """
        FINAL VERSION: Calls the appropriate summarizer based on the query type.
        """
        query_type = self.determine_query_type(query)

        if query_type == 'list_retrieval':
            self.debug("🚀 Running Comprehensive List Retrieval...")
            self._is_list_retrieval_mode = True
            results = self.smart_search_with_ai_reasoning(query, max_results=max_results)
            self._is_list_retrieval_mode = False
            # For list retrieval, it returns the documents for the original summarizer to handle.
            return results, False # Return results and a flag indicating it's not a final answer
        else: # enrichment_search
            self.debug("🚀 Running 'Search and Enrich' Workflow for a specific entity...")

            anchor_documents = self.smart_search_with_ai_reasoning(query, max_results=5)
            enriched_documents = self._enrich_results(anchor_documents)

            all_documents = anchor_documents + enriched_documents
            if not all_documents:
                return None, True # Return None if no documents found

            # --- THE CHANGE IS HERE ---
            # For enrichment, it calls the NEW summarizer and returns the final answer directly.
            final_answer = self.summarize_enriched_results_with_llm(query, all_documents)
            return final_answer, True # Return the final answer string and a flag
        
        
        
    def _enrich_results(self, anchor_documents: list) -> list:
        """
        Takes anchor documents, builds new smart queries from their metadata,
        and uses the main smart_search_with_ai_reasoning function to find related documents.
        """
        enriched_docs = []
        if not anchor_documents:
            return enriched_docs

        # Use a set to avoid running the same follow-up query multiple times
        follow_up_queries = set()

        for doc in anchor_documents:
            meta = doc.get('metadata', {})
            data_type = meta.get('data_type', '')
            new_query = ""

            # For students, build a new query to find their schedule
            if 'student' in data_type:
                course = meta.get('course')
                year = meta.get('year_level')
                section = meta.get('section')
                if course and year and section:
                    new_query = f"schedule for {course} year {year} section {section}"
                    follow_up_queries.add(new_query)

            # For faculty, build a new query to find their schedule
            elif 'faculty' in data_type or 'admin' in data_type:
                name = meta.get('full_name') or meta.get('adviser_name') or meta.get('staff_name')
                dept = meta.get('department')
                if name and dept:
                    new_query = f"schedule for faculty {name} from {dept} department"
                    follow_up_queries.add(new_query)

        if not follow_up_queries:
            self.debug("🔗 No follow-up queries generated for enrichment.")
            return enriched_docs

        self.debug(f"🔗 Generated {len(follow_up_queries)} unique follow-up quer(ies): {follow_up_queries}")

        # Perform a full smart search for each new query
        for query in follow_up_queries:
            self.debug(f"▶️ Executing follow-up smart search for: '{query}'")
            # This calls the main search function, making the enrichment "smart"
            results = self.smart_search_with_ai_reasoning(query, max_results=5)
            
            for res in results:
                # Add a reason to the result to show it came from enrichment
                res['match_reason'] = "Enriched by data connection"
                enriched_docs.append(res)
        
        return enriched_docs
        
        
    
        
    
    
        
        
    
    # =====================NAG AANALYZE NG INTENT NG QUERY TAPOS ISESEND SA FILTERING LOGIC========================
    def analyze_query_with_llm(self, query: str) -> dict:
        """
        FINAL VERSION: Uses more examples to make intent analysis highly reliable.
        """
        default_intent = {
            'intent': 'general', 'target_course': None, 'target_year': None,
            'target_section': None, 'target_person': None, 'target_subject': None,
            'data_type': None
        }

        system_prompt = f"""
        You are an expert query analyzer for a university's data system. Your task is to analyze the user's query and extract key information in a structured JSON format. Follow these rules precisely:
        1.  **Identify Intent:** Determine the user's primary goal. Possible intents are: 'person_search', 'schedule_search', 'course_specific', 'general'.
        2.  **Extract Entities:** Identify specific details in the query. The most important entity is 'target_person'.
        3.  **Determine Data Type:** Infer the most likely data type the user needs. Options: 'student', 'faculty', 'schedule'.
        4.  **Output Format:** Your response MUST be a single, valid JSON object and nothing else.
        5.  **Be Precise:** If you cannot extract a specific entity, its value must be `null`.

        Here are examples to guide you:

        - Query: "who is christine from bscs 1a"
        - Response:
              {{
                "intent": "person_search",
                "target_person": "Christine",
                "target_course": "BSCS",
                "target_year": "1",
                "target_section": "A",
                "target_subject": null,
                "data_type": "student"
              }}

        - Query: "what is the schedule for IT205"
        - Response:
              {{
                "intent": "schedule_search",
                "target_person": null,
                "target_course": null,
                "target_year": null,
                "target_section": null,
                "target_subject": "IT205",
                "data_type": "schedule"
              }}

        - Query: "does lee pace have a class on monday"
        - Response:
              {{
                "intent": "person_search",
                "target_person": "Lee Pace",
                "target_course": null,
                "target_year": null,
                "target_section": null,
                "target_subject": null,
                "data_type": "schedule"
              }}
        """
        user_prompt = f"Analyze the following user query: \"{query}\""

        # --- The rest of the function remains the same ---
        headers = {}
        payload = {}
        api_url = ""

        if self.api_mode == 'online':
            headers = {"Authorization": f"Bearer {self.mistral_api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "mistral-small-latest",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"}
            }
            api_url = self.mistral_api_url
        elif self.api_mode == 'offline':
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "mistral:instruct",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "format": "json",
                "stream": False
            }
            api_url = self.ollama_api_url

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            response_json = response.json()
            content_str = ""
            if 'choices' in response_json:
                content_str = response_json['choices'][0]['message']['content']
            elif 'message' in response_json:
                content_str = response_json['message']['content']
            if self.debug_mode:
                print(f"✅ LLM analysis raw response: {content_str}")
            parsed_json = json.loads(content_str.strip())
            for key in default_intent:
                if key not in parsed_json:
                    parsed_json[key] = None
            return parsed_json
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            if self.debug_mode:
                print(f"❌ LLM-based intent analysis failed: {e}. Falling back to general search.")
            return default_intent
        
    # ETO YUNG AI SEARCH NATIN

    def test_llm_search(self):
        """FINAL: Handles different return types from the workflow."""

        while True:
            query = input("\nAsk PDMAI anything (or type 'exit' to quit): ").strip()

            if query.lower() == 'exit':
                print("Returning to the main menu. Goodbye!")
                break

            if not query:
                continue

            # This call now returns either a list of documents or a final answer string
            response, is_final_answer = self.run_search_and_enrich_workflow(query)

            if not response:
                print("\nPDM AI: I couldn't find any information related to your query in the school's records.")
                continue

            if is_final_answer:
                # If the workflow returned the final answer, just print it.
                print("\nPDM AI: " + response)
            else:
                # If the workflow returned documents, use the ORIGINAL summarizer.
                summary = self.summarize_with_llm(query, response)
                print("\nPDM AI: " + summary)
   

    # 🆕 --- END OF NEW METHODS ---
    
    
    
    
    # 🆕 --- START OF NEW METHODS ---

    def toggle_debug_mode(self):
        """Toggles the debug mode on or off."""
        self.debug_mode = not self.debug_mode
        status = "ON" if self.debug_mode else "OFF"
        print(f"🔧 Debug mode is now {status}.")

    def toggle_api_mode(self):
        """Toggles the API mode between online and offline."""
        if self.api_mode == 'online':
            self.api_mode = 'offline'
        else:
            self.api_mode = 'online'
        print(f"📡 API mode is now set to: {self.api_mode.upper()}")
        if self.api_mode == 'offline':
            print("   (Ensure Ollama is running with the 'mistral:instruct' model available)")

    def manage_system_options(self):
        """Menu for managing system options like debug and API mode."""
        while True:
            print("\n⚙️ SYSTEM OPTIONS")
            print("="*50)
            print(f"  1. Toggle Debug Mode (Current: {'ON' if self.debug_mode else 'OFF'})")
            print(f"  2. Toggle API Mode (Current: {self.api_mode.upper()})")
            print("  3. ↩️  Back to Main Menu")
            
            try:
                choice = input("\n💡 Choose an option (1-3): ").strip()

                if choice == "1":
                    self.toggle_debug_mode()
                elif choice == "2":
                    self.toggle_api_mode()
                elif choice == "3":
                    break
                else:
                    print("❌ Invalid choice. Please select 1-3.")
            except KeyboardInterrupt:
                break
        print("\n↩️ Returning to main menu...")

    # 🆕 --- END OF NEW METHODS ---
        
    # ======================== INITIALIZATION & SETUP ========================
    
    def check_existing_data(self):
        """Check if there's already data in ChromaDB"""
        try:
            existing_collections = self.client.list_collections()
            if existing_collections:
                print("🗃️ Found existing data in ChromaDB:")
                for i, collection in enumerate(existing_collections, 1):
                    # When listing collections, ensure we use the correct embedding function if we need to interact
                    # with them beyond just getting count/name, though list_collections doesn't require it.
                    count = collection.count() 
                    collection_type = self.get_collection_type(collection.name)
                    print(f"  {i}. {collection.name} - {collection_type} ({count} records)")
                return existing_collections
            return []
        except Exception as e: # Catch specific exceptions if possible
            print(f"Error checking existing data: {e}")
            return []
    
    def get_collection_type(self, name):
        """Enhanced collection type display with curriculum, admin and faculty support"""
        parts = name.split('_')
        
        if len(parts) >= 2:
            base_type = parts[0]
            dept = parts[1] if len(parts) > 1 else ""
            
            dept_display = dept.replace('unclassified', 'Unclassified').upper() if dept else ''
            
            # CURRICULUM collection type
            if base_type == "curriculum":
                program = parts[2] if len(parts) > 2 else ""
                program_display = program.upper() if program else 'GENERAL'
                return f"Curriculum - {dept_display} {program_display}".strip()
            
            # STUDENTS collection type
            elif base_type == "students":
                course = parts[2] if len(parts) > 2 else ""
                year = parts[3] if len(parts) > 3 else ""
                section = parts[4] if len(parts) > 4 else ""
                
                course_display = course.upper() if course else ''
                year_display = year.replace('year', 'Year ') if year and 'year' in year else ''
                section_display = section.replace('sec', 'Section ').upper() if section and 'sec' in section else ''
                
                return f"Students - {dept_display} {course_display} {year_display} {section_display}".strip().replace('  ', ' ')
                
            # SCHEDULES collection type
            elif base_type == "schedules":
                course = parts[2] if len(parts) > 2 else ""
                year = parts[3] if len(parts) > 3 else ""
                section = parts[4] if len(parts) > 4 else ""
                
                course_display = course.upper() if course else ''
                year_display = year.replace('year', 'Year ') if year and 'year' in year else ''
                section_display = section.replace('sec', 'Section ').upper() if section and 'sec' in section else ''
                
                return f"COR Schedule - {dept_display} {course_display} {year_display} {section_display}".strip().replace('  ', ' ')
                
            # FACULTY collection type
            elif base_type == "faculty":
                # Check the full collection name for different faculty patterns
                full_name = "_".join(parts)
                
                if 'admin' in full_name:
                    faculty_type = 'admin'
                elif 'non_teaching_schedule' in full_name:
                    faculty_type = 'non_teaching_schedule'
                elif 'non_teaching' in full_name:
                    faculty_type = 'non_teaching'
                else:
                    faculty_type = parts[2] if len(parts) > 2 else "general"
                
                # Handle admin type with specific admin type distinction
                if faculty_type == 'admin':
                    # Check for admin type in the collection name
                    if 'board_member' in full_name:
                        return f"Board Member Staff - School Administration"
                    elif 'school_administrator' in full_name:
                        return f"School Administrator Staff - School Administration"
                    else:
                        return f"Administrative Staff - School Administration"
                
                # Special handling for non-teaching faculty schedules
                elif faculty_type == 'non_teaching_schedule':
                    non_teaching_dept_names = {
                        'REGISTRAR': 'Office of the Registrar',
                        'ACCOUNTING': 'Accounting & Finance',
                        'GUIDANCE': 'Guidance & Counseling',
                        'LIBRARY': 'Library Services',
                        'HEALTH_SERVICES': 'Health Services',
                        'MAINTENANCE_CUSTODIAL': 'Maintenance & Custodial',
                        'SECURITY': 'Security Services',
                        'SYSTEM_ADMIN': 'IT Services',
                        'ADMIN_SUPPORT': 'Administrative Support',
                    }
                    dept_name = non_teaching_dept_names.get(dept_display, dept_display)
                    return f"Non-Teaching Faculty Schedule - {dept_name}".strip()
                
                # Special handling for non-teaching faculty
                elif faculty_type == 'non_teaching':
                    non_teaching_dept_names = {
                        'REGISTRAR': 'Office of the Registrar',
                        'ACCOUNTING': 'Accounting & Finance',
                        'GUIDANCE': 'Guidance & Counseling',
                        'LIBRARY': 'Library Services',
                        'HEALTH_SERVICES': 'Health Services',
                        'MAINTENANCE_CUSTODIAL': 'Maintenance & Custodial',
                        'SECURITY': 'Security Services',
                        'SYSTEM_ADMIN': 'IT Services',
                        'ADMIN_SUPPORT': 'Administrative Support',
                    }
                    dept_name = non_teaching_dept_names.get(dept_display, dept_display)
                    return f"Non-Teaching Faculty - {dept_name}".strip()
                
                # Regular teaching faculty
                else:
                    faculty_type_display = faculty_type.replace('_', ' ').title()
                    return f"Faculty {faculty_type_display} - {dept_display}".strip().replace('  ', ' ')
        
        return f"Data Collection ({name})"

    
    def quick_setup(self):
        """Quick setup - check existing data or load new"""
        existing = self.check_existing_data()
        
        if existing:
            print(f"\n🚀 Ready to query! Found {len(existing)} data collections.")
            print("💡 You can search across all your data immediately.")
            self.data_loaded = True
            
            # Load all existing collections, ensuring the embedding function is passed
            for collection_info in existing:
                try:
                    # Get the collection with the specified embedding function
                    collection = self.client.get_collection(
                        name=collection_info.name, 
                        embedding_function=self.embedding_function
                    )
                    self.collections[collection.name] = collection
                except Exception as e:
                    print(f"⚠️ Could not load existing collection {collection_info.name}: {e}")
            
            return True
        else:
            print("📂 No existing data found. Let's load some files first...")
            return self.load_new_data()
    
    # ======================== FILE MANAGEMENT ========================
    
    def list_available_files(self):
        """List available files with smart type detection"""
        files = [f for f in os.listdir('.') 
                if (f.endswith('.xlsx') or f.endswith('.pdf')) and not f.startswith('~$')]
        
        if not files:
            print("❌ No Excel or PDF files found.")
            return []
        
        print("\n📁 Available Files:")
        for i, file in enumerate(files, 1):
            file_type = self.detect_file_type(file)
            print(f"  {i}. {file} - {file_type}")
        return files
    
    def detect_file_type(self, filename):
        """Smart file type detection"""
        ext = os.path.splitext(filename)[1].lower()
        filename_lower = filename.lower()
        
        if ext == ".xlsx":
            try:
                # Check filename patterns first
                if any(x in filename_lower for x in ['non_teaching_schedule', 'staff_schedule', 'admin_schedule']):
                    return "Non-Teaching Faculty Schedule (Excel)"
                elif any(x in filename_lower for x in ['teaching', 'faculty_resume', 'faculty_data']):
                    return "Teaching Faculty Data (Excel)"
                elif any(x in filename_lower for x in ['non_teaching', 'non-teaching', 'admin_faculty', 'registrar', 'accounting']):
                    return "Non-Teaching Faculty Data (Excel)"
                elif any(x in filename_lower for x in ['resume', 'cv']) and 'faculty' in filename_lower:
                    return "Teaching Faculty Data (Excel)"
                elif 'cor' in filename_lower:
                    return "COR Schedule (Excel)"
                elif any(x in filename_lower for x in ['schedule', 'class_schedule']):
                    return "Faculty Schedule (Excel)"
                elif any(x in filename_lower for x in ['student', 'year', 'bscs', 'bsit', 'bstm', 'bshm']):
                    return "Student Data (Excel)"
                
                # Check content if filename is unclear
                df_check = pd.read_excel(filename, header=None)
                if self.is_cor_file(df_check):
                    return "COR Schedule (Excel)"
                elif self.is_non_teaching_faculty_schedule_excel(df_check, silent=True):  # 🆕 ADD silent=True
                    return "Non-Teaching Faculty Schedule (Excel)"
                elif self.is_faculty_schedule_excel(df_check, silent=True):  # 🆕 ADD silent=True
                    return "Faculty Schedule (Excel)"
                elif self.is_admin_excel(df_check, silent=True):  # 🆕 ADD silent=True
                    return "Admin Data (Excel)"
                elif self.is_non_teaching_faculty_excel(df_check, silent=True):  # 🆕 ADD silent=True
                    return "Non-Teaching Faculty Data (Excel)"
                elif self.is_teaching_faculty_excel(df_check, silent=True):  # 🆕 ADD silent=True
                    return "Teaching Faculty Data (Excel)"
                elif self.is_faculty_excel(df_check):
                    return "Faculty Data (Excel)"
                else:
                    return "Student Data (Excel)"
            except:
                return "Excel File"
                
        elif ext == ".pdf":
            # Check filename patterns
            if any(x in filename_lower for x in ['resume', 'cv']):
                return "Faculty Data (PDF)"
            elif 'cor' in filename_lower:
                return "COR Schedule (PDF)"
            elif 'schedule' in filename_lower:
                return "Faculty Schedule (PDF)"
            elif any(x in filename_lower for x in ['student', 'year', 'synthetic']):
                return "Student Data (PDF)"
                
            # Check content
            if self.is_cor_pdf(filename):
                return "COR Schedule (PDF)"
            elif self.is_faculty_schedule_pdf(filename):
                return "Faculty Schedule (PDF)"
            elif self.is_faculty_pdf(filename):
                return "Faculty Data (PDF)"
            else:
                return "Student Data (PDF)"
        
        return "Unknown"
    
    # ======================== UNIVERSAL DATA EXTRACTION ========================
    
    def extract_universal_student_data(self, text_content, source_type):
        """
        Universal extractor for student data regardless of format.
        This function attempts to find the specified fields using flexible patterns.
        """
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        all_text = ' '.join(lines).upper() # Flatten text for broader pattern matching
        
        # Initialize required fields with None
        student_data = {
            'student_id': None,
            'surname': None,
            'first_name': None,
            'full_name': None,
            'year': None,
            'course': None,
            'section': None,
            'contact_number': None,
            'guardian_name': None,
            'guardian_contact': None
        }
        
        # For structured Excel, extract directly from column headers
        if source_type == 'excel_structured':
            extracted_from_structured = self.extract_from_structured_text(lines)
            for key, value in extracted_from_structured.items():
                if value:
                    student_data[key] = value
        
        # Define flexible patterns for unstructured data (PDFs, unstructured Excel)
        # Prioritize more specific patterns first
        patterns = {
            'student_id': [
                r'(?:STUDENT\s*ID|ID\s*NO|ID|STUDENT\s*NUMBER)[:\s]*([A-Z0-9-]+)', # "STUDENT ID: PDM-123456"
                r'([A-Z]{2,4}-\d{4,6})',  # Pattern like PDM-123456, BSCS-2023
                r'(\d{4,8})',  # Pure numbers, e.g., 20230001
                r'^\s*(\d{4,8})\s*$', # NEW: ID on its own line (pure numbers)
            ],
            'full_name': [ # Try to get full name first, then split
                r'(?:FULL\s*NAME|STUDENT\s*NAME|NAME)[:\s]*([A-Z][A-Z\s\.,-]+)', # "NAME: JOHN DOE"
                r'([A-Z][A-Z\s\.,-]+(?:,\s*[A-Z][A-Z\s\.,-]+)?)\s*(?:STUDENT ID|ID NO|YEAR|COURSE|SECTION|CONTACT)', # Name before another field
                r'([A-Z][A-Z\s\.,-]+)\s*\d{4,8}', # Name followed by a number (potential ID)
                r'^\s*([A-Z][A-Z\s\.,-]+(?:,\s*[A-Z][A-Z\s\.,-]+)?)\s*$', # NEW: Full name on its own line (e.g., "DOE, JOHN")
                r'^\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\s*$', # NEW: First Last or First Middle Last on its own line (e.g., "John Doe", "Mary Jane Smith")
            ],
            'year': [
                r'(?:YEAR\s*LEVEL|YEAR)[:\s]*([1-4])', # "YEAR: 2"
                r'([1-4])(?:ST|ND|RD|TH)?\s*YEAR', # "2ND YEAR"
                r'YEAR\s*(\d)', # "YEAR 2"
                r'\b(YEAR\s*[1-4])\b', # NEW: "YEAR 1", "YEAR 2" as a whole word
                r'\b([1-4])(?:ST|ND|RD|TH)\b', # NEW: Directly capture "1ST", "2ND" etc.
                r'\b([1-4])\s*(?:YR|YEAR)\b', # NEW: "1 YR", "2 YEAR"
                r'\bLEVEL\s*([1-4])\b', # NEW: "LEVEL 1"
                r'\b([1-4])\s*(?:LEVEL)\b', # NEW: "1 LEVEL"
            ],
            'course': [
                r'(?:COURSE|PROGRAM)[:\s]*([A-Z]{2,6})', # "COURSE: BSCS"
                r'(BS[A-Z]{2,4}|AB[A-Z]*|B[A-Z]{2,4})', # BSCS, ABPSY, BSTM
                r'\b(BSCS|BSIT|BSHM|BSTM|BSOA|BECED|BTLE)\b', # NEW: Specific common courses as whole words
                r'\b([A-Z]{2,6})\b(?=\s*(?:YEAR|SECTION|STUDENT ID|LEVEL))', # NEW: Course code followed by other student info indicators
                r'(?:BACHELOR\s*OF\s*SCIENCE\s*IN|BS\s*IN)\s*([A-Z\s]+)', # NEW: "Bachelor of Science in Computer Science"
                r'\b(?:PROGRAM|DEGREE)[:\s]*([A-Z\s]+)\b', # NEW: "PROGRAM: Computer Science"
                r'\b([A-Z]{2,6})\s*(?:PROGRAM|COURSE)\b', # NEW: "BSCS PROGRAM"
            ],
            'section': [
                r'(?:SECTION|SEC)[:\s]*([A-Z0-9-]+)', # "SECTION: A"
                r'SECTION\s+([A-Z0-9-]+)', # "SECTION A"
                r'\b(SEC\s*[A-Z0-9]+)\b', # NEW: "SEC A", "SEC B1"
                r'\b([A-Z])\b(?=\s*(?:YEAR|COURSE|STUDENT ID|LEVEL))', # NEW: Single letter section followed by other student info
                r'\b(SECTION\s*[A-Z0-9]+)\b', # NEW: "SECTION 1A"
                r'\b[A-Z]\s*(?:SECTION)\b', # NEW: "A SECTION"
                r'\b([A-Z0-9]+)\s*(?:SEC|SECTION)\b', # NEW: "1A SEC"
            ],
            'contact_number': [
                r'(?:CONTACT\s*NUMBER|PHONE|MOBILE|TEL|CONTACT)[:\s]*([\+\d\s()-]+)', # "CONTACT NUMBER: +639123456789"
                r'(\d{10,12})', # Pure 10-12 digit numbers (e.g., 9123456789)
                r'(\+\d{10,14})', # NEW: +639... format
                r'\b(?:PHONE|TEL)[:\s]*([\+\d\s()-]+)\b', # NEW: "PHONE: 123-456-7890"
                r'\bCELL\s*NO[:\s]*([\+\d\s()-]+)\b', # NEW: "CELL NO: 9123..."
            ],
            'guardian_name': [
                r'(?:GUARDIAN\s*NAME|PARENT\s*NAME|EMERGENCY\s*CONTACT\s*NAME)[:\s]*([A-Z][A-Z\s\.,-]+)',
                r'GUARDIAN[:\s]*([A-Z][A-Z\s\.,-]+)',
                r'(?:GUARDIAN|PARENT)\s*:\s*([A-Z][A-Z\s\.,-]+)', # NEW: "GUARDIAN: John Doe"
                r'\b(?:EMERGENCY\s*CONTACT)[:\s]*([A-Z][A-Z\s\.,-]+)\b', # NEW: "EMERGENCY CONTACT: Jane Doe"
            ],
            'guardian_contact': [
                r'(?:GUARDIAN\s*CONTACT|PARENT\s*CONTACT|EMERGENCY\s*CONTACT\s*NUMBER)[:\s]*([\+\d\s()-]+)',
                r'(\d{10,12})\s*(?:GUARDIAN|PARENT)', # Number followed by guardian/parent keyword
                r'(?:GUARDIAN|PARENT)\s*CONTACT[:\s]*([\+\d\s()-]+)', # NEW: "GUARDIAN CONTACT: 9123..."
                r'\b(?:EMERGENCY\s*CONTACT\s*NO)[:\s]*([\+\d\s()-]+)\b', # NEW: "EMERGENCY CONTACT NO: 9123..."
            ]
        }
        
        # Extract each field using patterns
        for field, field_patterns in patterns.items():
            if student_data[field] is not None: # Skip if already found by structured excel path
                continue

            for pattern in field_patterns:
                # Use re.IGNORECASE for all findall calls to be robust
                matches = re.findall(pattern, all_text, re.IGNORECASE) 
                if matches:
                    extracted_raw_value = matches[0]
                    # Determine the actual string value from the match result
                    if isinstance(extracted_raw_value, tuple):
                        # If it's a tuple, iterate from the end to find the first non-empty string
                        value_to_clean = next((v for v in reversed(extracted_raw_value) if v), '')
                    else:
                        value_to_clean = extracted_raw_value
                    
                    cleaned_value = self.clean_extracted_value(value_to_clean.strip(), field) # Strip here
                    if cleaned_value: # Only assign if cleaned value is not None
                        student_data[field] = cleaned_value
                        break
            
            # If not found by regex, try fuzzy matching in lines
            if not student_data[field]:
                fuzzy_value = self.fuzzy_field_extraction(lines, field)
                if fuzzy_value: # Only assign if fuzzy value is not None
                    student_data[field] = fuzzy_value
        
        # Post-process name splitting:
        # If full_name was found, ensure surname and first_name are populated.
        # If surname and first_name were found separately, combine for full_name.
        if student_data['full_name'] and not (student_data['surname'] and student_data['first_name']):
            student_data['surname'], student_data['first_name'] = self.split_full_name(student_data['full_name'])
        elif student_data['surname'] and student_data['first_name'] and not student_data['full_name']:
            student_data['full_name'] = f"{student_data['surname']}, {student_data['first_name']}"
        elif not student_data['full_name'] and not student_data['surname'] and not student_data['first_name']:
            # Fallback: try to find a name-like string if no specific pattern matched
            potential_name = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}(?:,\s*[A-Z][a-z]+)?)', text_content, re.IGNORECASE) # More robust name pattern
            if potential_name:
                student_data['full_name'] = self.clean_extracted_value(potential_name.group(0), 'full_name')
                student_data['surname'], student_data['first_name'] = self.split_full_name(potential_name.group(0))
        
        # Ensure all fields are strings (or appropriate types) before returning, to prevent NoneType errors in metadata
        for key, value in student_data.items():
            if value is None:
                student_data[key] = '' # Default to empty string for missing string fields
            elif key == 'year' and not isinstance(value, (int, str)): # For year, ensure it's a string or int
                student_data[key] = str(value) if value is not None else ''

        return student_data

    def clean_extracted_value(self, value, field_type):
        """Clean and validate extracted values, removing common noise and formatting."""
        if not value:
            return None
        
        value = value.strip()
        
        # Filter out common header values that might be picked up by mistake
        header_values = [
            'SURNAME', 'FIRST NAME', 'GUARDIAN NAME', 'CONTACT NUMBER', 'STUDENT ID', 
            'YEAR', 'COURSE', 'SECTION', 'ID', 'NAME', 'PROGRAM', 'TEL', 'MOBILE', 
            'PHONE', 'PARENT NAME', 'EMERGENCY CONTACT', 'DESCRIPTION', 'UNITS', 
            'DAY', 'TIME', 'ROOM', 'ADVISER', 'LEVEL', 'NO', 'STUDENT', 'ADDRESS',
            'COURSE CODE', 'SUBJECT', 'SCHEDULE', 'CLASS', 'INSTRUCTOR', 'ROOM NO',
            'FULL NAME', 'YEAR COURSE SECTION', 'ST', 'BSCS', 'BSIT', 'BSHM', 'BSTM', 'BSOA', 'BECED', 'BTLE', # Added common course names
            'COLLEGE', 'DEPARTMENT', 'ACADEMIC YEAR', 'SEMESTER', 'ENROLLMENT', 'REGISTRATION',
            'PDM-', 'YEAR COURSE SECTION PDM-', # Specific problematic strings from output
            'STUDENT ID NAME YEAR COURSE SECTION' # Another problematic header
        ]
        if value.upper() in header_values:
            return None
        
        if field_type == 'student_id':
            # Keep only alphanumeric and dashes, remove extra spaces
            cleaned = re.sub(r'[^A-Z0-9-]', '', value.upper())
            return cleaned if cleaned else None
        
        elif field_type in ['contact_number', 'guardian_contact']:
            # Clean phone numbers: keep digits and '+'
            cleaned = re.sub(r'[^\d\+]', '', value)
            # Ensure it's a plausible phone number length (e.g., 7-15 digits)
            if 7 <= len(cleaned) <= 15: 
                return cleaned
            return None # Discard if too short/long after cleaning
        
        elif field_type in ['full_name', 'guardian_name', 'surname', 'first_name']:
            # Clean names: keep letters, spaces, dots, commas, hyphens. Capitalize words.
            cleaned = re.sub(r'[^A-Za-z\s\.,-]', '', value).title()
            # Remove isolated single letters unless they are part of a valid initial
            cleaned = re.sub(r'\b[A-Za-z]\b(?!\.)', '', cleaned).strip() 
            return cleaned if cleaned and len(cleaned) > 1 else None # Discard very short strings
        
        elif field_type == 'year':
            # Extract just the number for year level (1-4)
            year_match = re.search(r'([1-4])', value)
            return year_match.group(1) if year_match else None
        
        elif field_type in ['course', 'section']:
            # Keep uppercase letters and numbers, remove spaces/special chars
            cleaned = re.sub(r'[^A-Z0-9]', '', value.upper())
            return cleaned if cleaned else None
        
        return value

    def fuzzy_field_extraction(self, lines, field_type):
        """
        Fuzzy extraction when pattern matching fails, looking for keywords and
        attempting to extract the value from the same line or nearby.
        """
        field_keywords = {
            'student_id': ['id', 'student no', 'student number', 'matric no', 's.i.d', 'std id', 'enrollment no'], # Added 'enrollment no'
            'full_name': ['name', 'student name', 'full name', 'student', 'examinee'], # Added 'examinee'
            'surname': ['surname', 'last name', 'family name'],
            'first_name': ['first name', 'given name'],
            'year': ['year', 'year level', 'level', 'yr', 'grade'], # Added 'grade'
            'course': ['course', 'program', 'degree', 'major', 'strand'], # Added 'strand'
            'section': ['section', 'sec', 'class', 'block'], # Added 'block'
            'contact_number': ['contact', 'phone', 'mobile', 'tel no', 'cell no', 'contact no'],
            'guardian_name': ['guardian', 'parent', 'emergency contact name', 'mother', 'father'],
            'guardian_contact': ['guardian contact', 'parent contact', 'emergency contact no', 'mother contact', 'father contact']
        }
        
        keywords = field_keywords.get(field_type, [])
        
        # Search in the current line and the next few lines
        for line_idx, line in enumerate(lines):
            line_upper = line.upper()
            for keyword in keywords:
                if keyword.upper() in line_upper:
                    # Try to extract value after colon, equals, or space
                    value = None
                    # Prioritize extracting after a colon or equals sign if present
                    if ':' in line:
                        value = line.split(':', 1)[1].strip()
                    elif '=' in line:
                        value = line.split('=', 1)[1].strip()
                    else:
                        # If no separator, try to get text after the keyword
                        parts = re.split(re.escape(keyword), line, 1, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            value = parts[1].strip()
                    
                    # If value is found in current line and is not just the keyword itself
                    if value and len(value) > 1 and value.upper() not in keywords:
                        cleaned_value = self.clean_extracted_value(value, field_type)
                        if cleaned_value:
                            return cleaned_value
                    
                    # If value not found or too short in current line, look in the next 1-5 lines (increased range)
                    for offset in range(1, 6): # Check next 5 lines
                        if line_idx + offset < len(lines):
                            next_line = lines[line_idx + offset].strip()
                            if next_line and len(next_line) > 1:
                                # Try to clean and return the next line as the value
                                cleaned_value = self.clean_extracted_value(next_line, field_type)
                                if cleaned_value:
                                    return cleaned_value
        return None

    def split_full_name(self, full_name):
        """Split full name into surname and first name (Surname, First Name or First Name Surname)"""
        if not full_name:
            return None, None
        
        full_name = full_name.strip()
        
        # Case 1: "Surname, First Name Middle Name"
        if ',' in full_name:
            parts = full_name.split(',', 1)
            surname = parts[0].strip()
            first_name_parts = parts[1].strip().split()
            first_name = first_name_parts[0] if first_name_parts else None
            return surname, first_name
        
        # Case 2: "First Name Middle Name Surname" (common for unstructured text)
        name_parts = full_name.split()
        if len(name_parts) >= 2:
            # Simple heuristic: last word is surname, rest is first name
            surname = name_parts[-1]
            first_name = ' '.join(name_parts[:-1])
            return surname, first_name
        elif len(name_parts) == 1:
            return name_parts[0], None # Only one part, assume it's the surname
        
        return None, None

    def split_into_student_records(self, text):
        """
        Split a large block of text (from PDF) into individual student records.
        This uses multiple strategies to identify record boundaries.
        """
        # Define patterns for strong record delimiters (Student ID or Full Name)
        # These patterns are now more comprehensive and include common name formats.
        delimiter_patterns = [
            r'(?:STUDENT\s*ID|ID\s*NO|ID|STUDENT\s*NUMBER)[:\s]*([A-Z0-9-]+)', # "STUDENT ID: PDM-123456"
            r'([A-Z]{2,4}-\d{4,6})',  # Pattern like PDM-123456, BSCS-2023
            r'(\d{4,8})',  # Pure numbers, e.g., 20230001
            r'(?:FULL\s*NAME|STUDENT\s*NAME|NAME)[:\s]*([A-Z][A-Z\s\.,-]+)', # "NAME: JOHN DOE"
            r'([A-Z][A-Z\s\.,-]+(?:,\s*[A-Z][A-Z\s\.,-]+)?)\s*(?:STUDENT ID|ID NO|YEAR|COURSE|SECTION|CONTACT)', # Name before another field
            r'([A-Z][A-Z\s\.,-]+)\s*\d{4,8}', # Name followed by a number (potential ID)
            r'^\s*([A-Z][a-z]+(?:[\s,\.]+[A-Z][a-z]+){1,})\s*$', # Full name on its own line (e.g., "DOE, JOHN" or "John Doe")
            r'^\s*(?:STUDENT|PUPIL)\s*INFORMATION\s*$', # NEW: "STUDENT INFORMATION" as a potential block start
            r'^\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?\s*$', # NEW: First Middle Last name on its own line
            r'^\s*PAGE\s*\d+\s*OF\s*\d+\s*$', # NEW: Page number as a potential delimiter (to avoid merging across pages)
            r'^\s*-+\s*\d+\s*-+\s*$', # NEW: Page number in dashes (e.g., --- 1 ---)
        ]
        
        all_lines = [line.strip() for line in text.split('\n') if line.strip()]
        records = []
        
        # Collect all potential starting points of records (line index)
        potential_record_starts = []
        for i, line in enumerate(all_lines):
            for pattern in delimiter_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    potential_record_starts.append(i)
                    break # Found a delimiter for this line, move to next line
        
        # Remove duplicates and sort
        potential_record_starts = sorted(list(set(potential_record_starts)))
        
        if not potential_record_starts:
            # Fallback: if no strong delimiters found, assume each "significant" block is a record
            # This is a very weak fallback, but better than nothing.
            current_block = []
            for line in all_lines:
                # Consider lines with more than 2 words or containing key student info keywords as significant
                if len(line.split()) > 2 or re.search(r'(ID|NAME|STUDENT|COURSE|YEAR|SECTION|CONTACT|GUARDIAN|PROGRAM|LEVEL)', line, re.IGNORECASE):
                    current_block.append(line)
                elif current_block: # If a significant block was being built and we hit an empty/insignificant line
                    records.append('\n'.join(current_block))
                    current_block = []
            if current_block: # Add the last block if any
                records.append('\n'.join(current_block))
            return records if records else [text.strip()] if text.strip() else []


        # Now, construct records based on these starting points
        for i, start_line_idx in enumerate(potential_record_starts):
            end_line_idx = len(all_lines)
            if i + 1 < len(potential_record_starts):
                end_line_idx = potential_record_starts[i+1]
            
            # Capture a block of lines for the current record
            record_lines = all_lines[start_line_idx:end_line_idx]
            
            # Join lines to form the record text
            record_content = '\n'.join(record_lines).strip()
            
            if record_content:
                records.append(record_content)
        
        # Filter out any records that are too short to be meaningful student data
        # A student record should have at least an ID/Name and maybe a few other fields.
        final_records = []
        for record_text in records:
            # A simple heuristic: if it contains an ID or a name-like pattern
            # or if it contains multiple key student info fields
            # Increased minimum length for a valid record
            if (re.search(r'(ID|NAME|STUDENT ID|FULL NAME|BSCS|BSIT|YEAR|SECTION|\d{4,8})', record_text, re.IGNORECASE) and
                len(record_text.splitlines()) > 2 and len(record_text.split()) > 10): # Must have a key identifier AND be sufficiently long
                final_records.append(record_text)
            elif (re.search(r'YEAR', record_text, re.IGNORECASE) and re.search(r'COURSE', record_text, re.IGNORECASE) and
                  len(record_text.splitlines()) > 2 and len(record_text.split()) > 10): # If year and course are found, also consider
                final_records.append(record_text)
            # Add a fallback for very simple records that might not hit the above, but are clearly not headers
            elif (re.search(r'^\s*([A-Z][a-z]+(?:[\s,\.]+[A-Z][a-z]+){1,})\s*$', record_text) and # Name on its own line
                  re.search(r'\d{4,8}', record_text)): # and an ID somewhere
                final_records.append(record_text)

        return final_records if final_records else [text.strip()] if text.strip() else []
    
    
    def extract_universal_teaching_faculty_data(self, text_content, source_type):
        """Enhanced universal extractor that processes structured key-value pairs properly."""
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        faculty_data = {
            'surname': None, 'first_name': None, 'date_of_birth': None, 'place_of_birth': None,
            'citizenship': None, 'sex': None, 'height': None, 'weight': None, 'blood_type': None,
            'religion': None, 'civil_status': None, 'address': None, 'zip_code': None,
            'phone': None, 'email': None, 'position': None, 'department': None, 'employment_status': None,
            'father_name': None, 'father_dob': None, 'father_occupation': None,
            'mother_name': None, 'mother_dob': None, 'mother_occupation': None,
            'spouse_name': None, 'spouse_dob': None, 'spouse_occupation': None,
            'gsis': None, 'philhealth': None
        }
        
        # DEBUG: Print all lines to see what we're working with
        print(f"🔍 DEBUG: All lines in faculty data:")
        for i, line in enumerate(lines[:50]):  # Show first 50 lines
            print(f"   {i}: {line}")
        
        # Process line by line for structured data
        for line in lines:
            if not line or len(line.strip()) == 0:
                continue
                
            line_clean = line.strip()
            
            # Skip section headers
            if line_clean.upper() in ['PERSONAL INFORMATION', 'CONTACT INFORMATION', 'OCCUPATIONAL INFORMATION', 'FAMILY BACKGROUND', 'GOVERNMENT IDS', 'GOVERNMENT INFORMATION']:
                continue
            
            # Process key-value pairs with enhanced detection
            parts = None
            
            if ':' in line_clean:
                parts = line_clean.split(':', 1)
            elif len(line_clean.split()) >= 2:
                words = line_clean.split()
                if len(words) >= 2:
                    first_word = words[0].lower()
                    # Enhanced keyword detection including more department variations
                    if first_word in ['full', 'date', 'place', 'citizenship', 'sex', 'height', 
                                    'weight', 'blood', 'religion', 'civil', 'address', 'zip', 
                                    'phone', 'email', 'position', 'department', 'employment',
                                    'father', 'mother', 'spouse', 'gsis', 'philhealth',
                                    'college', 'school', 'division', 'office']:
                        
                        if first_word == 'full' and len(words) > 2 and words[1].lower() == 'name':
                            parts = ['Full Name', ' '.join(words[2:])]
                        elif first_word == 'date' and len(words) > 3 and words[1].lower() == 'of':
                            parts = ['Date of Birth', ' '.join(words[3:])]
                        elif first_word == 'place' and len(words) > 3 and words[1].lower() == 'of':
                            parts = ['Place of Birth', ' '.join(words[3:])]
                        elif first_word == 'civil' and len(words) > 2 and words[1].lower() == 'status':
                            parts = ['Civil Status', ' '.join(words[2:])]
                        elif first_word == 'blood' and len(words) > 2 and words[1].lower() == 'type':
                            parts = ['Blood Type', ' '.join(words[2:])]
                        elif first_word == 'zip' and len(words) > 2 and words[1].lower() == 'code':
                            parts = ['Zip Code', ' '.join(words[2:])]
                        elif first_word == 'employment' and len(words) > 2 and words[1].lower() == 'status':
                            parts = ['Employment Status', ' '.join(words[2:])]
                        elif first_word == 'father' and 'name' in ' '.join(words[1:]).lower():
                            parts = ['Father Name', ' '.join(words[2:]) if len(words) > 2 else '']
                        elif first_word == 'mother' and 'name' in ' '.join(words[1:]).lower():
                            parts = ['Mother Name', ' '.join(words[2:]) if len(words) > 2 else '']
                        elif first_word == 'spouse' and 'name' in ' '.join(words[1:]).lower():
                            parts = ['Spouse Name', ' '.join(words[2:]) if len(words) > 2 else '']
                        elif first_word == 'gsis':
                            parts = ['GSIS', ' '.join(words[1:])]
                        elif first_word == 'philhealth':
                            parts = ['PhilHealth', ' '.join(words[1:])]
                        # ENHANCED: More department detection patterns
                        elif first_word in ['department', 'college', 'school', 'division', 'office']:
                            parts = ['Department', ' '.join(words[1:])]
                        else:
                            parts = [words[0], ' '.join(words[1:])]
            
            if parts and len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                if not value or value.lower() in ['n/a', 'na', '']:
                    continue
                
                # DEBUG: Show what we're processing
                print(f"🔍 Processing: {key} = {value}")
                
                # Map keys to faculty data fields - ENHANCED
                key_lower = key.lower()
                
                if 'full name' in key_lower:
                    if ',' in value:
                        name_parts = value.split(',', 1)
                        faculty_data['surname'] = name_parts[0].strip()
                        faculty_data['first_name'] = name_parts[1].strip()
                    else:
                        name_parts = value.split()
                        if len(name_parts) >= 2:
                            faculty_data['first_name'] = name_parts[0]
                            faculty_data['surname'] = ' '.join(name_parts[1:])
                            
                elif 'date of birth' in key_lower or key_lower == 'birthday':
                    faculty_data['date_of_birth'] = value
                elif 'place of birth' in key_lower:
                    faculty_data['place_of_birth'] = value
                elif 'citizenship' in key_lower:
                    faculty_data['citizenship'] = value
                elif key_lower == 'sex' or key_lower == 'gender':
                    faculty_data['sex'] = value
                elif 'height' in key_lower:
                    faculty_data['height'] = value
                elif 'weight' in key_lower:
                    faculty_data['weight'] = value
                elif 'blood type' in key_lower:
                    faculty_data['blood_type'] = value
                elif 'religion' in key_lower:
                    faculty_data['religion'] = value
                elif 'civil status' in key_lower or 'marital status' in key_lower:
                    faculty_data['civil_status'] = value
                elif 'address' in key_lower:
                    faculty_data['address'] = value
                elif 'zip code' in key_lower:
                    faculty_data['zip_code'] = value
                elif 'phone' in key_lower or 'mobile' in key_lower:
                    faculty_data['phone'] = value
                elif 'email' in key_lower:
                    faculty_data['email'] = value
                elif 'position' in key_lower:
                    faculty_data['position'] = value
                # ENHANCED: Better department detection
                elif any(dept_word in key_lower for dept_word in ['department', 'college', 'school', 'division', 'office']):
                    faculty_data['department'] = value
                    print(f"🎯 Found department: {value}")
                elif 'employment status' in key_lower:
                    faculty_data['employment_status'] = value
                elif 'father' in key_lower and 'name' in key_lower:
                    faculty_data['father_name'] = value
                elif 'mother' in key_lower and 'name' in key_lower:
                    faculty_data['mother_name'] = value
                elif 'spouse' in key_lower and 'name' in key_lower:
                    faculty_data['spouse_name'] = value
                elif 'gsis' in key_lower:
                    faculty_data['gsis'] = value
                elif 'philhealth' in key_lower or 'phil health' in key_lower:
                    faculty_data['philhealth'] = value
        
        # FALLBACK: Infer department from position if department is missing
        if not faculty_data['department'] and faculty_data['position']:
            faculty_data['department'] = self.infer_department_from_position(faculty_data['position'])
            print(f"🔍 Inferred department from position '{faculty_data['position']}': {faculty_data['department']}")
        
        # Clean extracted data
        for key, value in faculty_data.items():
            if value:
                faculty_data[key] = self.clean_teaching_faculty_value(value, key)
        
        print(f"🔍 Final faculty data: {faculty_data}")
        return faculty_data
    
    
    def infer_department_from_position(self, position):
        """Enhanced department inference from faculty position"""
        if not position:
            return None
        
        position_upper = position.upper()
        
        # Enhanced position-based mappings
        if 'DEAN' in position_upper:
            # College Dean - could be any department, check other context
            if any(word in position_upper for word in ['COMPUTER', 'TECHNOLOGY', 'IT']):
                return 'CCS'
            elif any(word in position_upper for word in ['BUSINESS', 'ADMIN']):
                return 'CBA'
            elif any(word in position_upper for word in ['HOSPITALITY', 'TOURISM']):
                return 'CHTM'
            elif any(word in position_upper for word in ['EDUCATION']):
                return 'CTE'
            elif any(word in position_upper for word in ['ENGINEERING']):
                return 'COE'
            elif any(word in position_upper for word in ['NURSING']):
                return 'CON'
            else:
                return 'ADMIN'  # General administrative position
        
        # Specific subject/department professors
        if any(word in position_upper for word in ['COMPUTER', 'IT', 'PROGRAMMING', 'SOFTWARE']):
            return 'CCS'
        elif any(word in position_upper for word in ['BUSINESS', 'ACCOUNTING', 'FINANCE', 'MARKETING']):
            return 'CBA'
        elif any(word in position_upper for word in ['HOSPITALITY', 'TOURISM', 'CULINARY']):
            return 'CHTM'
        elif any(word in position_upper for word in ['EDUCATION', 'TEACHING']):
            return 'CTE'
        elif any(word in position_upper for word in ['ENGINEERING', 'MECHANICAL', 'ELECTRICAL']):
            return 'COE'
        elif any(word in position_upper for word in ['NURSING', 'HEALTH']):
            return 'CON'
        
        return None
    
    
    def extract_faculty_line_by_line(self, lines, faculty_data):
        """Extract faculty data line by line when patterns fail"""
        field_keywords = {
            'surname': ['surname', 'last name', 'family name'],
            'first_name': ['first name', 'given name', 'firstname'],
            'position': ['position', 'title', 'rank', 'designation'],
            'department': ['department', 'college', 'dept'],
            'email': ['email', 'e-mail'],
            'phone': ['phone', 'mobile', 'contact', 'tel'],
            'date_of_birth': ['date of birth', 'birthday', 'birth date'],
            'sex': ['sex', 'gender'],
            'civil_status': ['civil status', 'marital status'],
        }
        
        for line in lines:
            line_upper = line.upper()
            
            # Check if line contains a colon (key-value format)
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().upper()
                    value = parts[1].strip()
                    
                    for field, keywords in field_keywords.items():
                        if faculty_data[field]:  # Skip if already found
                            continue
                        
                        for keyword in keywords:
                            if keyword.upper() in key:
                                cleaned_value = self.clean_teaching_faculty_value(value, field)
                                if cleaned_value:
                                    faculty_data[field] = cleaned_value
                                    print(f"   ✅ Line extraction found {field}: {cleaned_value}")
                                break
        
        return faculty_data
    

    def clean_teaching_faculty_value(self, value, field_type):
        """Enhanced cleaning for teaching faculty field values"""
        if not value or len(value.strip()) == 0:
            return None
        
        value = value.strip()
        
        # Filter out header values and common noise
        header_values = [
            'SURNAME', 'FIRST NAME', 'DATE OF BIRTH', 'PLACE OF BIRTH', 'CITIZENSHIP',
            'SEX', 'HEIGHT', 'WEIGHT', 'BLOOD TYPE', 'RELIGION', 'CIVIL STATUS',
            'ADDRESS', 'ZIP CODE', 'PHONE', 'EMAIL', 'POSITION', 'DEPARTMENT',
            'EMPLOYMENT STATUS', 'FATHER', 'MOTHER', 'SPOUSE', 'GSIS', 'PHILHEALTH',
            'PERSONAL INFORMATION', 'CONTACT INFORMATION', 'OCCUPATIONAL INFORMATION',
            'GOVERNMENT IDS', 'GOVERNMENT INFORMATION'
        ]
        if value.upper() in header_values:
            return None
        
        # Remove common noise phrases
        noise_phrases = ['DATE OF BIRTH', 'PLACE OF BIRTH', 'CONTACT NUMBER', 'PHONE NUMBER']
        for noise in noise_phrases:
            value = re.sub(re.escape(noise), '', value, flags=re.IGNORECASE).strip()
        
        if field_type in ['surname', 'first_name', 'father_name', 'mother_name', 'spouse_name']:
            cleaned = re.sub(r'[^A-Za-z\s\.,-]', '', value)
            cleaned = re.sub(r'\b(DATE|BIRTH|OF|PLACE)\b', '', cleaned, flags=re.IGNORECASE)
            cleaned = ' '.join(cleaned.split()).title()
            return cleaned if cleaned and len(cleaned) > 1 else None
        
        elif field_type in ['date_of_birth', 'father_dob', 'mother_dob', 'spouse_dob']:
            date_match = re.search(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', value)
            return date_match.group(1) if date_match else None
        
        elif field_type == 'sex':
            sex_upper = value.upper()
            if 'MALE' in sex_upper and 'FEMALE' not in sex_upper:
                return 'Male'
            elif 'FEMALE' in sex_upper:
                return 'Female'
            return None
        
        elif field_type == 'phone':
            phone_match = re.search(r'(\d{11}|\+63\d{10}|09\d{9})', value)
            return phone_match.group(1) if phone_match else None
        
        elif field_type == 'email':
            email_match = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', value)
            return email_match.group(1).lower() if email_match else None
        
        elif field_type == 'department':
            cleaned = re.sub(r'[^A-Za-z\s&]', '', value)
            cleaned = ' '.join(cleaned.split()).title()
            # FIX: Allow 2-character department codes like "IT"
            return self.standardize_department_name(cleaned) if len(cleaned) >= 2 else None
        
        elif field_type == 'position':
            cleaned = re.sub(r'[^A-Za-z\s\.]', '', value)
            cleaned = ' '.join(cleaned.split()).title()
            return cleaned if len(cleaned) > 2 else None
        
        # ENHANCED: Government ID cleaning with better formatting
        elif field_type in ['gsis', 'philhealth']:
            # Remove common prefixes and noise words
            cleaned = value.upper()
            
            # Remove common noise words that might appear before the number
            noise_words = ['NUMBER', 'NO', 'ID', 'GSIS', 'PHILHEALTH', 'PHIL', 'HEALTH']
            for word in noise_words:
                cleaned = re.sub(rf'\b{word}\b', '', cleaned).strip()
            
            # Keep only alphanumeric characters and dashes
            cleaned = re.sub(r'[^A-Z0-9\-]', '', cleaned)
            
            # Format GSIS numbers (typically 11 digits)
            if field_type == 'gsis' and cleaned:
                # GSIS numbers are usually 11 digits, format as XXX-XX-XXXXXX or similar
                if len(cleaned) == 11 and cleaned.isdigit():
                    return f"{cleaned[:2]}-{cleaned[2:9]}-{cleaned[9:]}"
                elif len(cleaned) >= 8:  # Accept other lengths but don't format
                    return cleaned
            
            # Format PhilHealth numbers (typically 12 digits)
            elif field_type == 'philhealth' and cleaned:
                # PhilHealth numbers are usually 12 digits, format as XX-XXXXXXXXX-X
                if len(cleaned) == 12 and cleaned.isdigit():
                    return f"{cleaned[:2]}-{cleaned[2:11]}-{cleaned[11:]}"
                elif len(cleaned) >= 8:  # Accept other lengths but don't format
                    return cleaned
            
            return cleaned if len(cleaned) >= 3 else None
        
        else:
            # For other fields, basic cleaning
            cleaned = ' '.join(value.split()).title()
            return cleaned if len(cleaned) > 1 else None
        
        
    def process_teaching_faculty_excel(self, filename):
        """Enhanced Teaching Faculty Excel processing with smart department inference"""
        try:
            faculty_info = self.extract_teaching_faculty_excel_info_smart(filename)
            
            if not faculty_info:
                print("❌ Could not extract teaching faculty data from Excel")
                return False
                
            # SMART DEPARTMENT INFERENCE: Try multiple approaches
            department = faculty_info.get('department', '')
            
            # Method 1: Direct department extraction (already done)
            if not department or department in ['N/A', 'NA', '']:
                # Method 2: Infer from position
                if faculty_info.get('position'):
                    inferred_dept = self.infer_department_from_position(faculty_info['position'])
                    if inferred_dept:
                        department = inferred_dept
                        print(f"🔍 Inferred department from position: {department}")
            
            # Method 3: Infer from email domain
            if not department or department in ['N/A', 'NA', '']:
                if faculty_info.get('email'):
                    inferred_dept = self.infer_department_from_email(faculty_info['email'])
                    if inferred_dept:
                        department = inferred_dept
                        print(f"🔍 Inferred department from email: {department}")
            
            # Method 4: Infer from name patterns (if following department naming)
            if not department or department in ['N/A', 'NA', '']:
                if faculty_info.get('surname') or faculty_info.get('first_name'):
                    inferred_dept = self.infer_department_from_name_context(filename)
                    if inferred_dept:
                        department = inferred_dept
                        print(f"🔍 Inferred department from filename context: {department}")
            
            # Method 5: Default based on position type
            if not department or department in ['N/A', 'NA', '']:
                if faculty_info.get('position'):
                    position_upper = faculty_info['position'].upper()
                    if 'DEAN' in position_upper:
                        department = 'ADMIN'
                    elif 'PROFESSOR' in position_upper or 'INSTRUCTOR' in position_upper:
                        department = 'CAS'  # Default to Arts & Sciences for general teaching
                    print(f"🔍 Default department assignment: {department}")
            
            # Final fallback
            if not department or department in ['N/A', 'NA', '']:
                department = 'UNKNOWN'
            
            formatted_text = self.format_teaching_faculty_info_enhanced(faculty_info)
            
            # Create smart metadata with the determined department
            full_name = ""
            if faculty_info.get('surname') and faculty_info.get('first_name'):
                full_name = f"{faculty_info['surname']}, {faculty_info['first_name']}"
            elif faculty_info.get('surname'):
                full_name = faculty_info['surname']
            elif faculty_info.get('first_name'):
                full_name = faculty_info['first_name']
            else:
                full_name = "Unknown Faculty"
            
            # Update the faculty_info with the determined department
            faculty_info['department'] = department
            
            metadata = {
                'full_name': full_name,
                'surname': faculty_info.get('surname') or '',
                'first_name': faculty_info.get('first_name') or '',
                'department': self.standardize_department_name(department),
                'position': faculty_info.get('position') or '',
                'employment_status': faculty_info.get('employment_status') or '',
                'email': faculty_info.get('email') or '',
                'phone': faculty_info.get('phone') or '',
                'data_type': 'teaching_faculty_excel',
                'faculty_type': 'teaching',
            }
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            
            # Update the formatted text to show the determined department
            updated_formatted_text = self.format_teaching_faculty_info_enhanced(faculty_info)
            
            self.store_with_smart_metadata(collection, [updated_formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            hierarchy_path = f"{self.get_department_display_name(metadata['department'])} > Teaching Faculty"
            print(f"✅ Loaded teaching faculty data into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   👨‍🏫 Faculty: {metadata['full_name']} ({metadata['position']})")
            return True
            
        except Exception as e:
            print(f"❌ Error processing teaching faculty Excel: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    
    def infer_department_from_name_context(self, filename):
        """Infer department from filename context"""
        filename_lower = filename.lower()
        
        if any(dept in filename_lower for dept in ['ccs', 'computer', 'cs']):
            return 'CCS'
        elif any(dept in filename_lower for dept in ['cba', 'business', 'admin']):
            return 'CBA'
        elif any(dept in filename_lower for dept in ['chtm', 'hospitality', 'tourism']):
            return 'CHTM'
        elif any(dept in filename_lower for dept in ['cte', 'education']):
            return 'CTE'
        elif any(dept in filename_lower for dept in ['coe', 'engineering']):
            return 'COE'
        elif any(dept in filename_lower for dept in ['con', 'nursing']):
            return 'CON'
        
        return None
        
    
    def infer_department_from_email(self, email):
        """Infer department from email domain or prefix"""
        if not email:
            return None
        
        email_lower = email.lower()
        
        # Check email prefixes that might indicate department
        if any(prefix in email_lower for prefix in ['cs.', 'ccs.', 'compsci', 'computer']):
            return 'CCS'
        elif any(prefix in email_lower for prefix in ['business', 'admin', 'acct', 'finance']):
            return 'CBA'
        elif any(prefix in email_lower for prefix in ['hospitality', 'tourism', 'hotel']):
            return 'CHTM'
        elif any(prefix in email_lower for prefix in ['education', 'teaching']):
            return 'CTE'
        elif any(prefix in email_lower for prefix in ['engineering']):
            return 'COE'
        elif any(prefix in email_lower for prefix in ['nursing', 'health']):
            return 'CON'
        
        return None
        
        
    def extract_teaching_faculty_excel_info_smart(self, filename):
        """Universal teaching faculty extraction that works with ANY Excel format"""
        try:
            df_full = pd.read_excel(filename, header=None)
            print(f"📋 Teaching Faculty Excel dimensions: {df_full.shape}")
            
            # DEBUG: Show the actual Excel content
            print(f"📋 Raw Excel content (first 20 rows):")
            for i in range(min(20, df_full.shape[0])):
                row_data = []
                for j in range(min(df_full.shape[1], 3)):  # Show first 3 columns
                    if pd.notna(df_full.iloc[i, j]):
                        row_data.append(f"'{str(df_full.iloc[i, j])}'")
                    else:
                        row_data.append("'N/A'")
                print(f"   Row {i}: {row_data}")
            
            # Convert entire sheet to text and use universal extractor
            all_text = ""
            for i in range(df_full.shape[0]):
                for j in range(df_full.shape[1]):
                    if pd.notna(df_full.iloc[i, j]):
                        all_text += str(df_full.iloc[i, j]) + " "
                all_text += "\n"
            
            faculty_info = self.extract_universal_teaching_faculty_data(all_text, 'excel_universal')
            
            if faculty_info and (faculty_info.get('surname') or faculty_info.get('first_name') or 
                            faculty_info.get('position') or faculty_info.get('department')):
                print("✅ Universal teaching faculty extraction successful")
                return faculty_info
            
            print("❌ Could not extract sufficient teaching faculty data")
            return None
            
        except Exception as e:
            print(f"❌ Error in teaching faculty extraction: {e}")
            return None
        
    
    def format_teaching_faculty_info_enhanced(self, faculty_info):
        """Enhanced teaching faculty formatting with clean display"""
        
        # Helper function to format field
        def format_field(value):
            if value and value not in ['None', 'N/A', '']:
                return value
            return 'N/A'
        
        text = f"""TEACHING FACULTY INFORMATION

    PERSONAL INFORMATION:
    Surname: {format_field(faculty_info.get('surname'))}
    First Name: {format_field(faculty_info.get('first_name'))}
    Date of Birth: {format_field(faculty_info.get('date_of_birth'))}
    Place of Birth: {format_field(faculty_info.get('place_of_birth'))}
    Citizenship: {format_field(faculty_info.get('citizenship'))}
    Sex: {format_field(faculty_info.get('sex'))}
    Height: {format_field(faculty_info.get('height'))}
    Weight: {format_field(faculty_info.get('weight'))}
    Blood Type: {format_field(faculty_info.get('blood_type'))}
    Religion: {format_field(faculty_info.get('religion'))}
    Civil Status: {format_field(faculty_info.get('civil_status'))}

    CONTACT INFORMATION:
    Address: {format_field(faculty_info.get('address'))}
    Zip Code: {format_field(faculty_info.get('zip_code'))}
    Phone: {format_field(faculty_info.get('phone'))}
    Email: {format_field(faculty_info.get('email'))}

    PROFESSIONAL INFORMATION:
    Position: {format_field(faculty_info.get('position'))}
    Department: {format_field(faculty_info.get('department'))}
    Employment Status: {format_field(faculty_info.get('employment_status'))}"""

        # Only show family info if at least one field has data
        family_fields = ['father_name', 'father_dob', 'father_occupation', 
                        'mother_name', 'mother_dob', 'mother_occupation',
                        'spouse_name', 'spouse_dob', 'spouse_occupation']
        
        has_family_data = any(faculty_info.get(field) and faculty_info.get(field) not in ['None', 'N/A', ''] 
                            for field in family_fields)
        
        if has_family_data:
            text += f"""

    FAMILY INFORMATION:
    Father's Name: {format_field(faculty_info.get('father_name'))}
    Father's Date of Birth: {format_field(faculty_info.get('father_dob'))}
    Father's Occupation: {format_field(faculty_info.get('father_occupation'))}

    Mother's Name: {format_field(faculty_info.get('mother_name'))}
    Mother's Date of Birth: {format_field(faculty_info.get('mother_dob'))}
    Mother's Occupation: {format_field(faculty_info.get('mother_occupation'))}

    Spouse's Name: {format_field(faculty_info.get('spouse_name'))}
    Spouse's Date of Birth: {format_field(faculty_info.get('spouse_dob'))}
    Spouse's Occupation: {format_field(faculty_info.get('spouse_occupation'))}"""

        # Only show government IDs if at least one is available
        gsis = faculty_info.get('gsis')
        philhealth = faculty_info.get('philhealth')
        
        if (gsis and gsis not in ['None', 'N/A', '']) or (philhealth and philhealth not in ['None', 'N/A', '']):
            text += f"""

    GOVERNMENT IDs:
    GSIS: {format_field(gsis)}
    PhilHealth: {format_field(philhealth)}"""
        
        return text.strip()
    
    
    def standardize_department_name(self, department):
        """Smart department standardization that handles abbreviations"""
        if not department:
            return 'UNKNOWN'
        
        dept_upper = department.upper().strip()
        
        # Handle direct abbreviations first
        if dept_upper == 'CCS':
            return 'CCS'
        # REMOVE THIS LINE: elif dept_upper == 'IT':
        #     return 'CCS'
        elif dept_upper == 'CTE':
            return 'CTE'
        elif dept_upper == 'CHTM':
            return 'CHTM'
        elif dept_upper == 'CBA':
            return 'CBA'
        elif dept_upper == 'COE':
            return 'COE'
        elif dept_upper == 'CAS':
            return 'CAS'
        elif dept_upper == 'CON':
            return 'CON'
        
        # Map full names to abbreviations
        dept_mappings = {
            'COLLEGE OF COMPUTER STUDIES': 'CCS',
            'COMPUTER STUDIES': 'CCS',
            'INFORMATION TECHNOLOGY': 'IT',  # Map full name to IT
            'COLLEGE OF EDUCATION': 'CTE',
            'EDUCATION': 'CTE',
            'COLLEGE OF HOSPITALITY': 'CHTM',
            'HOSPITALITY': 'CHTM',
            'TOURISM': 'CHTM',
            'COLLEGE OF BUSINESS': 'CBA',
            'BUSINESS': 'CBA',
            'OFFICE ADMINISTRATION': 'CBA',
            'COLLEGE OF ENGINEERING': 'COE',
            'ENGINEERING': 'COE',
            'COLLEGE OF ARTS': 'CAS',
            'ARTS AND SCIENCES': 'CAS',
            'COLLEGE OF NURSING': 'CON',
            'NURSING': 'CON',
        }
        
        # Check exact mappings
        for full_name, abbrev in dept_mappings.items():
            if full_name in dept_upper:
                return abbrev
        
        # If no mapping found, return as-is (cleaned)
        return dept_upper  # This will return "IT" as "IT"

    def intelligently_categorize_department(self, dept_upper):
        """Intelligently categorize unknown departments based on keywords"""
        
        # Technology-related keywords
        tech_keywords = ['COMPUTER', 'TECHNOLOGY', 'IT', 'DIGITAL', 'CYBER', 'SOFTWARE', 'DATA', 'PROGRAMMING']
        if any(keyword in dept_upper for keyword in tech_keywords):
            return 'CCS'
        
        # Education-related keywords
        education_keywords = ['EDUCATION', 'TEACHING', 'ELEMENTARY', 'SECONDARY', 'CHILDHOOD', 'PEDAGOGY', 'CURRICULUM']
        if any(keyword in dept_upper for keyword in education_keywords):
            return 'CTE'
        
        # Business-related keywords
        business_keywords = ['BUSINESS', 'MANAGEMENT', 'ADMINISTRATION', 'FINANCE', 'ACCOUNTING', 'MARKETING', 'ECONOMICS', 'OFFICE']
        if any(keyword in dept_upper for keyword in business_keywords):
            return 'CBA'
        
        # Hospitality/Tourism keywords
        hospitality_keywords = ['HOSPITALITY', 'TOURISM', 'HOTEL', 'CULINARY', 'RESTAURANT', 'FOOD', 'SERVICE', 'TRAVEL']
        if any(keyword in dept_upper for keyword in hospitality_keywords):
            return 'CHTM'
        
        # Engineering keywords
        engineering_keywords = ['ENGINEERING', 'MECHANICAL', 'ELECTRICAL', 'CIVIL', 'CHEMICAL', 'INDUSTRIAL', 'AEROSPACE']
        if any(keyword in dept_upper for keyword in engineering_keywords):
            return 'COE'
        
        # Health/Medical keywords
        health_keywords = ['NURSING', 'HEALTH', 'MEDICAL', 'MEDICINE', 'THERAPY', 'REHABILITATION', 'PHARMACY']
        if any(keyword in dept_upper for keyword in health_keywords):
            return 'CON'
        
        # Arts/Sciences keywords
        arts_keywords = ['ARTS', 'SCIENCES', 'LIBERAL', 'HUMANITIES', 'SOCIAL', 'PSYCHOLOGY', 'COMMUNICATION', 'ENGLISH', 'MATHEMATICS']
        if any(keyword in dept_upper for keyword in arts_keywords):
            return 'CAS'
        
        # If no category matches, create a new category
        return 'NEW_DEPT'


    # ======================== SMART HIERARCHY & METADATA ========================
    
    def extract_smart_metadata(self, text, file_type):
        """Enhanced smart extraction of organizational metadata from text"""
        metadata = {
            'course': None,
            'section': None, 
            'year_level': None,
            'department': None,
            'faculty_type': None,
            'data_type': file_type,
            'subject_codes': ''  # Change from list to string
        }
        
        text_upper = text.upper()
        
        # Enhanced course detection
        course_patterns = [
            r'COURSE[:\s]*([A-Z]{2,6})',
            r'PROGRAM[:\s]*([A-Z]{2,6})',
            r'BS[A-Z]{2,4}',
            r'AB[A-Z]*',
            r'BA[A-Z]*'
        ]
        
        for pattern in course_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                metadata['course'] = str(matches[0] if isinstance(matches[0], str) else matches[0])
                break
        
        # Enhanced section detection
        section_patterns = [
            r'SECTION[:\s]*([A-Z0-9-]+)',
            r'SEC[:\s]*([A-Z0-9-]+)',
            r'SECTION\s+([A-Z0-9-]+)'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                metadata['section'] = str(matches[0])
                break
        
        # Enhanced year level detection
        year_patterns = [
            r'YEAR\s*LEVEL[:\s]*([1-4])',
            r'YEAR[:\s]*([1-4])',
            r'([1-4])(?:ST|ND|RD|TH)?\s*YEAR'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                metadata['year_level'] = str(matches[0])
                break
        
        # Subject code extraction for schedules - convert to string
        subject_matches = re.findall(r'[A-Z]{2,4}\d{3}', text_upper)
        metadata['subject_codes'] = ', '.join(list(set(subject_matches)))  # Convert to comma-separated string
        
        # Smart department detection from course
        if metadata['course']:
            metadata['department'] = str(self.detect_department_from_course(metadata['course']))
        
        # Faculty type detection (enhanced)
        if 'faculty' in file_type.lower():
            if any(keyword in text.lower() for keyword in ['schedule', 'class schedule', 'teaching']):
                metadata['faculty_type'] = 'teaching'
            elif any(keyword in text.lower() for keyword in ['resume', 'cv', 'profile']):
                metadata['faculty_type'] = 'profile'
            else:
                metadata['faculty_type'] = 'general'
        
        # Ensure all values are valid types
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ''
            elif not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)
        
        return metadata
    
    def contextual_department_inference(self, course_upper):
        """Infer department from course structure and common patterns"""
        
        # Analyze degree prefix patterns
        if course_upper.startswith('BS'):
            suffix = course_upper[2:]  # Remove 'BS'
            
            # Technology indicators
            if any(indicator in suffix for indicator in ['TECH', 'DIGITAL', 'COMP', 'INFO', 'DATA', 'CYBER', 'SOFT', 'WEB']):
                return 'CCS'
            
            # Engineering indicators
            elif any(indicator in suffix for indicator in ['ENG', 'MECH', 'ELEC', 'CIV', 'IND', 'CHEM', 'AERO']):
                return 'COE'
            
            # Business indicators
            elif any(indicator in suffix for indicator in ['BUS', 'ADMIN', 'MANAGE', 'ACCT', 'FIN', 'MARKET', 'ECON']):
                return 'CBA'
            
            # Hospitality indicators
            elif any(indicator in suffix for indicator in ['HOST', 'TOUR', 'HOTEL', 'CULINARY', 'REST', 'FOOD']):
                return 'CHTM'
            
            # Health indicators
            elif any(indicator in suffix for indicator in ['NURS', 'HEALTH', 'MED', 'CARE', 'THERAPY']):
                return 'CON'
            
            # Education indicators
            elif any(indicator in suffix for indicator in ['ED', 'TEACH', 'CHILD', 'ELEM', 'SEC', 'SPEC']):
                return 'CED'
        
        elif course_upper.startswith('AB'):
            return 'CAS'  # Liberal arts
        
        elif course_upper.startswith('MA') or course_upper.startswith('MS'):
            # Master's degrees - try to infer from content
            if any(ed in course_upper for ed in ['ED', 'TEACH']):
                return 'CED'
            elif any(bus in course_upper for bus in ['BUS', 'ADMIN', 'MBA']):
                return 'CBA'
            elif any(tech in course_upper for tech in ['COMP', 'INFO', 'TECH']):
                return 'CCS'
        
        return 'UNKNOWN'

    def create_intelligent_category(self, course_upper):
        """Create intelligent category names for unknown courses"""
        
        # Try to create meaningful category based on course characteristics
        if course_upper.startswith('BS'):
            return 'EMERGING_BS'  # Bachelor of Science - Emerging Program
        elif course_upper.startswith('AB'):
            return 'EMERGING_AB'  # Bachelor of Arts - Emerging Program  
        elif course_upper.startswith('MA'):
            return 'GRADUATE_MA'  # Master of Arts Program
        elif course_upper.startswith('MS'):
            return 'GRADUATE_MS'  # Master of Science Program
        elif course_upper.startswith('PHD') or course_upper.startswith('DR'):
            return 'DOCTORAL'     # Doctoral Program
        else:
            return 'NEW_PROGRAM'  # Completely new program type

    def get_department_display_name(self, dept_code):
        """Enhanced department display names that handle new departments"""
        dept_names = {
            'CCS': 'College of Computer Studies',
            'IT': 'Information Technology Department',
            'COE': 'College of Engineering', 
            'CHTM': 'College of Hospitality & Tourism Management',
            'CBA': 'College of Business Administration',
            'CTE': 'College of Education',
            'CAS': 'College of Arts & Sciences',
            'CON': 'College of Nursing',
            'ADMIN': 'Administration',
            'NEW_DEPT': 'New Academic Department',
            'UNKNOWN': 'Unclassified Department',
            
            # Add non-teaching departments
            'REGISTRAR': 'Office of the Registrar',
            'ACCOUNTING': 'Accounting & Finance Office',
            'GUIDANCE': 'Guidance & Counseling Office',
            'LIBRARY': 'Library Services',
            'HEALTH_SERVICES': 'Health Services Office',
            'MAINTENANCE_CUSTODIAL': 'Maintenance & Custodial Services',
            'SECURITY': 'Security Services',
            'SYSTEM_ADMIN': 'Information Technology Services',
            'ADMIN_SUPPORT': 'Administrative Support Services',
        }
        return dept_names.get(dept_code, f'Department of {dept_code}')

    def process_non_teaching_faculty_pdf(self, filename):
        """Process Non-Teaching Faculty PDF file with smart organization"""
        try:
            faculty_data = self.extract_faculty_pdf_data(filename)
            if not faculty_data:
                print("❌ Could not extract non-teaching faculty data from PDF")
                return False
            
            # Try to infer department from content
            department = 'ADMIN_SUPPORT'
            lines = faculty_data.split('\n')
            
            # Look for department/position clues in PDF content
            for line in lines:
                line_upper = line.upper()
                if any(word in line_upper for word in ['REGISTRAR', 'REGISTRATION']):
                    department = 'REGISTRAR'
                    break
                elif any(word in line_upper for word in ['ACCOUNTING', 'FINANCE']):
                    department = 'ACCOUNTING'
                    break
                # ... add more department detection logic
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(faculty_data, 'non_teaching_faculty_pdf')
            metadata['faculty_type'] = 'non_teaching'
            metadata['department'] = self.standardize_non_teaching_department_name(department)
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [faculty_data], [metadata])
            self.collections[collection_name] = collection
            
            # Extract name from data for display
            lines = faculty_data.split('\n')
            faculty_name = lines[0] if lines else "Unknown Faculty"
            
            hierarchy_path = f"{self.get_non_teaching_department_display_name(metadata['department'])} > Non-Teaching Faculty"
            print(f"✅ Loaded non-teaching faculty resume into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   👨‍💼 Faculty: {faculty_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing non-teaching faculty PDF: {e}")
            return False
        
    def detect_department_from_course(self, course_code):
        """
        Enhanced smart department detection that handles new courses
        """
        if not course_code:
            return 'UNKNOWN'

        course_code_upper = str(course_code).upper().strip()

        # Handle full program names first
        if 'COMPUTER SCIENCE' in course_code_upper or 'INFORMATION TECHNOLOGY' in course_code_upper:
            return 'CCS'
        elif 'HOSPITALITY MANAGEMENT' in course_code_upper or 'TOURISM MANAGEMENT' in course_code_upper:
            return 'CHTM'
        elif 'BUSINESS ADMINISTRATION' in course_code_upper or 'OFFICE ADMINISTRATION' in course_code_upper:
            return 'CBA'
        elif 'EDUCATION' in course_code_upper:
            return 'CTE'
        elif 'ENGINEERING' in course_code_upper:
            return 'COE'
        elif 'NURSING' in course_code_upper:
            return 'CON'
        elif 'ARTS' in course_code_upper and 'SCIENCES' in course_code_upper:
            return 'CAS'

        # Handle known abbreviated codes
        known_courses = {
            'CCS': ['BSCS', 'BSIT'],
            'CHTM': ['BSHM', 'BSTM'],
            'CBA': ['BSBA', 'BSOA'],
            'CTE': ['BECED', 'BTLE'],
        }

        for dept, courses in known_courses.items():
            if course_code_upper in courses:
                return dept

        # SMART DETECTION: Analyze unknown course codes
        return self.intelligently_categorize_course(course_code_upper)
    
    
    def intelligently_categorize_course(self, course_code_upper):
        """Intelligently categorize unknown course codes"""
        
        # Analyze course code patterns and suffixes
        if course_code_upper.startswith('BS'):
            suffix = course_code_upper[2:]  # Remove 'BS'
            
            # Technology indicators
            tech_indicators = ['CS', 'IT', 'IS', 'SE', 'CE', 'CPE', 'CIS', 'MIT', 'TECH', 'COMP', 'DATA', 'CYBER']
            if any(indicator in suffix for indicator in tech_indicators):
                return 'CCS'
            
            # Business indicators
            business_indicators = ['BA', 'BM', 'FM', 'MM', 'AM', 'ECON', 'FIN', 'MKT', 'ACCT', 'ENT', 'OA', 'ADMIN']
            if any(indicator in suffix for indicator in business_indicators):
                return 'CBA'
            
            # Hospitality indicators
            hospitality_indicators = ['HM', 'TM', 'HTM', 'CHM', 'TOUR', 'HOSP', 'CULI', 'FB']
            if any(indicator in suffix for indicator in hospitality_indicators):
                return 'CHTM'
            
            # Education indicators
            education_indicators = ['ED', 'ELEM', 'SEC', 'ECE', 'TLE', 'SPED', 'TEACH']
            if any(indicator in suffix for indicator in education_indicators):
                return 'CTE'
            
            # Engineering indicators
            engineering_indicators = ['EE', 'ME', 'CE', 'IE', 'ChE', 'AE', 'ENG', 'TECH']
            if any(indicator in suffix for indicator in engineering_indicators):
                return 'COE'
            
            # Health indicators
            health_indicators = ['N', 'NURS', 'MED', 'PHAR', 'PT', 'RT', 'MT']
            if any(indicator in suffix for indicator in health_indicators):
                return 'CON'
            
            # If it's a BS degree but doesn't match known patterns, it might be sciences
            return 'CAS'
        
        elif course_code_upper.startswith('AB'):
            # Most AB degrees go to Arts & Sciences
            return 'CAS'
        
        elif course_code_upper.startswith('MA') or course_code_upper.startswith('MS'):
            # Graduate degrees - try to infer from content
            if any(keyword in course_code_upper for keyword in ['ED', 'TEACH']):
                return 'CTE'
            elif any(keyword in course_code_upper for keyword in ['BUS', 'ADMIN', 'MBA']):
                return 'CBA'
            elif any(keyword in course_code_upper for keyword in ['COMP', 'IT', 'TECH']):
                return 'CCS'
            else:
                return 'CAS'  # Default for graduate degrees
        
        # For completely unknown patterns
        return 'NEW_DEPT'
    

    def create_smart_collection_name(self, file_type, metadata):
        """Enhanced collection name creation that handles faculty properly"""
        base_name = ""
        if "student" in file_type.lower():
            base_name = "students"
        elif "cor" in file_type.lower() or "schedule" in file_type.lower():
            base_name = "schedules"
        elif "curriculum" in file_type.lower():  # NEW
            base_name = "curriculum"
        elif "teaching_faculty" in file_type.lower() or metadata.get('data_type') == 'teaching_faculty_excel':
            base_name = "faculty"
        elif "faculty" in file_type.lower():
            base_name = "faculty"
        else:
            base_name = "data_collection"

        # For curriculum, organize by department and program
        if base_name == "curriculum":
            department = metadata.get('department', 'unknown').lower()
            program = metadata.get('program', 'general').lower()
            
            # Clean program name for collection naming
            program_clean = re.sub(r'[^a-z0-9]', '', program.lower())
            
            if department and department not in ['unknown', 'new_dept']:
                name_parts = [base_name, department, program_clean]
            else:
                name_parts = [base_name, 'unclassified', program_clean]
            
            final_name = "_".join(filter(None, name_parts)).lower()
            return re.sub(r'_{2,}', '_', final_name).strip('_')
        
        # For faculty, organize by department and type
        elif base_name == "faculty":
            department = metadata.get('department', 'unknown').lower()
            faculty_type = metadata.get('faculty_type', 'general').lower()
            
            # Handle admin with admin_type distinction
            if faculty_type == 'admin':
                admin_type = metadata.get('admin_type', 'Administrator').lower().replace(' ', '_')
                faculty_type = admin_type  # Use admin_type as the faculty_type
            elif faculty_type == 'non_teaching_schedule':
                faculty_type = 'non_teaching_schedule'
            elif faculty_type == 'non_teaching':
                faculty_type = 'non_teaching'
            elif faculty_type == 'teaching':
                faculty_type = 'teaching'
            elif faculty_type == 'schedule':
                faculty_type = 'schedule'
            
            # Handle departments properly
            if department and department not in ['unknown', 'new_dept']:
                name_parts = [base_name, department, faculty_type]
            elif department == 'new_dept':
                name_parts = [base_name, 'newdept', faculty_type]
            else:
                name_parts = [base_name, 'unclassified', faculty_type]
            
            final_name = "_".join(filter(None, name_parts)).lower()
            return re.sub(r'_{2,}', '_', final_name).strip('_')
            
        # For students and schedules (existing logic)
        else:
            department = metadata.get('department', '').lower()
            course = metadata.get('course', '').lower()
            year = str(metadata.get('year_level', ''))
            section = metadata.get('section', '').lower()
            
            if course and course != 'unknown':
                course_match = re.match(r'^(BS[A-Z]{2,4}|AB[A-Z]{2,4}|B[A-Z]{2,4})', course.upper())
                if course_match:
                    course = course_match.group(1).lower()
                elif re.match(r'^[A-Z]{2,6}$', course.upper()):
                    course = course.lower()
                else:
                    course = 'newcourse'
            else:
                course = 'general'
            
            if department and department not in ['unknown', 'new_dept']:
                name_parts = [base_name, department]
            elif department == 'new_dept':
                name_parts = [base_name, 'newdept']
            else:
                name_parts = [base_name, 'unclassified']

            name_parts.append(course)

            if year and year not in ['0', '']:
                year_clean = re.sub(r'[^\d]', '', str(year))
                if year_clean:
                    name_parts.append(f"year{year_clean}")
            
            if section:
                section_clean = re.sub(r'[^a-zA-Z0-9]', '', section)
                if section_clean:
                    name_parts.append(f"sec{section_clean.lower()}")

            final_name = "_".join(filter(None, name_parts)).lower()
            return re.sub(r'_{2,}', '_', final_name).strip('_')

    def get_target_collection_name(self, data_type, metadata):
        """Helper method to get the collection name where data would be stored (for duplicate detection)"""
        try:
            # Map data_type to file_type for collection naming
            if data_type == 'curriculum':
                file_type = 'curriculum'
            elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
                file_type = 'faculty_schedule'
            elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
                file_type = 'faculty'
            elif data_type == 'student':
                file_type = 'student'
            elif data_type == 'cor_schedule':
                file_type = 'cor_schedule'
            else:
                file_type = 'unknown'
            
            # For faculty schedules, we need to set the right faculty_type in metadata
            if data_type == 'teaching_faculty_schedule':
                metadata_copy = metadata.copy()
                metadata_copy['faculty_type'] = 'schedule'
                return self.create_smart_collection_name(file_type, metadata_copy)
            elif data_type == 'non_teaching_faculty_schedule':
                metadata_copy = metadata.copy()
                metadata_copy['faculty_type'] = 'non_teaching_schedule'
                return self.create_smart_collection_name(file_type, metadata_copy)
            else:
                return self.create_smart_collection_name(file_type, metadata)
                
        except Exception as e:
            print(f"⚠️ Error getting target collection name: {e}")
            return 'unknown_collection'

    def store_with_smart_hierarchy(self, texts, metadata_list, base_type):
        """Store data with smart hierarchical organization"""
        try:
            # Group data by smart hierarchy
            hierarchy_groups = {}
            
            for text, metadata in zip(texts, metadata_list):
                # Create smart collection name based on hierarchy
                collection_name = self.create_smart_collection_name(base_type, metadata)
                
                if collection_name not in hierarchy_groups:
                    hierarchy_groups[collection_name] = {
                        'texts': [], 
                        'metadata': [],
                        'sample_meta': metadata
                    }
                
                hierarchy_groups[collection_name]['texts'].append(text)
                hierarchy_groups[collection_name]['metadata'].append(metadata)
            
            # Store each group in its own collection
            success_count = 0
            for collection_name, group_data in hierarchy_groups.items():
                try:
                    # Use get_or_create_collection with the consistent embedding function
                    collection = self.client.get_or_create_collection(
                        name=collection_name, 
                        embedding_function=self.embedding_function
                    )
                    self.store_with_smart_metadata(collection, group_data['texts'], group_data['metadata'])
                    self.collections[collection_name] = collection
                    
                    # Display organization info
                    sample = group_data['sample_meta']
                    hierarchy_path = f"{self.get_department_display_name(sample.get('department', 'Unknown'))} > {sample.get('course', 'Unknown')} > Year {sample.get('year_level', 'Unknown')} > Section {sample.get('section', 'Unknown')}"
                    
                    print(f"✅ Stored {len(group_data['texts'])} records in: {collection_name}")
                    print(f"   📁 Hierarchy: {hierarchy_path}")
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error storing {collection_name}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error in smart hierarchy storage: {e}")
            return False

    def store_with_smart_metadata(self, collection, texts, metadata_list):
        """Store embeddings with rich metadata for smart filtering"""
        for idx, (text, metadata) in enumerate(zip(texts, metadata_list)):
            # The embedding is generated by the collection's embedding_function implicitly
            # when you use collection.add. You don't need to explicitly call self.model.encode here.
            # However, if you *were* to encode manually, you'd use self.model.encode(text).tolist()
            
            # Create unique ID with metadata info
            doc_id = f"{metadata.get('course', 'unknown')}_{metadata.get('section', 'unknown')}_{idx}_{datetime.now().timestamp()}"
            
            collection.add(
                documents=[text],
                metadatas=[metadata],  # Store metadata for filtering
                ids=[doc_id]
            )
            
    def smart_search_with_ai_reasoning(self, query, max_results=50):
        """True AI-powered smart search with contextual understanding"""
        query_intent = self.analyze_query_intent(query)
        query_intent['query'] = query
        search_strategy = self.determine_search_strategy(query_intent)

        if self.debug_mode: print(f"🧠 AI Analysis: {query_intent['intent']} | Strategy: {search_strategy['type']}")

        all_results = []

        for name, collection_obj in self.collections.items():
            try:
                where_clause = self.build_smart_filters(query_intent, name)
                search_results_count = max(1, max_results // 2 if not search_strategy['broad'] else max_results)

                if where_clause and 'impossible_filter' in where_clause:
                    continue

                query_params = {
                    "query_texts": [query],
                    "n_results": search_results_count,
                }

                if where_clause:
                    query_params["where"] = where_clause

                try:
                    results = collection_obj.query(**query_params)
                except Exception as query_error:
                    # --- CHANGE IS HERE ---
                    if self.debug_mode:
                        print(f"⚠️ Skipping collection {name} due to query error: {query_error}")
                    try:
                        collection_obj = self.client.get_collection(
                            name=name,
                            embedding_function=self.embedding_function
                        )
                        self.collections[name] = collection_obj
                        results = collection_obj.query(**query_params)
                        if self.debug_mode:
                            print(f"✅ Successfully recovered collection {name}")
                    except Exception as recovery_error:
                        # --- AND CHANGE IS HERE ---
                        if self.debug_mode:
                            print(f"❌ Could not recover collection {name}: {recovery_error}")
                        continue

                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        metadata = results["metadatas"][0][i] if results["metadatas"][0] else {}
                        chroma_distance = results["distances"][0][i]

                        relevance_score = self.calculate_ai_relevance(query_intent, doc, metadata, chroma_distance)

                        min_threshold = 5 if query_intent['intent'] == 'person_search' else search_strategy['threshold']

                        if relevance_score >= min_threshold:
                            collection_type = self.get_collection_type(name)
                            hierarchy = self.get_proper_hierarchy_display(name, metadata)

                            all_results.append({
                                "source": collection_type,
                                "content": doc,
                                "metadata": metadata,
                                "hierarchy": hierarchy,
                                "relevance": relevance_score,
                                "match_reason": self.explain_match(query_intent, doc, metadata)
                            })
                            if self.debug_mode:
                                print(f"✅ Added result with relevance {relevance_score}")
                        else:
                            if self.debug_mode:
                                print(f"❌ Rejected result with relevance {relevance_score} (threshold: {min_threshold})")

            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error searching {name}: {e}")

        if self.debug_mode:
            print(f"🔍 Total results before final filtering: {len(all_results)}")

        all_results.sort(key=lambda x: x['relevance'], reverse=True)
        final_results = all_results[:max_results]

        if self.debug_mode:
            print(f"🔍 Final results after sorting: {len(final_results)}")
            
        return final_results
    
    def get_proper_hierarchy_display(self, collection_name, metadata):
        """Get proper hierarchy display for any collection type"""
        
        # Curriculum hierarchy
        if 'curriculum' in collection_name.lower():
            department = metadata.get('department', 'Unknown')
            program = metadata.get('program', 'Unknown Program')
            year_level = metadata.get('year_level', '')
            
            dept_display = self.get_department_display_name(department)
            
            if year_level:
                return f"{dept_display} > {program} Curriculum > Year {year_level}"
            else:
                return f"{dept_display} > {program} Curriculum"
        
        # Faculty hierarchy
        elif 'faculty' in collection_name.lower():
            if 'admin' in collection_name.lower():
                admin_type = metadata.get('admin_type', 'Administrator')
                return f"Administration > {admin_type}s"
            elif 'non_teaching' in collection_name.lower():
                if 'schedule' in collection_name.lower():
                    dept_display = self.get_non_teaching_department_display_name(metadata.get('department', 'Unknown'))
                    return f"{dept_display} > Non-Teaching Faculty Schedules"
                else:
                    dept_display = self.get_non_teaching_department_display_name(metadata.get('department', 'Unknown'))
                    return f"{dept_display} > Non-Teaching Faculty"
            else:
                dept_display = self.get_department_display_name(metadata.get('department', 'Unknown'))
                if 'schedule' in collection_name.lower():
                    return f"{dept_display} > Teaching Faculty Schedules"
                else:
                    return f"{dept_display} > Teaching Faculty"
        
        # COR Schedule hierarchy
        elif 'schedule' in collection_name.lower() or metadata.get('data_type', '').startswith('cor'):
            dept_display = self.get_department_display_name(metadata.get('department', 'Unknown'))
            course = metadata.get('course', 'Unknown')
            year = metadata.get('year_level', 'Unknown')
            section = metadata.get('section', 'Unknown')
            return f"{dept_display} > {course} > Year {year} > Section {section}"
        
        # Student data hierarchy
        elif 'student' in collection_name.lower():
            dept_display = self.get_department_display_name(metadata.get('department', 'Unknown'))
            course = metadata.get('course', 'Unknown')
            year = metadata.get('year_level', 'Unknown')
            section = metadata.get('section', 'Unknown')
            return f"{dept_display} > {course} > Year {year} > Section {section}"
        
        # Fallback for unknown types
        else:
            dept_display = self.get_department_display_name(metadata.get('department', 'Unknown'))
            return f"{dept_display} > Unknown Collection Type"
    
    def check_collection_health(self):
        """Check health of all collections"""
        print("🔍 Checking collection health...")
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                # Try a simple query
                test_query = collection.query(query_texts=["test"], n_results=1)
                print(f"✅ {name}: {count} records, query OK")
            except Exception as e:
                print(f"❌ {name}: ERROR - {e}")
                print(f"   Suggestion: Delete and reload this collection")

    def analyze_query_intent(self, query):
        """
        Analyze what the user is really looking for using an LLM-based approach.
        """
        # Get the structured analysis from the LLM
        analyzed_data = self.analyze_query_with_llm(query)

        # Populate the intent dictionary from the LLM's response
        intent = {
            'intent': analyzed_data.get('intent', 'general'),
            'target_course': analyzed_data.get('target_course'),
            'target_year': analyzed_data.get('target_year'),
            'target_section': analyzed_data.get('target_section'),
            'target_person': analyzed_data.get('target_person'),
            'target_subject': analyzed_data.get('target_subject'),
            'data_type': analyzed_data.get('data_type'),
            'specificity': 'low',  # Start with low specificity
            'query': query
        }

        # Keep the existing specificity calculation logic, as it's still very useful.
        specific_elements = sum([
            1 for x in [intent['target_course'], intent['target_year'],
                    intent['target_section'], intent['target_person'], intent['target_subject']] if x
        ])

        if specific_elements >= 3:
            intent['specificity'] = 'high'
        elif specific_elements >= 1:
            intent['specificity'] = 'medium'
        else:
            intent['specificity'] = 'low'

        if self.debug_mode:
            print(f"🤖 AI Intent Analysis Result: {intent}")

        return intent

    def determine_search_strategy(self, query_intent):
        """Universal smart search strategy determination"""

        # --- NEW ENHANCEMENT: Override for list retrieval mode ---
        # This allows the new workflow to force a threshold of 0 for complete lists.
        if hasattr(self, '_is_list_retrieval_mode') and self._is_list_retrieval_mode:
            return {'type': 'list_retrieval', 'broad': True, 'threshold': 0}
        # --- END OF ENHANCEMENT ---

        # --- ORIGINAL CODE (UNCHANGED) ---
        strategy = {
            'type': 'balanced',
            'broad': True,
            'threshold': 30
        }
        
        if query_intent['specificity'] == 'high':
            strategy = {
                'type': 'precise',
                'broad': False,
                'threshold': 70
            }
        
        elif query_intent['intent'] == 'person_search':
            strategy = {
                'type': 'person_focused',
                'broad': False,
                'threshold': 25
            }
        
        elif query_intent['specificity'] == 'medium' and any([
            query_intent['target_person'], 
            query_intent['target_subject'],
            query_intent['target_course']
        ]):
            strategy = {
                'type': 'focused',
                'broad': False,
                'threshold': 50
            }
        
        elif query_intent['specificity'] == 'low':
            strategy = {
                'type': 'broad',
                'broad': True,
                'threshold': 25
            }
         
        return strategy

    def build_smart_filters(self, query_intent, collection_name):
        """Build dynamic filters based on AI analysis with correct $and operator."""
        
        # --- NEW ENHANCEMENT: Handle person searches flexibly ---
        # If the user is looking for a person, we should not restrict the search
        # based on the AI's data_type guess (e.g., 'student' vs 'faculty').
        # Instead, we search all relevant collections and let the relevance score decide.
        if query_intent['intent'] == 'person_search':
            # We still build filters for other details provided with the name,
            # like course or year, but we deliberately ignore the data_type filter.
            person_search_conditions = []
            if query_intent['target_course']:
                person_search_conditions.append({'course': query_intent['target_course']})
            if query_intent['target_year']:
                try:
                    year_val = int(query_intent['target_year'])
                    person_search_conditions.append({"$or": [{'year_level': str(year_val)}, {'year_level': year_val}]})
                except (ValueError, TypeError):
                    person_search_conditions.append({'year_level': str(query_intent['target_year'])})
            if query_intent['target_section']:
                person_search_conditions.append({'section': query_intent['target_section']})
            
            # Construct and return the filter for the person search, bypassing the original logic.
            if not person_search_conditions:
                return {} # Return empty filter to search everywhere for the name
            elif len(person_search_conditions) == 1:
                return person_search_conditions[0]
            else:
                return {"$and": person_search_conditions}
        # --- END OF ENHANCEMENT ---


        # --- ORIGINAL CODE (UNCHANGED) ---
        # This original logic will now only run for non-person searches (e.g., schedule_search),
        # where the data_type filter is still essential.
        conditions = []
        
        # Only apply filters if we have specific targets
        if query_intent['target_course']:
            conditions.append({'course': query_intent['target_course']})
        
        if query_intent['target_year']:
            # Metadata year_level can be int or str, so we check for both
            try:
                year_val = int(query_intent['target_year'])
                conditions.append({"$or": [{'year_level': str(year_val)}, {'year_level': year_val}]})
            except (ValueError, TypeError):
                conditions.append({'year_level': str(query_intent['target_year'])})

        if query_intent['target_section']:
            conditions.append({'section': query_intent['target_section']})
        
        # Collection-specific filtering logic
        if query_intent['data_type']:
            if query_intent['data_type'] == 'student' and 'faculty' in collection_name:
                return {'impossible_filter': 'skip'}
            elif query_intent['data_type'] == 'faculty' and 'student' in collection_name:
                return {'impossible_filter': 'skip'}
            elif query_intent['data_type'] == 'schedule' and 'student' in collection_name:
                return {'impossible_filter': 'skip'}

        # Construct the final where_clause
        if not conditions:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}


    def calculate_ai_relevance(self, query_intent, document, metadata, chroma_distance):
        """Calculate relevance score using AI reasoning, with improved single-name query handling."""
        score = 0
        doc_upper = document.upper()

        # --- NEW ENHANCEMENT 1: GUARANTEE CONSISTENT PERSON SEARCH ---
        # This logic runs first. If it's a clear person search with a name match,
        # it returns a high score immediately, fixing the inconsistency issue.
        if query_intent['intent'] == 'person_search' and query_intent['target_person']:
            target_person_upper = query_intent['target_person'].upper()
            if (metadata.get('full_name') and target_person_upper in metadata['full_name'].upper()) or \
               (target_person_upper in doc_upper):
                if self.debug_mode: print(f"🎯 DEFINITIVE PERSON MATCH. Score set to 95.")
                return 95

        # --- NEW ENHANCEMENT 2: GUARANTEE COMPLETE LISTS ---
        # This logic adds a large boost for broad category queries to ensure all
        # relevant documents (like all 30 students) get a high score.
        is_list_query = bool(query_intent.get('target_course') or query_intent.get('target_year'))
        is_person_query = bool(query_intent.get('target_person'))
        if is_list_query and not is_person_query:
            course_match = (query_intent['target_course'] and metadata.get('course') and query_intent['target_course'].upper() in str(metadata.get('course')).upper())
            year_match = (query_intent['target_year'] and str(metadata.get('year_level')) == str(query_intent['target_year']))
            
            if course_match and year_match:
                score += 80
                if self.debug_mode: print(f"🎯 STRONG CATEGORY MATCH (Course & Year). +80")
            elif course_match or year_match:
                score += 60
                if self.debug_mode: print(f"🎯 CATEGORY MATCH (Course or Year). +60")
        
        # --- ORIGINAL CODE (UNCHANGED) ---
        # The original logic below is preserved and acts as the baseline.
        
        # Convert ChromaDB distance to a base semantic score
        semantic_base_score = max(0, 70 - (chroma_distance * 2))
        score += semantic_base_score

        # ENHANCED Person search scoring
        if query_intent['target_person']:
            target_person_upper = query_intent['target_person'].upper()
            name_parts = query_intent['target_person'].split()

            if self.debug_mode:
                print(f"🔍 Looking for person: '{target_person_upper}' in document")

            # --- NEW LOGIC FOR SINGLE-NAME QUERIES ---
            # If the query is just a single name, give a large boost for matches in name fields.
            if len(name_parts) == 1:
                if metadata.get('first_name') and target_person_upper == metadata['first_name'].upper():
                    score += 85
                    if self.debug_mode: print(f"🎯 Found exact first name in metadata: +85")
                elif metadata.get('full_name') and target_person_upper in metadata['full_name'].upper():
                    score += 75
                    if self.debug_mode: print(f"🎯 Found single name within full name in metadata: +75")

            # --- Existing multi-word name logic ---
            else:
                full_match_found_in_doc = False
                # Priority 1: Boost for exact matches in metadata fields
                if metadata.get('full_name') and target_person_upper in metadata['full_name'].upper():
                    score += 80
                    full_match_found_in_doc = True
                    if self.debug_mode: print(f"🎯 Found exact name in 'full_name' metadata: +80")
                elif metadata.get('surname') and target_person_upper in metadata['surname'].upper():
                    score += 75
                    full_match_found_in_doc = True
                    if self.debug_mode: print(f"🎯 Found name in 'surname' metadata: +75")

                # Priority 2: Boost for exact match in the document content
                if target_person_upper in doc_upper:
                    score += 60
                    full_match_found_in_doc = True
                    if self.debug_mode: print(f"🎯 Found exact name in document content: +60")

                # Priority 3: If no full match found, check for partial matches
                if not full_match_found_in_doc:
                    if len(name_parts) > 1:
                        partial_matches = 0
                        for part in name_parts:
                            if len(part) > 2:
                                if part.upper() in doc_upper:
                                    partial_matches += 1
                                    score += 35
                                    if self.debug_mode: print(f"🎯 Partial match '{part}' in document: +35")
                                elif metadata.get('full_name') and part.upper() in metadata['full_name'].upper():
                                    partial_matches += 1
                                    score += 40
                                    if self.debug_mode: print(f"🎯 Partial match '{part}' in metadata: +40")

                        if partial_matches > 1:
                            score += 25
                            if self.debug_mode: print(f"🎯 Multiple name parts matched: +25")

        # Scoring for other fields
        if query_intent['target_course'] and query_intent['target_course'] in doc_upper:
            score += 25
        if query_intent['target_year'] and str(query_intent['target_year']) in doc_upper:
            score += 20
        if query_intent['target_section'] and query_intent['target_section'] in doc_upper:
            score += 20
        if metadata:
            if query_intent['target_course'] and metadata.get('course') and query_intent['target_course'] in metadata['course'].upper():
                score += 15
            if query_intent['target_year'] and str(metadata.get('year_level')) == str(query_intent['target_year']):
                score += 15
            if query_intent['target_section'] and metadata.get('section') and query_intent['target_section'] in metadata['section'].upper():
                score += 15

        final_score = max(0, min(100, score))
        if self.debug_mode:
            print(f"🔍 Final relevance score: {final_score} (raw: {score})")
            
        return final_score

    def rank_and_filter_results(self, results, query_intent, max_results):
        """AI-powered ranking and filtering of results"""
        
        # FIX: Lower minimum relevance and add debug info
        if query_intent['specificity'] == 'high':
            min_relevance = 8
        elif query_intent['intent'] == 'person_search':
            min_relevance = 5  # Lower threshold for person searches
        else:
            min_relevance = 5  # Lower default threshold
        
        print(f"🔍 Filtering {len(results)} results with min_relevance: {min_relevance}")
        
        # Remove results that don't meet minimum relevance
        filtered_results = []
        for r in results:
            print(f"🔍 Result relevance: {r['relevance']} (min: {min_relevance})")
            if r['relevance'] >= min_relevance:
                filtered_results.append(r)
            else:
                print(f"🔍 Filtered out result with relevance {r['relevance']}")
        
        print(f"🔍 After filtering: {len(filtered_results)} results remain")
        
        # Sort by relevance score
        filtered_results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Apply intelligent deduplication if needed
        if query_intent['intent'] == 'person_search':
            # For person searches, prioritize unique individuals
            seen_names = set()
            unique_results = []
            for result in filtered_results:
                doc_upper = result['content'].upper()
                # Extract name from document
                if 'FULL NAME:' in doc_upper:
                    name_start = doc_upper.find('FULL NAME:') + len('FULL NAME:')
                    name_line = doc_upper[name_start:doc_upper.find('\n', name_start) if '\n' in doc_upper[name_start:] else len(doc_upper)].strip()
                    if name_line not in seen_names:
                        seen_names.add(name_line)
                        unique_results.append(result)
                else:
                    # If no FULL NAME found, include the result
                    unique_results.append(result)
            filtered_results = unique_results
        
        final_results = filtered_results[:max_results]
        print(f"🔍 Final results count: {len(final_results)}")
        return final_results

    def explain_match(self, query_intent, document, metadata):
        """Explain why this result matches the query"""
        reasons = []
        
        if query_intent['target_course'] and metadata.get('course') == query_intent['target_course']:
            reasons.append(f"Matches course: {query_intent['target_course']}")
        
        if query_intent['target_year'] and metadata.get('year_level') == query_intent['target_year']:
            reasons.append(f"Matches year: {query_intent['target_year']}")
        
        if query_intent['target_section'] and metadata.get('section') == query_intent['target_section']:
            reasons.append(f"Matches section: {query_intent['target_section']}")
        
        return " | ".join(reasons) if reasons else "General relevance match"
    
    
    def check_for_duplicates(self, new_data, data_type, metadata):
        """ENHANCED: Check for duplicates across AND within collections"""
        duplicate_found = False
        similar_records = []
        
        print(f"🔍 IMMEDIATE DUPLICATE CHECK: Scanning {len(self.collections)} collections...")
        
        # Get the target collection name where this would be stored
        target_collection_name = self.create_smart_collection_name(data_type, metadata)
        checking_name = self.get_entity_name_for_display(metadata, data_type)
        print(f"   📝 Checking: {checking_name} ({data_type})")
        print(f"   🎯 Target collection: {target_collection_name}")
        
        # Check ALL existing collections (including the target collection)
        for collection_name, collection in self.collections.items():
            try:
                all_docs = collection.get()
                
                if not all_docs["documents"]:
                    continue
                    
                collection_type = self.get_collection_type(collection_name)
                print(f"   📊 Scanning {collection_type}: {len(all_docs['documents'])} records")
                
                # Check each existing record
                for i, existing_doc in enumerate(all_docs["documents"]):
                    existing_metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                    
                    # MULTIPLE DETECTION STRATEGIES
                    is_duplicate = False
                    match_reason = ""
                    
                    # Strategy 1: Metadata-based detection
                    if self.is_metadata_duplicate(metadata, existing_metadata, data_type):
                        is_duplicate = True
                        match_reason = "Same identifying information"
                    
                    # Strategy 2: Content-based detection  
                    elif self.is_content_duplicate(new_data, existing_doc):
                        is_duplicate = True
                        match_reason = "Identical or very similar content"
                    
                    # Strategy 3: Name-based detection (fuzzy matching)
                    elif self.is_name_duplicate(metadata, existing_metadata):
                        is_duplicate = True
                        match_reason = "Same or similar name"
                    
                    if is_duplicate:
                        existing_name = self.get_entity_name_for_display(existing_metadata, existing_metadata.get('data_type', ''))
                        
                        # ENHANCED: Show if it's same collection or different collection
                        if collection_name == target_collection_name:
                            print(f"   🚨 DUPLICATE DETECTED: {existing_name} (SAME COLLECTION - {match_reason})")
                        else:
                            print(f"   🚨 DUPLICATE DETECTED: {existing_name} (DIFFERENT COLLECTION - {match_reason})")
                        
                        similar_records.append({
                            'collection': collection_name,
                            'collection_type': collection_type,
                            'metadata': existing_metadata,
                            'content_preview': existing_doc[:300] + "..." if len(existing_doc) > 300 else existing_doc,
                            'doc_index': i,
                            'match_reason': match_reason,
                            'existing_name': existing_name,
                            'same_collection': collection_name == target_collection_name
                        })
                        
                        duplicate_found = True
                        
            except Exception as e:
                print(f"⚠️ Error checking duplicates in {collection_name}: {e}")
        
        if duplicate_found:
            same_collection_count = sum(1 for r in similar_records if r.get('same_collection', False))
            different_collection_count = len(similar_records) - same_collection_count
            
            print(f"⚠️ IMMEDIATE DETECTION: Found {len(similar_records)} duplicate(s) for {checking_name}")
            print(f"   📁 Same collection: {same_collection_count}")
            print(f"   📁 Different collections: {different_collection_count}")
        else:
            print(f"✅ CLEAR: No duplicates found for {checking_name}")
        
        return duplicate_found, similar_records
    
    def is_name_duplicate(self, new_meta, existing_meta):
        """Check for name-based duplicates with fuzzy matching"""
        
        new_name = self.get_entity_name_for_display(new_meta, new_meta.get('data_type', ''))
        existing_name = self.get_entity_name_for_display(existing_meta, existing_meta.get('data_type', ''))
        
        # Exact name match
        if new_name.upper() == existing_name.upper():
            print(f"      🎯 MATCH: Same Name ({new_name})")
            return True
        
        # Fuzzy name match (for slight variations)
        if self.fuzzy_name_match(new_name, existing_name, threshold=0.9):
            print(f"      🎯 MATCH: Similar Name ({new_name} ≈ {existing_name})")
            return True
        
        return False
    
    def is_content_duplicate(self, new_data, existing_doc):
        """Check for content-based duplicates"""
        
        # Convert new_data to string for comparison
        if isinstance(new_data, dict):
            new_content = str(new_data)
        elif isinstance(new_data, str):
            new_content = new_data
        else:
            new_content = str(new_data)
        
        # Exact match
        if new_content == existing_doc:
            print(f"      🎯 MATCH: Identical Content")
            return True
        
        # High similarity (98%+)
        if len(new_content) > 100 and len(existing_doc) > 100:
            similarity = self.calculate_text_similarity(new_content, existing_doc)
            if similarity > 0.98:
                print(f"      🎯 MATCH: High Content Similarity ({similarity:.1%})")
                return True
        
        return False
    
    
    def is_metadata_duplicate(self, new_meta, existing_meta, data_type):
        """Check for duplicates based on key metadata fields"""
        
        if data_type == 'student':
            # Student ID is primary key
            new_id = str(new_meta.get('student_id', '')).strip().upper()
            existing_id = str(existing_meta.get('student_id', '')).strip().upper()
            
            if new_id and existing_id and new_id == existing_id:
                print(f"      🎯 MATCH: Same Student ID ({new_id})")
                return True
            
            # Secondary: Name + Course + Year
            new_name = str(new_meta.get('full_name', '')).strip().upper()
            existing_name = str(existing_meta.get('full_name', '')).strip().upper()
            new_course = str(new_meta.get('course', '')).strip().upper()
            existing_course = str(existing_meta.get('course', '')).strip().upper()
            new_year = str(new_meta.get('year_level', '')).strip()
            existing_year = str(existing_meta.get('year_level', '')).strip()
            
            if (new_name and existing_name and new_name == existing_name and
                new_course and existing_course and new_course == existing_course and
                new_year and existing_year and new_year == existing_year):
                print(f"      🎯 MATCH: Same Name+Course+Year ({new_name})")
                return True
        
        
        elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
            # Faculty: Name + Department
            new_name = str(new_meta.get('full_name', '')).strip().upper()
            existing_name = str(existing_meta.get('full_name', '')).strip().upper()
            new_dept = str(new_meta.get('department', '')).strip().upper()
            existing_dept = str(existing_meta.get('department', '')).strip().upper()
            
            if (new_name and existing_name and new_name == existing_name and
                new_dept and existing_dept and new_dept == existing_dept):
                print(f"      🎯 MATCH: Same Faculty Name+Dept ({new_name} in {new_dept})")
                return True
            
            # Secondary: Email
            new_email = str(new_meta.get('email', '')).strip().lower()
            existing_email = str(existing_meta.get('email', '')).strip().lower()
            
            if new_email and existing_email and new_email == existing_email:
                print(f"      🎯 MATCH: Same Email ({new_email})")
                return True
        
        elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
            # Schedule: Staff Name + Department
            new_staff = str(new_meta.get('adviser_name', new_meta.get('staff_name', ''))).strip().upper()
            existing_staff = str(existing_meta.get('adviser_name', existing_meta.get('staff_name', ''))).strip().upper()
            new_dept = str(new_meta.get('department', '')).strip().upper()
            existing_dept = str(existing_meta.get('department', '')).strip().upper()
            
            if (new_staff and existing_staff and new_staff == existing_staff and
                new_dept and existing_dept and new_dept == existing_dept):
                print(f"      🎯 MATCH: Same Schedule ({new_staff} in {new_dept})")
                return True
        
        elif data_type == 'cor_schedule':
            # COR: Course + Year + Section + Adviser
            new_course = str(new_meta.get('course', '')).strip().upper()
            existing_course = str(existing_meta.get('course', '')).strip().upper()
            new_year = str(new_meta.get('year_level', '')).strip()
            existing_year = str(existing_meta.get('year_level', '')).strip()
            new_section = str(new_meta.get('section', '')).strip().upper()
            existing_section = str(existing_meta.get('section', '')).strip().upper()
            
            if (new_course == existing_course and new_year == existing_year and 
                new_section == existing_section):
                print(f"      🎯 MATCH: Same COR ({new_course} Y{new_year} Sec{new_section})")
                return True
            
        elif data_type == 'curriculum':
        # Curriculum: Program + Department
            new_program = str(new_meta.get('program', '')).strip().upper()
            existing_program = str(existing_meta.get('program', '')).strip().upper()
            new_dept = str(new_meta.get('department', '')).strip().upper()
            existing_dept = str(existing_meta.get('department', '')).strip().upper()
            
            if (new_program and existing_program and new_program == existing_program and
                new_dept and existing_dept and new_dept == existing_dept):
                print(f"      🎯 MATCH: Same Curriculum ({new_program} in {new_dept})")
                return True
    
        return False
    
    
    def get_entity_name_for_display(self, metadata, data_type):
        """Get the main identifier for display purposes"""
        if data_type == 'student':
            name = metadata.get('full_name', 'Unknown Student')
            student_id = metadata.get('student_id', '')
            if student_id:
                return f"{name} (ID: {student_id})"
            return name
        
        elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
            return metadata.get('full_name', 'Unknown Faculty')
        
        elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
            staff_name = metadata.get('adviser_name', metadata.get('staff_name', 'Unknown Staff'))
            dept = metadata.get('department', '')
            if dept:
                return f"{staff_name} ({dept})"
            return staff_name
        
        elif data_type == 'cor_schedule':
            course = metadata.get('course', '')
            year = metadata.get('year_level', '')
            section = metadata.get('section', '')
            return f"{course} Year {year} Section {section}"
        
        elif data_type == 'curriculum':
            program = metadata.get('program', 'Unknown Program')
            dept = metadata.get('department', '')
            if dept:
                return f"{program} Curriculum ({dept})"
            return f"{program} Curriculum"
        
        return 'Unknown Entity'
    
        
    def is_duplicate_record(self, new_metadata, existing_metadata, new_data, existing_doc, data_type):
        """Enhanced duplicate checking with multiple strategies"""
        
        # Strategy 1: Check by data type and key identifying fields
        if data_type == 'student':
            return self.is_duplicate_student(new_metadata, existing_metadata)
        elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
            return self.is_duplicate_faculty(new_metadata, existing_metadata, data_type)
        elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
            return self.is_duplicate_schedule(new_metadata, existing_metadata, data_type)
        elif data_type == 'cor_schedule':
            return self.is_duplicate_cor(new_metadata, existing_metadata)
        
        # Strategy 2: Content-based duplicate detection (fallback)
        return self.is_duplicate_content(new_data, existing_doc)
    
    def is_duplicate_content(self, new_data, existing_doc):
        """Content-based duplicate detection for complex data"""
        if isinstance(new_data, dict):
            # Convert dict to string for comparison
            new_content = str(new_data)
        elif isinstance(new_data, str):
            new_content = new_data
        else:
            new_content = str(new_data)
        
        # Check for very high content similarity
        if len(new_content) > 200 and len(existing_doc) > 200:
            similarity = self.calculate_text_similarity(new_content, existing_doc)
            if similarity > 0.95:  # 95% similar content
                print(f"      🎯 High Content Similarity: {similarity:.2%}")
                return True
        
        # Check for exact content match
        if new_content == existing_doc:
            print(f"      🎯 Identical Content")
            return True
        
        return False
    
    def is_duplicate_schedule(self, new_meta, existing_meta, data_type):
        """Check if two schedule records are duplicates"""
        # Get staff/adviser name
        if data_type == 'teaching_faculty_schedule':
            new_staff = str(new_meta.get('adviser_name', new_meta.get('staff_name', ''))).strip().upper()
            existing_staff = str(existing_meta.get('adviser_name', existing_meta.get('staff_name', ''))).strip().upper()
        else:
            new_staff = str(new_meta.get('staff_name', new_meta.get('adviser_name', ''))).strip().upper()
            existing_staff = str(existing_meta.get('staff_name', existing_meta.get('adviser_name', ''))).strip().upper()
        
        new_dept = str(new_meta.get('department', '')).strip().upper()
        existing_dept = str(existing_meta.get('department', '')).strip().upper()
        
        if (new_staff and existing_staff and new_staff == existing_staff and
            new_dept and existing_dept and new_dept == existing_dept):
            print(f"      🎯 Same Schedule: {new_staff} in {new_dept}")
            return True
        
        return False
    
    def is_duplicate_cor(self, new_meta, existing_meta):
        """Check if two COR records are duplicates"""
        new_course = str(new_meta.get('course', '')).strip().upper()
        existing_course = str(existing_meta.get('course', '')).strip().upper()
        new_year = str(new_meta.get('year_level', '')).strip()
        existing_year = str(existing_meta.get('year_level', '')).strip()
        new_section = str(new_meta.get('section', '')).strip().upper()
        existing_section = str(existing_meta.get('section', '')).strip().upper()
        new_adviser = str(new_meta.get('adviser', '')).strip().upper()
        existing_adviser = str(existing_meta.get('adviser', '')).strip().upper()
        
        if (new_course == existing_course and new_year == existing_year and 
            new_section == existing_section and new_adviser == existing_adviser):
            print(f"      🎯 Same COR: {new_course} Year {new_year} Section {new_section}")
            return True
        
        return False
    
    
    
    
    def is_duplicate_student(self, new_meta, existing_meta):
        """Check if two student records are duplicates"""
        # Check by Student ID (most reliable)
        new_id = str(new_meta.get('student_id', '')).strip().upper()
        existing_id = str(existing_meta.get('student_id', '')).strip().upper()
        
        if new_id and existing_id and new_id == existing_id:
            print(f"      🎯 Same Student ID: {new_id}")
            return True
        
        # Check by Name + Course + Year (secondary check)
        new_name = str(new_meta.get('full_name', '')).strip().upper()
        existing_name = str(existing_meta.get('full_name', '')).strip().upper()
        new_course = str(new_meta.get('course', '')).strip().upper()
        existing_course = str(existing_meta.get('course', '')).strip().upper()
        new_year = str(new_meta.get('year_level', '')).strip()
        existing_year = str(existing_meta.get('year_level', '')).strip()
        
        if (new_name and existing_name and new_name == existing_name and
            new_course and existing_course and new_course == existing_course and
            new_year and existing_year and new_year == existing_year):
            print(f"      🎯 Same Name+Course+Year: {new_name}, {new_course}, Year {new_year}")
            return True
        
        return False
    
    def is_duplicate_faculty(self, new_meta, existing_meta, data_type):
        """Check if two faculty records are duplicates"""
        # Check by Full Name + Department
        new_name = str(new_meta.get('full_name', '')).strip().upper()
        existing_name = str(existing_meta.get('full_name', '')).strip().upper()
        new_dept = str(new_meta.get('department', '')).strip().upper()
        existing_dept = str(existing_meta.get('department', '')).strip().upper()
        
        if (new_name and existing_name and new_name == existing_name and
            new_dept and existing_dept and new_dept == existing_dept):
            print(f"      🎯 Same Faculty: {new_name} in {new_dept}")
            return True
        
        # Check by Email (if available)
        new_email = str(new_meta.get('email', '')).strip().lower()
        existing_email = str(existing_meta.get('email', '')).strip().lower()
        
        if new_email and existing_email and new_email == existing_email:
            print(f"      🎯 Same Email: {new_email}")
            return True
        
        # Fuzzy name matching for slight variations
        if new_name and existing_name and self.fuzzy_name_match(new_name, existing_name):
            print(f"      🎯 Similar Names: {new_name} ≈ {existing_name}")
            return True
        
        return False
    
    def fuzzy_name_match(self, name1, name2, threshold=0.85):
        """Enhanced fuzzy name matching"""
        if not name1 or not name2:
            return False
        
        # Remove common prefixes/suffixes and split into parts
        name1_clean = name1.replace(',', ' ').replace('.', ' ')
        name2_clean = name2.replace(',', ' ').replace('.', ' ')
        
        name1_parts = set(part for part in name1_clean.split() if len(part) > 1)
        name2_parts = set(part for part in name2_clean.split() if len(part) > 1)
        
        if not name1_parts or not name2_parts:
            return False
        
        # Calculate similarity based on matching parts
        intersection = len(name1_parts.intersection(name2_parts))
        union = len(name1_parts.union(name2_parts))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def process_with_duplicate_check(self, filename, data_type):
        """Process file with automatic duplicate detection and handling"""
        
        # Extract data first (reuse existing extraction methods)
        extracted_data = None
        
        try:
            if data_type == 'student':
                extracted_data = self.extract_student_data_for_duplicate_check(filename)
            elif data_type == 'teaching_faculty':
                extracted_data = self.extract_teaching_faculty_excel_info_smart(filename)
            elif data_type == 'admin':
                extracted_data = self.extract_teaching_faculty_excel_info_smart(filename)
            elif data_type == 'non_teaching_faculty':
                extracted_data = self.extract_teaching_faculty_excel_info_smart(filename)
            elif data_type == 'cor_schedule':
                extracted_data = self.extract_cor_excel_info_smart(filename)
            elif data_type == 'teaching_faculty_schedule':
                extracted_data = self.extract_teaching_faculty_schedule_info_smart(filename)
            elif data_type == 'non_teaching_faculty_schedule':
                extracted_data = self.extract_non_teaching_faculty_schedule_info_smart(filename)
            elif data_type == 'curriculum':  # ADD THIS CASE
                extracted_data = self.extract_curriculum_excel_info_smart(filename)
            else:
                print(f"❌ Unknown data type: {data_type}")
                return False
                
            if not extracted_data:
                print(f"❌ Could not extract data from {filename}")
                return False
            
            # Create metadata for duplicate checking
            metadata = self.create_metadata_for_duplicate_check(extracted_data, data_type)
            
            # Check for duplicates
            has_duplicates, similar_records = self.check_for_duplicates(extracted_data, data_type, metadata)
            
            if has_duplicates:
                return self.handle_duplicate_found(filename, extracted_data, similar_records, data_type)
            else:
                # No duplicates - proceed with normal processing
                print(f"✅ No duplicates found. Processing {filename}...")
                return self.process_file_normally(filename, data_type)
                
        except Exception as e:
            print(f"❌ Error in duplicate checking for {filename}: {e}")
            return False
    
    def extract_student_data_for_duplicate_check(self, filename):
        """Extract student data specifically for duplicate checking"""
        try:
            # Try structured first
            df = pd.read_excel(filename)
            if self.is_structured_student_data(df):
                # Extract first student record for checking
                df = pd.read_excel(filename)
                if len(df) > 0:
                    first_row = df.iloc[0]
                    return {
                        'student_id': str(first_row.get('Student ID', first_row.get('ID', ''))),
                        'full_name': str(first_row.get('Full Name', first_row.get('Name', ''))),
                        'surname': str(first_row.get('Surname', '')),
                        'first_name': str(first_row.get('First Name', '')),
                        'course': str(first_row.get('Course', '')),
                        'year_level': str(first_row.get('Year', first_row.get('Year Level', ''))),
                        'section': str(first_row.get('Section', '')),
                    }
            else:
                # Try unstructured extraction
                xl_file = pd.ExcelFile(filename)
                all_text = ""
                for sheet_name in xl_file.sheet_names:
                    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
                    for row in df.values:
                        row_text = ' '.join([str(cell) for cell in row if pd.notna(cell)])
                        all_text += row_text + "\n"
                
                # Extract first student record
                student_records = self.split_into_student_records(all_text)
                if student_records:
                    return self.extract_universal_student_data(student_records[0], 'excel_unstructured')
            
            return None
            
        except Exception as e:
            print(f"❌ Error extracting student data for duplicate check: {e}")
            return None

    def create_metadata_for_duplicate_check(self, extracted_data, data_type):
        """Create standardized metadata for duplicate checking"""
        try:
            print(f"🔍 Creating metadata for {data_type} duplicate check...")
            
            if data_type == 'student':
                metadata = {
                    'student_id': str(extracted_data.get('student_id', '')).strip(),
                    'full_name': str(extracted_data.get('full_name', '')).strip(),
                    'course': str(extracted_data.get('course', '')).strip().upper(),
                    'year_level': str(extracted_data.get('year_level', extracted_data.get('year', ''))).strip(),
                    'section': str(extracted_data.get('section', '')).strip().upper(),
                    'data_type': 'student'
                }
                print(f"   📊 Student: {metadata['full_name']} ({metadata['course']} Year {metadata['year_level']})")
                return metadata
                
            elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
                full_name = ""
                if extracted_data.get('surname') and extracted_data.get('first_name'):
                    full_name = f"{extracted_data['surname']}, {extracted_data['first_name']}"
                elif extracted_data.get('full_name'):
                    full_name = extracted_data['full_name']
                
                metadata = {
                    'full_name': full_name.strip(),
                    'surname': str(extracted_data.get('surname', '')).strip(),
                    'first_name': str(extracted_data.get('first_name', '')).strip(),
                    'department': str(extracted_data.get('department', '')).strip().upper(),
                    'position': str(extracted_data.get('position', '')).strip(),
                    'email': str(extracted_data.get('email', '')).strip().lower(),
                    'data_type': data_type
                }
                
                if data_type == 'admin':
                    metadata['admin_type'] = str(extracted_data.get('admin_type', '')).strip()
                
                print(f"   📊 Faculty: {metadata['full_name']} in {metadata['department']}")
                return metadata
                
            elif data_type == 'cor_schedule':
                metadata = {
                    'course': str(extracted_data['program_info'].get('Program', '')).strip().upper(),
                    'year_level': str(extracted_data['program_info'].get('Year Level', '')).strip(),
                    'section': str(extracted_data['program_info'].get('Section', '')).strip().upper(),
                    'adviser': str(extracted_data['program_info'].get('Adviser', '')).strip(),
                    'data_type': 'cor_schedule'
                }
                print(f"   📊 COR: {metadata['course']} Year {metadata['year_level']} Section {metadata['section']}")
                return metadata
                
            elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
                name_field = 'adviser_name' if data_type == 'teaching_faculty_schedule' else 'staff_name'
                staff_name = str(extracted_data.get(name_field, '')).strip()
                
                metadata = {
                    'staff_name': staff_name,
                    'adviser_name': staff_name,  # Store both for compatibility
                    'department': str(extracted_data.get('department', '')).strip().upper(),
                    'data_type': data_type
                }
                print(f"   📊 Schedule: {staff_name} in {metadata['department']}")
                return metadata
                
            elif data_type == 'curriculum':
                curriculum_info = extracted_data.get('curriculum_info', {})
                subjects = extracted_data.get('subjects', [])
                
                metadata = {
                    'program': str(curriculum_info.get('program', '')).strip().upper(),
                    'department': str(curriculum_info.get('department', '')).strip().upper(),
                    'total_subjects': len(subjects),
                    'total_units': self.calculate_total_units(subjects),
                    'curriculum_type': 'academic_program',
                    'data_type': 'curriculum'
                }
                print(f"   📊 Curriculum: {metadata['program']} in {metadata['department']} ({metadata['total_subjects']} subjects)")
                return metadata
                
            else:
                print(f"   ⚠️ Unknown data type: {data_type}")
                return {'data_type': data_type}
                
        except Exception as e:
            print(f"❌ Error creating metadata for duplicate check: {e}")
            return {'data_type': data_type}

    def process_file_normally(self, filename, data_type):
        """Process file using the normal methods (fallback to existing code)"""
        try:
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == ".xlsx":
                df_check = pd.read_excel(filename, header=None)
                
                if data_type == 'curriculum':
                    return self.process_curriculum_excel(filename)
                elif data_type == 'cor_schedule':
                    return self.process_cor_excel(filename)
                elif data_type == 'admin':
                    return self.process_admin_excel(filename)
                elif data_type == 'teaching_faculty':
                    return self.process_teaching_faculty_excel(filename)
                elif data_type == 'non_teaching_faculty':
                    return self.process_non_teaching_faculty_excel(filename)
                elif data_type == 'teaching_faculty_schedule':
                    return self.process_teaching_faculty_schedule_excel(filename)
                elif data_type == 'non_teaching_faculty_schedule':
                    return self.process_non_teaching_faculty_schedule_excel(filename)
                elif data_type == 'student':
                    return self.process_student_excel(filename)
                else:
                    return self.process_student_excel(filename)  # Default fallback
                    
            elif ext == ".pdf":
                if data_type == 'curriculum':
                    # Curriculum PDFs would need separate handling if needed
                    print("📄 Curriculum PDF processing not implemented yet")
                    return False
                elif data_type == 'cor_schedule':
                    return self.process_cor_pdf(filename)
                elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
                    return self.process_faculty_pdf(filename)
                elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
                    return self.process_faculty_schedule_pdf(filename)
                else:
                    return self.process_student_pdf(filename)
            
            return False
            
        except Exception as e:
            print(f"❌ Error in normal file processing: {e}")
            return False

    def handle_duplicate_found(self, filename, new_data, similar_records, data_type):
        """Handle when duplicates are found"""
        print(f"\n⚠️ DUPLICATE DETECTED!")
        print(f"📁 File: {filename}")
        print(f"📊 Data Type: {data_type.replace('_', ' ').title()}")
        print(f"🔍 Found {len(similar_records)} similar record(s):")
        
        # Show existing records
        for i, record in enumerate(similar_records, 1):
            collection_type = self.get_collection_type(record['collection'])
            print(f"\n   {i}. Similar record in: {collection_type}")
            
            if data_type == 'student':
                print(f"      Student ID: {record['metadata'].get('student_id', 'Unknown')}")
                print(f"      Name: {record['metadata'].get('full_name', 'Unknown')}")
                print(f"      Course: {record['metadata'].get('course', 'Unknown')}")
            elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
                print(f"      Name: {record['metadata'].get('full_name', 'Unknown')}")
                print(f"      Department: {record['metadata'].get('department', 'Unknown')}")
                print(f"      Position: {record['metadata'].get('position', 'Unknown')}")
            elif data_type == 'cor_schedule':
                print(f"      Course: {record['metadata'].get('course', 'Unknown')}")
                print(f"      Year/Section: {record['metadata'].get('year_level', 'Unknown')}/{record['metadata'].get('section', 'Unknown')}")
                print(f"      Adviser: {record['metadata'].get('adviser', 'Unknown')}")
            elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
                print(f"      Staff: {record['metadata'].get('staff_name', record['metadata'].get('adviser_name', 'Unknown'))}")
                print(f"      Department: {record['metadata'].get('department', 'Unknown')}")
        
        print(f"\n💡 What would you like to do?")
        print(f"   1. 🚫 Skip loading (keep existing data)")
        print(f"   2. 🔄 Replace existing data with new file")
        print(f"   3. 📝 Load as new record anyway")
        print(f"   4. 🔍 View detailed comparison")
        
        while True:
            try:
                choice = input("\n👉 Choose option (1-4): ").strip()
                
                if choice == "1":
                    print(f"✅ Skipped loading duplicate data from {filename}")
                    return True
                elif choice == "2":
                    return self.replace_existing_record(filename, new_data, similar_records, data_type)
                elif choice == "3":
                    print(f"✅ Loading as new record...")
                    return self.process_file_normally(filename, data_type)
                elif choice == "4":
                    self.show_detailed_comparison(new_data, similar_records, data_type)
                    # Continue the loop to ask again
                else:
                    print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                    
            except KeyboardInterrupt:
                print(f"\n❌ Cancelled. Skipping {filename}")
                return False
            except Exception as e:
                print(f"❌ Error handling input: {e}")
                return False

    def replace_existing_record(self, filename, new_data, similar_records, data_type):
        """Replace existing record with new data"""
        try:
            print(f"\n🔄 Replacing existing record(s)...")
            
            # Delete the old record(s)
            deleted_count = 0
            for record in similar_records:
                collection_name = record['collection']
                if collection_name in self.collections:
                    if self.delete_collection(collection_name):
                        deleted_count += 1
                        collection_type = self.get_collection_type(collection_name)
                        print(f"   🗑️ Deleted: {collection_type}")
            
            print(f"✅ Deleted {deleted_count} old record(s)")
            
            # Process the new file
            print(f"📥 Processing new data from {filename}...")
            success = self.process_file_normally(filename, data_type)
            
            if success:
                print(f"✅ Successfully replaced with new data!")
            else:
                print(f"❌ Failed to process new file")
                
            return success
            
        except Exception as e:
            print(f"❌ Error replacing record: {e}")
            return False

    def show_detailed_comparison(self, new_data, similar_records, data_type):
        """Show detailed comparison between new and existing data"""
        print(f"\n📊 DETAILED COMPARISON")
        print("=" * 70)
        
        print(f"\n🆕 NEW DATA (from file):")
        print("-" * 35)
        self.display_data_details(new_data, data_type)
        
        for i, record in enumerate(similar_records, 1):
            collection_type = self.get_collection_type(record['collection'])
            print(f"\n📋 EXISTING RECORD {i} (in {collection_type}):")
            print("-" * 50)
            self.display_data_details(record['metadata'], data_type)
        
        print("=" * 70)

    def display_data_details(self, data, data_type):
        """Display data details in a formatted way"""
        try:
            if data_type == 'student':
                print(f"   Student ID: {data.get('student_id', 'N/A')}")
                print(f"   Name: {data.get('full_name', 'N/A')}")
                print(f"   Course: {data.get('course', 'N/A')}")
                print(f"   Year: {data.get('year_level', data.get('year', 'N/A'))}")
                print(f"   Section: {data.get('section', 'N/A')}")
                
            elif data_type in ['teaching_faculty', 'admin', 'non_teaching_faculty']:
                print(f"   Name: {data.get('full_name', 'N/A')}")
                print(f"   Surname: {data.get('surname', 'N/A')}")
                print(f"   First Name: {data.get('first_name', 'N/A')}")
                print(f"   Department: {data.get('department', 'N/A')}")
                print(f"   Position: {data.get('position', 'N/A')}")
                print(f"   Email: {data.get('email', 'N/A')}")
                if data_type == 'admin':
                    print(f"   Admin Type: {data.get('admin_type', 'N/A')}")
            
            elif data_type == 'curriculum':
                if isinstance(data, dict) and 'curriculum_info' in data:
                    curr_info = data['curriculum_info']
                    subjects = data.get('subjects', [])
                    print(f"   Program: {curr_info.get('program', 'N/A')}")
                    print(f"   Department: {curr_info.get('department', 'N/A')}")
                    print(f"   Total Subjects: {len(subjects)}")
                    print(f"   Total Units: {self.calculate_total_units(subjects)}")
                else:
                    print(f"   Program: {data.get('program', 'N/A')}")
                    print(f"   Department: {data.get('department', 'N/A')}")
                    print(f"   Total Subjects: {data.get('total_subjects', 'N/A')}")
                    print(f"   Total Units: {data.get('total_units', 'N/A')}")
            
            elif data_type == 'cor_schedule':
                if isinstance(data, dict) and 'program_info' in data:
                    prog_info = data['program_info']
                    print(f"   Program: {prog_info.get('Program', 'N/A')}")
                    print(f"   Year Level: {prog_info.get('Year Level', 'N/A')}")
                    print(f"   Section: {prog_info.get('Section', 'N/A')}")
                    print(f"   Adviser: {prog_info.get('Adviser', 'N/A')}")
                    print(f"   Subjects: {len(data.get('schedule', []))}")
                else:
                    print(f"   Course: {data.get('course', 'N/A')}")
                    print(f"   Year Level: {data.get('year_level', 'N/A')}")
                    print(f"   Section: {data.get('section', 'N/A')}")
                    print(f"   Adviser: {data.get('adviser', 'N/A')}")
                    
            elif data_type in ['teaching_faculty_schedule', 'non_teaching_faculty_schedule']:
                name_field = 'adviser_name' if data_type == 'teaching_faculty_schedule' else 'staff_name'
                print(f"   Staff Name: {data.get(name_field, data.get('staff_name', 'N/A'))}")
                print(f"   Department: {data.get('department', 'N/A')}")
                if 'schedule' in data:
                    print(f"   Schedule Items: {len(data['schedule'])}")
            else:
                # Generic display
                for key, value in data.items():
                    if value and str(value).strip() not in ['', 'N/A', 'None']:
                        print(f"   {key.replace('_', ' ').title()}: {value}")
                        
        except Exception as e:
            print(f"   ❌ Error displaying data: {e}")

    # ======================== STUDENT DATA PROCESSING ========================
    
    def process_student_excel(self, filename):
        """Process Student Excel file with universal extraction"""
        try:
            # Try to read as structured data first
            df = pd.read_excel(filename)
            
            if self.is_structured_student_data(df):
                return self.process_structured_student_excel(filename)  # Pass filename, not df
            else:
                return self.process_unstructured_student_excel(filename)
        
        except Exception as e:
            print(f"❌ Error processing student Excel: {e}")
            return False

    def is_structured_student_data(self, df):
        """Check if Excel has structured column headers"""
        if df.empty:
            return False
        
        # Check if columns contain expected student data headers
        columns_upper = [str(col).upper() for col in df.columns]
        expected_fields = ['STUDENT', 'NAME', 'YEAR', 'COURSE', 'SECTION', 'CONTACT']
        
        matches = sum(1 for field in expected_fields 
                     if any(field in col for col in columns_upper))
        
        return matches >= 4  # At least 4 expected fields found

    def process_structured_student_excel(self, file_path):
        """
        Process structured student data from Excel files by directly mapping columns.
        Prioritizes structured data extraction over universal text parsing.
        """
        try:
            df = pd.read_excel(file_path)
            all_student_data = []
            texts = []
            metadata_list = []

            # Enhanced column mapping with more variations
            column_mapping = {
                'student id': 'student_id',
                'id no': 'student_id',
                'id': 'student_id',
                'student number': 'student_id',
                'full name': 'full_name',
                'name': 'full_name',
                'student name': 'full_name',
                'surname': 'surname',
                'last name': 'surname',
                'family name': 'surname',
                'first name': 'first_name',
                'given name': 'first_name',
                'firstname': 'first_name',
                'year': 'year',
                'year level': 'year',
                'yr': 'year',
                'level': 'year',
                'course': 'course',
                'program': 'course',
                'degree': 'course',
                'course program': 'course',
                'section': 'section',
                'sec': 'section',
                'class': 'section',
                'contact number': 'contact_number',
                'phone': 'contact_number',
                'mobile': 'contact_number',
                'tel no': 'contact_number',
                'phone number': 'contact_number',
                'mobile number': 'contact_number',
                'contact': 'contact_number',
                'guardian name': 'guardian_name',
                'parent name': 'guardian_name',
                'guardian': 'guardian_name',
                'parent': 'guardian_name',
                'emergency contact name': 'guardian_name',
                'guardian contact': 'guardian_contact',
                'guardian contact number': 'guardian_contact',
                'guardian contact no': 'guardian_contact',
                'guardian phone': 'guardian_contact',
                'guardian mobile': 'guardian_contact',
                'guardian tel': 'guardian_contact',
                'parent contact': 'guardian_contact',
                'parent contact number': 'guardian_contact',
                'parent contact no': 'guardian_contact',
                'parent phone': 'guardian_contact',
                'parent mobile': 'guardian_contact',
                'emergency contact': 'guardian_contact',
                'emergency contact number': 'guardian_contact',
                'emergency contact no': 'guardian_contact',
                'emergency phone': 'guardian_contact',
                'guardian\'s contact': 'guardian_contact',
                'guardian\'s contact number': 'guardian_contact',
                'guardian\'s phone': 'guardian_contact',
                'guardians contact': 'guardian_contact',
                'guardians contact number': 'guardian_contact',
                'guardians phone': 'guardian_contact',
            }

            # Standardize DataFrame column names to lowercase for easier matching
            original_columns = list(df.columns)
            df.columns = [str(col).lower().strip() for col in df.columns]
            
            # Debug: Print column names to see what we're working with
            print(f"📋 Original Excel columns: {original_columns}")
            print(f"📋 Standardized columns: {list(df.columns)}")
            
            # Find potential guardian contact columns that might not be in our mapping
            guardian_contact_candidates = [col for col in df.columns if 'guardian' in col or 'parent' in col or 'emergency' in col]
            print(f"🔍 Guardian/Parent related columns found: {guardian_contact_candidates}")

            for index, row in df.iterrows():
                student_data = {
                    'student_id': None, 'surname': None, 'first_name': None, 'full_name': None,
                    'year': None, 'course': None, 'section': None, 'contact_number': None,
                    'guardian_name': None, 'guardian_contact': None
                }

                # Prioritize direct extraction from DataFrame columns
                for col_header, data_key in column_mapping.items():
                    if col_header in df.columns and col_header in row.index and pd.notna(row[col_header]):
                        raw_value = str(row[col_header]).strip()
                        if raw_value and raw_value.lower() not in ['nan', '', 'null']:
                            cleaned_value = self.clean_extracted_value(raw_value, data_key)
                            if cleaned_value:
                                student_data[data_key] = cleaned_value

                # Special handling for guardian contact - try fuzzy matching
                if not student_data['guardian_contact']:
                    guardian_contact_value = self.find_guardian_contact_fuzzy(row, df.columns)
                    if guardian_contact_value:
                        cleaned_contact = self.clean_extracted_value(guardian_contact_value, 'guardian_contact')
                        if cleaned_contact:
                            student_data['guardian_contact'] = cleaned_contact

                # Enhanced course detection - try to extract from filename if missing
                if not student_data['course']:
                    filename_course = self.extract_course_from_filename(file_path)
                    if filename_course:
                        student_data['course'] = filename_course

                # Post-process name splitting
                if student_data['full_name'] and not (student_data['surname'] and student_data['first_name']):
                    student_data['surname'], student_data['first_name'] = self.split_full_name(student_data['full_name'])
                elif student_data['surname'] and student_data['first_name'] and not student_data['full_name']:
                    student_data['full_name'] = f"{student_data['surname']}, {student_data['first_name']}"

                # Ensure all fields are properly formatted
                for key, value in student_data.items():
                    if value is None:
                        student_data[key] = ''
                    elif key == 'year':
                        if isinstance(value, str) and value.isdigit():
                            student_data[key] = value
                        elif isinstance(value, (int, float)):
                            student_data[key] = str(int(value))
                        else:
                            student_data[key] = ''
                    elif not isinstance(value, str):
                        student_data[key] = str(value)

                # Only process records with essential data
                if student_data['student_id'] or student_data['full_name']:
                    formatted_text = self.format_student_data(student_data)
                    metadata = self.create_student_metadata(student_data)
                    
                    texts.append(formatted_text)
                    metadata_list.append(metadata)

            # Store using the smart hierarchy system
            if texts:
                print(f"📊 Processed {len(texts)} student records")
                return self.store_with_smart_hierarchy(texts, metadata_list, 'students')
            else:
                print("❌ No valid student data found in Excel file")
                return False
                
        except Exception as e:
            print(f"❌ Error processing structured student Excel file {file_path}: {e}")
            return False

    def process_unstructured_student_excel(self, filename):
        """Process unstructured Excel data"""
        try:
            # Read all sheets and all data as text
            xl_file = pd.ExcelFile(filename)
            all_text = ""
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
                for row in df.values:
                    row_text = ' '.join([str(cell) for cell in row if pd.notna(cell)])
                    all_text += row_text + "\n"
            
            # Split into potential student records
            student_records = self.split_into_student_records(all_text)
            
            texts = []
            metadata_list = []
            
            for record in student_records:
                student_data = self.extract_universal_student_data(record, 'excel_unstructured')
                
                if student_data['student_id']:
                    formatted_text = self.format_student_data(student_data)
                    metadata = self.create_student_metadata(student_data)
                    
                    texts.append(formatted_text)
                    metadata_list.append(metadata)
            
            if texts:
                return self.store_with_smart_hierarchy(texts, metadata_list, 'students')
            else:
                print("❌ No valid student data found")
                return False
        
        except Exception as e:
            print(f"❌ Error processing unstructured Excel: {e}")
            return False

    def process_student_pdf(self, filename):
        """Process Student PDF with universal extraction"""
        try:
            doc = fitz.open(filename)
            all_text = ""
            for page in doc:
                all_text += page.get_text() + "\n"
            doc.close()
            
            # Split into potential student records
            student_records = self.split_into_student_records(all_text)
            
            texts = []
            metadata_list = []
            
            for record in student_records:
                student_data = self.extract_universal_student_data(record, 'pdf')
                
                # Only process if we found a student ID or a full name
                if student_data['student_id'] or student_data['full_name']:
                    formatted_text = self.format_student_data(student_data)
                    metadata = self.create_student_metadata(student_data)
                    
                    texts.append(formatted_text)
                    metadata_list.append(metadata)
            
            if texts:
                return self.store_with_smart_hierarchy(texts, metadata_list, 'students')
            else:
                print("❌ No valid student data found in PDF.")
                return False
        
        except Exception as e:
            print(f"❌ Error processing student PDF: {e}")
            return False

    def format_student_data(self, student_data):
        """Format extracted student data consistently"""
        return f"""
Student ID: {student_data.get('student_id', 'N/A')}
Full Name: {student_data.get('full_name', 'N/A')}
Surname: {student_data.get('surname', 'N/A')}
First Name: {student_data.get('first_name', 'N/A')}
Year: {student_data.get('year', 'N/A')}
Course: {student_data.get('course', 'N/A')}
Section: {student_data.get('section', 'N/A')}
Contact Number: {student_data.get('contact_number', 'N/A')}
Guardian Name: {student_data.get('guardian_name', 'N/A')}
Guardian Contact: {student_data.get('guardian_contact', 'N/A')}
""".strip()

    def create_student_metadata(self, student_data):
        """Create metadata from extracted student data - ensure all values are valid types"""
        # Ensure all metadata values are valid ChromaDB types (str, int, float, bool, None)
        metadata = {
            'course': str(student_data.get('course', '')),
            'section': str(student_data.get('section', '')),
            'year_level': str(student_data.get('year', '')),  # Keep as string for consistency
            'student_id': str(student_data.get('student_id', '')),
            'full_name': str(student_data.get('full_name', '')),
            'surname': str(student_data.get('surname', '')),
            'first_name': str(student_data.get('first_name', '')),
            'data_type': 'student_universal',
            'department': str(self.detect_department_from_course(student_data.get('course', '')) or '')
        }
        
        # Convert year_level to int if it's a valid number, otherwise keep as string
        try:
            if metadata['year_level'].isdigit():
                metadata['year_level'] = int(metadata['year_level'])
            elif not metadata['year_level']:
                metadata['year_level'] = 0
        except (ValueError, AttributeError):
            metadata['year_level'] = 0

        return metadata

    # ======================== COR PROCESSING ========================
    
    
    def extract_cor_excel_info_smart(self, filename):
        """Universal COR extraction that works with ANY Excel format"""
        try:
            # Read the entire Excel file
            df_full = pd.read_excel(filename, header=None)
            print(f"📋 COR Excel dimensions: {df_full.shape}")
            
            # Universal extraction - scan entire sheet for all data
            cor_info = self.extract_cor_universal_scan(df_full, filename)
            
            if cor_info and cor_info['program_info']['Program']:
                print("✅ Universal COR extraction successful")
                return cor_info
            
            print("❌ Could not extract COR data from any format")
            return None
            
        except Exception as e:
            print(f"❌ Error in universal COR extraction: {e}")
            return None

    def extract_cor_universal_scan(self, df, filename):
        """Universal scanner that finds COR data regardless of format"""
        try:
            program_info = {'Program': '', 'Year Level': '', 'Section': '', 'Adviser': ''}
            schedule_data = []
            total_units = None
            
            # STEP 1: Universal Program Info Extraction
            program_info = self.scan_for_program_info(df, filename)
            print(f"📋 Extracted Program Info: {program_info}")
            
            # STEP 2: Universal Schedule Extraction  
            schedule_data = self.scan_for_schedule_data(df)
            print(f"📋 Found {len(schedule_data)} subjects")
            
            # STEP 3: Universal Total Units Extraction
            total_units = self.scan_for_total_units(df)
            print(f"📋 Total Units: {total_units}")
            
            return {
                'program_info': program_info,
                'schedule': schedule_data,
                'total_units': total_units
            }
            
        except Exception as e:
            print(f"⚠️ Universal scan failed: {e}")
            return None
        
    def scan_for_program_info(self, df, filename):
        """Scan entire sheet for program information"""
        program_info = {'Program': '', 'Year Level': '', 'Section': '', 'Adviser': ''}
        
        # Define search patterns for each field
        search_patterns = {
            'Program': [
                (r'PROGRAM\s*[:\-]?\s*(.+)', 1),
                (r'COURSE\s*[:\-]?\s*(.+)', 1),
                (r'DEGREE\s*[:\-]?\s*(.+)', 1),
                (r'^(BS[A-Z]{2,4}|AB[A-Z]{2,4})$', 0),  # Direct course code
            ],
            'Year Level': [
                (r'YEAR\s*LEVEL\s*[:\-]?\s*(.+)', 1),
                (r'YEAR\s*[:\-]?\s*(.+)', 1),
                (r'LEVEL\s*[:\-]?\s*(.+)', 1),
                (r'^([1-4])(?:ST|ND|RD|TH)?\s*YEAR$', 1),
                (r'^([1-4])$', 0),  # Just a number 1-4
            ],
            'Section': [
                (r'SECTION\s*[:\-]?\s*(.+)', 1),
                (r'SEC\s*[:\-]?\s*(.+)', 1),
                (r'CLASS\s*[:\-]?\s*(.+)', 1),
                (r'^([A-Z])$', 0),  # Single letter
            ],
            'Adviser': [
                (r'ADVISER\s*[:\-]?\s*(.+)', 1),
                (r'ADVISOR\s*[:\-]?\s*(.+)', 1),
                (r'FACULTY\s*ADVISER\s*[:\-]?\s*(.+)', 1),
                (r'INSTRUCTOR\s*[:\-]?\s*(.+)', 1),
            ]
        }
        
        # Scan every cell in the sheet
        for i in range(min(df.shape[0], 30)):  # First 30 rows
            for j in range(min(df.shape[1], 15)):  # First 15 columns
                if pd.notna(df.iloc[i, j]):
                    cell_value = str(df.iloc[i, j]).strip()
                    cell_upper = cell_value.upper()
                    
                    # Check each program info field
                    for field, patterns in search_patterns.items():
                        if program_info[field]:  # Skip if already found
                            continue
                            
                        for pattern, group_idx in patterns:
                            match = re.search(pattern, cell_upper)
                            if match:
                                if group_idx == 0:  # Use entire match
                                    value = match.group(0)
                                else:  # Use specific group
                                    value = match.group(group_idx).strip()
                                
                                # Clean and validate the value
                                cleaned_value = self.clean_program_info_value(value, field)
                                if cleaned_value:
                                    program_info[field] = cleaned_value
                                    print(f"   Found {field}: {cleaned_value} at ({i},{j})")
                                    break
                    
                    # Also check adjacent cells for values
                    if not program_info['Program'] or not program_info['Year Level'] or not program_info['Section'] or not program_info['Adviser']:
                        self.check_adjacent_cells(df, i, j, cell_upper, program_info)
        
        # Fallback: Extract from filename if still missing critical info
        if not program_info['Program']:
            filename_course = self.extract_course_from_filename(filename)
            if filename_course:
                program_info['Program'] = filename_course
                print(f"   Found Program from filename: {filename_course}")
        
        return program_info
    
    
    def check_adjacent_cells(self, df, row, col, cell_text, program_info):
        """Check adjacent cells for program info values"""
        keywords = {
            'Program': ['PROGRAM', 'COURSE', 'DEGREE'],
            'Year Level': ['YEAR', 'LEVEL'],
            'Section': ['SECTION', 'SEC', 'CLASS'],
            'Adviser': ['ADVISER', 'ADVISOR', 'FACULTY', 'INSTRUCTOR']
        }
        
        for field, field_keywords in keywords.items():
            if program_info[field]:  # Skip if already found
                continue
                
            if any(keyword in cell_text for keyword in field_keywords):
                # Check adjacent cells (right, below, diagonal)
                adjacent_positions = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
                
                for dr, dc in adjacent_positions:
                    new_row, new_col = row + dr, col + dc
                    if (new_row < df.shape[0] and new_col < df.shape[1] and 
                        pd.notna(df.iloc[new_row, new_col])):
                        
                        adjacent_value = str(df.iloc[new_row, new_col]).strip()
                        cleaned_value = self.clean_program_info_value(adjacent_value, field)
                        
                        if cleaned_value:
                            program_info[field] = cleaned_value
                            print(f"   Found {field} adjacent: {cleaned_value}")
                            break
                        
    
    def clean_program_info_value(self, value, field):
        """Clean and validate program info values"""
        if not value or len(value.strip()) == 0:
            return None
        
        value = value.strip()
        
        if field == 'Program':
            # Smart program extraction - use whatever is actually in the COR
            value_upper = value.upper().strip()
            
            # Remove common prefixes/suffixes that aren't part of the actual program
            clean_value = re.sub(r'^(PROGRAM|COURSE|DEGREE)[:\s]*', '', value_upper)
            clean_value = re.sub(r'\s*(PROGRAM|COURSE|DEGREE)$', '', clean_value)
            
            # Remove extra whitespace and clean up
            clean_value = ' '.join(clean_value.split())
            
            # If it's a reasonable length and contains letters, use it as-is
            if 2 <= len(clean_value) <= 50 and re.search(r'[A-Z]', clean_value):
                return clean_value
            
            # If original value is better, use that
            if 2 <= len(value) <= 50:
                return value.strip()
            
            return None
        
        elif field == 'Year Level':
            # Extract year number
            year_match = re.search(r'([1-4])', value)
            return year_match.group(1) if year_match else None
        
        elif field == 'Section':
            # Extract section letter/number - FIX: More precise extraction
            value_upper = value.upper().strip()
            
            # Remove common prefixes
            clean_section = re.sub(r'^(SECTION|SEC)[:\s]*', '', value_upper)
            
            # Look for single letter or letter-number combination
            section_match = re.search(r'^([A-Z][0-9]?|[0-9][A-Z]?)$', clean_section)
            if section_match:
                return section_match.group(1)
            
            # Fallback: if it's just one or two characters, use it
            if 1 <= len(clean_section) <= 2 and clean_section.isalnum():
                return clean_section
            
            # If original value looks like a section, clean it up
            if len(value) <= 3:
                clean_original = re.sub(r'[^A-Z0-9]', '', value.upper())
                if clean_original:
                    return clean_original
            
            return None
        
        elif field == 'Adviser':
            # Clean adviser name
            if len(value) > 3 and not any(char.isdigit() for char in value):
                return value.title()
            return None
        
        return value
    
    def scan_for_schedule_data(self, df):
        """Universal schedule data extraction"""
        schedule_data = []
        
        # Find potential schedule headers
        header_keywords = ['SUBJECT', 'CODE', 'DESCRIPTION', 'UNITS', 'DAY', 'TIME', 'ROOM']
        schedule_start_row = -1
        
        # Look for a row with multiple schedule keywords
        for i in range(df.shape[0]):
            row_text = ' '.join([str(df.iloc[i, j]) for j in range(min(df.shape[1], 10)) if pd.notna(df.iloc[i, j])]).upper()
            keyword_count = sum(1 for keyword in header_keywords if keyword in row_text)
            
            if keyword_count >= 4:  # Found header row
                schedule_start_row = i + 1
                print(f"   Found schedule header at row {i}, data starts at {schedule_start_row}")
                break
        
        if schedule_start_row == -1:
            # Fallback: look for subject codes pattern
            for i in range(df.shape[0]):
                if pd.notna(df.iloc[i, 0]):
                    cell_value = str(df.iloc[i, 0]).strip()
                    if re.match(r'^[A-Z]{2,4}\s*\d{3}[A-Z]?$', cell_value.upper()):
                        schedule_start_row = i
                        print(f"   Found schedule data starting at row {i} (subject code pattern)")
                        break
        
        if schedule_start_row >= 0:
            # Extract schedule using flexible column mapping
            schedule_data = self.extract_schedule_flexible(df, schedule_start_row)
        
        return schedule_data
    
    
    def extract_schedule_flexible(self, df, start_row):
        """Flexible schedule extraction that adapts to column layout"""
        schedule_data = []
        
        # Try different column arrangements
        column_arrangements = [
            # Standard arrangement
            {'subject_col': 0, 'desc_col': 1, 'type_col': 2, 'units_col': 3, 'day_col': 4, 'time_start_col': 5, 'time_end_col': 6, 'room_col': 7},
            # No type column
            {'subject_col': 0, 'desc_col': 1, 'units_col': 2, 'day_col': 3, 'time_start_col': 4, 'time_end_col': 5, 'room_col': 6},
            # Combined time column
            {'subject_col': 0, 'desc_col': 1, 'type_col': 2, 'units_col': 3, 'day_col': 4, 'time_col': 5, 'room_col': 6},
            # Minimal columns
            {'subject_col': 0, 'desc_col': 1, 'units_col': 2, 'day_col': 3, 'time_col': 4, 'room_col': 5},
        ]
        
        for arrangement in column_arrangements:
            test_data = self.try_column_arrangement(df, start_row, arrangement)
            if len(test_data) > 0:
                schedule_data = test_data
                print(f"   Using column arrangement: {arrangement}")
                break
        
        return schedule_data

    def try_column_arrangement(self, df, start_row, arrangement):
        """Try a specific column arrangement"""
        schedule_data = []
        
        for i in range(start_row, df.shape[0]):
            # Stop at empty rows or footer text
            if self.is_schedule_end_row(df, i):
                break
            
            # Extract subject data based on arrangement
            subject_entry = {}
            
            # Subject Code (required)
            if arrangement['subject_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['subject_col']]):
                subject_code = str(df.iloc[i, arrangement['subject_col']]).strip()
                if not re.match(r'^[A-Z]{2,4}\s*\d{3}[A-Z]?', subject_code.upper()):
                    continue  # Not a valid subject code
                subject_entry['Subject Code'] = subject_code
            else:
                continue
            
            # Description
            if arrangement['desc_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['desc_col']]):
                subject_entry['Description'] = str(df.iloc[i, arrangement['desc_col']]).strip()
            
            # Type (if exists)
            if 'type_col' in arrangement and arrangement['type_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['type_col']]):
                subject_entry['Type'] = str(df.iloc[i, arrangement['type_col']]).strip()
            
            # Units
            if arrangement['units_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['units_col']]):
                units_value = str(df.iloc[i, arrangement['units_col']]).strip()
                if re.match(r'^\d+(\.\d+)?$', units_value):
                    subject_entry['Units'] = units_value
            
            # Day
            if arrangement['day_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['day_col']]):
                subject_entry['Day'] = str(df.iloc[i, arrangement['day_col']]).strip()
            
            # Time handling
            if 'time_col' in arrangement:  # Single time column
                if arrangement['time_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['time_col']]):
                    time_value = str(df.iloc[i, arrangement['time_col']]).strip()
                    # Split time range
                    time_parts = re.split(r'[-–]|to|TO', time_value)
                    if len(time_parts) == 2:
                        subject_entry['Time Start'] = time_parts[0].strip()
                        subject_entry['Time End'] = time_parts[1].strip()
                    else:
                        subject_entry['Time Start'] = time_value
                        subject_entry['Time End'] = time_value
            else:  # Separate time columns
                if 'time_start_col' in arrangement and arrangement['time_start_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['time_start_col']]):
                    subject_entry['Time Start'] = str(df.iloc[i, arrangement['time_start_col']]).strip()
                
                if 'time_end_col' in arrangement and arrangement['time_end_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['time_end_col']]):
                    subject_entry['Time End'] = str(df.iloc[i, arrangement['time_end_col']]).strip()
            
            # Room
            if arrangement['room_col'] < df.shape[1] and pd.notna(df.iloc[i, arrangement['room_col']]):
                subject_entry['Room'] = str(df.iloc[i, arrangement['room_col']]).strip()
            
            schedule_data.append(subject_entry)
        
        return schedule_data
    
    
    def is_schedule_end_row(self, df, row):
        """Check if this row indicates end of schedule data"""
        if row >= df.shape[0]:
            return True
        
        # Check if row is empty
        if all(pd.isna(df.iloc[row, j]) for j in range(df.shape[1])):
            return True
        
        # Check for footer keywords
        first_cell = str(df.iloc[row, 0]) if pd.notna(df.iloc[row, 0]) else ""
        footer_keywords = ['TOTAL', 'GENERATED', 'PRINTED', 'PAGE', 'END']
        
        return any(keyword in first_cell.upper() for keyword in footer_keywords)

    def scan_for_total_units(self, df):
        """Find total units anywhere in the sheet"""
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if pd.notna(df.iloc[i, j]):
                    cell_value = str(df.iloc[i, j]).upper()
                    if 'TOTAL' in cell_value and ('UNIT' in cell_value or 'CREDIT' in cell_value):
                        # Look for number in nearby cells
                        for di in range(-1, 2):
                            for dj in range(-1, 4):
                                ni, nj = i + di, j + dj
                                if (0 <= ni < df.shape[0] and 0 <= nj < df.shape[1] and 
                                    pd.notna(df.iloc[ni, nj])):
                                    potential_units = str(df.iloc[ni, nj]).strip()
                                    if re.match(r'^\d+(\.\d+)?$', potential_units):
                                        return potential_units
        return None

    
    def extract_cor_standard_format(self, df):
        """Extract from standard COR format (original method enhanced)"""
        try:
            program_info = {
                'Program': '',
                'Year Level': '',
                'Section': '',
                'Adviser': ''
            }
            
            # Look for standard positions first
            if df.shape[0] >= 4 and df.shape[1] >= 2:
                if pd.notna(df.iloc[0, 1]): 
                    raw_program = str(df.iloc[0, 1]).strip()
                    # Clean program name - extract just the course code
                    program_match = re.search(r'(BS[A-Z]{2,4}|AB[A-Z]{2,4})', raw_program.upper())
                    program_info['Program'] = program_match.group(1) if program_match else raw_program
                    
                if pd.notna(df.iloc[1, 1]): 
                    raw_year = str(df.iloc[1, 1]).strip()
                    # Clean year level - extract just the number
                    year_match = re.search(r'([1-4])', raw_year)
                    program_info['Year Level'] = year_match.group(1) if year_match else raw_year
                    
                if pd.notna(df.iloc[2, 1]): 
                    program_info['Section'] = str(df.iloc[2, 1]).strip()
                    
                if pd.notna(df.iloc[3, 1]): 
                    program_info['Adviser'] = str(df.iloc[3, 1]).strip()
            
            # Debug output
            print(f"📋 Extracted Program Info: {program_info}")
            
            # Try to find schedule data starting from row 5
            schedule_data = []
            total_units = None
            
            # Look for header row with subject information
            header_row = -1
            for i in range(4, min(10, df.shape[0])):
                row_text = ' '.join([str(df.iloc[i, j]) for j in range(min(8, df.shape[1])) if pd.notna(df.iloc[i, j])]).upper()
                if 'SUBJECT' in row_text and 'DESCRIPTION' in row_text:
                    header_row = i + 1
                    print(f"📋 Found schedule header at row {i}, data starts at row {header_row}")
                    break
            
            if header_row > 0:
                schedule_data, total_units = self.extract_schedule_from_rows(df, header_row)
                print(f"📋 Extracted {len(schedule_data)} subjects")
            
            return {
                'program_info': program_info,
                'schedule': schedule_data,
                'total_units': total_units
            }
            
        except Exception as e:
            print(f"⚠️ Standard format extraction failed: {e}")
            return None
        
    
    def find_schedule_header_row(self, df):
        """Find the row that contains schedule headers"""
        schedule_keywords = ['SUBJECT CODE', 'SUBJECT', 'DESCRIPTION', 'UNITS', 'DAY', 'TIME', 'ROOM']
        
        for i in range(df.shape[0]):
            row_text = ' '.join([str(df.iloc[i, j]) for j in range(min(df.shape[1], 10)) if pd.notna(df.iloc[i, j])]).upper()
            keyword_count = sum(1 for keyword in schedule_keywords if keyword in row_text)
            
            # If we find at least 4 schedule keywords in a row, it's likely the header
            if keyword_count >= 4:
                return i + 1  # Return next row as data start
        
        return -1

    def extract_schedule_from_rows(self, df, start_row):
        """Extract schedule data from rows starting at start_row"""
        schedule_data = []
        total_units = None
        
        # Define column mapping flexibility
        possible_mappings = [
            # Standard 8-column format
            {'Subject Code': 0, 'Description': 1, 'Type': 2, 'Units': 3, 'Day': 4, 'Time Start': 5, 'Time End': 6, 'Room': 7},
            # Alternative 7-column format (no separate time start/end)
            {'Subject Code': 0, 'Description': 1, 'Type': 2, 'Units': 3, 'Day': 4, 'Time': 5, 'Room': 6},
            # Compact format
            {'Subject Code': 0, 'Description': 1, 'Units': 2, 'Day': 3, 'Time': 4, 'Room': 5},
        ]
        
        # Try each mapping
        for mapping in possible_mappings:
            test_schedule = self.try_schedule_mapping(df, start_row, mapping)
            if len(test_schedule) > 0:
                schedule_data = test_schedule
                break
        
        # Look for total units
        for i in range(start_row, df.shape[0]):
            for j in range(df.shape[1]):
                if pd.notna(df.iloc[i, j]):
                    cell_value = str(df.iloc[i, j]).upper()
                    if 'TOTAL UNITS' in cell_value or 'TOTAL:' in cell_value:
                        # Look for the number in nearby cells
                        for offset in range(1, 4):
                            if j + offset < df.shape[1] and pd.notna(df.iloc[i, j + offset]):
                                potential_units = str(df.iloc[i, j + offset]).strip()
                                if re.match(r'^\d+(\.\d+)?$', potential_units):
                                    total_units = potential_units
                                    break
                        if total_units:
                            break
            if total_units:
                break
        
        return schedule_data, total_units

    def try_schedule_mapping(self, df, start_row, mapping):
        """Try a specific column mapping for schedule data"""
        schedule_data = []
        max_cols = max(mapping.values()) + 1
        
        for i in range(start_row, df.shape[0]):
            # Stop if we hit empty rows or "Total Units"
            if all(pd.isna(df.iloc[i, j]) for j in range(min(max_cols, df.shape[1]))):
                break
            
            first_cell = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
            if any(keyword in first_cell.upper() for keyword in ['TOTAL', 'GENERATED', 'PRINTED']):
                break
            
            # Extract subject data
            subject_entry = {}
            valid_entry = False
            
            for field, col_idx in mapping.items():
                if col_idx < df.shape[1] and pd.notna(df.iloc[i, col_idx]):
                    value = str(df.iloc[i, col_idx]).strip()
                    subject_entry[field] = value
                    if field == 'Subject Code' and value:
                        valid_entry = True
            
            # Handle time fields specially
            if 'Time' in mapping and 'Time Start' not in mapping:
                # Split single time field into start and end
                time_value = subject_entry.get('Time', '')
                if '-' in time_value or 'to' in time_value.lower():
                    time_parts = re.split(r'[-–]|to', time_value, 1)
                    if len(time_parts) == 2:
                        subject_entry['Time Start'] = time_parts[0].strip()
                        subject_entry['Time End'] = time_parts[1].strip()
                        del subject_entry['Time']
            
            if valid_entry and subject_entry.get('Subject Code'):
                schedule_data.append(subject_entry)
        
        return schedule_data

    def extract_cor_multisheet_format(self, filename):
        """Handle multi-sheet Excel files"""
        try:
            xl_file = pd.ExcelFile(filename)
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
                
                # Try standard format on this sheet
                cor_info = self.extract_cor_standard_format(df)
                if cor_info and cor_info['program_info']['Program']:
                    return cor_info
                
                # Try flexible format on this sheet
                cor_info = self.extract_cor_flexible_format(df)
                if cor_info and cor_info['program_info']['Program']:
                    return cor_info
            
            return None
            
        except Exception as e:
            print(f"⚠️ Multi-sheet extraction failed: {e}")
            return None

    def extract_cor_table_format(self, df):
        """Extract from table-style format where data is in structured rows/columns"""
        try:
            # This method handles cases where the COR is formatted as a single table
            # with program info in the first few rows and schedule below
            
            program_info = {'Program': '', 'Year Level': '', 'Section': '', 'Adviser': ''}
            schedule_data = []
            total_units = None
            
            # Scan the entire sheet for program information and schedule data
            program_keywords = {
                'program': ['PROGRAM', 'COURSE', 'DEGREE'],
                'year': ['YEAR', 'LEVEL'],
                'section': ['SECTION', 'SEC'],
                'adviser': ['ADVISER', 'ADVISOR', 'FACULTY']
            }
            
            # Extract program info from anywhere in the sheet
            for i in range(min(df.shape[0], 30)):  # Check first 30 rows
                for j in range(min(df.shape[1], 10)):  # Check first 10 columns
                    if pd.notna(df.iloc[i, j]):
                        cell_text = str(df.iloc[i, j]).upper().strip()
                        
                        for info_type, keywords in program_keywords.items():
                            if any(keyword in cell_text for keyword in keywords):
                                # Look for value in adjacent cells
                                value = None
                                for dj in range(1, 4):  # Check next 3 columns
                                    if j + dj < df.shape[1] and pd.notna(df.iloc[i, j + dj]):
                                        potential_value = str(df.iloc[i, j + dj]).strip()
                                        if potential_value and not any(kw in potential_value.upper() for kw_list in program_keywords.values() for kw in kw_list):
                                            value = potential_value
                                            break
                                
                                if value:
                                    if info_type == 'program':
                                        program_info['Program'] = value
                                    elif info_type == 'year':
                                        program_info['Year Level'] = value
                                    elif info_type == 'section':
                                        program_info['Section'] = value
                                    elif info_type == 'adviser':
                                        program_info['Adviser'] = value
            
            # Find and extract schedule
            header_row = self.find_schedule_header_row(df)
            if header_row >= 0:
                schedule_data, total_units = self.extract_schedule_from_rows(df, header_row)
            
            return {
                'program_info': program_info,
                'schedule': schedule_data,
                'total_units': total_units
            }
            
        except Exception as e:
            print(f"⚠️ Table format extraction failed: {e}")
            return None
    
    
    def format_cor_info_enhanced(self, cor_info):
        """Enhanced COR formatting with more detailed information"""
        text = f"""COR (Certificate of Registration) - Class Schedule

    STUDENT INFORMATION:
    Program: {cor_info['program_info']['Program']}
    Year Level: {cor_info['program_info']['Year Level']}
    Section: {cor_info['program_info']['Section']}
    Adviser: {cor_info['program_info']['Adviser']}
    Total Units: {cor_info.get('total_units', 'N/A')}

    ENROLLED SUBJECTS ({len(cor_info['schedule'])} subjects):
    """
        
        if cor_info['schedule']:
            for i, course in enumerate(cor_info['schedule'], 1):
                text += f"""
    Subject {i}:
    • Subject Code: {course.get('Subject Code', 'N/A')}
    • Description: {course.get('Description', 'N/A')}
    • Type: {course.get('Type', 'N/A')}
    • Units: {course.get('Units', 'N/A')}
    • Schedule: {course.get('Day', 'N/A')} {course.get('Time Start', 'N/A')}-{course.get('Time End', 'N/A')}
    • Room: {course.get('Room', 'N/A')}
    """
        else:
            text += "\nNo subjects found in schedule."
        
        return text.strip()
    
    
    def extract_cor_flexible_format(self, df):
        """Extract COR info by searching for keywords anywhere in the sheet"""
        try:
            program_info = {
                'Program': '',
                'Year Level': '',
                'Section': '',
                'Adviser': ''
            }
            
            # Search for program info throughout the sheet
            for i in range(min(20, df.shape[0])):
                for j in range(min(10, df.shape[1])):
                    if pd.notna(df.iloc[i, j]):
                        cell_value = str(df.iloc[i, j]).strip().upper()
                        
                        # Look for program info
                        if cell_value in ['PROGRAM:', 'COURSE:', 'DEGREE:']:
                            if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                                program_info['Program'] = str(df.iloc[i, j + 1]).strip()
                            elif i + 1 < df.shape[0] and pd.notna(df.iloc[i + 1, j]):
                                program_info['Program'] = str(df.iloc[i + 1, j]).strip()
                        
                        elif cell_value in ['YEAR LEVEL:', 'YEAR:', 'LEVEL:']:
                            if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                                program_info['Year Level'] = str(df.iloc[i, j + 1]).strip()
                            elif i + 1 < df.shape[0] and pd.notna(df.iloc[i + 1, j]):
                                program_info['Year Level'] = str(df.iloc[i + 1, j]).strip()
                        
                        elif cell_value in ['SECTION:', 'SEC:']:
                            if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                                program_info['Section'] = str(df.iloc[i, j + 1]).strip()
                            elif i + 1 < df.shape[0] and pd.notna(df.iloc[i + 1, j]):
                                program_info['Section'] = str(df.iloc[i + 1, j]).strip()
                        
                        elif cell_value in ['ADVISER:', 'ADVISOR:', 'FACULTY ADVISER:']:
                            if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                                program_info['Adviser'] = str(df.iloc[i, j + 1]).strip()
                            elif i + 1 < df.shape[0] and pd.notna(df.iloc[i + 1, j]):
                                program_info['Adviser'] = str(df.iloc[i + 1, j]).strip()
            
            # Find schedule data
            schedule_data = []
            total_units = None
            
            # Look for schedule headers
            header_row = self.find_schedule_header_row(df)
            if header_row >= 0:
                schedule_data, total_units = self.extract_schedule_from_rows(df, header_row)
            
            return {
                'program_info': program_info,
                'schedule': schedule_data,
                'total_units': total_units
            }
            
        except Exception as e:
            print(f"⚠️ Flexible format extraction failed: {e}")
            return None
    
    def process_cor_excel(self, filename):
        """Enhanced COR Excel processing with intelligent format detection"""
        try:
            # Try multiple approaches to detect COR format
            cor_info = self.extract_cor_excel_info_smart(filename)
            
            if not cor_info or not cor_info['program_info']['Program']:
                print("❌ Could not extract COR data from Excel")
                return False
                
            formatted_text = self.format_cor_info_enhanced(cor_info)
            
            # Create smart metadata - FIX: Convert list to string
            subject_codes_list = [course.get('Subject Code', '') for course in cor_info['schedule'] if course.get('Subject Code')]
            subject_codes_string = ', '.join(subject_codes_list)  # Convert list to comma-separated string
            
            metadata = {
                'course': cor_info['program_info']['Program'],
                'section': cor_info['program_info']['Section'],
                'year_level': cor_info['program_info']['Year Level'],
                'adviser': cor_info['program_info']['Adviser'],
                'data_type': 'cor_excel',
                'subject_codes': subject_codes_string,  # Now a string instead of list
                'total_units': str(cor_info.get('total_units', '')),  # Ensure it's a string
                'subject_count': len(cor_info['schedule']),  # Add subject count as metadata
                'department': self.detect_department_from_course(cor_info['program_info']['Program'])
            }
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('schedules', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            hierarchy_path = f"{self.get_department_display_name(metadata['department'])} > {metadata['course']} > Year {metadata['year_level']} > Section {metadata['section']}"
            print(f"✅ Loaded COR schedule into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   📚 Subjects: {metadata['subject_count']}, Total Units: {metadata['total_units']}")
            print(f"   📋 Subject Codes: {subject_codes_string}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing COR Excel: {e}")
            return False
   
    def process_cor_pdf(self, filename):
        """Process COR PDF file with smart organization"""
        cor_info = self.extract_cor_pdf_info(filename)
        if not cor_info:
            print("❌ Could not extract COR data from PDF")
            return False
        
        formatted_text = self.format_cor_info(cor_info)
        
        # Create smart metadata
        metadata = {
            'course': cor_info['program_info']['Program'],
            'section': cor_info['program_info']['Section'],
            'year_level': cor_info['program_info']['Year Level'],
            'adviser': cor_info['program_info']['Adviser'],
            'data_type': 'cor_pdf',
            'subject_codes': [course.get('Subject Code', '') for course in cor_info['schedule'] if course.get('Subject Code')]
        }
        
        # Smart department detection
        metadata['department'] = self.detect_department_from_course(metadata['course'])
        
        # Store with hierarchy
        collection_name = self.create_smart_collection_name('schedules', metadata)
        # Use get_or_create_collection with the consistent embedding function
        collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_function
        )
        self.store_with_smart_metadata(collection, [formatted_text], [metadata])
        self.collections[collection_name] = collection
        
        hierarchy_path = f"{self.get_department_display_name(metadata['department'])} > {metadata['course']} > Year {metadata['year_level']} > Section {metadata['section']}"
        print(f"✅ Loaded COR schedule into: {collection_name}")
        print(f"   📁 Hierarchy: {hierarchy_path}")
        return True
   
   # ======================== FACULTY PROCESSING ========================
   
    def process_faculty_excel(self, filename):
        """Process Faculty Excel file with smart organization"""
        try:
            df = pd.read_excel(filename, header=None)
            
            # Extract faculty data from columns A and B (rows 1-45)
            faculty_data = {}
            current_section = ""
            
            for i in range(min(45, len(df))):  # Process up to row 45 or end of file
                field_name = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""
                field_value = str(df.iloc[i, 1]).strip() if pd.notna(df.iloc[i, 1]) and str(df.iloc[i, 1]).strip() != 'nan' else ""
                
                # Skip empty rows
                if not field_name:
                    continue
                
                # Check if this is a section header (ALL CAPS)
                if field_name.isupper() and not field_value:
                    current_section = field_name
                    faculty_data[current_section] = {}
                else:
                    # Add field to current section or general data
                    if current_section:
                        faculty_data[current_section][field_name] = field_value
                    else:
                        faculty_data[field_name] = field_value
            
            # Format as text for ChromaDB storage
            formatted_text = self.format_faculty_excel_data(faculty_data)
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(formatted_text, 'faculty_excel')
            metadata['faculty_type'] = 'profile'
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            # Use get_or_create_collection with the consistent embedding function
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            # Extract name for display
            faculty_name = faculty_data.get("PERSONAL INFORMATION", {}).get("Full Name", "Unknown Faculty")
            
            print(f"✅ Loaded faculty data into: {collection_name}")
            print(f"   Faculty: {faculty_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty Excel: {e}")
            return False
        
    def process_faculty_pdf(self, filename):
        """Process Faculty PDF file (Resume) with smart organization"""
        try:
            faculty_data = self.extract_faculty_pdf_data(filename)
            if not faculty_data:
                print("❌ Could not extract faculty data from PDF")
                return False
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(faculty_data, 'faculty_pdf')
            metadata['faculty_type'] = 'profile'
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            # Use get_or_create_collection with the consistent embedding function
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [faculty_data], [metadata])
            self.collections[collection_name] = collection
            
            # Extract name from data for display
            lines = faculty_data.split('\n')
            faculty_name = lines[0] if lines else "Unknown Faculty"
            
            print(f"✅ Loaded faculty resume into: {collection_name}")
            print(f"   Faculty: {faculty_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty PDF: {e}")
            return False

    def process_faculty_schedule_excel(self, filename):
        """Process Faculty Schedule Excel file with smart organization"""
        try:
            df = pd.read_excel(filename, header=None)
            
            # Extract adviser name from first row
            adviser_name = ""
            for i in range(min(3, len(df))):
                for j in range(min(10, df.shape[1])):
                    cell_value = str(df.iloc[i, j]) if pd.notna(df.iloc[i, j]) else ""
                    if "ADVISER" in cell_value.upper():
                        # Look for the adviser name in nearby cells
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            adviser_name = str(df.iloc[i, j + 1])
                        break
            
            # Extract schedule data (simplified for now)
            schedule_data = []
            # ... (schedule extraction logic) ...
            
            # Format for ChromaDB
            formatted_text = self.format_faculty_schedule_data(adviser_name, schedule_data)
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(formatted_text, 'faculty_schedule_excel')
            metadata['faculty_type'] = 'teaching'
            metadata['adviser'] = adviser_name
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            # Use get_or_create_collection with the consistent embedding function
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            print(f"✅ Loaded faculty schedule into: {collection_name}")
            print(f"   Adviser: {adviser_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty schedule Excel: {e}")
            return False

    def process_faculty_schedule_pdf(self, filename):
        """Process Faculty Schedule PDF file with smart organization"""
        try:
            doc = fitz.open(filename)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            # Find adviser name
            adviser_name = "Unknown"
            for line in lines:
                if "NAME OF ADVISER:" in line:
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            name_part = parts[1].strip()
                            if name_part:
                                adviser_name = name_part
                                break
            
            # Format the schedule data
            formatted_text = f"FACULTY CLASS SCHEDULE\nName of Adviser: {adviser_name}\n\n"
            formatted_text += full_text
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(formatted_text, 'faculty_schedule_pdf')
            metadata['faculty_type'] = 'teaching'
            metadata['adviser'] = adviser_name
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            # Use get_or_create_collection with the consistent embedding function
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            print(f"✅ Loaded faculty schedule into: {collection_name}")
            print(f"   Adviser: {adviser_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty schedule PDF: {e}")
            return False

   # ======================== FILE TYPE DETECTION ========================
   
    def is_cor_file(self, df):
        """Check if Excel is a COR file"""
        try:
            return (df.iloc[0, 0] == "Program:" and 
                    df.iloc[1, 0] == "Year Level:" and 
                    df.iloc[2, 0] == "Section:" and 
                    df.iloc[3, 0] == "Adviser:")
        except:
            return False
    
    def is_cor_pdf(self, filename):
        """Check if PDF is a COR file by looking for schedule keywords"""
        try:
            doc = fitz.open(filename)
            first_page = doc[0].get_text().lower()
            doc.close()
            
            # Look for COR-specific keywords
            cor_keywords = ["program:", "year level:", "section:", "adviser:", "subject code", "description", "units"]
            cor_count = sum(1 for keyword in cor_keywords if keyword in first_page)
            
            # Must have multiple COR-specific indicators and NOT be a class schedule
            return cor_count >= 4 and "class schedule" not in first_page
        except:
            return False

    def is_faculty_excel(self, df):
        """Check if Excel is a Faculty file"""
        try:
            # Check if first column contains faculty-specific headers
            first_col = df.iloc[:, 0].astype(str).str.upper()
            faculty_keywords = ["PERSONAL INFORMATION", "CONTACT INFORMATION", "OCCUPATIONAL INFORMATION"]
            return any(keyword in first_col.values for keyword in faculty_keywords)
        except:
            return False

    def is_faculty_pdf(self, filename):
        """Check if PDF is a Faculty file by looking for resume keywords"""
        try:
            doc = fitz.open(filename)
            first_page = doc[0].get_text().lower()
            doc.close()
            
            # Look for resume-specific keywords
            resume_keywords = ["professional profile", "education", "professional experience", "certifications", "email:", "phone:"]
            return any(keyword in first_page for keyword in resume_keywords)
        except:
            return False

    def is_faculty_schedule_excel(self, df):
        """Enhanced check for Faculty Schedule Excel files"""
        try:
            df_str = df.astype(str)
            first_few_rows = ' '.join(df_str.iloc[:15].values.flatten()).upper()
            
            # Enhanced detection patterns
            faculty_schedule_indicators = [
                "NAME OF ADVISER", "ADVISER", "CLASS SCHEDULE", 
                "FACULTY SCHEDULE", "TEACHING SCHEDULE", "INSTRUCTOR"
            ]
            
            # Look for day layout (key indicator of schedules)
            day_indicators = ["MON", "TUE", "WED", "THU", "FRI", "SAT", 
                            "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"]
            
            has_faculty_indicator = any(indicator in first_few_rows for indicator in faculty_schedule_indicators)
            has_day_layout = sum(1 for day in day_indicators if day in first_few_rows) >= 3  # At least 3 days
            
            # Should NOT have student data indicators
            student_indicators = ["STUDENT ID", "CONTACT NUMBER", "GUARDIAN", "YEAR LEVEL", "COURSE SECTION"]
            has_student_indicator = any(indicator in first_few_rows for indicator in student_indicators)
            
            # Enhanced: Check for time patterns (schedules have time slots)
            has_time_pattern = bool(re.search(r'\d{1,2}:\d{2}.*?(?:AM|PM|am|pm)', first_few_rows))
            
            result = (has_faculty_indicator or has_day_layout) and not has_student_indicator
            if has_time_pattern:
                result = True  # Strong indicator
                
            return result
            
        except Exception as e:
            print(f"🔍 Error in faculty schedule detection: {e}")
            return False

    def is_faculty_schedule_pdf(self, filename):
        """Check if PDF is a Faculty Schedule file"""
        try:
            doc = fitz.open(filename)
            first_page = doc[0].get_text().lower()
            doc.close()
            
            schedule_keywords = ["class schedule", "name of adviser", "subject/s time"]
            faculty_schedule_indicators = ["mon tue wed thu fri", "subject/s", "time mon tue"]
            
            has_schedule_title = any(keyword in first_page for keyword in schedule_keywords)
            has_day_layout = any(indicator in first_page for indicator in faculty_schedule_indicators)
            
            return has_schedule_title and has_day_layout
        except:
            return False

    # ======================== HELPER METHODS ========================
    
    def extract_cor_excel_info(self, filename):
        """Extract COR information from Excel"""
        raw_df = pd.read_excel(filename, header=None)
        
        program_info = {
            'Program': raw_df.iloc[0, 1] if pd.notna(raw_df.iloc[0, 1]) else "",
            'Year Level': raw_df.iloc[1, 1] if pd.notna(raw_df.iloc[1, 1]) else "",
            'Section': raw_df.iloc[2, 1] if pd.notna(raw_df.iloc[2, 1]) else "",
            'Adviser': raw_df.iloc[3, 1] if pd.notna(raw_df.iloc[3, 1]) else ""
        }
        
        schedule_df = pd.read_excel(filename, header=4)
        schedule_data = []
        
        for _, row in schedule_df.iterrows():
            if pd.notna(row.iloc[0]) and not "Generated on:" in str(row.iloc[0]):
                schedule_data.append({
                    'Subject Code': row.iloc[0],
                    'Description': row.iloc[1],
                    'Type': row.iloc[2],
                    'Units': row.iloc[3],
                    'Day': row.iloc[4],
                    'Time Start': row.iloc[5],
                    'Time End': row.iloc[6],
                    'Room': row.iloc[7]
                })
        
        total_units = None
        for i in range(1, 5):
            try:
                if "Total Units:" in str(schedule_df.iloc[-i, 2]):
                    total_units = schedule_df.iloc[-i, 3]
                    break
            except:
                pass
        
        return {
            'program_info': program_info,
            'schedule': schedule_data,
            'total_units': total_units
        }
    
    def extract_cor_pdf_info(self, filename):
        """Extract COR information from PDF"""
        try:
            doc = fitz.open(filename)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            # Extract program info
            program_info = {
                'Program': '',
                'Year Level': '',
                'Section': '',
                'Adviser': ''
            }
            
            for i, line in enumerate(lines):
                if line == 'Program:' and i + 1 < len(lines):
                    program_info['Program'] = lines[i + 1]
                elif line == 'Year Level:' and i + 1 < len(lines):
                    program_info['Year Level'] = lines[i + 1]
                elif line == 'Section:' and i + 1 < len(lines):
                    program_info['Section'] = lines[i + 1]
                elif line == 'Adviser:' and i + 1 < len(lines):
                    program_info['Adviser'] = lines[i + 1]
            
            # Extract schedule data
            schedule_data = []
            total_units = None
            
            # Find where the data starts after the header
            data_start = -1
            for i, line in enumerate(lines):
                if line == 'Room':  # This is the last header field
                    data_start = i + 1
                    break
            
            if data_start > 0:
                i = data_start
                while i + 7 < len(lines):
                    if lines[i] == 'Total Units':
                        total_units = lines[i + 1] if i + 1 < len(lines) else None
                        break
                    
                    if 'Generated on:' in lines[i]:
                        break
                    
                    try:
                        subject_entry = {
                            'Subject Code': lines[i],
                            'Description': lines[i + 1],
                            'Type': lines[i + 2],
                            'Units': lines[i + 3],
                            'Day': lines[i + 4],
                            'Time Start': lines[i + 5],
                            'Time End': lines[i + 6],
                            'Room': lines[i + 7]
                        }
                        
                        schedule_data.append(subject_entry)
                        i += 8
                        
                    except IndexError:
                        break
            
            return {
                'program_info': program_info,
                'schedule': schedule_data,
                'total_units': total_units
            }
            
        except Exception as e:
            print(f"❌ Error extracting COR PDF: {e}")
            return None

    def format_cor_info(self, cor_info):
        """Format COR information as text"""
        text = f"""
    Program: {cor_info['program_info']['Program']}
    Year Level: {cor_info['program_info']['Year Level']}
    Section: {cor_info['program_info']['Section']}
    Adviser: {cor_info['program_info']['Adviser']}
    Total Units: {cor_info['total_units']}

    Schedule:
    """
        for course in cor_info['schedule']:
            if course.get('Subject Code') and str(course['Subject Code']).lower() != 'nan':
                text += f"""
    - {course['Subject Code']} ({course.get('Type', 'N/A')}): {course.get('Description', 'N/A')}
    Day: {course.get('Day', 'N/A')}, Time: {course.get('Time Start', 'N/A')} to {course.get('Time End', 'N/A')}
    Room: {course.get('Room', 'N/A')}, Units: {course.get('Units', 'N/A')}
    """
        return text.strip()

    def extract_from_structured_text(self, lines):
        """Extract data from structured column:value format"""
        student_data = {
            'student_id': None, 'surname': None, 'first_name': None, 'full_name': None,
            'year': None, 'course': None, 'section': None, 'contact_number': None,
            'guardian_name': None, 'guardian_contact': None
        }
        
        # Map common column names to our fields
        field_mapping = {
            'STUDENT ID': 'student_id', 'STUDENT_ID': 'student_id', 'ID': 'student_id',
            'STUDENT NUMBER': 'student_id', 'STUDENT NO': 'student_id',
            'NAME': 'full_name', 'STUDENT NAME': 'full_name', 'FULL NAME': 'full_name',
            'SURNAME': 'surname', 'LAST NAME': 'surname',
            'FIRST NAME': 'first_name', 'FIRSTNAME': 'first_name', 'GIVEN NAME': 'first_name',
            'YEAR': 'year', 'YEAR LEVEL': 'year',
            'COURSE': 'course', 'PROGRAM': 'course', 'DEGREE': 'course',
            'SECTION': 'section', 'SEC': 'section', 'CLASS': 'section',
            'CONTACT NUMBER': 'contact_number', 'PHONE': 'contact_number', 'MOBILE': 'contact_number', 'TEL NO': 'contact_number',
            'GUARDIAN NAME': 'guardian_name', 'GUARDIAN': 'guardian_name', 'PARENT NAME': 'guardian_name',
            'EMERGENCY CONTACT NAME': 'guardian_name',
            'GUARDIAN CONTACT': 'guardian_contact', "GUARDIAN'S CONTACT NUMBER": 'guardian_contact',
            'GUARDIAN CONTACT NUMBER': 'guardian_contact', 'PARENT CONTACT': 'guardian_contact',
            'EMERGENCY CONTACT NO': 'guardian_contact'
        }
        
        # Extract values from lines
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    column_name = parts[0].strip().upper()
                    value = parts[1].strip()
                    
                    # Skip header values and empty values
                    if (value and value != 'nan' and 
                        value.upper() not in field_mapping.keys() and # Avoid picking up column names as values
                        column_name in field_mapping):
                        
                        field = field_mapping[column_name]
                        student_data[field] = self.clean_extracted_value(value, field)
        
        # Post-process: if we have separate surname/first name, combine for full name
        if student_data['surname'] and student_data['first_name'] and not student_data['full_name']:
            student_data['full_name'] = f"{student_data['surname']}, {student_data['first_name']}"
        elif student_data['full_name'] and not (student_data['surname'] and student_data['first_name']):
            # Split full name if we don't have separate fields
            student_data['surname'], student_data['first_name'] = self.split_full_name(student_data['full_name'])
        
        return student_data
    
    def extract_faculty_pdf_data(self, filename):
        """Extract faculty resume data from PDF"""
        try:
            doc = fitz.open(filename)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            # Clean and format the resume text
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            # Format as a structured resume
            formatted_text = "FACULTY RESUME\n\n"
            current_section = ""
            
            for line in lines:
                # Detect section headers
                if any(keyword in line.upper() for keyword in ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'CERTIFICATIONS', 'PROFILE', 'CONTACT INFORMATION', 'PERSONAL INFORMATION']):
                    current_section = line
                    formatted_text += f"\n{line}\n" + "-" * len(line) + "\n"
                else:
                    formatted_text += f"{line}\n"
            
            return formatted_text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting faculty PDF: {e}")
            return None

    def format_faculty_excel_data(self, faculty_data):
        """Format faculty Excel data as text"""
        text = ""
        
        for section, data in faculty_data.items():
            if isinstance(data, dict):
                text += f"\n{section}:\n"
                for field, value in data.items():
                    if value:  # Only include non-empty values
                        text += f"  {field}: {value}\n"
            else:
                if data:  # Only include non-empty values
                    text += f"{section}: {data}\n"
        
        return text.strip()

    def format_faculty_schedule_data(self, adviser_name, schedule_data):
        """Format faculty schedule data as text"""
        text = f"FACULTY CLASS SCHEDULE\nName of Adviser: {adviser_name}\n\n"
        
        if schedule_data:
            text += "WEEKLY SCHEDULE:\n"
            for entry in schedule_data:
                text += f"Day: {entry.get('Day', 'N/A')}\n"
                text += f"Time: {entry.get('Time', 'N/A')}\n"
                text += f"Subject: {entry.get('Subject', 'N/A')}\n"
                text += f"Class: {entry.get('Class', 'N/A')}\n"
                text += "-" * 30 + "\n"
        
        return text.strip()
    
    
    def is_teaching_faculty_excel(self, df):
        """Enhanced check for Teaching Faculty Excel files"""
        try:
            # Convert first 20 rows to text
            first_rows_text = ""
            for i in range(min(20, df.shape[0])):
                for j in range(df.shape[1]):  # Fix: Remove min() wrapper
                    if pd.notna(df.iloc[i, j]):
                        first_rows_text += str(df.iloc[i, j]).upper() + " "
            
            print(f"🔍 Checking if file is teaching faculty: {first_rows_text[:200]}...")
            
            # Teaching faculty indicators - more specific
            teaching_faculty_indicators = [
                "SURNAME", "FIRST NAME", "DATE OF BIRTH", "PLACE OF BIRTH", 
                "CITIZENSHIP", "BLOOD TYPE", "GSIS", "PHILHEALTH", "FATHER", "MOTHER"
            ]
            
            # Count indicators
            indicator_count = sum(1 for indicator in teaching_faculty_indicators if indicator in first_rows_text)
            print(f"🔍 Found {indicator_count} teaching faculty indicators")
            
            # Student data exclusions - make this stronger
            student_indicators = ["STUDENT ID", "GUARDIAN", "YEAR LEVEL", "COURSE SECTION", "PDM-"]
            has_student_indicator = any(indicator in first_rows_text for indicator in student_indicators)
            print(f"🔍 Has student indicators: {has_student_indicator}")
            
            # Teaching faculty should have personal info that students don't have
            is_faculty = indicator_count >= 4 and not has_student_indicator
            print(f"🔍 Is teaching faculty: {is_faculty}")
            
            return is_faculty
            
        except Exception as e:
            print(f"🔍 Error in teaching faculty detection: {e}")
            return False

    # ======================== FILE PROCESSING CONTROLLER ========================
    
    
    def process_teaching_faculty_schedule_excel(self, filename):
        """Process Teaching Faculty Schedule Excel with universal extraction"""
        try:
            faculty_schedule_info = self.extract_teaching_faculty_schedule_info_smart(filename)
            
            if not faculty_schedule_info:
                print("❌ Could not extract teaching faculty schedule data from Excel")
                return False
                
            formatted_text = self.format_teaching_faculty_schedule_enhanced(faculty_schedule_info)
            
            # Create smart metadata
            adviser_name = faculty_schedule_info.get('adviser_name', 'Unknown Faculty')
            department = faculty_schedule_info.get('department', 'UNKNOWN')
            
            metadata = {
                'adviser_name': adviser_name,
                'full_name': adviser_name,
                'department': self.standardize_department_name(department),
                'data_type': 'teaching_faculty_schedule_excel',
                'faculty_type': 'schedule',
                'total_subjects': len(faculty_schedule_info.get('schedule', [])),
                'days_teaching': len(set(item.get('day', '') for item in faculty_schedule_info.get('schedule', []) if item.get('day')))
            }
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            hierarchy_path = f"{self.get_department_display_name(metadata['department'])} > Faculty Schedules"
            print(f"✅ Loaded faculty schedule into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   👨‍🏫 Faculty: {adviser_name}")
            print(f"   📚 Subjects: {metadata['total_subjects']}, Days: {metadata['days_teaching']}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing teaching faculty schedule Excel: {e}")
            import traceback
            traceback.print_exc()
            return False

    def extract_teaching_faculty_schedule_info_smart(self, filename):
        """Universal teaching faculty schedule extraction that works with ANY Excel format"""
        try:
            df_full = pd.read_excel(filename, header=None)
            print(f"📋 Faculty Schedule Excel dimensions: {df_full.shape}")
            
            # DEBUG: Show the actual Excel content
            print(f"📋 Raw Excel content (first 15 rows):")
            for i in range(min(15, df_full.shape[0])):
                row_data = []
                for j in range(min(df_full.shape[1], 6)):  # Show first 6 columns
                    if pd.notna(df_full.iloc[i, j]):
                        row_data.append(f"'{str(df_full.iloc[i, j])}'")
                    else:
                        row_data.append("'N/A'")
                print(f"   Row {i}: {row_data}")
            
            # STEP 1: Extract adviser name and department
            adviser_info = self.extract_adviser_info_from_schedule(df_full)
            print(f"📋 Extracted Adviser Info: {adviser_info}")
            
            # STEP 2: Extract schedule data
            schedule_data = self.extract_schedule_data_from_faculty_excel(df_full)
            print(f"📋 Found {len(schedule_data)} scheduled classes")
            
            return {
                'adviser_name': adviser_info.get('name', 'Unknown Faculty'),
                'department': adviser_info.get('department', 'UNKNOWN'),
                'schedule': schedule_data
            }
            
        except Exception as e:
            print(f"❌ Error in teaching faculty schedule extraction: {e}")
            return None

    def extract_adviser_info_from_schedule(self, df):
        """Extract adviser name and department from faculty schedule"""
        adviser_info = {'name': 'Unknown Faculty', 'department': 'UNKNOWN'}
        
        # Search for adviser information in first 20 rows
        for i in range(min(20, df.shape[0])):
            for j in range(min(df.shape[1], 10)):
                if pd.notna(df.iloc[i, j]):
                    cell_value = str(df.iloc[i, j]).strip().upper()
                    
                    # Look for adviser name patterns
                    if any(keyword in cell_value for keyword in ['NAME OF ADVISER', 'ADVISER', 'FACULTY NAME', 'INSTRUCTOR']):
                        # Check adjacent cells for the actual name
                        name_value = None
                        
                        # Check right cell
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            potential_name = str(df.iloc[i, j + 1]).strip()
                            if len(potential_name) > 3 and not any(keyword in potential_name.upper() for keyword in ['ADVISER', 'NAME', 'FACULTY']):
                                name_value = potential_name
                        
                        # Check below cell
                        if not name_value and i + 1 < df.shape[0] and pd.notna(df.iloc[i + 1, j]):
                            potential_name = str(df.iloc[i + 1, j]).strip()
                            if len(potential_name) > 3:
                                name_value = potential_name
                        
                        # Check if current cell contains the name after colon
                        if not name_value and ':' in cell_value:
                            parts = str(df.iloc[i, j]).split(':', 1)
                            if len(parts) > 1:
                                potential_name = parts[1].strip()
                                if len(potential_name) > 3:
                                    name_value = potential_name
                        
                        if name_value:
                            adviser_info['name'] = name_value.title()
                            print(f"🎯 Found adviser name: {name_value}")
                    
                    # Look for department information
                    if any(keyword in cell_value for keyword in ['DEPARTMENT', 'COLLEGE', 'DEPT']):
                        dept_value = None
                        
                        # Check adjacent cells for department
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            potential_dept = str(df.iloc[i, j + 1]).strip()
                            if len(potential_dept) > 1:
                                dept_value = potential_dept
                        
                        if dept_value:
                            adviser_info['department'] = dept_value.upper()
                            print(f"🎯 Found department: {dept_value}")
        
        return adviser_info

    def extract_schedule_data_from_faculty_excel(self, df):
        """Enhanced extraction that captures ALL schedule data including time-only rows"""
        schedule_data = []
        
        # Find the schedule table headers
        schedule_start_row = -1
        day_columns = {}
        time_column = -1
        subject_column = -1
        
        # Search for the header row with SUBJECT/S, TIME, and days
        days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
        
        for i in range(min(10, df.shape[0])):  # Check first 10 rows for headers
            row_cells = []
            for j in range(min(df.shape[1], 15)):
                if pd.notna(df.iloc[i, j]):
                    row_cells.append(str(df.iloc[i, j]).upper().strip())
                else:
                    row_cells.append('')
            
            # Check for column headers
            for j, cell in enumerate(row_cells):
                # Find TIME column
                if 'TIME' in cell and time_column == -1:
                    time_column = j
                    print(f"🕐 Found TIME column at position {j}")
                
                # Find SUBJECT column
                if any(keyword in cell for keyword in ['SUBJECT', 'COURSE']) and subject_column == -1:
                    subject_column = j
                    print(f"📚 Found SUBJECT column at position {j}")
                
                # Find day columns
                for day in days:
                    if day in cell and j not in day_columns:
                        day_columns[j] = self.standardize_day_name(day)
                        print(f"📅 Found {day} column at position {j}")
            
            # If we found headers, next row is data start
            if len(day_columns) >= 3 and time_column >= 0:  # Need at least 3 days and time column
                schedule_start_row = i + 1
                print(f"🎯 Found schedule header at row {i}, data starts at row {schedule_start_row}")
                break
        
        if schedule_start_row == -1:
            print("⚠️ Could not find proper schedule table structure")
            return []
        
        print(f"📋 Day columns mapping: {day_columns}")
        print(f"📋 Time column: {time_column}, Subject column: {subject_column}")
        
        # Build subject lookup from rows that have both subject and time
        subject_lookup = {}  # time -> subject mapping
        
        # First pass: collect all subjects and their time slots
        for i in range(schedule_start_row, df.shape[0]):
            current_time = None
            current_subject = None
            
            # Get time
            if time_column >= 0 and time_column < df.shape[1] and pd.notna(df.iloc[i, time_column]):
                time_cell = str(df.iloc[i, time_column]).strip()
                if time_cell and time_cell.upper() not in ['N/A', 'NONE', '']:
                    current_time = time_cell
            
            # Get subject (if available)
            if subject_column >= 0 and subject_column < df.shape[1] and pd.notna(df.iloc[i, subject_column]):
                subject_cell = str(df.iloc[i, subject_column]).strip()
                if subject_cell and subject_cell.upper() not in ['N/A', 'NONE', '']:
                    current_subject = subject_cell
            
            # Map time to subject for lookup
            if current_time and current_subject:
                subject_lookup[current_time] = current_subject
                print(f"🔗 Mapped time '{current_time}' to subject '{current_subject}'")
        
        print(f"📋 Built subject lookup with {len(subject_lookup)} time slots")
        
        # Second pass: extract all schedule entries
        consecutive_empty_rows = 0
        
        for i in range(schedule_start_row, df.shape[0]):
            # Get the time slot for this row
            current_time = None
            if time_column >= 0 and time_column < df.shape[1] and pd.notna(df.iloc[i, time_column]):
                time_cell = str(df.iloc[i, time_column]).strip()
                if time_cell and time_cell.upper() not in ['N/A', 'NONE', '']:
                    current_time = time_cell
            
            # Skip rows without valid time
            if not current_time:
                continue
            
            # Get subject from lookup or current row
            current_subject = subject_lookup.get(current_time)
            if not current_subject:
                # Try to get from current row
                if subject_column >= 0 and subject_column < df.shape[1] and pd.notna(df.iloc[i, subject_column]):
                    subject_cell = str(df.iloc[i, subject_column]).strip()
                    if subject_cell and subject_cell.upper() not in ['N/A', 'NONE', '']:
                        current_subject = subject_cell
            
            # If we still don't have a subject, use a generic one
            if not current_subject:
                current_subject = f"Class at {current_time}"
            
            # Check each day column for classes
            found_classes_this_row = False
            for col_idx, day in day_columns.items():
                if col_idx < df.shape[1] and pd.notna(df.iloc[i, col_idx]):
                    class_section = str(df.iloc[i, col_idx]).strip()
                    
                    # If there's a meaningful entry in this day column
                    if class_section and class_section.upper() not in ['N/A', 'NONE', '']:
                        # Determine the actual subject
                        if current_subject.startswith("Class at") and class_section in subject_lookup.values():
                            # The class section name matches a known subject
                            actual_subject = class_section
                        elif current_subject.startswith("Class at"):
                            # Use the class section as the subject
                            actual_subject = f"Course: {class_section}"
                        else:
                            # Use the mapped subject
                            actual_subject = current_subject
                        
                        schedule_entry = {
                            'day': day,
                            'time': current_time,
                            'subject': actual_subject,
                            'section_class': class_section,
                            'full_description': f"{actual_subject} - {class_section}" if actual_subject != class_section else actual_subject
                        }
                        
                        schedule_data.append(schedule_entry)
                        found_classes_this_row = True
                        print(f"📚 Added: {day} {current_time} - {actual_subject} (Section: {class_section})")
            
            # Track consecutive empty rows to know when to stop
            if found_classes_this_row:
                consecutive_empty_rows = 0
            else:
                consecutive_empty_rows += 1
                
            # Stop if we hit too many consecutive empty rows (but be more lenient)
            if consecutive_empty_rows >= 8:  # Increased from 5 to 8
                print(f"🛑 Stopping after {consecutive_empty_rows} consecutive empty rows")
                break
        
        print(f"📋 Total extracted classes: {len(schedule_data)}")
        return schedule_data

    def standardize_day_name(self, day):
        """Standardize day names to full format"""
        day_mapping = {
            'MON': 'Monday', 'MONDAY': 'Monday',
            'TUE': 'Tuesday', 'TUESDAY': 'Tuesday', 'TUES': 'Tuesday',
            'WED': 'Wednesday', 'WEDNESDAY': 'Wednesday',
            'THU': 'Thursday', 'THURSDAY': 'Thursday', 'THURS': 'Thursday',
            'FRI': 'Friday', 'FRIDAY': 'Friday',
            'SAT': 'Saturday', 'SATURDAY': 'Saturday'
        }
        return day_mapping.get(day.upper(), day.title())

    def format_teaching_faculty_schedule_enhanced(self, schedule_info):
        """Clean, structured formatting for teaching faculty schedule"""
        text = f"""TEACHING FACULTY CLASS SCHEDULE

    FACULTY INFORMATION:
    Name of Adviser: {schedule_info.get('adviser_name', 'Unknown Faculty')}
    Department: {schedule_info.get('department', 'Unknown Department')}

    WEEKLY TEACHING SCHEDULE ({len(schedule_info.get('schedule', []))} scheduled classes):
    """
        
        if schedule_info.get('schedule'):
            # Group by day for better display
            by_day = {}
            for item in schedule_info['schedule']:
                day = item.get('day', 'Unknown Day')
                if day not in by_day:
                    by_day[day] = []
                by_day[day].append(item)
            
            # Display schedule day by day in proper order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            
            for day in day_order:
                if day in by_day:
                    text += f"\n{day.upper()}:\n"
                    # Sort by time
                    by_day[day].sort(key=lambda x: self.parse_time_for_sorting(x.get('time', '')))
                    
                    for item in by_day[day]:
                        time_display = item.get('time', 'No Time')
                        subject_display = item.get('subject', 'No Subject')
                        section_display = item.get('section_class', '')
                        
                        if section_display:
                            text += f"  • {time_display} - {subject_display} (Section: {section_display})\n"
                        else:
                            text += f"  • {time_display} - {subject_display}\n"
            
            # Show unique subjects summary
            unique_subjects = {}
            for item in schedule_info['schedule']:
                subject = item.get('subject', '')
                if subject and subject not in unique_subjects:
                    sections = set()
                    for s in schedule_info['schedule']:
                        if s.get('subject') == subject and s.get('section_class'):
                            sections.add(s.get('section_class'))
                    unique_subjects[subject] = list(sections)
            
            if unique_subjects:
                text += f"\nSUBJECTS TAUGHT ({len(unique_subjects)} unique subjects):\n"
                for subject, sections in sorted(unique_subjects.items()):
                    if sections:
                        text += f"• {subject} (Sections: {', '.join(sorted(sections))})\n"
                    else:
                        text += f"• {subject}\n"
        else:
            text += "\nNo schedule data found."
        
        return text.strip()
    
    
    def parse_time_for_sorting(self, time_str):
        """Parse time string for sorting with proper 12/24 hour handling"""
        try:
            # Extract first time from range like "08:00 - 08:30"
            if ' - ' in time_str:
                time_part = time_str.split(' - ')[0].strip()
            else:
                time_part = time_str.strip()
            
            # Handle different time formats
            if ':' in time_part:
                hour_min = time_part.split(':')
                hour = int(hour_min[0])
                minute = int(hour_min[1][:2])  # Handle cases like "09:00 AM"
                
                # Handle 12-hour vs 24-hour format
                # If hour is 1-6, it's likely PM (afternoon) in a work schedule
                if 1 <= hour <= 6:
                    hour += 12  # Convert to PM
                
                return hour * 60 + minute
            else:
                return 9999  # Put unparseable times at end
        except Exception as e:
            print(f"⚠️ Error parsing time {time_str}: {e}")
            return 9999
    
    
    def process_file(self, filename):
        """Enhanced file processing controller with curriculum, duplicate detection and admin support"""
        ext = os.path.splitext(filename)[1].lower()
        
        try:
            if ext == ".xlsx":
                df_check = pd.read_excel(filename, header=None)
                
                # Check curriculum FIRST (most specific academic content)
                if self.is_curriculum_excel(df_check):
                    print("🔍 Detected as Curriculum Excel")
                    return self.process_with_duplicate_check(filename, 'curriculum')
                # Check COR (most specific schedule)
                elif self.is_cor_file(df_check):
                    print("🔍 Detected as COR Excel")
                    return self.process_with_duplicate_check(filename, 'cor_schedule')
                # Check non-teaching faculty schedule FIRST
                elif self.is_non_teaching_faculty_schedule_excel(df_check):
                    print("🔍 Detected as Non-Teaching Faculty Schedule Excel")
                    return self.process_with_duplicate_check(filename, 'non_teaching_faculty_schedule')
                # Check teaching faculty schedule
                elif self.is_faculty_schedule_excel(df_check):
                    print("🔍 Detected as Teaching Faculty Schedule Excel")
                    return self.process_with_duplicate_check(filename, 'teaching_faculty_schedule')
                # Check admin BEFORE other faculty types
                elif self.is_admin_excel(df_check):
                    print("🔍 Detected as Admin Excel")
                    return self.process_with_duplicate_check(filename, 'admin')
                # Check non-teaching faculty BEFORE teaching faculty
                elif self.is_non_teaching_faculty_excel(df_check):
                    print("🔍 Detected as Non-Teaching Faculty Excel")
                    return self.process_with_duplicate_check(filename, 'non_teaching_faculty')
                # Then check teaching faculty
                elif self.is_teaching_faculty_excel(df_check):
                    print("🔍 Detected as Teaching Faculty Excel")
                    return self.process_with_duplicate_check(filename, 'teaching_faculty')
                elif self.is_faculty_excel(df_check):
                    print("🔍 Detected as Faculty Excel")
                    return self.process_with_duplicate_check(filename, 'teaching_faculty')  # Treat as teaching faculty
                else:
                    print("🔍 Detected as Student Excel")
                    return self.process_with_duplicate_check(filename, 'student')
                    
            elif ext == ".pdf":
                # NOTE: Curriculum PDFs could be added here later if needed
                if self.is_cor_pdf(filename):
                    print("🔍 Detected as COR PDF")
                    return self.process_with_duplicate_check(filename, 'cor_schedule')
                elif self.is_faculty_schedule_pdf(filename):
                    print("🔍 Detected as Faculty Schedule PDF")
                    return self.process_with_duplicate_check(filename, 'teaching_faculty_schedule')
                elif self.is_faculty_pdf(filename):
                    print("🔍 Detected as Faculty PDF")
                    return self.process_with_duplicate_check(filename, 'teaching_faculty')
                else:
                    print("🔍 Detected as Student PDF")
                    return self.process_with_duplicate_check(filename, 'student')
                    
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
        return False
    
    def scan_for_duplicates_in_existing_data(self):
        """Scan all existing collections for potential duplicates"""
        print("🔍 Scanning for duplicates in existing data...")
        
        all_records = []
        duplicates_found = []
        
        # Collect all records with their metadata
        for collection_name, collection in self.collections.items():
            try:
                all_docs = collection.get()
                collection_type = self.get_collection_type(collection_name)
                
                for i, metadata in enumerate(all_docs["metadatas"]):
                    record = {
                        'collection': collection_name,
                        'collection_type': collection_type,
                        'metadata': metadata,
                        'content': all_docs["documents"][i],
                        'index': i
                    }
                    all_records.append(record)
                    
            except Exception as e:
                print(f"⚠️ Error reading {collection_name}: {e}")
        
        # Compare all records
        for i, record1 in enumerate(all_records):
            for j, record2 in enumerate(all_records[i+1:], i+1):
                if self.are_records_duplicates(record1, record2):
                    duplicates_found.append((record1, record2))
        
        if duplicates_found:
            print(f"\n⚠️ Found {len(duplicates_found)} potential duplicate pairs:")
            for i, (rec1, rec2) in enumerate(duplicates_found, 1):
                print(f"\n{i}. Potential duplicate:")
                print(f"   Record 1: {rec1['collection_type']} - {rec1['metadata'].get('full_name', 'Unknown')}")
                print(f"   Record 2: {rec2['collection_type']} - {rec2['metadata'].get('full_name', 'Unknown')}")
            
            self.handle_existing_duplicates(duplicates_found)
        else:
            print("✅ No duplicates found in existing data!")
    
    def scan_and_clean_existing_duplicates(self):
        """Enhanced scanner that finds duplicates within same collections"""
        print("🔍 Scanning for duplicates in existing data...")
        
        duplicates_found = []
        total_records = 0
        
        # Check each collection individually for internal duplicates
        for collection_name, collection in self.collections.items():
            try:
                all_docs = collection.get()
                collection_type = self.get_collection_type(collection_name)
                records_in_collection = len(all_docs["documents"])
                total_records += records_in_collection
                
                print(f"   📊 Scanning {collection_type}: {records_in_collection} records")
                
                if records_in_collection <= 1:
                    continue  # Skip collections with 1 or no records
                
                # Compare each record with every other record in the same collection
                for i in range(records_in_collection):
                    for j in range(i + 1, records_in_collection):
                        doc1 = all_docs["documents"][i]
                        doc2 = all_docs["documents"][j]
                        meta1 = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                        meta2 = all_docs["metadatas"][j] if j < len(all_docs["metadatas"]) else {}
                        
                        # Check if these two records are duplicates
                        if self.are_records_identical(doc1, doc2, meta1, meta2):
                            print(f"   🚨 DUPLICATE FOUND: Records {i} and {j} in {collection_type}")
                            
                            record1 = {
                                'collection_name': collection_name,
                                'collection_type': collection_type,
                                'document': doc1,
                                'metadata': meta1,
                                'doc_index': i,
                                'global_id': f"{collection_name}_{i}"
                            }
                            
                            record2 = {
                                'collection_name': collection_name,
                                'collection_type': collection_type,
                                'document': doc2,
                                'metadata': meta2,
                                'doc_index': j,
                                'global_id': f"{collection_name}_{j}"
                            }
                            
                            duplicates_found.append((record1, record2))
                            
            except Exception as e:
                print(f"⚠️ Error scanning {collection_name}: {e}")
        
        print(f"📊 Scanned {total_records} total records across {len(self.collections)} collections")
        
        if duplicates_found:
            print(f"\n⚠️ Found {len(duplicates_found)} duplicate pairs!")
            
            # Show details about each duplicate pair
            for i, (rec1, rec2) in enumerate(duplicates_found, 1):
                print(f"\n🔍 Duplicate Pair {i}:")
                print(f"   Collection: {rec1['collection_type']}")
                print(f"   Records: #{rec1['doc_index']} and #{rec2['doc_index']}")
                
                # Extract staff/student name for display
                staff_name_1 = self.extract_name_from_metadata(rec1['metadata'])
                staff_name_2 = self.extract_name_from_metadata(rec2['metadata'])
                
                print(f"   Name: {staff_name_1}")
                print(f"   Department: {rec1['metadata'].get('department', 'Unknown')}")
                
                # Show preview of content difference
                if rec1['document'] == rec2['document']:
                    print(f"   Status: IDENTICAL CONTENT")
                else:
                    similarity = self.calculate_text_similarity(rec1['document'], rec2['document'])
                    print(f"   Similarity: {similarity:.1%}")
            
            self.handle_existing_duplicates(duplicates_found)
        else:
            print("✅ No duplicates found in existing data!")
            
    
    def are_records_identical(self, doc1, doc2, meta1, meta2):
        """Check if two records are identical or near-identical"""
        
        # Method 1: Exact content match
        if doc1 == doc2:
            print(f"      🎯 IDENTICAL CONTENT")
            return True
        
        # Method 2: High similarity content (99%+ match)
        if len(doc1) > 100 and len(doc2) > 100:
            similarity = self.calculate_text_similarity(doc1, doc2)
            if similarity > 0.99:
                print(f"      🎯 HIGH SIMILARITY: {similarity:.1%}")
                return True
        
        # Method 3: Same metadata key fields
        if self.have_identical_metadata(meta1, meta2):
            print(f"      🎯 IDENTICAL METADATA")
            return True
        
        return False
    
    def have_identical_metadata(self, meta1, meta2):
        """Check if two metadata records represent the same entity"""
        
        # For any type of record, check name fields
        name1 = self.extract_name_from_metadata(meta1)
        name2 = self.extract_name_from_metadata(meta2)
        
        if name1 and name2 and name1.upper() == name2.upper():
            # Same name - check department/course context
            dept1 = str(meta1.get('department', '')).strip().upper()
            dept2 = str(meta2.get('department', '')).strip().upper()
            
            if dept1 and dept2 and dept1 == dept2:
                return True
            
            # For students, also check course and year
            course1 = str(meta1.get('course', '')).strip().upper()
            course2 = str(meta2.get('course', '')).strip().upper()
            year1 = str(meta1.get('year_level', '')).strip()
            year2 = str(meta2.get('year_level', '')).strip()
            
            if course1 and course2 and course1 == course2 and year1 == year2:
                return True
        
        return False
    
    def extract_name_from_metadata(self, metadata):
        """Extract the primary name from metadata regardless of record type"""
        
        # Try different name fields in order of preference
        name_fields = [
            'full_name', 'adviser_name', 'staff_name', 'student_name',
            'surname', 'first_name'
        ]
        
        for field in name_fields:
            name = metadata.get(field, '')
            if name and str(name).strip():
                return str(name).strip()
        
        # If surname and first_name exist separately, combine them
        surname = metadata.get('surname', '')
        first_name = metadata.get('first_name', '')
        if surname and first_name:
            return f"{surname}, {first_name}"
        
        return 'Unknown'
            
    def are_duplicate_records(self, record1, record2):
        """Check if two records are duplicates"""
        # Skip if same collection (shouldn't happen with current logic, but safety check)
        if record1['collection_name'] == record2['collection_name']:
            return False
        
        meta1 = record1['metadata']
        meta2 = record2['metadata']
        
        # Check by data type and key fields
        data_type1 = meta1.get('data_type', '')
        data_type2 = meta2.get('data_type', '')
        
        # Only compare same data types
        if data_type1 != data_type2:
            return False
        
        # Student duplicates
        if data_type1 == 'student_universal':
            student_id1 = str(meta1.get('student_id', '')).strip().upper()
            student_id2 = str(meta2.get('student_id', '')).strip().upper()
            name1 = str(meta1.get('full_name', '')).strip().upper()
            name2 = str(meta2.get('full_name', '')).strip().upper()
            
            # Same student ID or same name + course + year
            if student_id1 and student_id2 and student_id1 == student_id2:
                return True
            if name1 and name2 and name1 == name2:
                course1 = str(meta1.get('course', '')).strip().upper()
                course2 = str(meta2.get('course', '')).strip().upper()
                year1 = str(meta1.get('year_level', '')).strip()
                year2 = str(meta2.get('year_level', '')).strip()
                return course1 == course2 and year1 == year2
        
        # Faculty duplicates (teaching, non-teaching, admin)
        elif data_type1 in ['teaching_faculty_excel', 'non_teaching_faculty_excel', 'admin_excel']:
            name1 = str(meta1.get('full_name', '')).strip().upper()
            name2 = str(meta2.get('full_name', '')).strip().upper()
            dept1 = str(meta1.get('department', '')).strip().upper()
            dept2 = str(meta2.get('department', '')).strip().upper()
            
            # Same name + department
            if name1 and name2 and name1 == name2 and dept1 == dept2:
                return True
        
        # Schedule duplicates
        elif data_type1 in ['teaching_faculty_schedule_excel', 'non_teaching_faculty_schedule_excel']:
            staff1 = str(meta1.get('staff_name', meta1.get('adviser_name', ''))).strip().upper()
            staff2 = str(meta2.get('staff_name', meta2.get('adviser_name', ''))).strip().upper()
            dept1 = str(meta1.get('department', '')).strip().upper()
            dept2 = str(meta2.get('department', '')).strip().upper()
            
            # Same staff name + department
            if staff1 and staff2 and staff1 == staff2 and dept1 == dept2:
                return True
        
        # COR schedule duplicates
        elif data_type1 == 'cor_excel':
            course1 = str(meta1.get('course', '')).strip().upper()
            course2 = str(meta2.get('course', '')).strip().upper()
            year1 = str(meta1.get('year_level', '')).strip()
            year2 = str(meta2.get('year_level', '')).strip()
            section1 = str(meta1.get('section', '')).strip().upper()
            section2 = str(meta2.get('section', '')).strip().upper()
            
            # Same course + year + section
            if course1 == course2 and year1 == year2 and section1 == section2:
                return True
        
        # Content-based duplicate check as fallback
        doc1 = record1['document']
        doc2 = record2['document']
        
        # If documents are exactly the same, they're duplicates
        if doc1 == doc2:
            return True
        
        # Check for very high similarity (> 95% same content)
        if len(doc1) > 100 and len(doc2) > 100:
            similarity = self.calculate_text_similarity(doc1, doc2)
            return similarity > 0.95
        
        return False
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate text similarity percentage"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Simple character-by-character comparison
            len1, len2 = len(text1), len(text2)
            if len1 == 0 and len2 == 0:
                return 1.0
            if len1 == 0 or len2 == 0:
                return 0.0
            
            # Count matching characters
            min_len = min(len1, len2)
            max_len = max(len1, len2)
            
            matches = 0
            for i in range(min_len):
                if text1[i] == text2[i]:
                    matches += 1
            
            # Calculate similarity as ratio of matches to longer text
            similarity = matches / max_len
            return similarity
            
        except Exception:
            return 0.0

    def handle_existing_duplicates(self, duplicates_found):
        """Handle existing duplicates found in the database"""
        print(f"\n📋 Found {len(duplicates_found)} duplicate pairs:")
        
        for i, (record1, record2) in enumerate(duplicates_found, 1):
            print(f"\n🔍 Duplicate Pair {i}:")
            print(f"   Record A: {record1['collection_type']}")
            print(f"     Name: {record1['metadata'].get('full_name', record1['metadata'].get('staff_name', record1['metadata'].get('adviser_name', 'Unknown')))}")
            print(f"     Department: {record1['metadata'].get('department', 'Unknown')}")
            
            print(f"   Record B: {record2['collection_type']}")  
            print(f"     Name: {record2['metadata'].get('full_name', record2['metadata'].get('staff_name', record2['metadata'].get('adviser_name', 'Unknown')))}")
            print(f"     Department: {record2['metadata'].get('department', 'Unknown')}")
        
        print(f"\n💡 How do you want to handle these duplicates?")
        print(f"   1. 🗑️ Delete all duplicate collections automatically")
        print(f"   2. 🔍 Review each duplicate individually") 
        print(f"   3. ❌ Skip - keep duplicates as they are")
        
        try:
            choice = input("\n👉 Choose option (1-3): ").strip()
            
            if choice == "1":
                self.auto_delete_duplicates(duplicates_found)
            elif choice == "2":
                self.review_duplicates_individually(duplicates_found)
            elif choice == "3":
                print("✅ Keeping duplicates as they are")
            else:
                print("❌ Invalid choice. Keeping duplicates as they are")
                
        except Exception as e:
            print(f"❌ Error handling duplicates: {e}")
    
    def auto_delete_duplicates(self, duplicates_found):
        """Delete duplicates within same collections by recreating them"""
        print(f"\n🗑️ Auto-deleting duplicates...")
        
        # Group duplicates by collection
        collections_with_duplicates = {}
        
        for record1, record2 in duplicates_found:
            collection_name = record1['collection_name']
            if collection_name not in collections_with_duplicates:
                collections_with_duplicates[collection_name] = set()
            
            # Mark the second record for deletion (keep first)
            collections_with_duplicates[collection_name].add(record2['doc_index'])
        
        # Recreate each collection without the duplicate records
        recreated_count = 0
        for collection_name, indices_to_remove in collections_with_duplicates.items():
            if self.recreate_collection_without_duplicates(collection_name, indices_to_remove):
                collection_type = self.get_collection_type(collection_name)
                print(f"   🔄 Cleaned duplicates from: {collection_type}")
                recreated_count += 1
        
        print(f"✅ Cleaned duplicates from {recreated_count} collections!")
        
    def recreate_collection_without_duplicates(self, collection_name, indices_to_remove):
        """Recreate a collection without specified duplicate records"""
        try:
            # Get all data from the collection
            collection = self.collections[collection_name]
            all_docs = collection.get()
            
            print(f"     📊 Original records: {len(all_docs['documents'])}")
            print(f"     🗑️ Removing records at indices: {sorted(indices_to_remove)}")
            
            # Filter out duplicate records
            filtered_docs = []
            filtered_metadata = []
            filtered_ids = []
            
            for i, (doc, metadata) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
                if i not in indices_to_remove:  # Keep only non-duplicate records
                    filtered_docs.append(doc)
                    filtered_metadata.append(metadata)
                    filtered_ids.append(f"doc_{len(filtered_docs)}_{datetime.now().timestamp()}")
            
            print(f"     📊 Keeping {len(filtered_docs)} unique records")
            
            # Delete the old collection
            self.delete_collection(collection_name)
            
            # Recreate with filtered data
            new_collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            if filtered_docs:
                new_collection.add(
                    documents=filtered_docs,
                    metadatas=filtered_metadata,
                    ids=filtered_ids
                )
            
            # Update our collections reference
            self.collections[collection_name] = new_collection
            
            return True
            
        except Exception as e:
            print(f"     ❌ Error recreating collection {collection_name}: {e}")
            return False
    
        
    def review_duplicates_individually(self, duplicates_found):
        """Review each duplicate pair individually"""
        for i, (record1, record2) in enumerate(duplicates_found, 1):
            print(f"\n🔍 Reviewing Duplicate Pair {i} of {len(duplicates_found)}:")
            print("=" * 60)
            
            print(f"Record A: {record1['collection_type']}")
            print(f"  Collection: {record1['collection_name']}")
            self.display_record_summary(record1)
            
            print(f"\nRecord B: {record2['collection_type']}")
            print(f"  Collection: {record2['collection_name']}")
            self.display_record_summary(record2)
            
            print(f"\n💡 What do you want to do?")
            print(f"   1. 🗑️ Delete Record A")
            print(f"   2. 🗑️ Delete Record B") 
            print(f"   3. ⏭️ Skip (keep both)")
            print(f"   4. 🛑 Stop reviewing")
            
            try:
                choice = input(f"\n👉 Choose option (1-4): ").strip()
                
                if choice == "1":
                    if self.delete_collection(record1['collection_name']):
                        print(f"✅ Deleted Record A")
                elif choice == "2":
                    if self.delete_collection(record2['collection_name']):
                        print(f"✅ Deleted Record B")
                elif choice == "3":
                    print(f"⏭️ Skipped - keeping both records")
                elif choice == "4":
                    print(f"🛑 Stopped reviewing")
                    break
                else:
                    print(f"❌ Invalid choice. Skipping this pair")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_record_summary(self, record):
        """Display a summary of a record"""
        metadata = record['metadata']
        
        # Show key identifying information
        name = metadata.get('full_name', metadata.get('staff_name', metadata.get('adviser_name', 'Unknown')))
        dept = metadata.get('department', 'Unknown')
        data_type = metadata.get('data_type', 'Unknown')
        
        print(f"    Name: {name}")
        print(f"    Department: {dept}")
        print(f"    Data Type: {data_type}")
        
        # Show preview of content
        content_preview = record['document'][:150] + "..." if len(record['document']) > 150 else record['document']
        print(f"    Preview: {content_preview}")            
    

    def are_records_duplicates(self, record1, record2):
        """Check if two records are duplicates"""
        # Don't compare records from the same collection
        if record1['collection'] == record2['collection']:
            return False
        
        # Compare based on record type
        meta1, meta2 = record1['metadata'], record2['metadata']
        
        # Name-based comparison
        name1 = str(meta1.get('full_name', '')).strip().upper()
        name2 = str(meta2.get('full_name', '')).strip().upper()
        
        if name1 and name2 and self.fuzzy_name_match(name1, name2):
            # Additional checks based on context
            dept1 = str(meta1.get('department', '')).strip().upper()
            dept2 = str(meta2.get('department', '')).strip().upper()
            
            # If same name and department, likely duplicate
            if dept1 and dept2 and dept1 == dept2:
                return True
        
        return False
    

    def load_new_data(self):
        """Load new data from files"""
        files = self.list_available_files()
        if not files:
            return False
            
        try:
            choice = int(input("\n🔢 Enter file number to load: ").strip())
            if 1 <= choice <= len(files):
                filename = files[choice - 1]
                success = self.process_file(filename)
                if success:
                    self.data_loaded = True
                    print("✅ Data loaded successfully!")
                return success
        except ValueError:
            print("❌ Invalid input.")
        return False
    
    
    def process_non_teaching_faculty_excel(self, filename):
        """Process Non-Teaching Faculty Excel with smart department inference"""
        try:
            faculty_info = self.extract_teaching_faculty_excel_info_smart(filename)  # Reuse existing extraction
            
            if not faculty_info:
                print("❌ Could not extract non-teaching faculty data from Excel")
                return False
                
            # SMART DEPARTMENT INFERENCE: Try multiple approaches for non-teaching
            department = faculty_info.get('department', '')
            
            # Method 1: Direct department extraction (already done)
            if not department or department in ['N/A', 'NA', '']:
                # Method 2: Infer from position (non-teaching specific)
                if faculty_info.get('position'):
                    inferred_dept = self.infer_non_teaching_department_from_position(faculty_info['position'])
                    if inferred_dept:
                        department = inferred_dept
                        print(f"🔍 Inferred non-teaching department from position: {department}")
            
            # Method 3: Infer from email domain (if applicable)
            if not department or department in ['N/A', 'NA', '']:
                if faculty_info.get('email'):
                    inferred_dept = self.infer_non_teaching_department_from_email(faculty_info['email'])
                    if inferred_dept:
                        department = inferred_dept
                        print(f"🔍 Inferred non-teaching department from email: {department}")
            
            # Method 4: Default based on position type (non-teaching fallback)
            if not department or department in ['N/A', 'NA', '']:
                if faculty_info.get('position'):
                    department = self.default_non_teaching_department_assignment(faculty_info['position'])
                    print(f"🔍 Default non-teaching department assignment: {department}")
            
            # Final fallback
            if not department or department in ['N/A', 'NA', '']:
                department = 'ADMIN_SUPPORT'  # Default non-teaching category
            
            formatted_text = self.format_non_teaching_faculty_info_enhanced(faculty_info)
            
            # Create smart metadata with the determined department
            full_name = ""
            if faculty_info.get('surname') and faculty_info.get('first_name'):
                full_name = f"{faculty_info['surname']}, {faculty_info['first_name']}"
            elif faculty_info.get('surname'):
                full_name = faculty_info['surname']
            elif faculty_info.get('first_name'):
                full_name = faculty_info['first_name']
            else:
                full_name = "Unknown Faculty"
            
            # Update the faculty_info with the determined department
            faculty_info['department'] = department
            
            metadata = {
                'full_name': full_name,
                'surname': faculty_info.get('surname') or '',
                'first_name': faculty_info.get('first_name') or '',
                'department': self.standardize_non_teaching_department_name(department),
                'position': faculty_info.get('position') or '',
                'employment_status': faculty_info.get('employment_status') or '',
                'email': faculty_info.get('email') or '',
                'phone': faculty_info.get('phone') or '',
                'data_type': 'non_teaching_faculty_excel',
                'faculty_type': 'non_teaching',
            }
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            
            # Update the formatted text to show the determined department
            updated_formatted_text = self.format_non_teaching_faculty_info_enhanced(faculty_info)
            
            self.store_with_smart_metadata(collection, [updated_formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            hierarchy_path = f"{self.get_non_teaching_department_display_name(metadata['department'])} > Non-Teaching Faculty"
            print(f"✅ Loaded non-teaching faculty data into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   👨‍💼 Faculty: {metadata['full_name']} ({metadata['position']})")
            return True
            
        except Exception as e:
            print(f"❌ Error processing non-teaching faculty Excel: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def infer_non_teaching_department_from_position(self, position):
        """Infer non-teaching department from faculty position"""
        if not position:
            return None
        
        position_upper = position.upper()
        
        # Non-teaching department mappings based on position
        if any(word in position_upper for word in ['REGISTRAR', 'REGISTRATION', 'RECORDS']):
            return 'REGISTRAR'
        elif any(word in position_upper for word in ['ACCOUNTING', 'ACCOUNTANT', 'FINANCE', 'CASHIER', 'TREASURER']):
            return 'ACCOUNTING'
        elif any(word in position_upper for word in ['GUIDANCE', 'COUNSELOR', 'COUNSELLING']):
            return 'GUIDANCE'
        elif any(word in position_upper for word in ['LIBRARY', 'LIBRARIAN']):  # ADD LIBRARIAN HERE
            return 'LIBRARY'
        elif any(word in position_upper for word in ['HEALTH', 'NURSE', 'MEDICAL', 'CLINIC']):
            return 'HEALTH_SERVICES'
        elif any(word in position_upper for word in ['MAINTENANCE', 'CUSTODIAL', 'JANITOR', 'CLEANER', 'FACILITIES']):
            return 'MAINTENANCE_CUSTODIAL'
        elif any(word in position_upper for word in ['SECURITY', 'GUARD']):
            return 'SECURITY'
        elif any(word in position_upper for word in ['SYSTEM ADMIN', 'IT SUPPORT', 'NETWORK', 'COMPUTER TECHNICIAN', 'IT STAFF']):
            return 'SYSTEM_ADMIN'
        elif any(word in position_upper for word in ['ADMIN', 'ADMINISTRATIVE', 'SECRETARY', 'ASSISTANT']):
            return 'ADMIN_SUPPORT'
        
        return None
    
    def infer_non_teaching_department_from_email(self, email):
        """Infer non-teaching department from email domain or prefix"""
        if not email:
            return None
        
        email_lower = email.lower()
        
        # Check email prefixes that might indicate non-teaching department
        if any(prefix in email_lower for prefix in ['registrar', 'records', 'enrollment']):
            return 'REGISTRAR'
        elif any(prefix in email_lower for prefix in ['accounting', 'finance', 'cashier']):
            return 'ACCOUNTING'
        elif any(prefix in email_lower for prefix in ['guidance', 'counselor']):
            return 'GUIDANCE'
        elif any(prefix in email_lower for prefix in ['library', 'lib']):
            return 'LIBRARY'
        elif any(prefix in email_lower for prefix in ['health', 'clinic', 'nurse']):
            return 'HEALTH_SERVICES'
        elif any(prefix in email_lower for prefix in ['maintenance', 'facilities']):
            return 'MAINTENANCE_CUSTODIAL'
        elif any(prefix in email_lower for prefix in ['security', 'guard']):
            return 'SECURITY'
        elif any(prefix in email_lower for prefix in ['admin', 'it', 'support']):
            return 'SYSTEM_ADMIN'
        
        return None

    def default_non_teaching_department_assignment(self, position):
        """Default department assignment for non-teaching faculty"""
        if not position:
            return 'ADMIN_SUPPORT'
        
        position_upper = position.upper()
        
        # Very broad categorizations as fallback
        if any(word in position_upper for word in ['STAFF', 'OFFICER', 'COORDINATOR']):
            return 'ADMIN_SUPPORT'
        elif any(word in position_upper for word in ['TECHNICIAN', 'SPECIALIST']):
            return 'SYSTEM_ADMIN'
        else:
            return 'ADMIN_SUPPORT'  # Ultimate fallback
    
    def standardize_non_teaching_department_name(self, department):
        """Standardize non-teaching department names"""
        if not department:
            return 'ADMIN_SUPPORT'
        
        dept_upper = department.upper().strip()
        
        # Handle direct abbreviations and full names
        dept_mappings = {
            'REGISTRAR': 'REGISTRAR',
            'REGISTRATION': 'REGISTRAR', 
            'RECORDS': 'REGISTRAR',
            'ACCOUNTING': 'ACCOUNTING',
            'FINANCE': 'ACCOUNTING',
            'CASHIER': 'ACCOUNTING',
            'GUIDANCE': 'GUIDANCE',
            'COUNSELING': 'GUIDANCE',
            'COUNSELLING': 'GUIDANCE',
            'LIBRARY': 'LIBRARY',
            'LIBRARIAN': 'LIBRARY',
            'HEALTH SERVICES': 'HEALTH_SERVICES',
            'HEALTH': 'HEALTH_SERVICES',
            'MEDICAL': 'HEALTH_SERVICES',
            'CLINIC': 'HEALTH_SERVICES',
            'MAINTENANCE': 'MAINTENANCE_CUSTODIAL',  # Keep original mapping
            'CUSTODIAL': 'MAINTENANCE_CUSTODIAL',
            'FACILITIES': 'MAINTENANCE_CUSTODIAL',
            'SECURITY': 'SECURITY',
            'SYSTEM ADMIN': 'SYSTEM_ADMIN',
            'IT SUPPORT': 'SYSTEM_ADMIN',
            'ADMIN SUPPORT': 'ADMIN_SUPPORT',
            'ADMINISTRATIVE': 'ADMIN_SUPPORT',
        }
        
        # Check exact mappings first
        if dept_upper in dept_mappings:
            return dept_mappings[dept_upper]
        
        # Check partial matches
        for key, value in dept_mappings.items():
            if key in dept_upper:
                return value
        
        # Default fallback
        return 'ADMIN_SUPPORT'

    def get_non_teaching_department_display_name(self, dept_code):
        """Display names for non-teaching departments"""
        dept_names = {
            'REGISTRAR': 'Office of the Registrar',
            'ACCOUNTING': 'Accounting & Finance Office',
            'GUIDANCE': 'Guidance & Counseling Office',
            'LIBRARY': 'Library Services',
            'HEALTH_SERVICES': 'Health Services Office',
            'MAINTENANCE_CUSTODIAL': 'Maintenance & Custodial Services',
            'SECURITY': 'Security Services',
            'SYSTEM_ADMIN': 'Information Technology Services',
            'ADMIN_SUPPORT': 'Administrative Support Services',
        }
        return dept_names.get(dept_code, f'Non-Teaching Services - {dept_code}')
    
    
    def format_non_teaching_faculty_info_enhanced(self, faculty_info):
        """Enhanced non-teaching faculty formatting with clean display"""
        
        # Helper function to format field
        def format_field(value):
            if value and value not in ['None', 'N/A', '']:
                return value
            return 'N/A'
        
        text = f"""NON-TEACHING FACULTY INFORMATION

        PERSONAL INFORMATION:
        Surname: {format_field(faculty_info.get('surname'))}
        First Name: {format_field(faculty_info.get('first_name'))}
        Date of Birth: {format_field(faculty_info.get('date_of_birth'))}
        Place of Birth: {format_field(faculty_info.get('place_of_birth'))}
        Citizenship: {format_field(faculty_info.get('citizenship'))}
        Sex: {format_field(faculty_info.get('sex'))}
        Height: {format_field(faculty_info.get('height'))}
        Weight: {format_field(faculty_info.get('weight'))}
        Blood Type: {format_field(faculty_info.get('blood_type'))}
        Religion: {format_field(faculty_info.get('religion'))}
        Civil Status: {format_field(faculty_info.get('civil_status'))}

        CONTACT INFORMATION:
        Address: {format_field(faculty_info.get('address'))}
        Zip Code: {format_field(faculty_info.get('zip_code'))}
        Phone: {format_field(faculty_info.get('phone'))}
        Email: {format_field(faculty_info.get('email'))}

        PROFESSIONAL INFORMATION:
        Position: {format_field(faculty_info.get('position'))}
        Department: {format_field(faculty_info.get('department'))}
        Employment Status: {format_field(faculty_info.get('employment_status'))}"""

        # Only show family info if at least one field has data
        family_fields = ['father_name', 'father_dob', 'father_occupation', 
                        'mother_name', 'mother_dob', 'mother_occupation',
                        'spouse_name', 'spouse_dob', 'spouse_occupation']
        
        has_family_data = any(faculty_info.get(field) and faculty_info.get(field) not in ['None', 'N/A', ''] 
                            for field in family_fields)
        
        if has_family_data:
            text += f"""

        FAMILY INFORMATION:
        Father's Name: {format_field(faculty_info.get('father_name'))}
        Father's Date of Birth: {format_field(faculty_info.get('father_dob'))}
        Father's Occupation: {format_field(faculty_info.get('father_occupation'))}

        Mother's Name: {format_field(faculty_info.get('mother_name'))}
        Mother's Date of Birth: {format_field(faculty_info.get('mother_dob'))}
        Mother's Occupation: {format_field(faculty_info.get('mother_occupation'))}

        Spouse's Name: {format_field(faculty_info.get('spouse_name'))}
        Spouse's Date of Birth: {format_field(faculty_info.get('spouse_dob'))}
        Spouse's Occupation: {format_field(faculty_info.get('spouse_occupation'))}"""

        # Only show government IDs if at least one is available
        gsis = faculty_info.get('gsis')
        philhealth = faculty_info.get('philhealth')
        
        if (gsis and gsis not in ['None', 'N/A', '']) or (philhealth and philhealth not in ['None', 'N/A', '']):
            text += f"""

        GOVERNMENT IDs:
        GSIS: {format_field(gsis)}
        PhilHealth: {format_field(philhealth)}"""
        
        return text.strip()
    
    def is_non_teaching_faculty_excel(self, df):
        """Check for Non-Teaching Faculty Excel files"""
        try:
            # Convert first 20 rows to text
            first_rows_text = ""
            for i in range(min(20, df.shape[0])):
                for j in range(df.shape[1]):
                    if pd.notna(df.iloc[i, j]):
                        first_rows_text += str(df.iloc[i, j]).upper() + " "
            
            print(f"🔍 Checking if file is non-teaching faculty...")
            print(f"🔍 First 200 chars: {first_rows_text[:200]}")  # Debug line
            
            # Non-teaching faculty indicators (same personal data as teaching faculty)
            faculty_indicators = [
                "SURNAME", "FIRST NAME", "DATE OF BIRTH", "PLACE OF BIRTH", 
                "CITIZENSHIP", "BLOOD TYPE", "GSIS", "PHILHEALTH"
            ]
            
            # UPDATED: More comprehensive non-teaching specific position indicators
            non_teaching_positions = [
                "REGISTRAR", "ACCOUNTING", "GUIDANCE", "LIBRARY", "LIBRARIAN", "HEALTH", 
                "MAINTENANCE", "SECURITY", "SYSTEM ADMIN", "ADMINISTRATIVE",
                "CASHIER", "SECRETARY", "ASSISTANT", "CLERK", "JANITOR", "GUARD",
                "IT SUPPORT", "NETWORK ADMIN", "COUNSELOR", "NURSE"
            ]
            
            # Count faculty indicators
            faculty_indicator_count = sum(1 for indicator in faculty_indicators if indicator in first_rows_text)
            
            # Check for non-teaching positions
            has_non_teaching_position = any(pos in first_rows_text for pos in non_teaching_positions)
            
            # Student data exclusions
            student_indicators = ["STUDENT ID", "GUARDIAN", "YEAR LEVEL", "COURSE SECTION", "PDM-"]
            has_student_indicator = any(indicator in first_rows_text for indicator in student_indicators)
            
            # Should have faculty structure AND non-teaching position indicators
            is_non_teaching = (faculty_indicator_count >= 4 and 
                            has_non_teaching_position and 
                            not has_student_indicator)
            
            print(f"🔍 Faculty indicators: {faculty_indicator_count}")
            print(f"🔍 Has non-teaching position: {has_non_teaching_position}")
            print(f"🔍 Detected positions: {[pos for pos in non_teaching_positions if pos in first_rows_text]}")
            print(f"🔍 Is non-teaching faculty: {is_non_teaching}")
            
            return is_non_teaching
            
        except Exception as e:
            print(f"🔍 Error in non-teaching faculty detection: {e}")
            return False
    
    def is_non_teaching_faculty_schedule_excel(self, df):
        """Enhanced check for Non-Teaching Faculty Schedule Excel files"""
        try:
            df_str = df.astype(str)
            first_few_rows = ' '.join(df_str.iloc[:15].values.flatten()).upper()
            
            print(f"🔍 Checking if file is non-teaching faculty schedule...")
            
            # Non-teaching faculty schedule indicators
            non_teaching_schedule_indicators = [
                "NAME OF STAFF", "STAFF SCHEDULE", "NON-TEACHING SCHEDULE", 
                "ADMINISTRATIVE SCHEDULE", "SUPPORT STAFF SCHEDULE",
                "LIBRARIAN SCHEDULE", "REGISTRAR SCHEDULE", "ACCOUNTING SCHEDULE"
            ]
            
            # General schedule indicators
            general_schedule_indicators = [
                "NAME OF ADVISER", "ADVISER", "CLASS SCHEDULE", 
                "FACULTY SCHEDULE", "TEACHING SCHEDULE", "INSTRUCTOR"
            ]
            
            # Look for day layout (key indicator of schedules)
            day_indicators = ["MON", "TUE", "WED", "THU", "FRI", "SAT", 
                            "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"]
            
            # Non-teaching position indicators
            non_teaching_positions = [
                "REGISTRAR", "ACCOUNTING", "GUIDANCE", "LIBRARY", "LIBRARIAN", "HEALTH", 
                "MAINTENANCE", "SECURITY", "SYSTEM ADMIN", "ADMINISTRATIVE",
                "CASHIER", "SECRETARY", "ASSISTANT", "CLERK", "JANITOR", "GUARD"
            ]
            
            has_non_teaching_schedule_indicator = any(indicator in first_few_rows for indicator in non_teaching_schedule_indicators)
            has_general_schedule_indicator = any(indicator in first_few_rows for indicator in general_schedule_indicators)
            has_day_layout = sum(1 for day in day_indicators if day in first_few_rows) >= 3  # At least 3 days
            has_non_teaching_position = any(pos in first_few_rows for pos in non_teaching_positions)
            
            # Should NOT have student data indicators
            student_indicators = ["STUDENT ID", "CONTACT NUMBER", "GUARDIAN", "YEAR LEVEL", "COURSE SECTION"]
            has_student_indicator = any(indicator in first_few_rows for indicator in student_indicators)
            
            # Enhanced: Check for time patterns (schedules have time slots)
            has_time_pattern = bool(re.search(r'\d{1,2}:\d{2}.*?(?:AM|PM|am|pm)', first_few_rows))
            
            # Non-teaching faculty schedule criteria
            is_non_teaching_schedule = (
                (has_non_teaching_schedule_indicator or 
                (has_general_schedule_indicator and has_non_teaching_position) or
                (has_day_layout and has_non_teaching_position)) and 
                not has_student_indicator
            )
            
            if has_time_pattern and has_non_teaching_position:
                is_non_teaching_schedule = True  # Strong indicator
            
            print(f"🔍 Has non-teaching schedule indicators: {has_non_teaching_schedule_indicator}")
            print(f"🔍 Has general schedule indicators: {has_general_schedule_indicator}")
            print(f"🔍 Has day layout: {has_day_layout}")
            print(f"🔍 Has non-teaching position: {has_non_teaching_position}")
            print(f"🔍 Has time pattern: {has_time_pattern}")
            print(f"🔍 Is non-teaching faculty schedule: {is_non_teaching_schedule}")
            
            return is_non_teaching_schedule
            
        except Exception as e:
            print(f"🔍 Error in non-teaching faculty schedule detection: {e}")
            return False
        
    def process_non_teaching_faculty_schedule_excel(self, filename):
        """Process Non-Teaching Faculty Schedule Excel with universal extraction"""
        try:
            faculty_schedule_info = self.extract_non_teaching_faculty_schedule_info_smart(filename)
            
            if not faculty_schedule_info:
                print("❌ Could not extract non-teaching faculty schedule data from Excel")
                return False
                
            formatted_text = self.format_non_teaching_faculty_schedule_enhanced(faculty_schedule_info)
            
            # Create smart metadata
            staff_name = faculty_schedule_info.get('staff_name', 'Unknown Staff')
            department = faculty_schedule_info.get('department', 'UNKNOWN')
            
            metadata = {
                'staff_name': staff_name,
                'full_name': staff_name,
                'department': self.standardize_non_teaching_department_name(department),
                'data_type': 'non_teaching_faculty_schedule_excel',
                'faculty_type': 'non_teaching_schedule',
                'total_shifts': len(faculty_schedule_info.get('schedule', [])),
                'days_working': len(set(item.get('day', '') for item in faculty_schedule_info.get('schedule', []) if item.get('day')))
            }
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            hierarchy_path = f"{self.get_non_teaching_department_display_name(metadata['department'])} > Non-Teaching Faculty Schedules"
            print(f"✅ Loaded non-teaching faculty schedule into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   👨‍💼 Staff: {staff_name}")
            print(f"   📅 Shifts: {metadata['total_shifts']}, Days: {metadata['days_working']}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing non-teaching faculty schedule Excel: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_non_teaching_faculty_schedule_info_smart(self, filename):
        """Universal non-teaching faculty schedule extraction that works with ANY Excel format"""
        try:
            df_full = pd.read_excel(filename, header=None)
            print(f"📋 Non-Teaching Faculty Schedule Excel dimensions: {df_full.shape}")
            
            # DEBUG: Show the actual Excel content
            print(f"📋 Raw Excel content (first 15 rows):")
            for i in range(min(15, df_full.shape[0])):
                row_data = []
                for j in range(min(df_full.shape[1], 6)):  # Show first 6 columns
                    if pd.notna(df_full.iloc[i, j]):
                        row_data.append(f"'{str(df_full.iloc[i, j])}'")
                    else:
                        row_data.append("'N/A'")
                print(f"   Row {i}: {row_data}")
            
            # STEP 1: Extract staff name and department
            staff_info = self.extract_non_teaching_staff_info_from_schedule(df_full)
            print(f"📋 Extracted Staff Info: {staff_info}")
            
            # STEP 2: Extract schedule data
            schedule_data = self.extract_schedule_data_from_non_teaching_faculty_excel(df_full)
            print(f"📋 Found {len(schedule_data)} scheduled shifts/duties")
            
            return {
                'staff_name': staff_info.get('name', 'Unknown Staff'),
                'department': staff_info.get('department', 'UNKNOWN'),
                'schedule': schedule_data
            }
            
        except Exception as e:
            print(f"❌ Error in non-teaching faculty schedule extraction: {e}")
            return None
    
    def extract_non_teaching_staff_info_from_schedule(self, df):
        """Extract staff name and department from non-teaching faculty schedule"""
        staff_info = {'name': 'Unknown Staff', 'department': 'UNKNOWN'}
        
        # Search for staff information in first 10 rows (more focused)
        for i in range(min(10, df.shape[0])):
            for j in range(min(df.shape[1], 10)):
                if pd.notna(df.iloc[i, j]):
                    cell_value = str(df.iloc[i, j]).strip()
                    cell_upper = cell_value.upper()
                    
                    # Look for staff name patterns with more specific matching
                    if any(keyword in cell_upper for keyword in ['NAME OF FACULTY', 'FACULTY NAME', 'NAME OF STAFF', 'STAFF NAME']):
                        # Check right cell for name
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            potential_name = str(df.iloc[i, j + 1]).strip()
                            if len(potential_name) > 2 and potential_name.upper() not in ['N/A', 'NA', '']:
                                staff_info['name'] = potential_name.title()
                                print(f"🎯 Found staff name: {potential_name}")
                    
                    # Look for department with more specific matching
                    if any(keyword in cell_upper for keyword in ['DEPARTMENT:', 'DEPT:']):
                        # Check right cell for department
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            potential_dept = str(df.iloc[i, j + 1]).strip()
                            if len(potential_dept) > 1 and potential_dept.upper() not in ['N/A', 'NA', '']:
                                staff_info['department'] = potential_dept.upper()
                                print(f"🎯 Found department: {potential_dept}")
        
        return staff_info
    
    def extract_schedule_data_from_non_teaching_faculty_excel(self, df):
        """Simple and accurate extraction treating each assignment as individual slots"""
        schedule_data = []
        
        # Find the schedule table headers
        schedule_start_row = -1
        day_columns = {}
        time_column = -1
        
        # Search for the header row with TIME, and days
        days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
        
        for i in range(min(10, df.shape[0])):
            row_cells = []
            for j in range(min(df.shape[1], 15)):
                if pd.notna(df.iloc[i, j]):
                    row_cells.append(str(df.iloc[i, j]).upper().strip())
                else:
                    row_cells.append('')
            
            # Check for column headers
            for j, cell in enumerate(row_cells):
                # Find TIME column
                if 'TIME' in cell and time_column == -1:
                    time_column = j
                    print(f"🕐 Found TIME column at position {j}")
                
                # Find day columns
                for day in days:
                    if day in cell and j not in day_columns:
                        day_columns[j] = self.standardize_day_name(day)
                        print(f"📅 Found {day} column at position {j}")
            
            # If we found headers, next row is data start
            if len(day_columns) >= 3 and time_column >= 0:
                schedule_start_row = i + 1
                print(f"🎯 Found schedule header at row {i}, data starts at row {schedule_start_row}")
                break
        
        if schedule_start_row == -1:
            print("⚠️ Could not find proper schedule table structure")
            return []
        
        print(f"📋 Day columns mapping: {day_columns}")
        print(f"📋 Time column: {time_column}")
        
        # Find the actual end of data
        data_end_row = df.shape[0]
        for i in range(schedule_start_row, df.shape[0]):
            if not self.has_valid_time(df, i, time_column):
                data_end_row = i
                break
        
        print(f"📋 Data ends at row: {data_end_row}")
        
        # SIMPLE APPROACH: Extract each assignment as individual time slots
        for col_idx, day in day_columns.items():
            print(f"\n🔍 Processing {day} column (index {col_idx}):")
            
            # Find all assignments for this day
            assignments = []
            
            for i in range(schedule_start_row, data_end_row):
                # Get time for this row
                current_time = None
                if time_column >= 0 and time_column < df.shape[1] and pd.notna(df.iloc[i, time_column]):
                    time_cell = str(df.iloc[i, time_column]).strip()
                    if time_cell and time_cell.upper() not in ['N/A', 'NONE', '']:
                        current_time = time_cell
                
                if not current_time:
                    continue
                
                # Check if this day column has an assignment
                if col_idx < df.shape[1] and pd.notna(df.iloc[i, col_idx]):
                    assignment_cell = str(df.iloc[i, col_idx]).strip()
                    if assignment_cell and assignment_cell.upper() not in ['N/A', 'NONE', '']:
                        assignments.append((i, current_time, assignment_cell))
                        print(f"   Row {i}: Found assignment '{assignment_cell}' at time {current_time}")
            
            # Create individual schedule entries for each assignment
            for row_idx, time_slot, assignment in assignments:
                schedule_entry = {
                    'day': day,
                    'time': time_slot,  # Use the exact time slot from Excel
                    'duty': f"Task: {assignment}",
                    'assignment': assignment,
                    'full_description': f"Task: {assignment}"
                }
                
                schedule_data.append(schedule_entry)
                print(f"📋 Added schedule entry: {day} {time_slot} - Task: {assignment}")
        
        print(f"📋 Total extracted assignments: {len(schedule_data)}")
        return schedule_data
    
    def combine_merged_time_slots(self, time_slots):
        """Combine time slots preserving original 12-hour format"""
        if not time_slots:
            return ""
        
        if len(time_slots) == 1:
            return time_slots[0]
        
        try:
            print(f"🔍 Combining {len(time_slots)} merged slots")
            print(f"    First few: {time_slots[:3]}")
            print(f"    Last few: {time_slots[-3:] if len(time_slots) > 3 else time_slots}")
            
            # Sort time slots to ensure proper chronological order
            def get_chronological_time(slot):
                try:
                    if ' - ' in slot:
                        start_time = slot.split(' - ')[0].strip()
                    else:
                        start_time = slot.strip()
                    
                    if ':' in start_time:
                        hour, minute = start_time.split(':')
                        hour = int(hour)
                        minute = int(minute)
                        
                        # For sorting only - don't change display format
                        if 1 <= hour <= 6:
                            hour_24 = hour + 12
                        else:
                            hour_24 = hour
                        
                        return hour_24 * 60 + minute
                    return 9999
                except:
                    return 9999
            
            sorted_slots = sorted(time_slots, key=get_chronological_time)
            print(f"    Sorted order: {sorted_slots[:3]} ... {sorted_slots[-3:] if len(sorted_slots) > 3 else sorted_slots}")
            
            # Get start time from first slot (keep original format)
            first_slot = sorted_slots[0]
            if ' - ' in first_slot:
                start_time = first_slot.split(' - ')[0].strip()
            else:
                start_time = first_slot.strip()
            
            # Get end time from last slot (keep original format)
            last_slot = sorted_slots[-1]
            if ' - ' in last_slot:
                end_time = last_slot.split(' - ')[1].strip()
            else:
                # If no end time, add 30 minutes but keep 12-hour format
                end_time = self.add_30_minutes_12hour(last_slot)
            
            combined = f"{start_time} - {end_time}"
            print(f"🔍 Combined result: {combined}")
            return combined
            
        except Exception as e:
            print(f"⚠️ Error combining merged time slots: {e}")
            return f"{time_slots[0]} - {time_slots[-1]}"
        
    def add_30_minutes_12hour(self, time_str):
        """Add 30 minutes keeping 12-hour format"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                hour = int(parts[0])
                minute = int(parts[1][:2])
                
                minute += 30
                if minute >= 60:
                    minute -= 60
                    hour += 1
                
                # Keep in 12-hour format - don't convert to 24-hour
                if hour > 12:
                    hour = hour - 12
                elif hour == 0:
                    hour = 12
                
                return f"{hour:02d}:{minute:02d}"
            else:
                return time_str
        except:
            return time_str
        
    def format_time_properly(self, time_str):
        """Keep original time format from Excel (12-hour format)"""
        # Don't convert to 24-hour format - keep original
        return time_str.strip()
    
    def find_merged_cell_end(self, df, start_row, col_idx, time_column, data_end_row, assignment_starts):
        """Find where a merged cell ends with better accuracy"""
        
        # Get the start times of other assignments to know where this one should end
        other_assignment_rows = [row for row, _, _ in assignment_starts if row > start_row]
        
        # Look for the actual end of THIS specific merged cell
        merged_end_row = start_row
        
        # Check each row after the start to see if it's part of the same assignment block
        for i in range(start_row + 1, data_end_row):
            if not self.has_valid_time(df, i, time_column):
                break
            
            # If we hit another assignment start, stop here
            if i in [row for row, _, _ in assignment_starts]:
                break
            
            # Check if this cell has content
            cell_value = None
            if col_idx < df.shape[1] and pd.notna(df.iloc[i, col_idx]):
                cell_value = str(df.iloc[i, col_idx]).strip()
            
            # If it's empty/N/A, it's part of the merged cell
            if not cell_value or cell_value.upper() in ['N/A', 'NONE']:
                merged_end_row = i
            else:
                # Found different content, stop here
                break
        
        # Limit to reasonable shift length (max 8 hours = 16 slots)
        max_slots = 16
        if merged_end_row - start_row > max_slots:
            merged_end_row = start_row + max_slots
            print(f"   ⚠️ Limited merged cell to {max_slots} slots for reasonable shift length")
        
        print(f"   📋 Detected merged cell from row {start_row} to {merged_end_row}")
        return merged_end_row
    
    def combine_consecutive_time_slots(self, time_slots):
        """Combine only truly consecutive time slots"""
        if not time_slots:
            return ""
        
        if len(time_slots) == 1:
            return time_slots[0]
        
        try:
            # Sort time slots
            def parse_start_time(slot):
                try:
                    if ' - ' in slot:
                        start = slot.split(' - ')[0].strip()
                    else:
                        start = slot.strip()
                    
                    if ':' in start:
                        hour, minute = start.split(':')
                        return int(hour) * 60 + int(minute)
                    return 9999
                except:
                    return 9999
            
            sorted_slots = sorted(time_slots, key=parse_start_time)
            
            # Check if they're truly consecutive (30-minute intervals)
            consecutive_groups = []
            current_group = [sorted_slots[0]]
            
            for i in range(1, len(sorted_slots)):
                prev_slot = current_group[-1]
                curr_slot = sorted_slots[i]
                
                # Extract end time of previous and start time of current
                try:
                    if ' - ' in prev_slot:
                        prev_end = prev_slot.split(' - ')[1].strip()
                    else:
                        prev_end = None
                    
                    if ' - ' in curr_slot:
                        curr_start = curr_slot.split(' - ')[0].strip()
                    else:
                        curr_start = curr_slot
                    
                    # If previous end time matches current start time, they're consecutive
                    if prev_end and prev_end == curr_start:
                        current_group.append(curr_slot)
                    else:
                        # Not consecutive, start new group
                        consecutive_groups.append(current_group)
                        current_group = [curr_slot]
                except:
                    # Error parsing, start new group
                    consecutive_groups.append(current_group)
                    current_group = [curr_slot]
            
            # Add the last group
            if current_group:
                consecutive_groups.append(current_group)
            
            # Combine each consecutive group
            combined_ranges = []
            for group in consecutive_groups:
                if len(group) == 1:
                    combined_ranges.append(group[0])
                else:
                    # Get start of first and end of last
                    first_slot = group[0]
                    last_slot = group[-1]
                    
                    start_time = first_slot.split(' - ')[0] if ' - ' in first_slot else first_slot
                    end_time = last_slot.split(' - ')[1] if ' - ' in last_slot else last_slot
                    
                    combined_ranges.append(f"{start_time} - {end_time}")
            
            # Join multiple ranges with &
            result = " & ".join(combined_ranges)
            print(f"🔍 Combined {len(time_slots)} slots into: {result}")
            return result
            
        except Exception as e:
            print(f"⚠️ Error combining consecutive slots: {e}")
            return time_slots[0] if time_slots else ""
    
    
    def find_merged_cell_boundaries(self, df, start_row, col_idx, time_column, data_end_row, schedule_start_row):
        """Find the actual boundaries of a merged cell by looking for the next explicit assignment"""
        block_start = start_row
        block_end = start_row
        
        # Look backwards - stop when we find another explicit assignment or reach start
        for i in range(start_row - 1, schedule_start_row - 1, -1):
            if not self.has_valid_time(df, i, time_column):
                break
            
            # Check if this row has an explicit assignment
            if col_idx < df.shape[1] and pd.notna(df.iloc[i, col_idx]):
                cell_value = str(df.iloc[i, col_idx]).strip()
                if cell_value and cell_value.upper() not in ['N/A', 'NONE']:
                    # Found another assignment, this is the boundary
                    break
            
            block_start = i
        
        # Look forwards - stop when we find another explicit assignment or reasonable end
        consecutive_na_count = 0
        for i in range(start_row + 1, data_end_row):
            if not self.has_valid_time(df, i, time_column):
                break
            
            # Check if this row has an explicit assignment
            if col_idx < df.shape[1] and pd.notna(df.iloc[i, col_idx]):
                cell_value = str(df.iloc[i, col_idx]).strip()
                if cell_value and cell_value.upper() not in ['N/A', 'NONE']:
                    # Found another assignment, stop here
                    break
                else:
                    consecutive_na_count += 1
            else:
                consecutive_na_count += 1
            
            # If we've seen too many consecutive N/A rows, probably end of merged cell
            if consecutive_na_count > 3:  # Reasonable limit
                break
            
            block_end = i
        
        return block_start, block_end
    
    
    def has_valid_time(self, df, row_idx, time_column):
        """Check if a row has valid time data"""
        if row_idx >= df.shape[0] or time_column < 0 or time_column >= df.shape[1]:
            return False
        
        if pd.notna(df.iloc[row_idx, time_column]):
            time_cell = str(df.iloc[row_idx, time_column]).strip()
            return time_cell and time_cell.upper() not in ['N/A', 'NONE', '']
        
        return False
    
    def add_schedule_entry(self, schedule_data, day, assignment, time_slots):
        """Add a schedule entry with combined time slots"""
        if not time_slots or not assignment:
            return
        
        combined_time = self.combine_time_slots_smart(time_slots)
        
        schedule_entry = {
            'day': day,
            'time': combined_time,
            'duty': f"Task: {assignment}",
            'assignment': assignment,
            'full_description': f"Task: {assignment}"
        }
        
        schedule_data.append(schedule_entry)
        print(f"📋 Added schedule entry: {day} {combined_time} - Task: {assignment}")
        
    def combine_time_slots_smart(self, time_slots):
        """Smart combination of time slots that properly handles gaps"""
        if not time_slots:
            return ""
        
        if len(time_slots) == 1:
            return time_slots[0]
        
        try:
            print(f"🔍 Combining {len(time_slots)} time slots: {time_slots[:3]}{'...' if len(time_slots) > 3 else ''}")
            
            # Sort time slots by start time to ensure proper order
            def get_start_time_minutes(slot):
                try:
                    if ' - ' in slot:
                        start_time = slot.split(' - ')[0].strip()
                    else:
                        start_time = slot.strip()
                    
                    if ':' in start_time:
                        hour, minute = start_time.split(':')
                        hour = int(hour)
                        minute = int(minute)
                        
                        # Handle 12-hour format (assume 1-6 is PM)
                        if 1 <= hour <= 6:
                            hour += 12
                        
                        return hour * 60 + minute
                    return 9999
                except:
                    return 9999
            
            sorted_slots = sorted(time_slots, key=get_start_time_minutes)
            
            # Find continuous ranges
            ranges = []
            current_start = None
            current_end = None
            
            for slot in sorted_slots:
                if ' - ' in slot:
                    start_time, end_time = slot.split(' - ')
                    start_time = start_time.strip()
                    end_time = end_time.strip()
                    
                    if current_start is None:
                        # First slot
                        current_start = start_time
                        current_end = end_time
                    else:
                        # Check if this slot continues from previous
                        if start_time == current_end:
                            # Continuous - extend the range
                            current_end = end_time
                        else:
                            # Gap found - save current range and start new
                            ranges.append(f"{current_start} - {current_end}")
                            current_start = start_time
                            current_end = end_time
            
            # Add the final range
            if current_start and current_end:
                ranges.append(f"{current_start} - {current_end}")
            
            # Combine ranges
            if len(ranges) == 1:
                combined = ranges[0]
            elif len(ranges) <= 3:
                combined = " & ".join(ranges)
            else:
                # Too many ranges, show first and last with count
                combined = f"{ranges[0]} & {ranges[-1]} (+ {len(ranges)-2} more periods)"
            
            print(f"🔍 Combined result: {combined}")
            return combined
        
        except Exception as e:
            print(f"⚠️ Error combining time slots: {e}")
            # Simple fallback
            first_time = time_slots[0].split(' - ')[0] if ' - ' in time_slots[0] else time_slots[0]
            last_time = time_slots[-1].split(' - ')[1] if ' - ' in time_slots[-1] else time_slots[-1]
            return f"{first_time} - {last_time}"
        
    def add_30_minutes(self, time_str):
        """Add 30 minutes to a time string with proper format handling"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                hour = int(parts[0])
                minute = int(parts[1][:2])  # Handle any trailing text
                
                minute += 30
                if minute >= 60:
                    minute -= 60
                    hour += 1
                
                # Handle hour overflow and format properly
                if hour > 23:
                    hour = hour - 24
                
                return f"{hour:02d}:{minute:02d}"
            else:
                return time_str
        except:
            return time_str
        
    def debug_time_conversion(self, time_slots):
        """Debug helper to see time conversion process"""
        print(f"\n🔍 DEBUG: Time conversion process")
        for i, slot in enumerate(time_slots):
            if ' - ' in slot:
                start, end = slot.split(' - ')
                print(f"    Slot {i}: {slot}")
                print(f"      Start: {start} -> {self.format_time_properly(start)}")
                print(f"      End: {end} -> {self.format_time_properly(end)}")
            else:
                print(f"    Slot {i}: {slot}")
    
    def combine_time_slots(self, time_slots):
        """Combine consecutive time slots into a single time range"""
        if not time_slots:
            return ""
        
        if len(time_slots) == 1:
            return time_slots[0]
        
        # Parse first and last time slots to get the range
        try:
            # Handle formats like "08:00 - 08:30"
            first_slot = time_slots[0]
            last_slot = time_slots[-1]
            
            # Extract start time from first slot
            if ' - ' in first_slot:
                start_time = first_slot.split(' - ')[0].strip()
            else:
                start_time = first_slot
            
            # Extract end time from last slot
            if ' - ' in last_slot:
                end_time = last_slot.split(' - ')[1].strip()
            else:
                end_time = last_slot
            
            return f"{start_time} - {end_time}"
        
        except Exception as e:
            print(f"⚠️ Error combining time slots: {e}")
            return " + ".join(time_slots)  # Fallback: just join with +
        
    
    
    def format_non_teaching_faculty_schedule_enhanced(self, schedule_info):
        """Clean, structured formatting for non-teaching faculty schedule"""
        text = f"""NON-TEACHING FACULTY WORK SCHEDULE

        STAFF INFORMATION:
        Name of Staff: {schedule_info.get('staff_name', 'Unknown Staff')}
        Department: {schedule_info.get('department', 'Unknown Department')}

        WEEKLY WORK SCHEDULE ({len(schedule_info.get('schedule', []))} scheduled assignments):
        """
        
        if schedule_info.get('schedule'):
            # Group by day for better display
            by_day = {}
            for item in schedule_info['schedule']:
                day = item.get('day', 'Unknown Day')
                if day not in by_day:
                    by_day[day] = []
                by_day[day].append(item)
            
            # Display schedule day by day in proper order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            
            for day in day_order:
                if day in by_day:
                    text += f"\n{day.upper()}:\n"
                    # Sort by time - FIXED: Remove underscore
                    by_day[day].sort(key=lambda x: self.parse_time_for_sorting(x.get('time', '')))
                    
                    for item in by_day[day]:
                        time_display = item.get('time', 'No Time')
                        duty_display = item.get('duty', 'No Duty')
                        assignment_display = item.get('assignment', '')
                        
                        if assignment_display:
                            text += f"  • {time_display} - {duty_display} (Assignment: {assignment_display})\n"
                        else:
                            text += f"  • {time_display} - {duty_display}\n"
            
            # Show unique duties summary
            unique_duties = {}
            for item in schedule_info['schedule']:
                duty = item.get('duty', '')
                if duty and duty not in unique_duties:
                    assignments = set()
                    for s in schedule_info['schedule']:
                        if s.get('duty') == duty and s.get('assignment'):
                            assignments.add(s.get('assignment'))
                    unique_duties[duty] = list(assignments)
            
            if unique_duties:
                text += f"\nDUTIES ASSIGNED ({len(unique_duties)} unique duties):\n"
                for duty, assignments in sorted(unique_duties.items()):
                    if assignments:
                        text += f"• {duty} (Assignments: {', '.join(sorted(assignments))})\n"
                    else:
                        text += f"• {duty}\n"
        else:
            text += "\nNo schedule data found."
        
        return text.strip()
    
    def is_admin_excel(self, df, silent=False):
        """Check for Admin Excel files - Only School Admin & Board Members"""
        try:
            # Convert first 20 rows to text
            first_rows_text = ""
            for i in range(min(20, df.shape[0])):
                for j in range(df.shape[1]):
                    if pd.notna(df.iloc[i, j]):
                        first_rows_text += str(df.iloc[i, j]).upper() + " "
            
            if not silent:
                print(f"🔍 Checking if file is admin...")
            
            # Admin indicators (same personal data structure as faculty)
            admin_indicators = [
                "SURNAME", "FIRST NAME", "DATE OF BIRTH", "PLACE OF BIRTH", 
                "CITIZENSHIP", "BLOOD TYPE", "GSIS", "PHILHEALTH"
            ]
            
            # UPDATED: More specific admin position indicators
            admin_positions = [
                "SCHOOL ADMIN", "SCHOOL ADMINISTRATOR", "SCHOOL ADMINISTRATION",
                "BOARD MEMBER", "BOARD OF DIRECTORS", "BOARD DIRECTOR",
                "BOARD", "ADMINISTRATOR", "ADMIN"
            ]
            
            # Count admin indicators
            admin_indicator_count = sum(1 for indicator in admin_indicators if indicator in first_rows_text)
            
            # Check for specific admin positions
            has_admin_position = any(pos in first_rows_text for pos in admin_positions)
            
            # Student data exclusions
            student_indicators = ["STUDENT ID", "GUARDIAN", "YEAR LEVEL", "COURSE SECTION", "PDM-"]
            has_student_indicator = any(indicator in first_rows_text for indicator in student_indicators)
            
            # 🆕 UPDATED: More specific faculty exclusions (exclude MEMBER to allow BOARD MEMBER)
            faculty_indicators = ["PROFESSOR", "INSTRUCTOR", "TEACHER", "LECTURER", "DEAN", "CHAIR", "FACULTY"]
            has_faculty_indicator = any(indicator in first_rows_text for indicator in faculty_indicators)
            
            # 🆕 SPECIAL: If it's a board member, override faculty detection
            is_board_member = any(pos in first_rows_text for pos in ["BOARD MEMBER", "BOARD OF DIRECTORS", "BOARD DIRECTOR"])
            
            # Should have faculty structure AND specific admin position indicators
            # AND NOT be teaching faculty (unless it's board member)
            is_admin = (admin_indicator_count >= 4 and 
                    has_admin_position and 
                    not has_student_indicator and
                    (not has_faculty_indicator or is_board_member))  # 🆕 Allow board members
            
            if not silent:
                print(f"🔍 Admin indicators: {admin_indicator_count}")
                print(f"🔍 Has admin position: {has_admin_position}")
                print(f"🔍 Has faculty position: {has_faculty_indicator}")
                print(f"🔍 Is board member: {is_board_member}")
                print(f"🔍 Detected positions: {[pos for pos in admin_positions if pos in first_rows_text]}")
                print(f"🔍 Is admin: {is_admin}")
            
            return is_admin
            
        except Exception as e:
            if not silent:
                print(f"🔍 Error in admin detection: {e}")
            return False
        
    def standardize_admin_department_name(self, department):
        """Standardize admin department names for collection naming"""
        if not department:
            return 'ADMIN'
        
        dept_upper = department.upper().strip()
        
        # Clean up common admin department patterns
        dept_mappings = {
            'SCHOOL ADMIN': 'SCHOOL_ADMIN',
            'SCHOOL ADMINISTRATOR': 'SCHOOL_ADMIN', 
            'SCHOOL ADMINISTRATION': 'SCHOOL_ADMIN',
            'BOARD MEMBER': 'BOARD',
            'BOARD OF DIRECTORS': 'BOARD',
            'BOARD DIRECTOR': 'BOARD',
            'ADMINISTRATOR': 'ADMIN',
            'ADMINISTRATION': 'ADMIN',
        }
        
        # Check exact mappings
        for full_name, clean_name in dept_mappings.items():
            if full_name in dept_upper:
                return clean_name
        
        # Remove spaces and special characters for collection naming
        cleaned = re.sub(r'[^A-Z0-9]', '_', dept_upper)
        cleaned = re.sub(r'_{2,}', '_', cleaned).strip('_')
        
        return cleaned if cleaned else 'ADMIN'
        
    def infer_admin_position_type(self, position):
        """Infer specific admin position type"""
        if not position:
            return 'School Administrator'
        
        position_upper = position.upper()
        
        # Specific admin position mappings
        if any(word in position_upper for word in ['BOARD MEMBER', 'BOARD DIRECTOR', 'BOARD OF DIRECTORS']):
            return 'Board Member'
        elif any(word in position_upper for word in ['SCHOOL ADMIN', 'SCHOOL ADMINISTRATOR', 'ADMINISTRATOR', 'ADMIN']):
            return 'School Administrator'
        else:
            return 'School Administrator'  # Default fallback
        
    def process_admin_excel(self, filename):
        """Process Admin Excel - Only School Admin & Board Members"""
        try:
            admin_info = self.extract_teaching_faculty_excel_info_smart(filename)
            
            if not admin_info:
                print("❌ Could not extract admin data from Excel")
                return False
            
            # Validate it's actually admin position
            position = admin_info.get('position', '')
            admin_position_type = self.infer_admin_position_type(position)
            
            # Set department as ADMIN for all admin personnel
            clean_department = self.standardize_admin_department_name(admin_info.get('department', 'ADMIN'))
            admin_info['department'] = clean_department
            admin_info['admin_type'] = admin_position_type
            
            formatted_text = self.format_admin_info_enhanced(admin_info)
            
            # Create smart metadata
            full_name = ""
            if admin_info.get('surname') and admin_info.get('first_name'):
                full_name = f"{admin_info['surname']}, {admin_info['first_name']}"
            elif admin_info.get('surname'):
                full_name = admin_info['surname']
            elif admin_info.get('first_name'):
                full_name = admin_info['first_name']
            else:
                full_name = "Unknown Administrator"
            
            metadata = {
                'full_name': full_name,
                'surname': admin_info.get('surname') or '',
                'first_name': admin_info.get('first_name') or '',
                'department': clean_department,  # 🆕 Use clean department name
                'position': admin_info.get('position') or '',
                'admin_type': admin_position_type,
                'employment_status': admin_info.get('employment_status') or '',
                'email': admin_info.get('email') or '',
                'phone': admin_info.get('phone') or '',
                'data_type': 'admin_excel',
                'faculty_type': 'admin',
            }
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            hierarchy_path = f"Administration > {admin_position_type}s"
            print(f"✅ Loaded admin data into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   👨‍💼 Administrator: {metadata['full_name']} ({admin_position_type})")
            return True
            
        except Exception as e:
            print(f"❌ Error processing admin Excel: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def format_admin_info_enhanced(self, admin_info):
        """Enhanced admin formatting with clean display"""
        
        def format_field(value):
            if value and value not in ['None', 'N/A', '']:
                return value
            return 'N/A'
        
        text = f"""ADMINISTRATIVE STAFF INFORMATION

        PERSONAL INFORMATION:
        Surname: {format_field(admin_info.get('surname'))}
        First Name: {format_field(admin_info.get('first_name'))}
        Date of Birth: {format_field(admin_info.get('date_of_birth'))}
        Place of Birth: {format_field(admin_info.get('place_of_birth'))}
        Citizenship: {format_field(admin_info.get('citizenship'))}
        Sex: {format_field(admin_info.get('sex'))}
        Height: {format_field(admin_info.get('height'))}
        Weight: {format_field(admin_info.get('weight'))}
        Blood Type: {format_field(admin_info.get('blood_type'))}
        Religion: {format_field(admin_info.get('religion'))}
        Civil Status: {format_field(admin_info.get('civil_status'))}

        CONTACT INFORMATION:
        Address: {format_field(admin_info.get('address'))}
        Zip Code: {format_field(admin_info.get('zip_code'))}
        Phone: {format_field(admin_info.get('phone'))}
        Email: {format_field(admin_info.get('email'))}

        ADMINISTRATIVE INFORMATION:
        Position: {format_field(admin_info.get('position'))}
        Admin Type: {format_field(admin_info.get('admin_type'))}
        Department: Administration
        Employment Status: {format_field(admin_info.get('employment_status'))}"""

        # Add family info if exists
        family_fields = ['father_name', 'father_dob', 'father_occupation', 
                        'mother_name', 'mother_dob', 'mother_occupation',
                        'spouse_name', 'spouse_dob', 'spouse_occupation']
        
        has_family_data = any(admin_info.get(field) and admin_info.get(field) not in ['None', 'N/A', ''] 
                            for field in family_fields)
        
        if has_family_data:
            text += f"""

        FAMILY INFORMATION:
        Father's Name: {format_field(admin_info.get('father_name'))}
        Father's Date of Birth: {format_field(admin_info.get('father_dob'))}
        Father's Occupation: {format_field(admin_info.get('father_occupation'))}

        Mother's Name: {format_field(admin_info.get('mother_name'))}
        Mother's Date of Birth: {format_field(admin_info.get('mother_dob'))}
        Mother's Occupation: {format_field(admin_info.get('mother_occupation'))}

        Spouse's Name: {format_field(admin_info.get('spouse_name'))}
        Spouse's Date of Birth: {format_field(admin_info.get('spouse_dob'))}
        Spouse's Occupation: {format_field(admin_info.get('spouse_occupation'))}"""

        # Add government IDs if exist
        gsis = admin_info.get('gsis')
        philhealth = admin_info.get('philhealth')
        
        if (gsis and gsis not in ['None', 'N/A', '']) or (philhealth and philhealth not in ['None', 'N/A', '']):
            text += f"""

        GOVERNMENT IDs:
        GSIS: {format_field(gsis)}
        PhilHealth: {format_field(philhealth)}"""
        
        return text.strip()
    
    def is_curriculum_excel(self, df, silent=False):
        """Check for Curriculum Excel files"""
        try:
            # Convert first 20 rows to text
            first_rows_text = ""
            for i in range(min(20, df.shape[0])):
                for j in range(df.shape[1]):
                    if pd.notna(df.iloc[i, j]):
                        first_rows_text += str(df.iloc[i, j]).upper() + " "
            
            if not silent:
                print(f"🔍 Checking if file is curriculum...")
            
            # Curriculum indicators
            curriculum_indicators = [
                "CURRICULUM", "COURSE CURRICULUM", "ACADEMIC CURRICULUM",
                "CURRICULUM GUIDE", "PROGRAM CURRICULUM", "SYLLABUS"
            ]
            
            # Subject-related indicators
            subject_indicators = [
                "SUBJECT CODE", "SUBJECT NAME", "COURSE CODE", "COURSE NAME",
                "UNITS", "HOURS", "SEMESTER", "YEAR LEVEL", "PREREQUISITE"
            ]
            
            # Academic structure indicators
            academic_indicators = [
                "1ST YEAR", "2ND YEAR", "3RD YEAR", "4TH YEAR",
                "FIRST SEMESTER", "SECOND SEMESTER", "SUMMER",
                "MAJOR", "MINOR", "ELECTIVE", "CORE"
            ]
            
            # Count indicators
            has_curriculum_title = any(indicator in first_rows_text for indicator in curriculum_indicators)
            subject_indicator_count = sum(1 for indicator in subject_indicators if indicator in first_rows_text)
            academic_indicator_count = sum(1 for indicator in academic_indicators if indicator in first_rows_text)
            
            # Should NOT have student or faculty indicators
            student_indicators = ["STUDENT ID", "GUARDIAN", "CONTACT NUMBER"]
            faculty_indicators = ["FACULTY", "PROFESSOR", "INSTRUCTOR", "ADVISER"]
            
            has_student_indicator = any(indicator in first_rows_text for indicator in student_indicators)
            has_faculty_indicator = any(indicator in first_rows_text for indicator in faculty_indicators)
            
            # Curriculum detection logic
            is_curriculum = (
                (has_curriculum_title or subject_indicator_count >= 3 or academic_indicator_count >= 2) and
                not has_student_indicator and not has_faculty_indicator
            )
            
            if not silent:
                print(f"🔍 Curriculum title: {has_curriculum_title}")
                print(f"🔍 Subject indicators: {subject_indicator_count}")
                print(f"🔍 Academic indicators: {academic_indicator_count}")
                print(f"🔍 Is curriculum: {is_curriculum}")
            
            return is_curriculum
            
        except Exception as e:
            if not silent:
                print(f"🔍 Error in curriculum detection: {e}")
            return False
        
    def extract_curriculum_excel_info_smart(self, filename):
        """Universal curriculum extraction that works with ANY Excel format"""
        try:
            df_full = pd.read_excel(filename, header=None)
            print(f"📋 Curriculum Excel dimensions: {df_full.shape}")
            
            # DEBUG: Show the actual Excel content
            print(f"📋 Raw Excel content (first 15 rows):")
            for i in range(min(15, df_full.shape[0])):
                row_data = []
                for j in range(min(df_full.shape[1], 8)):  # Show first 8 columns
                    if pd.notna(df_full.iloc[i, j]):
                        row_data.append(f"'{str(df_full.iloc[i, j])}'")
                    else:
                        row_data.append("'N/A'")
                print(f"   Row {i}: {row_data}")
            
            # STEP 1: Extract curriculum metadata (program, department)
            curriculum_info = self.extract_curriculum_metadata(df_full, filename)
            print(f"📋 Extracted Curriculum Info: {curriculum_info}")
            
            # STEP 2: Extract curriculum subjects
            subjects_data = self.extract_curriculum_subjects(df_full)
            print(f"📋 Found {len(subjects_data)} curriculum subjects")
            
            return {
                'curriculum_info': curriculum_info,
                'subjects': subjects_data
            }
            
        except Exception as e:
            print(f"❌ Error in curriculum extraction: {e}")
            return None

    def extract_curriculum_metadata(self, df, filename):
        """Enhanced curriculum metadata extraction"""
        curriculum_info = {
            'program': '',
            'department': '',
            'total_years': 4,
            'curriculum_year': '',
            'year_level': ''
        }
        
        # Search for curriculum information in first 20 rows
        for i in range(min(20, df.shape[0])):
            for j in range(min(df.shape[1], 10)):
                if pd.notna(df.iloc[i, j]):
                    cell_value = str(df.iloc[i, j]).strip()
                    cell_upper = cell_value.upper()
                    
                    # Look for program information - ENHANCED
                    if any(keyword in cell_upper for keyword in ['COURSE:', 'PROGRAM:', 'DEGREE:']):
                        # Check adjacent cells for the actual program
                        program_value = None
                        
                        # Check right cell first
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            potential_program = str(df.iloc[i, j + 1]).strip()
                            if len(potential_program) > 2 and not any(x in potential_program.upper() for x in ['MAJOR', 'MINOR', 'CORE']):
                                program_value = potential_program
                        
                        # Check if current cell contains the program after colon
                        if not program_value and ':' in cell_value:
                            parts = cell_value.split(':', 1)
                            if len(parts) > 1:
                                potential_program = parts[1].strip()
                                if len(potential_program) > 2:
                                    program_value = potential_program
                        
                        if program_value:
                            # Extract course code from full program name
                            if 'BS' in program_value.upper() and 'COMPUTER' in program_value.upper():
                                curriculum_info['program'] = 'BSCS'
                            elif 'COMPUTER SCIENCE' in program_value.upper():
                                curriculum_info['program'] = 'BSCS'
                            else:
                                # Try to extract BS course pattern
                                course_match = re.search(r'BS\s*([A-Z]+)', program_value.upper())
                                if course_match:
                                    curriculum_info['program'] = f"BS{course_match.group(1)}"
                                else:
                                    curriculum_info['program'] = program_value.upper()
                            
                            print(f"🎯 Found program: {program_value} -> {curriculum_info['program']}")
                    
                    # Look for year level information - ENHANCED
                    if any(keyword in cell_upper for keyword in ['YEAR LEVEL:', 'YEAR:']):
                        year_value = None
                        
                        # Check right cell
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            potential_year = str(df.iloc[i, j + 1]).strip()
                            if potential_year:
                                year_value = potential_year
                        
                        # Check if current cell contains year after colon
                        if not year_value and ':' in cell_value:
                            parts = cell_value.split(':', 1)
                            if len(parts) > 1:
                                potential_year = parts[1].strip()
                                if potential_year:
                                    year_value = potential_year
                        
                        if year_value:
                            # Extract year number
                            year_match = re.search(r'([1-4])', year_value)
                            if year_match:
                                curriculum_info['year_level'] = year_match.group(1)
                            curriculum_info['curriculum_year'] = year_value
                            print(f"🎯 Found year level: {year_value}")
        
        # Infer department from program
        if curriculum_info['program'] and not curriculum_info['department']:
            inferred_dept = self.detect_department_from_course(curriculum_info['program'])
            curriculum_info['department'] = inferred_dept
            print(f"🎯 Inferred department from program: {inferred_dept}")
        
        # Fallback: infer from filename
        if not curriculum_info['program']:
            filename_program = self.extract_course_from_filename(filename)
            if filename_program:
                curriculum_info['program'] = filename_program
                print(f"🎯 Inferred program from filename: {filename_program}")
        
        return curriculum_info
    
    def extract_curriculum_subjects(self, df):
        """Enhanced curriculum subjects extraction with better column mapping"""
        subjects_data = []
        
        # Find the header row
        header_row = -1
        column_mapping = {}
        
        # Enhanced field mappings
        field_mappings = {
            'year_level': ['YEAR LEVEL', 'YEAR', 'YR', 'LEVEL'],
            'semester': ['SEMESTER', 'SEM', 'TERM', 'PERIOD'],
            'subject_code': ['SUBJECT CODE', 'COURSE CODE', 'CODE', 'SUBJ CODE'],
            'subject_name': ['SUBJECT NAME', 'COURSE NAME', 'SUBJECT', 'COURSE TITLE', 'DESCRIPTION', 'TITLE'],
            'type': ['TYPE', 'CATEGORY', 'CLASSIFICATION', 'KIND'],
            'hours_per_week': ['HOURS/WEEK', 'HOURS PER WEEK', 'HOURS', 'HRS/WK', 'CONTACT HOURS'],
            'units': ['UNITS', 'CREDITS', 'CREDIT UNITS', 'CR']
        }
        
        # Search for header row
        for i in range(min(10, df.shape[0])):
            row_text = ' '.join([str(df.iloc[i, j]) for j in range(min(df.shape[1], 15)) if pd.notna(df.iloc[i, j])]).upper()
            
            # Check if this row contains multiple field headers
            header_count = 0
            for field, possible_headers in field_mappings.items():
                for header in possible_headers:
                    if header in row_text:
                        header_count += 1
                        break
            
            if header_count >= 4:  # Found header row
                header_row = i
                print(f"🎯 Found curriculum header at row {i}")
                break
        
        if header_row == -1:
            print("⚠️ Could not find curriculum header row")
            return []
        
        # ENHANCED: Map columns more precisely
        header_cells = []
        for j in range(df.shape[1]):
            if pd.notna(df.iloc[header_row, j]):
                header_text = str(df.iloc[header_row, j]).strip().upper()
                header_cells.append((j, header_text))
            else:
                header_cells.append((j, ''))
        
        print(f"📋 Header cells: {header_cells}")
        
        # Map each field to the best matching column
        for field, possible_headers in field_mappings.items():
            best_match = None
            best_score = 0
            
            for col_idx, header_text in header_cells:
                for possible_header in possible_headers:
                    if possible_header in header_text:
                        # Prioritize exact matches
                        score = len(possible_header) if header_text == possible_header else len(possible_header) - 1
                        if score > best_score:
                            best_score = score
                            best_match = col_idx
            
            if best_match is not None:
                column_mapping[field] = best_match
                print(f"🎯 Mapped {field} to column {best_match} ({header_cells[best_match][1]})")
        
        # Special handling for subject_name if it's mapped to same column as subject_code
        if (column_mapping.get('subject_name') == column_mapping.get('subject_code') and 
            'subject_name' in column_mapping):
            # Look for the next column that might contain subject names
            subject_code_col = column_mapping['subject_code']
            for col_idx in range(subject_code_col + 1, df.shape[1]):
                if col_idx not in column_mapping.values():
                    # Check if this column contains descriptive text (likely subject names)
                    sample_value = None
                    for row_idx in range(header_row + 1, min(header_row + 5, df.shape[0])):
                        if pd.notna(df.iloc[row_idx, col_idx]):
                            sample_value = str(df.iloc[row_idx, col_idx]).strip()
                            break
                    
                    if sample_value and len(sample_value) > 10:  # Likely a descriptive name
                        column_mapping['subject_name'] = col_idx
                        print(f"🎯 Remapped subject_name to column {col_idx} (descriptive content)")
                        break
        
        print(f"📋 Final column mapping: {column_mapping}")
        
        # Extract subject data with enhanced logic
        for i in range(header_row + 1, df.shape[0]):
            # Skip empty rows
            if all(pd.isna(df.iloc[i, j]) for j in range(df.shape[1])):
                continue
            
            # Skip footer rows
            first_cell = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
            if any(keyword in first_cell.upper() for keyword in ['TOTAL', 'SUMMARY', 'NOTE', 'LEGEND']):
                break
            
            # Extract subject data
            subject_entry = {}
            valid_entry = False
            
            for field, col_idx in column_mapping.items():
                if col_idx < df.shape[1] and pd.notna(df.iloc[i, col_idx]):
                    value = str(df.iloc[i, col_idx]).strip()
                    if value and value.upper() not in ['N/A', 'NONE', 'TBA', 'TBD']:
                        cleaned_value = self.clean_curriculum_value(value, field)
                        if cleaned_value:
                            subject_entry[field] = cleaned_value
                            if field in ['subject_code', 'subject_name']:
                                valid_entry = True
            
            # Handle semester inheritance (if semester cell is empty, use previous semester)
            if 'semester' not in subject_entry or not subject_entry['semester']:
                if subjects_data:  # Use semester from previous subject
                    subject_entry['semester'] = subjects_data[-1].get('semester', '1st Semester')
                else:
                    subject_entry['semester'] = '1st Semester'
            
            # Set defaults for missing fields
            for field in field_mappings.keys():
                if field not in subject_entry:
                    subject_entry[field] = self.get_default_curriculum_value(field)
            
            # Only add if we have essential fields
            if valid_entry and (subject_entry.get('subject_code') or subject_entry.get('subject_name')):
                subjects_data.append(subject_entry)
                print(f"📚 Added subject: {subject_entry.get('subject_code', 'N/A')} - {subject_entry.get('subject_name', 'N/A')}")
        
        return subjects_data
    
    def clean_curriculum_value(self, value, field_type):
        """Enhanced cleaning for curriculum field values"""
        if not value or len(value.strip()) == 0:
            return None
        
        value = value.strip()
        
        if field_type == 'subject_name':
            # Keep subject names as-is but clean them properly
            if len(value) > 1 and value.upper() not in ['N/A', 'NONE', 'TBA', 'TBD']:
                return value.title()  # Proper case
            return None
        
        elif field_type == 'subject_code':
            # Clean subject codes (keep alphanumeric and dashes)
            cleaned = re.sub(r'[^A-Z0-9\-]', '', value.upper())
            return cleaned if cleaned and len(cleaned) >= 2 else None
        
        elif field_type == 'semester':
            # Enhanced semester standardization
            value_upper = value.upper()
            if any(term in value_upper for term in ['1ST', 'FIRST', '1']):
                return '1st Semester'
            elif any(term in value_upper for term in ['2ND', 'SECOND', '2']):
                return '2nd Semester'
            elif any(term in value_upper for term in ['SUMMER', 'SUM', 'MID']):
                return 'Summer'
            else:
                return value.strip()
        
        elif field_type == 'type':
            # Enhanced type standardization
            value_upper = value.upper()
            if any(term in value_upper for term in ['MAJOR', 'CORE', 'PROFESSIONAL']):
                return 'Major'
            elif any(term in value_upper for term in ['MINOR', 'ELECTIVE']):
                return 'Elective'
            elif any(term in value_upper for term in ['GEN', 'GENERAL', 'EDUCATION']):
                return 'General Education'
            elif any(term in value_upper for term in ['LAB', 'LABORATORY']):
                return 'Laboratory'
            elif 'PE' in value_upper:
                return 'Physical Education'
            elif 'NSTP' in value_upper:
                return 'NSTP'
            else:
                return value.title()
        
        elif field_type == 'year_level':
            # Extract year number (1-4)
            year_match = re.search(r'([1-4])', value)
            return year_match.group(1) if year_match else '1'
        
        elif field_type in ['hours_per_week', 'units']:
            # Extract numeric values
            numeric_match = re.search(r'(\d+(?:\.\d+)?)', value)
            return numeric_match.group(1) if numeric_match else '3'
        
        return value

    def get_default_curriculum_value(self, field):
        """Get default values for missing curriculum fields"""
        defaults = {
            'year_level': '1',
            'course': 'General',
            'semester': '1st Semester',
            'subject_code': 'N/A',
            'subject_name': 'Unknown Subject',
            'type': 'Core',
            'hours_per_week': '3',
            'units': '3'
        }
        return defaults.get(field, 'N/A')
    
    
    def process_curriculum_excel(self, filename):
        """Process Curriculum Excel with smart organization"""
        try:
            curriculum_info = self.extract_curriculum_excel_info_smart(filename)
            
            if not curriculum_info or not curriculum_info.get('subjects'):
                print("❌ Could not extract curriculum data from Excel")
                return False
            
            formatted_text = self.format_curriculum_info_enhanced(curriculum_info)
            
            # Create smart metadata
            program = curriculum_info['curriculum_info'].get('program', 'Unknown Program')
            department = curriculum_info['curriculum_info'].get('department', 'Unknown Department')
            
            metadata = {
                'program': program,
                'department': self.standardize_department_name(department),
                'total_subjects': len(curriculum_info['subjects']),
                'data_type': 'curriculum_excel',
                'curriculum_type': 'academic_program',
                'subjects_by_year': self.count_subjects_by_year(curriculum_info['subjects']),
                'total_units': self.calculate_total_units(curriculum_info['subjects'])
            }
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('curriculum', metadata)
            collection = self.client.get_or_create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            hierarchy_path = f"{self.get_department_display_name(metadata['department'])} > {metadata['program']} Curriculum"
            print(f"✅ Loaded curriculum into: {collection_name}")
            print(f"   📁 Hierarchy: {hierarchy_path}")
            print(f"   📚 Program: {metadata['program']}")
            print(f"   📊 Subjects: {metadata['total_subjects']}, Total Units: {metadata['total_units']}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing curriculum Excel: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def count_subjects_by_year(self, subjects):
        """Count subjects by year level"""
        year_counts = {}
        for subject in subjects:
            year = subject.get('year_level', '1')
            year_counts[year] = year_counts.get(year, 0) + 1
        return str(year_counts)  # Convert to string for ChromaDB

    def calculate_total_units(self, subjects):
        """Calculate total units in curriculum"""
        total = 0
        for subject in subjects:
            try:
                units = float(subject.get('units', '3'))
                total += units
            except (ValueError, TypeError):
                total += 3  # Default if can't parse
        return int(total)

    def format_curriculum_info_enhanced(self, curriculum_info):
        """Enhanced curriculum formatting with proper year level display"""
        
        curr_info = curriculum_info['curriculum_info']
        subjects = curriculum_info['subjects']
        
        # Get proper program and year display
        program = curr_info.get('program', 'Unknown Program')
        year_level = curr_info.get('year_level', '')
        curriculum_year = curr_info.get('curriculum_year', '')
        
        # Use curriculum_year for display if available, otherwise year_level
        year_display = curriculum_year if curriculum_year else (f"{year_level} Year" if year_level else "Unknown Year")
        
        text = f"""ACADEMIC CURRICULUM

        PROGRAM INFORMATION:
        Program: {program}
        Year Level: {year_display}
        Department: {curr_info.get('department', 'Unknown Department')}
        Total Subjects: {len(subjects)}
        Total Units: {self.calculate_total_units(subjects)}

        CURRICULUM STRUCTURE:
        """
        
        if subjects:
            # Group subjects by semester
            by_semester = {}
            for subject in subjects:
                semester = subject.get('semester', '1st Semester')
                if semester not in by_semester:
                    by_semester[semester] = []
                by_semester[semester].append(subject)
            
            # Display by semester
            for semester in sorted(by_semester.keys()):
                text += f"\n{semester.upper()}:\n"
                
                for subject in by_semester[semester]:
                    code = subject.get('subject_code', 'N/A')
                    name = subject.get('subject_name', 'Unknown Subject')
                    units = subject.get('units', '3')
                    hours = subject.get('hours_per_week', '3')
                    subject_type = subject.get('type', 'Core')
                    
                    text += f"  • {code} - {name}\n"
                    text += f"    Type: {subject_type}, Units: {units}, Hours/Week: {hours}\n"
            
            # Summary by subject type
            type_counts = {}
            for subject in subjects:
                subject_type = subject.get('type', 'Core')
                type_counts[subject_type] = type_counts.get(subject_type, 0) + 1
            
            if type_counts:
                text += f"\nSUBJECT DISTRIBUTION:\n"
                for subject_type, count in sorted(type_counts.items()):
                    text += f"• {subject_type}: {count} subjects\n"
        else:
            text += "\nNo subjects found in curriculum."
        
        return text.strip()

    def get_year_display(self, year):
        """Get display name for year level"""
        year_names = {
            '1': 'FIRST YEAR',
            '2': 'SECOND YEAR', 
            '3': 'THIRD YEAR',
            '4': 'FOURTH YEAR'
        }
        

        
    # ======================== SEARCH & QUERY ========================
    
    def search_all_collections(self, query, max_results=15):
        """Search across all loaded collections"""
        all_results = []
        
        for name, collection in self.collections.items():
            try:
                # When querying, ChromaDB uses the embedding function associated with the collection.
                # Ensure it's set during get_or_create_collection.
                results = collection.query(query_texts=[query], n_results=max_results)
                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        metadata = results["metadatas"][0][i] if results["metadatas"][0] else {}
                        collection_type = self.get_collection_type(name)
                        all_results.append({
                            "source": collection_type, 
                            "content": doc,
                            "metadata": metadata,
                            "hierarchy": f"{self.get_department_display_name(metadata.get('department', 'Unknown'))} > {metadata.get('course', 'Unknown')} > {metadata.get('section', 'Unknown')}"
                        })
            except Exception as e:
                print(f"⚠️ Error searching {name}: {e}")
        
        return all_results

    def smart_search(self):
        """True AI-powered search interface"""
        query = input("\n🧠 Enter your search query: ").strip()
        if not query:
            return
        
        try:
            limit_input = input("🔢 Max results (default 15): ").strip()
            max_results = int(limit_input) if limit_input else 15
        except ValueError:
            max_results = 50
        
        print(f"\n🔍 AI is analyzing your query...")
        
        # Use the new AI-powered search
        results = self.smart_search_with_ai_reasoning(query, max_results)
        
        if results:
            print(f"\n✅ Found {len(results)} highly relevant results:")
            for i, result in enumerate(results, 1):
                print(f"\n📄 Result {i} (Relevance: {result['relevance']}) - {result['source']}:")
                print(f"📁 {result.get('hierarchy', 'N/A')}")
                print(f"🎯 Match: {result['match_reason']}")
                print("-" * 60)
                print(result['content'])
        else:
            print("❌ No relevant results found. Try rephrasing your query.")

    def exact_match_search(self): # Removed 'query' parameter as it's taken from input
        """Perform exact text matching across all collections"""
        query = input("\n📝 Enter exact text to find: ").strip()
        if not query:
            return
                
        print(f"\n🔍 Searching for exact matches: '{query}'")
        matches = []
        for name, collection in self.collections.items():
            try:
                # Get all documents from collection
                all_docs = collection.get() # Fetches all documents, can be slow for large collections
                for doc in all_docs["documents"]:
                    # Perform a case-insensitive search for flexibility
                    if query.lower() in doc.lower():
                        metadata = all_docs["metadatas"][all_docs["documents"].index(doc)] # Get corresponding metadata
                        collection_type = self.get_collection_type(name)
                        
                        # ENHANCED: Use the centralized hierarchy display method
                        hierarchy = self.get_proper_hierarchy_display(name, metadata)

                        matches.append({
                            "source": collection_type, 
                            "content": doc,
                            "metadata": metadata,
                            "hierarchy": hierarchy
                        })
            except Exception as e:
                print(f"⚠️ Error in exact search for {name}: {e}")
                
        if matches:
            print(f"\n✅ Found {len(matches)} exact matches:")
            for i, match in enumerate(matches, 1):
                print(f"\n📄 Match {i} (from {match['source']}):")
                print(f"📁 {match.get('hierarchy', 'N/A')}")
                print("-" * 60)
                print(match['content'])
        else:
            print("❌ No exact matches found.")


    def search_specific_collection(self, collection_name, query, max_results=50):
        """Search in a specific collection"""
        if collection_name not in self.collections:
            print(f"❌ Collection '{collection_name}' not found")
            return []
        
        try:
            collection = self.collections[collection_name]
            # Use the consistent embedding function when querying
            results = collection.query(query_texts=[query], n_results=max_results)
            return results["documents"][0] if results["documents"] and results["documents"][0] else []
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []

    # ======================== COLLECTION MANAGEMENT ========================
    
    def delete_collection(self, collection_name):
        """Delete a specific collection"""
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            print(f"✅ Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error deleting collection {collection_name}: {e}")
            return False

    def delete_all_collections(self):
        """Delete all collections"""
        try:
            existing_collections = self.client.list_collections()
            deleted_count = 0
            
            for collection in existing_collections:
                try:
                    self.client.delete_collection(name=collection.name)
                    if collection.name in self.collections:
                        del self.collections[collection.name]
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Could not delete {collection.name}: {e}")
            
            print(f"✅ Deleted {deleted_count} collections")
            self.data_loaded = False
            return True
        except Exception as e:
            print(f"❌ Error deleting collections: {e}")
            return False

    def manage_collections(self):
        """Collection management interface"""
        existing = self.check_existing_data()
        
        if not existing:
            print("❌ No collections found to manage.")
            return
        
        print("\n📚 COLLECTION MANAGEMENT")
        print("=" * 50)
        print("1. 🗑️  Delete specific collection")
        print("2. 🗑️  Delete all collections")
        print("3. 📋 Show collection details")
        print("4. ↩️  Back to main menu")
        
        try:
            choice = input("\n💡 Choose an option (1-4): ").strip()
            
            if choice == "1":
                self.delete_specific_collection_menu()
            elif choice == "2":
                self.delete_all_collections_menu()
            elif choice == "3":
                self.show_all_collections()
            elif choice == "4":
                return
            else:
                print("❌ Invalid choice.")
        except KeyboardInterrupt:
            print("\n↩️ Returning to main menu...")

    def delete_specific_collection_menu(self):
        """Menu for deleting specific collections"""
        existing = self.client.list_collections()
        
        if not existing:
            print("❌ No collections found.")
            return
        
        print(f"\n📚 Available Collections:")
        for i, collection in enumerate(existing, 1):
            count = collection.count()
            collection_type = self.get_collection_type(collection.name)
            print(f"  {i}. {collection_type} ({count} records)")
        
        try:
            choice = int(input("\n🔢 Enter collection number to delete (0 to cancel): ").strip())
            
            if choice == 0:
                print("❌ Cancelled.")
                return
            
            if 1 <= choice <= len(existing):
                collection_to_delete = existing[choice - 1]
                collection_type = self.get_collection_type(collection_to_delete.name)
                
                # Confirm deletion
                confirm = input(f"\n⚠️  Are you sure you want to delete '{collection_type}'? (yes/no): ").strip().lower()
                
                if confirm in ['yes', 'y']:
                    if self.delete_collection(collection_to_delete.name):
                        print(f"✅ Successfully deleted '{collection_type}'")
                        
                        # Ask if they want to reload this type of data
                        reload = input(f"\n🔄 Do you want to reload {collection_type} from files? (yes/no): ").strip().lower()
                        if reload in ['yes', 'y']:
                            self.reload_specific_data_type(collection_to_delete.name)
                    else:
                        print(f"❌ Failed to delete '{collection_type}'")
                else:
                    print("❌ Deletion cancelled.")
            else:
                print("❌ Invalid selection.")
                
        except ValueError:
            print("❌ Invalid input.")

    def delete_all_collections_menu(self):
        """Menu for deleting all collections with confirmation"""
        existing = self.client.list_collections()
        
        if not existing:
            print("❌ No collections found.")
            return
        
        print(f"\n⚠️  WARNING: This will delete ALL {len(existing)} collections:")
        for collection in existing:
            collection_type = self.get_collection_type(collection.name)
            count = collection.count()
            print(f"   • {collection_type} ({count} records)")
        
        confirm = input(f"\n⚠️  Are you sure you want to delete ALL collections? Type 'DELETE ALL' to confirm: ").strip()
        
        if confirm == 'DELETE ALL':
            if self.delete_all_collections():
                print("✅ All collections deleted successfully!")
                
                # Ask if they want to reload data
                reload = input(f"\n🔄 Do you want to load fresh data now? (yes/no): ").strip().lower()
                if reload in ['yes', 'y']:
                    self.load_new_data()
            else:
                print("❌ Failed to delete all collections.")
        else:
            print("❌ Deletion cancelled.")

    def reload_specific_data_type(self, collection_name):
        """Reload a specific type of data"""
        # Map collection names to file types
        file_type_map = {
            "students_excel": "student Excel",
            "students_pdf": "student PDF", 
            "schedules_excel": "COR Excel",
            "schedules_pdf": "COR PDF",
            "faculty_excel": "faculty Excel",
            "faculty_pdf": "faculty PDF",
            "faculty_schedules_excel": "faculty schedule Excel",
            "faculty_schedules_pdf": "faculty schedule PDF"
        }
        
        data_type = file_type_map.get(collection_name, "unknown")
        print(f"\n🔍 Looking for {data_type} files...")
        
        files = self.list_available_files()
        if not files:
            print("❌ No files available to load.")
            return
        
        # Filter files by type if possible
        relevant_files = []
        for i, file in enumerate(files):
            file_type = self.detect_file_type(file)
            if data_type.lower() in file_type.lower():
                relevant_files.append((i + 1, file, file_type))
        
        if relevant_files:
            print(f"\n📁 Found {len(relevant_files)} relevant files:")
            for idx, file, ftype in relevant_files:
                print(f"  {idx}. {file} - {ftype}")
            
            try:
                choice = int(input(f"\n🔢 Enter file number to load: ").strip())
                if any(choice == idx for idx, _, _ in relevant_files):
                    filename = files[choice - 1]
                    self.process_file(filename)
                else:
                    print("❌ Invalid selection.")
            except ValueError:
                print("❌ Invalid input.")
        else:
            print(f"❌ No {data_type} files found. Showing all available files:")
            choice = int(input("\n🔢 Enter file number to load: ").strip())
            if 1 <= choice <= len(files):
                filename = files[choice - 1]
                self.process_file(filename)

    # ======================== USER INTERFACE ========================
    
    def show_search_options(self):
        """Display search options to user"""
        print("\n🔍 SEARCH OPTIONS:")
        print("1. 🔎 Smart Search")
        print("2. 📝 Exact Match Search")
        print("3. 🤖 GROUP 3 AI (DITO YUNG AI MISMO IGNORE IBA)") # 🆕 RENAMED/ADDED
        print("4. 📊 Browse by Collection")
        print("5. 📂 Load More Data")
        print("6. 📋 Show All Collections")
        print("7. 🗑️ Manage Collections")
        print("8. 🧹 Clean Existing Duplicates")
        print("9. ⚙️ System Options") # 🆕 ADDED
        print("10. ❌ Exit")

        if self.collections:
            if self.debug_mode:
                print(f"\n📚 Loaded Collections:")
                for name in self.collections.keys():
                    count = self.collections[name].count()
                    collection_type = self.get_collection_type(name)
                    print(f"   • {collection_type} ({count} records)")

    def browse_collections(self):
        """Browse data by collection"""
        if not self.collections:
            print("❌ No collections available.")
            return
        
        print(f"\n📚 Available Collections:")
        collection_list = list(self.collections.keys())
        for i, name in enumerate(collection_list, 1):
            count = self.collections[name].count()
            collection_type = self.get_collection_type(name)
            print(f"  {i}. {collection_type} ({count} records)")
        
        try:
            choice = int(input("\n🔢 Choose collection number: ").strip())
            if 1 <= choice <= len(collection_list):
                collection_name = collection_list[choice - 1]
                collection_type = self.get_collection_type(collection_name)
                query = input(f"\n🔍 Search in '{collection_type}': ").strip()
                
                if query:
                    results = self.search_specific_collection(collection_name, query)
                    if results:
                        print(f"\n✅ Found {len(results)} results in {collection_type}:")
                        for i, result in enumerate(results, 1):
                            print(f"\n📄 Result {i}:")
                            print("-" * 60)
                            print(result)
                    else:
                        print("❌ No results found.")
        except ValueError:
            print("❌ Invalid input.")

    def show_all_collections(self):
        """Show detailed info about all collections"""
        if not self.collections:
            print("❌ No collections loaded.")
            return
        
        print(f"\n📊 COLLECTION DETAILS:")
        print("=" * 60)
        
        for name, collection in self.collections.items():
            collection_type = self.get_collection_type(name)
            count = collection.count()
            print(f"\n📁 {collection_type}")
            print(f"   Collection ID: {name}")
            print(f"   Records: {count}")
            
            # Show sample data
            try:
                # Use the consistent embedding function when getting sample data
                sample = collection.get(limit=1) 
                if sample["documents"]:
                    print(f"   Sample data:")
                    sample_text = sample["documents"][0][:200] + "..." if len(sample["documents"][0]) > 200 else sample["documents"][0]
                    print(f"   {sample_text}")
            except Exception as e:
                print(f"   Could not retrieve sample data: {e}")
            print("-" * 40)

    def run_query_interface(self):
        """Main query interface"""
        if not self.data_loaded:
            if self.debug_mode:
                print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*70)
        print("🎯 SMART STUDENT DATA SYSTEM - READY!")
        print("="*70)
        
        while True:
            self.show_search_options()
            
            try:
                choice = input("\n💡 Choose an option (1-10): ").strip()
                
                if choice == "1":
                    self.smart_search()
                elif choice == "2":
                    self.exact_match_search()
                elif choice == "3": # 🆕 HANDLES NEW OPTION
                    self.test_llm_search()
                elif choice == "4":
                    self.browse_collections()
                elif choice == "5":
                    self.load_new_data()
                elif choice == "6":
                    self.show_all_collections()
                elif choice == "7":
                    self.manage_collections()
                elif choice == "8":
                    self.scan_and_clean_existing_duplicates()
                elif choice == "9": # 🆕 HANDLES NEW OPTION
                    self.manage_system_options()
                elif choice == "10":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-10.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
    def extract_course_from_filename(self, filename):
        """Extract course information from filename"""
        filename_upper = os.path.basename(filename).upper()
        
        # Common course patterns in filenames
        course_patterns = [
            r'(BSCS|BSIT|BSHM|BSTM|BSOA|BECED|BTLE)',
            r'BS([A-Z]{2,4})',  # BS followed by 2-4 letters
            r'AB([A-Z]{2,4})',  # AB followed by 2-4 letters
        ]
        
        for pattern in course_patterns:
            match = re.search(pattern, filename_upper)
            if match:
                if match.group(0).startswith(('BS', 'AB')):
                    return match.group(0)
                else:
                    return f"BS{match.group(1)}"  # Add BS prefix if missing
        
        return None
    
    def find_guardian_contact_fuzzy(self, row, columns):
        """
        Find guardian contact using fuzzy matching when exact mapping fails
        """
        # Look for columns that might contain guardian contact info
        potential_columns = []
        
        for col in columns:
            col_lower = col.lower()
            # Check for guardian/parent contact indicators
            if any(keyword in col_lower for keyword in ['guardian', 'parent', 'emergency']) and \
            any(contact_word in col_lower for contact_word in ['contact', 'phone', 'mobile', 'tel', 'number']):
                potential_columns.append(col)
        
        # Try to find a phone number in these potential columns
        for col in potential_columns:
            if col in row.index and pd.notna(row[col]):
                value = str(row[col]).strip()
                # Check if it looks like a phone number (contains digits and is reasonable length)
                if re.search(r'\d{7,}', value):  # At least 7 digits
                    return value
        
        # Fallback: look for any column with "guardian" or "parent" that has a phone-like pattern
        for col in columns:
            col_lower = col.lower()
            if ('guardian' in col_lower or 'parent' in col_lower) and col in row.index and pd.notna(row[col]):
                value = str(row[col]).strip()
                if re.search(r'\d{7,}', value):
                    return value
        
        return None
    
    
    def debug_search(self, query):
        """Debug search to see what's happening"""
        print(f"\n🔧 DEBUG: Searching for '{query}'")
        
        for name, collection_obj in self.collections.items():
            print(f"\n🔧 Checking collection: {name}")
            try:
                # Get a sample document to see structure
                sample = collection_obj.get(limit=1)
                if sample["documents"]:
                    print(f"🔧 Sample document preview: {sample['documents'][0][:200]}...")
                    print(f"🔧 Sample metadata: {sample['metadatas'][0] if sample['metadatas'] else 'No metadata'}")
                
                # Try a simple query
                results = collection_obj.query(query_texts=[query], n_results=1)
                if results["documents"] and results["documents"][0]:
                    print(f"🔧 ChromaDB found result with distance: {results['distances'][0][0]}")
                else:
                    print(f"🔧 ChromaDB found no results")
                    
            except Exception as e:
                print(f"🔧 Error in collection {name}: {e}")
    
    def debug_simple_search(self, query):
        """Simple debug search"""
        print(f"🔧 Simple search for: {query}")
        
        for name, collection in self.collections.items():
            try:
                results = collection.query(query_texts=[query], n_results=1)
                if results["documents"] and results["documents"][0]:
                    print(f"✅ Found in {name}: {results['documents'][0][0][:100]}...")
                    print(f"   Distance: {results['distances'][0][0]}")
                else:
                    print(f"❌ Nothing in {name}")
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
    
# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 SMART STUDENT DATA SYSTEM")
    print("="*60)
    print("📚 Features:")
    print("   • Universal Data Extraction (Any Format)")
    print("   • Smart Hierarchical Organization") 
    print("   • Student Data (Excel & PDF)")
    print("   • COR Schedules (Excel & PDF)")
    print("   • Faculty Data & Schedules")
    print("   • Intelligent Search & Query")
    print("="*60)
    
    system = SmartStudentDataSystem()
    
    # Check existing data, don't auto-load
    existing = system.check_existing_data()
    
    if existing:
        # Load existing collections into memory
        for collection_info in existing:
            try:
                collection = system.client.get_collection(
                    name=collection_info.name, 
                    embedding_function=system.embedding_function
                )
                system.collections[collection.name] = collection
            except Exception as e:
                print(f"⚠️ Could not load existing collection {collection_info.name}: {e}")
                
        print(f"\n🚀 Ready to query! Found {len(existing)} data collections.")
    else:
        print("\n📂 No existing data found.")
        print("💡 Use 'Load More Data' option in the menu to add files.")
    
    # 🆕 ALWAYS set data_loaded = True so user can access the menu
    system.data_loaded = True
    system.run_query_interface()  # ✅ Fixed: Use main_menu() instead of run_query_interface()