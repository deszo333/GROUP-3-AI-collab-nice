from __future__ import annotations
import json, re, time, os
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime


# -------------------------------
# LLM Service (with retries)
# -------------------------------


class TrainingSystem:
    def __init__(self, training_file: str = "training_data.json"):
        self.training_file = training_file
        self.training_data = self._load_training_data()
        
    def _load_training_data(self) -> dict:
        """Load existing training data or create new structure."""
        try:
            with open(self.training_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "successful_queries": [],
                "failed_queries": [],
                "query_patterns": {},
                "metadata": {"created": datetime.now().isoformat(), "version": "1.0"}
            }
    
    def _save_training_data(self):
        """Save training data to file."""
        with open(self.training_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2, ensure_ascii=False)
    
    def record_query_result(self, query: str, plan: dict, results_count: int, 
                            success: bool, execution_time: float, error_msg: str = None):
        """Record a query execution result for training."""
        record = {
            "query": query,
            "plan": plan,
            "results_count": results_count,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error_message": error_msg
        }
        
        if success and results_count > 0:
            self.training_data["successful_queries"].append(record)
        else:
            self.training_data["failed_queries"].append(record)
        
        # Extract and store query patterns
        self._extract_query_patterns(query, plan, success)
        self._save_training_data()
    
    def _extract_query_patterns(self, query: str, plan: dict, success: bool):
        """Extract patterns from queries for learning."""
        query_lower = query.lower()
        
        # Extract key patterns
        patterns = {
            "has_year_filter": any(year in query_lower for year in ['1st', '2nd', '3rd', '4th', 'year 1', 'year 2']),
            "has_program_filter": any(prog in query_lower for prog in ['bscs', 'bstm', 'computer science', 'tourism']),
            "is_random_request": 'random' in query_lower,
            "is_multi_condition": any(word in query_lower for word in ['and', 'or', 'both']),
            "has_name_search": any(char.isupper() for char in query if char.isalpha()),
            "plan_steps": len(plan.get('plan', [])) if isinstance(plan, dict) else 0
        }
        
        pattern_key = f"year:{patterns['has_year_filter']}_prog:{patterns['has_program_filter']}_rand:{patterns['is_random_request']}_multi:{patterns['is_multi_condition']}"
        
        if pattern_key not in self.training_data["query_patterns"]:
            self.training_data["query_patterns"][pattern_key] = {
                "successful": 0, "failed": 0, "examples": []
            }
        
        if success:
            self.training_data["query_patterns"][pattern_key]["successful"] += 1
        else:
            self.training_data["query_patterns"][pattern_key]["failed"] += 1
        
        # Keep only recent examples (max 5)
        examples = self.training_data["query_patterns"][pattern_key]["examples"]
        examples.append({"query": query, "success": success})
        if len(examples) > 5:
            examples.pop(0)
    
    def get_training_insights(self) -> str:
        """Generate insights from training data."""
        total_success = len(self.training_data["successful_queries"])
        total_failed = len(self.training_data["failed_queries"])
        success_rate = total_success / (total_success + total_failed) * 100 if (total_success + total_failed) > 0 else 0
        
        insights = [
            f"📊 Training Summary:",
            f"   • Success Rate: {success_rate:.1f}% ({total_success}/{total_success + total_failed})",
            f"   • Successful Queries: {total_success}",
            f"   • Failed Queries: {total_failed}",
            "",
            "🔍 Pattern Analysis:"
        ]
        
        for pattern, data in self.training_data["query_patterns"].items():
            total = data["successful"] + data["failed"]
            pattern_success = data["successful"] / total * 100 if total > 0 else 0
            insights.append(f"   • {pattern}: {pattern_success:.1f}% success ({data['successful']}/{total})")
        
        return "\n".join(insights)
    
    def suggest_plan_improvements(self, query: str) -> Optional[dict]:
        """Suggest plan improvements based on training data."""
        query_lower = query.lower()
        
        # Check for common failure patterns
        if 'random' in query_lower and ('and' in query_lower or 'or' in query_lower):
            return {
                "suggestion": "For random queries with multiple conditions, use separate steps instead of complex filters",
                "recommended_approach": "Split into individual searches per condition"
            }
        
        return None
    
    
class LLMService:
    def __init__(self, config: dict):
        self.api_mode = config.get('api_mode', 'online')
        self.debug_mode = config.get('debug_mode', False)
        self.mistral_api_key = config.get('mistral_api_key')
        self.mistral_api_url = config.get('mistral_api_url', 'https://api.mistral.ai/v1/chat/completions')
        self.ollama_api_url = config.get('ollama_api_url', 'http://localhost:11434/api/chat')
        self.planner_model = config.get('planner_model')
        self.synth_model   = config.get('synth_model')

    def _prepare_request(self, messages: list, json_mode: bool, phase: str = "planner"):
        headers, payload, api_url = {}, {}, ""
        model_override = self.planner_model if phase == "planner" else self.synth_model

        if self.api_mode == 'online':
            api_url = self.mistral_api_url
            headers = {"Authorization": f"Bearer {self.mistral_api_key}", "Content-Type": "application/json"}
            payload = {"model": model_override or "mistral-small-latest", "messages": messages}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
        else:
            api_url = self.ollama_api_url
            headers = {"Content-Type": "application/json"}
            payload = {"model": model_override or "mistral:instruct", "messages": messages, "stream": False}
            # FIX: Add proper JSON mode instruction for Ollama
            if json_mode:
                payload["format"] = "json"
                # Make the instruction more forceful
                if messages and messages[0].get("role") == "system":
                    # ✨ REVISED INSTRUCTION
                    messages[0]["content"] += (
                        "\n\nIMPORTANT: Your response MUST be a single, valid JSON object and nothing else. "
                        "Do not include any text, explanations, or markdown formatting before or after the JSON."
                    )
        return api_url, headers, payload

    def execute(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False,
                history: Optional[List[dict]] = None, retries: int = 2, phase: str = "planner") -> str:
    
        # --- ✨ CORRECTED MESSAGE ORDERING START ✨ ---
        # The system prompt must always be the first message in the list.
        messages = [{"role": "system", "content": system_prompt}]
        
        # The conversation history comes after the system prompt.
        if history:
            messages.extend(history)
        
        # The new user query is always the last message.
        messages.append({"role": "user", "content": user_prompt})
        # --- ✨ CORRECTED MESSAGE ORDERING END ✨ ---

        api_url, headers, payload = self._prepare_request(messages, json_mode, phase=phase)
        if not api_url:
            return "Configuration Error: API URL is not set."

        if self.debug_mode:
            print(f"🧠 LLMService → {self.api_mode.upper()} | phase={phase} | json={json_mode}")

        last_err = None
        for attempt in range(retries + 1):
            try:
                # The payload now correctly uses the ordered 'messages' list
                payload["messages"] = messages 
                resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
                resp.raise_for_status()
                rj = resp.json()
                if 'choices' in rj and rj['choices']:
                    return rj['choices'][0]['message']['content'].strip()
                if 'message' in rj and 'content' in rj['message']:
                    return rj['message']['content'].strip()
                raise ValueError("No content in LLM response")
            except Exception as e:
                last_err = e
                if self.debug_mode:
                    print(f"⚠️ LLM attempt {attempt+1}/{retries+1} failed: {e}")
                if attempt < retries:
                    time.sleep(1)
                    
        return f"Error: Could not connect to the AI service. Details: {last_err}"

# -------------------------------
# Prompts
# -------------------------------
PROMPT_TEMPLATES = {
    "planner_agent": r"""
            You are a **Planner AI**.  
            Your ONLY job is to map a user query to ONE tool call.  
            You must ALWAYS respond with a **single valid JSON object**.

        AVAILABLE TOOLS:
        
        
        1. get_person_schedule(person_name: str = null,program: str = null,year_level: int = null,section: str = null)
            - Use when the query is about schedules.  
            - If about an individual → use `person_name`.  
            - If about a group (e.g., "schedules of BSIT students", "2nd year BSIT A schedule") → use `program`, `year_level`, and/or `section`.

        2. get_person_schedule(person_name: str): Use ONLY when the user's main goal is to retrieve a full class or work schedule.
        3. answer_question_about_person(person_name: str, question: str): Use for any other specific questions about a named person, like "what is their contact number?", "what is their student ID?", or "what is their employment status?".
        4. get_adviser_info(program: str, year_level: int): Use when the user asks who the adviser is for a group.
        5. list_students(program: str, year_level: int): Use to get a list of students.
        6. find_faculty_by_class_count(find_most: bool): Use for "who has the most" or "fewest" classes.
        7. verify_student_adviser(student_name: str, adviser_name: str): Use to fact-check if a student is advised by a specific faculty member.
        8. compare_schedules(person_a_name: str, person_b_name: str): Use for schedule conflicts.
        9. search_database(query_text: str): A general-purpose search for simple "who is..." lookups that don't fit other tools.
        10. get_student_grades(student_name: str = None, program: str = None, year_level: int = None): Use for ANY query about student grades. Can find grades for a single student by name, or for a group by program, year level, or both. Can be used to answer questions intelligece, smartest students, etc.
        11. find_people(role: str, employment_status: str): A flexible tool to find faculty/staff. THIS IS YOUR PRIMARY PROBLEM-SOLVING TOOL.
            - IF the user mentions 'books', 'research', or 'library', you MUST set the `role` to 'Librarian'.
            - IF the user mentions 'electricity', 'maintenance', or 'broken', you MUST set the `role` to 'Maintenance Tech'.
            - IF the user asks about 'permanent' or 'contractual' staff, you MUST use the `employment_status` parameter.
        ---
        EXAMPLE 1 (Specific Question):
        User Query: "what is the contact number of Lee Pace?"
        Your JSON Response:
        {{
            "tool_name": "answer_question_about_person",
            "parameters": {{
                "person_name": "Lee Pace",
                "question": "What is his contact number?"
            }}
        }}
        ---
        EXAMPLE 2 (Schedule Question):
        User Query: "what is the schedule of Lee Pace?"
        Your JSON Response:
        {{
            "tool_name": "get_person_schedule",
            "parameters": {{
                "person_name": "Lee Pace"
            }}
        }}
        ---
        CRITICAL FINAL INSTRUCTION:
        Your entire response MUST be a single, raw JSON object containing "tool_name" and "parameters". Start immediately with `{{` and end with `}}`.
    """,
    "final_synthesizer": r"""
        You are an AI Analyst. Your answer must be based ONLY on the "Factual Documents" provided.

        INSTRUCTIONS:
        - Synthesize information from all documents to create a complete answer.
        - **Entity Linking Rule (CRITICAL):** You must actively try to link entities across documents. If one document mentions 'Dr. Deborah' as an adviser and another document lists a faculty member named 'Deborah K. Lewis', you MUST assume they are the same person. Synthesize their information into a single, coherent description and state the connection clearly (e.g., "The adviser, Dr. Deborah, is Professor Deborah K. Lewis."). Do not present them as two different people unless the documents give conflicting information.
        - Infer logical connections. For example, if a student document and a class schedule share the same program, year, and section, you MUST state that the schedule applies to that student.
        - **Name Interpretation Rule:** When a user asks about a person using a single name (e.g., "who is Lee"), you must summarize information for all individuals where that name appears as a first OR last name. If you find a match on a last name (e.g., "Michelle Lee"), you MUST include this person in your summary and clarify their role. Do not restrict your answer to only first-name matches.
        - If data is truly missing, state that clearly.
        - Cite the source_collection for key facts using [source_collection_name].
        - If status is 'empty': Do NOT say "status empty". Instead, use the 'summary' to inform the user conversationally that you couldn't find information. You can suggest an alternative query.
        - If status is 'error': Do NOT show the technical error message. Instead, use the 'summary' to apologize for the technical difficulty in a simple, user-friendly way.
        - Be conversational and natural in your response.
        - If the data is complete, provide the full list of that

        ---
        Factual Documents:
        {context}
        ---
        User's Query:
        {query}
        ---
        Your concise analysis (with citations):
    """
}

# -------------------------------
# AIAnalyst (Planner + Synthesizer)
# -------------------------------
class AIAnalyst:
    def __init__(self, collections: Dict[str, Any], llm_config: Optional[dict] = None):
        self.collections = collections or {}
        self.debug_mode = bool((llm_config or {}).get("debug_mode", False))
        self.llm = LLMService(llm_config or {})
        self.db_schema_summary = "Schema not generated yet."
        self.REVERSE_SCHEMA_MAP = self._create_reverse_schema_map()
        self._generate_db_schema()
        self.training_system = TrainingSystem()
        self.dynamic_examples = self._load_dynamic_examples()
        self.available_tools = {
            "get_person_schedule": self.get_person_schedule,
            "get_adviser_info": self.get_adviser_info,
            "find_faculty_by_class_count": self.find_faculty_by_class_count,
            "verify_student_adviser": self.verify_student_adviser,
            "search_database": self.search_database,
            "resolve_person_entity": self.resolve_person_entity,
            "list_students": self.list_students, # <-- ADD THIS LINE
            "find_people": self.find_people, # <-- ADD THIS LINE
            "compare_schedules": self.compare_schedules, # <-- ADD THIS LINE
            "answer_question_about_person": self.answer_question_about_person, # <-- NEW
            "get_student_grades": self.get_student_grades # <-- NEW,
            }
        
        
        
    # ✨ ADD/REPLACE THESE METHODS IN YOUR AIAnalyst CLASS
    
    def get_student_grades(self, student_name: str = None, program: str = None, year_level: int = None) -> List[dict]:

        """
        Finds grade documents for a specific student by name, OR for a group of students
        by any combination of program and/or year level.
        """
        self.debug(f"🛠️ Running final grade tool for name='{student_name}', program='{program}', year='{year_level}'")

        # --- PRIORITY 1: Handle search by a specific student's name ---
        if student_name:
            self.debug(f"-> Prioritizing search by name: {student_name}")
            entity = self.resolve_person_entity(name=student_name)
            if not entity or not entity.get("primary_document"):
                return [{"status": "error", "summary": f"Could not find a student named '{student_name}'."}]
            
            student_id = entity["primary_document"].get("metadata", {}).get("student_id")
            if not student_id:
                return [{"status": "error", "summary": f"Found '{student_name}' but they are missing a student ID."}]
                
            grade_docs = self.search_database(filters={"student_id": student_id}, collection_filter="_grades")
            if not grade_docs:
                return [entity["primary_document"], {"status": "empty", "summary": f"Found student '{student_name}' but could not find any grade information for them."}]
            
            return [entity["primary_document"]] + grade_docs

        # --- PRIORITY 2: Handle search by a group (program and/or year) ---
        if program or year_level:
            self.debug(f"-> Searching for group: program={program}, year_level={year_level}")
            student_filters = {}
            if program:
                student_filters['program'] = program
            if year_level:
                student_filters['year_level'] = year_level

            student_docs = self.list_students(**student_filters)
            if not student_docs or "status" in (student_docs[0] or {}).get("metadata", {}):
                return [{"status": "empty", "summary": f"I couldn't find any students matching those criteria."}]
                
            student_ids = [doc.get("metadata", {}).get("student_id") for doc in student_docs if doc.get("metadata", {}).get("student_id")]
            if not student_ids:
                return [{"status": "empty", "summary": "Found students, but they are missing IDs needed to find grades."}]

            grade_filters = {"student_id": {"$in": student_ids}}
            grade_docs = self.search_database(filters=grade_filters, collection_filter="_grades")
            
            
            if not grade_docs:
                return student_docs + [{"status": "empty", "summary": "Could not find any grade information for the specified students."}]

            return student_docs + grade_docs
            
            
            
        if not student_name and not program and not year_level:
            self.debug("-> No filters provided. Retrieving all grade documents.")
            all_grade_docs = self.search_database(collection_filter="_grades")
            if not all_grade_docs:
                return [{"status": "empty", "summary": "I could not find any grade documents in the database."}]
            return all_grade_docs

            

        # --- FINAL FALLBACK ---
        return [{"status": "error", "summary": "To get grades, please provide a specific student's name, a program, or a year level."}]
    
    
    
    
    
    
    def answer_question_about_person(self, person_name: str, question: str) -> List[dict]:
        """
        Finds all documents for a specific person and then uses the Synthesizer
        to answer a specific question based on that person's data.
        """
        self.debug(f"🛠️ Running QA tool: Answering '{question}' for '{person_name}'")

        # Step 1: Use our most robust tool to find all documents for the person.
        person_docs = self.search_database(query=person_name)
        if not person_docs:
            return [{"status": "empty", "summary": f"I could not find any information for a person named '{person_name}'."}]
        
        # Step 2: Create a temporary, focused context for the Synthesizer.
        context_for_qa = json.dumps({
            "status": "success",
            "data": person_docs
        }, indent=2, ensure_ascii=False)
        
        # Step 3: Create a specific prompt to force the AI to answer ONLY the question.
        qa_prompt = PROMPT_TEMPLATES["final_synthesizer"].format(
            context=context_for_qa,
            # We use the specific question here, not the original user query
            query=question
        )

        # Step 4: Call the LLM to perform the specific Question-Answering task.
        specific_answer = self.llm.execute(
            system_prompt="You are a helpful assistant that answers specific questions based ONLY on the provided Factual Documents.",
            user_prompt=qa_prompt,
            phase="synth" # Use the synthesizer model
        )

        # Step 5: Return the specific answer as the tool's result.
        # The 'content' is the answer, and we pass along the original docs as evidence.
        return [
            {"source_collection": "qa_result", "content": specific_answer, "metadata": {}}
        ] + person_docs
        
    
    
    
    def find_people(self, name: str = None, role: str = None, department: str = None, employment_status: str = None) -> List[dict]:
        self.debug(f"🛠️ Running tool: find_people with filters: role={role}, status={employment_status}")
        filters = {}
        if role: filters['position'] = role
        if department: filters['department'] = department
        if employment_status: filters['employment_status'] = employment_status
        if not filters and not name: return [{"status": "error", "summary": "Please provide criteria to find people."}]
        return self.search_database(query_text=name, filters=filters, collection_filter="faculty")

    def compare_schedules(self, person_a_name: str, person_b_name: str) -> List[dict]:
        """Compares the schedules of two different people to find conflicts or similarities."""
        self.debug(f"🛠️ Running smart tool: compare_schedules for '{person_a_name}' and '{person_b_name}'")
        
        # Get all available documents for Person A using the unified tool
        docs_a = self.get_person_schedule(person_name=person_a_name)

        # Get all available documents for Person B using the unified tool
        docs_b = self.get_person_schedule(person_name=person_b_name)
        
        # Combine all retrieved documents for the synthesizer to analyze
        return docs_a + docs_b
    
    
    def list_students(self, program: str = None, year_level: int = None, section: str = None) -> List[dict]:
        """Lists students based on optional filters for program, year_level, and section."""
        self.debug(f"🛠️ Running smart tool: list_students with filters program={program}, year_level={year_level}, section={section}")
        
        filters = {}
        if program:
            filters['program'] = program
        if year_level:
            filters['year_level'] = year_level
        if section:
            filters['section'] = section
            
        if not filters:
            # This prevents returning the entire student database on a vague query like "list students"
            return [{"status": "empty", "summary": "Please specify a program or year level to list students."}]

        return self.search_database(filters=filters, collection_filter="students")

    def get_person_schedule(self, person_name: str = None, program: str = None, year_level: int = None, section: str = None) -> List[dict]:
        """
        Unified schedule tool:
        - If given a specific person_name → resolve entity and get their schedule.
        - If given group filters (program/year/section) → get group schedules.
        """

        self.debug(f"🛠️ Running schedule tool for person='{person_name}', program={program}, year={year_level}, section={section}")

        # --- CASE 1: Group query (program/year/section provided, or 'student' keyword without a specific name) ---
        if program or year_level or section or (person_name and "student" in person_name.lower() and len(person_name.split()) <= 2):
            filters = {}
            if program: filters["program"] = program
            if year_level: filters["year_level"] = year_level
            if section: filters["section"] = section

            # Try to infer program if query was like "BSIT student"
            if person_name and "student" in person_name.lower() and not program:
                prog_guess = person_name.split()[0].upper()
                filters["program"] = prog_guess

            self.debug(f"-> Running group schedule search with filters={filters}")
            schedule_docs = self.search_database(filters=filters, collection_filter="schedules")

            if not schedule_docs:
                return [{"status": "empty", "summary": "No schedules found for the specified group."}]
            return schedule_docs

        # --- CASE 2: Specific person (normal behavior) ---
        if person_name:
            self.debug(f"-> Resolving entity for person: {person_name}")
            entity = self.resolve_person_entity(name=person_name)
            if not entity or not entity.get("primary_document"):
                return [{"status": "error", "summary": f"Could not find anyone matching '{person_name}'."}]
            
            person_record = entity["primary_document"]
            primary_name = entity["primary_name"]
            aliases = entity["aliases"]

            if primary_name not in aliases:
                aliases = [primary_name] + aliases

            meta = person_record.get("metadata", {})
            source_collection = person_record.get("source_collection", "")
            self.debug(f"-> Precisely identified '{primary_name}' via entity resolution.")

            if "student" in source_collection:
                schedule_filters = {
                    "program": meta.get("program") or meta.get("course"),
                    "year_level": meta.get("year_level") or meta.get("year"),
                    "section": meta.get("section")
                }
                if not all(schedule_filters.values()):
                    return [person_record, {"status": "error", "summary": "Student record is missing key details."}]
                schedule_docs = self.search_database(filters=schedule_filters, collection_filter="schedules")

            elif "faculty" in source_collection:
                schedule_filters = {
                    "$or": [
                        {"adviser": {"$in": aliases}},
                        {"staff_name": {"$in": aliases}},
                        {"full_name": {"$in": aliases}}
                    ]
                }
                schedule_docs = self.search_database(filters=schedule_filters, collection_filter="schedules")

                if not schedule_docs:
                    schedule_docs = self.search_database(filters=schedule_filters, collection_filter="faculty_library_non_teaching_schedule")

            else:
                return [person_record, {"status": "error", "summary": "Could not determine if they are a student or faculty."}]

            if not schedule_docs:
                return [person_record, {"status": "empty", "summary": f"Found {primary_name} but could not find a matching schedule."}]

            return [person_record] + schedule_docs

        # --- FINAL FALLBACK ---
        return [{"status": "error", "summary": "Please provide a person's name or a group filter (program, year, section)."}]


    def get_adviser_info(self, program: str, year_level: int) -> List[dict]:
        """Finds the adviser for a group of students and retrieves their faculty profile."""
        self.debug(f"🛠️ Running smart tool: get_adviser_info for {program} Year {year_level}")
        
        schedule_docs = self.search_database(filters={"program": program, "year_level": year_level}, collection_filter="schedules")
        if not schedule_docs or "adviser" not in schedule_docs[0].get("metadata", {}):
            return [{"status": "empty", "summary": f"Could not find a schedule or adviser for {program} Year {year_level}."}]
        
        adviser_name = schedule_docs[0]["metadata"]["adviser"]
        adviser_profile = self.resolve_person_entity(name=adviser_name)
        faculty_doc = self.search_database(query=adviser_profile["primary_name"], collection_filter="faculty")
        
        return schedule_docs + faculty_doc

    def find_faculty_by_class_count(self, find_most: bool = True) -> List[dict]:
        """Finds the faculty who teaches the most or fewest subjects."""
        self.debug(f"🛠️ Running smart tool: find_faculty_by_class_count (find_most={find_most})")
        
        schedule_docs = self.search_database(collection_filter="schedules", query_text="class schedule")
        if not schedule_docs:
            return [{"status": "empty", "summary": "No schedule documents were found to analyze."}]

        adviser_counts = {}
        for doc in schedule_docs:
            meta = doc.get("metadata", {})
            adviser = meta.get("adviser")
            subject_count = meta.get("subject_count", 0)
            if adviser and subject_count > 0:
                adviser_counts[adviser] = adviser_counts.get(adviser, 0) + subject_count

        if not adviser_counts:
            return [{"status": "empty", "summary": "Found schedules, but could not determine adviser counts."}]

        sorted_advisers = sorted(adviser_counts.items(), key=lambda item: item[1], reverse=find_most)
        target_adviser_name, count = sorted_advisers[0]
        
        summary_doc = {
            "source_collection": "analysis_result",
            "content": f"The faculty with the {'most' if find_most else 'fewest'} classes is {target_adviser_name} with {count} subject(s).",
            "metadata": {"status": "success"}
        }
        
        faculty_profile = self.search_database(query=target_adviser_name, collection_filter="faculty")
        
        return [summary_doc] + faculty_profile

    def verify_student_adviser(self, student_name: str, adviser_name: str) -> List[dict]:
        
        """
        Verifies if a given adviser is the correct one for a given student by
        resolving both entities and comparing their aliases for any overlap.
        """
        self.debug(f"🛠️ Running smart tool: verify_student_adviser for '{student_name}' and '{adviser_name}'")
        
        # 1. Get the student's actual schedule to find their official adviser's name.
        student_schedule_docs = self.get_person_schedule(person_name=student_name)
        
        actual_adviser_name = None
        for doc in student_schedule_docs:
            if "schedule" in doc.get("source_collection", ""):
                actual_adviser_name = doc.get("metadata", {}).get("adviser")
                break
                
        if not actual_adviser_name:
            return [{"status": "empty", "summary": f"Could not find an official adviser for {student_name}."}]

        # 2. Resolve BOTH the official adviser and the claimed adviser to get their full identities.
        self.debug(f"   -> Official adviser is '{actual_adviser_name}'. Resolving...")
        official_adviser_entity = self.resolve_person_entity(name=actual_adviser_name)
        
        self.debug(f"   -> Claimed adviser is '{adviser_name}'. Resolving...")
        claimed_adviser_entity = self.resolve_person_entity(name=adviser_name)
        
        # Convert the lists of aliases to sets for easy comparison.
        official_aliases = set(official_adviser_entity.get("aliases", []))
        claimed_aliases = set(claimed_adviser_entity.get("aliases", []))
        
        # 3. Compare the sets of aliases. If there's any name in common, it's a match.
        is_match = not official_aliases.isdisjoint(claimed_aliases)
        
        summary_content = (
            f"Verification result: The claim that {adviser_name} advises {student_name} is {'CORRECT' if is_match else 'INCORRECT'}. "
            f"The official adviser on record is {actual_adviser_name}."
        )
        summary_doc = {"source_collection": "analysis_result", "content": summary_content, "metadata": {"status": "success"}}
        
        # Return the summary and the original documents for full context.
        return [summary_doc] + student_schedule_docs
    def get_distinct_combinations(self, collection_filter: str, fields: List[str], filters: dict) -> dict:
        """
        Finds all unique combinations of values for the given fields 
        in a collection after applying a filter. Useful for finding all 
        year/section pairs for a given program.
        """
        self.debug(f"🛠️ get_distinct_combinations | collection='{collection_filter}' | fields={fields} | filters={filters}")
        
        where_clause = {}
        if filters:
            # This logic is simplified for the tool's purpose.
            # It can be expanded if more complex filters are needed.
            key, value = next(iter(filters.items()))
            standard_key = self.REVERSE_SCHEMA_MAP.get(key, key)
            possible_keys = list(set([standard_key] + [orig for orig, std in self.REVERSE_SCHEMA_MAP.items() if std == standard_key]))
            where_clause = {"$or": [{k: {"$eq": value}} for k in possible_keys]}

        unique_combinations = set()
        
        # Create a map to find original field names from standard ones
        # e.g., "year_level" -> ["year", "yr", "yearlvl"]
        field_map = {
            std_field: list(set([std_field] + [orig for orig, std in self.REVERSE_SCHEMA_MAP.items() if std == std_field]))
            for std_field in fields
        }

        for name, coll in self.collections.items():
            if collection_filter in name:
                try:
                    results = coll.get(where=where_clause, include=["metadatas"])
                    for meta in results.get("metadatas", []):
                        combo_values = []
                        for std_field in fields:
                            found_value = None
                            # Check all possible original keys for the standard field
                            for original_key in field_map[std_field]:
                                if original_key in meta:
                                    found_value = meta[original_key]
                                    break
                            combo_values.append(found_value)
                        
                        combo = tuple(combo_values)
                        if all(item is not None for item in combo):
                            unique_combinations.add(combo)
                except Exception as e:
                    self.debug(f"⚠️ Error during get_distinct_combinations in {name}: {e}")

        combinations_list = [dict(zip(fields, combo)) for combo in sorted(list(unique_combinations))]
        
        self.debug(f"✅ Found {len(combinations_list)} distinct combinations.")
        return {"status": "success", "combinations": combinations_list}
        
    def _fuzzy_name_match(self, name1, name2, threshold=0.5):
        """A simplified fuzzy match for entity resolution within the analyst."""
        if not name1 or not name2:
            return False
        
        # Clean names by removing titles and splitting
        name1_clean = re.sub(r'^(DR|PROF|MR|MS|MRS)\.?\s*', '', name1.upper()).replace(',', '')
        name2_clean = re.sub(r'^(DR|PROF|MR|MS|MRS)\.?\s*', '', name2.upper()).replace(',', '')
        
        name1_parts = set(name1_clean.split())
        name2_parts = set(name2_clean.split())
        
        if not name1_parts or not name2_parts:
            return False
        
        intersection = len(name1_parts.intersection(name2_parts))
        union = len(name1_parts.union(name2_parts))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    # 🆕 NEW TOOL FOR THE AI PLANNER
    # 🆕 REVISED TOOL FOR THE AI PLANNER
    def resolve_person_entity(self, name: str) -> dict:
        """
        Finds all documents for a person using a robust, multi-pronged search
        that handles variations like middle initials.
        """
        self.debug(f"🕵️  Resolving entity for: '{name}'")
        
        # --- ✨ FINAL UPGRADE: MULTI-PRONGED SEARCH ---
        # 1. Prepare multiple search terms to be robust.
        original_query = name.lower()
        
        # Create a version of the name with titles and punctuation (like the "Q.") removed.
        name_clean = re.sub(r'^(DR|PROF|MR|MS|MRS)\.?\s*', '', name.upper()).replace(',', '')
        # Also remove any single-letter words, like middle initials
        name_parts_no_initials = [part for part in name_clean.split() if len(part) > 1]
        cleaned_query = ' '.join(name_parts_no_initials).lower()

        self.debug(f"   -> Performing multi-pronged search for: ['{original_query}', '{cleaned_query}']")

        # 2. Perform a search for each variation to cast a wide net.
        results1 = self.search_database(query=original_query)
        results2 = self.search_database(query=cleaned_query)

        # 3. Combine and de-duplicate all results.
        all_results = results1 + results2
        initial_results = list({doc['content']: doc for doc in all_results}.values())
        # --- ✨ END OF UPGRADE ✨ ---
        
        if not initial_results:
            return {} # Return empty if no documents are found

        # The rest of the function's logic for gathering and fuzzy-matching aliases is correct.
        potential_names, primary_name = {name.title()}, name.title()
        for result in initial_results:
            meta = result.get('metadata', {})
            fields = ['full_name', 'adviser', 'staff_name']
            for field in fields:
                if meta.get(field): potential_names.add(str(meta[field]).strip().title())
                
        resolved_aliases = {primary_name}
        for p_name in potential_names:
            if self._fuzzy_name_match(primary_name, p_name):
                resolved_aliases.add(p_name)
                if len(p_name) > len(primary_name): primary_name = p_name

        # Logic to find the best primary document
        best_doc = None
        longest_name_len = 0
        for doc in initial_results:
            full_name = doc.get("metadata", {}).get("full_name", "")
            if full_name in resolved_aliases and len(full_name) > longest_name_len:
                best_doc = doc
                longest_name_len = len(full_name)
        
        if not best_doc and initial_results:
            best_doc = initial_results[0]
            
        final_primary_name = best_doc.get("metadata", {}).get("full_name", primary_name)

        self.debug(f"✅ Entity resolved: Primary='{final_primary_name}', Aliases={list(resolved_aliases)}")
        return {
            "primary_name": final_primary_name,
            "aliases": list(resolved_aliases),
            "primary_document": best_doc
        }
        


    def debug(self, *args):
        if self.debug_mode:
            print(*args)
            
            
    def _load_dynamic_examples(self) -> str:
        """Loads training examples from a JSON file, returns as a formatted string."""
        file_path = "dynamic_examples.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                example_strings = []
                for example in data.get("examples", []):
                    example_str = f"""
        
        **EXAMPLE (User-Provided):**
        User Query: "{example['query']}"
        Your JSON Response:
        {json.dumps(example['plan'], indent=2, ensure_ascii=False)}
        """
                    example_strings.append(example_str)
                return "".join(example_strings)
        except FileNotFoundError:
            self.debug(f"⚠️ {file_path} not found. Starting with no dynamic examples.")
            return ""
        except json.JSONDecodeError:
            self.debug(f"❌ Error decoding {file_path}. Starting with no dynamic examples.")
            return ""

    def _save_dynamic_example(self, query: str, plan: dict):
        """Adds a new example to the JSON file."""
        file_path = "dynamic_examples.json"
        data = {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"examples": []}

        # Check for duplicate
        for ex in data["examples"]:
            if ex["query"] == query:
                self.debug("Duplicate query found. Not saving.")
                return

        data["examples"].append({"query": query, "plan": plan})

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.debug("✅ New training example saved to dynamic_examples.json.")

    def _repair_json(self, text: str) -> Optional[dict]:
        if not text: return None
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m: return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    def _create_reverse_schema_map(self) -> dict:
        """Creates a map from standard names to possible original names."""
        mappings = {
            'program': ('course',),
            'year_level': ('year', 'yr', 'yearlvl'),
            'full_name': ('name', 'student_name'),
            'section': ('sec',),
            'adviser': ('advisor', 'faculty'),
            'student_id': ('stud_id', 'id', 'student_number')
        }
        reverse_map = {}
        for standard_name, original_names in mappings.items():
            for original_name in original_names:
                reverse_map[original_name] = standard_name
        return reverse_map

    def _normalize_schema(self, schema_dict: dict) -> dict:
        """Uses the reverse map to standardize field names for the AI."""
        def std(field: str) -> str:
            return self.REVERSE_SCHEMA_MAP.get(field.lower(), field)
            
        norm = {}
        for coll, fields in schema_dict.items():
            norm[coll] = sorted(list({std(f) for f in fields}))
        return norm

    def _generate_db_schema(self):
        if not self.collections:
            self.db_schema_summary = "No collections loaded."
            return

        FIELDS_TO_HINT = ['position', 'department', 'program', 'faculty_type', 'admin_type', 'employment_status']
        HINT_LIMIT = 7
        
        raw = {}
        value_hints = {}

        for name, coll in self.collections.items():
            try:
                sample = coll.get(limit=100, include=["metadatas"])

                if sample and sample.get("metadatas") and sample["metadatas"]:
                    
                    metadatas_list = sample["metadatas"]
                    raw[name] = list(metadatas_list[0].keys())
                    value_hints[name] = {}

                    for field in FIELDS_TO_HINT:
                        unique_values = set()
                        for meta in metadatas_list:
                            if field in meta and meta[field]:
                                unique_values.add(str(meta[field]))
                        
                        if unique_values:
                            hint_list = sorted(list(unique_values))
                            value_hints[name][field] = hint_list[:HINT_LIMIT]
                else:
                    raw[name] = []
            
            except Exception as e:
                self.debug(f"Schema inspect failed for {name}: {e}")
                raw[name] = []

        norm = self._normalize_schema(raw)
        
        schema_hints = {
            "subjects_by_year": '(format: a dictionary string, not filterable by year)'
        }
        
        parts = []
        for name, fields in norm.items():
            described_fields = [f"{field} {schema_hints[field]}" if field in schema_hints else field for field in fields]
            parts.append(f"- {name}: {described_fields}")

            if name in value_hints and value_hints[name]:
                hint_parts = []
                for field, values in value_hints[name].items():
                    hint_parts.append(f"'{field}' can be {values}")
                if hint_parts:
                    parts.append(f"   (Hint: {', '.join(hint_parts)})")

        self.db_schema_summary = "\n".join(parts)
        self.debug("✅ DB Schema for planner:\n", self.db_schema_summary)
        
        
        
        
        
        
    

    def _resolve_placeholders(self, params: dict, step_results: dict) -> dict:
        """Recursively search for and replace placeholders, aware of schema normalization."""
        resolved_params = json.loads(json.dumps(params))

        # Map standard -> originals
        forward_map = {}
        for original, standard in self.REVERSE_SCHEMA_MAP.items():
            forward_map.setdefault(standard, []).append(original)

        def normalize_for_search(key: str, value: Any):
            """
            Turn a single scalar into a forgiving filter dict for ChromaDB.
            This version simplifies the output to avoid overly complex `$in` lists.
            """
            COURSE_ALIASES = {
                "BSCS": ["BSCS", "BS COMPUTER SCIENCE", "BS Computer Science"],
                "BSTM": ["BSTM", "BS TOURISM MANAGEMENT", "BS Tourism Management"],
                "BSOA": ["BSOA", "BS OFFICE ADMINISTRATION", "BS Office Administration"],
                "BECED": ["BECED", "BACHELOR OF EARLY CHILDHOOD EDUCATION", "Bachelor of Early Childhood Education"],
                "BSIT": ["BSIT", "BS INFORMATION TECHNOLOGY", "BS Information Technology"],
                "BSHM": ["BSHM", "BS HOSPITALITY MANAGEMENT", "BS Hospitality Management"],
                "BTLE": ["BTLE", "BACHELOR OF TECHNOLOGY AND LIVELIHOOD EDUCATION", "Bachelor of Technology and Livelihood Education"]
            }
            
            # If the placeholder already produced an operator dict, pass it through
            if isinstance(value, dict):
                if any(op in value for op in ("$in", "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$nin")):
                    return value

            # From here, treat 'value' as a single scalar and expand to variants
            scalars: List[Any] = [value] if value is not None else []
            out: List[Any] = []

            if key == "program":
                for v in scalars:
                    v_str_u = str(v).upper()
                    matched = False
                    for prog_key, alias_list in COURSE_ALIASES.items():
                        alias_upper = [a.upper() for a in alias_list]
                        if v_str_u == prog_key or v_str_u in alias_upper:
                            out.extend(alias_list)
                            matched = True
                            break
                    if not matched:
                        out.append(v)
                # Ensure all values are strings for ChromaDB's `$in` operator
                return {"$in": [str(x) for x in list(dict.fromkeys(out))]}

            if key == "year_level":
                     
                for v in scalars:
                    vs = str(v).strip()
                    out.extend([
                        vs,
                        f"Year {vs}",
                        f"{vs}st Year", f"{vs}nd Year", f"{vs}rd Year", f"{vs}th Year"
                    ])
                    if vs == "1": out.extend(["1st Year", "First Year", "Year I"])
                    if vs == "2": out.extend(["2nd Year", "Second Year", "Year II"])
                    if vs == "3": out.extend(["3rd Year", "Third Year", "Year III"])
                    if vs == "4": out.extend(["4th Year", "Fourth Year", "Year IV"])
                return {"$in": list(dict.fromkeys(out))}
            
            if key == "section":
                for v in scalars:
                    vs = str(v).upper().strip()
                    out.extend([vs, f"SEC {vs}", f"Section {vs}"])
                return {"$in": [str(x) for x in list(dict.fromkeys(out))]}

            # Default: return as-is (scalar $eq), ensuring it's a string
            return {"$eq": str(value)}

        def resolve(obj):
            if isinstance(obj, dict):
                for k, v_item in list(obj.items()):
                    obj[k] = resolve(v_item)
            elif isinstance(obj, list):
                for i, item in enumerate(list(obj)):
                    obj[i] = resolve(item)
            elif isinstance(obj, str) and obj.startswith('$'):
                parts = obj.strip('$').split('_from_step_')
                if len(parts) == 2:
                    key_to_find, step_num_str = parts
                    step_num = int(step_num_str)
                    self.debug(f"   -> Resolving placeholder: looking for '{key_to_find}' in results of step {step_num}")
                    if step_num in step_results and step_results[step_num]:
                        step_result = step_results[step_num]
                        # Check if the result is a dictionary (from resolve_person_entity)
                    if isinstance(step_result, dict):
                        if key_to_find in step_result:
                            return step_result[key_to_find] # Return the value (e.g., the aliases list) directly

                    # Otherwise, assume it's a list of docs (from search_database)
                    elif isinstance(step_result, list) and len(step_result) > 0:
                        metadata = step_result[0].get("metadata", {})
                        if key_to_find in metadata:
                            # We don't need normalize_for_search here because the prompt example
                            # for students just uses the direct value.
                            return metadata[key_to_find]
                    # --- ✨ CORRECTED LOGIC END ✨ ---

                        if key_to_find in metadata:
                            return normalize_for_search(key_to_find, metadata[key_to_find])
                        
                        for original_key in forward_map.get(key_to_find, []):
                            if original_key in metadata:
                                self.debug(f"   -> Found value using original key '{original_key}' for standard key '{key_to_find}'")
                                return normalize_for_search(key_to_find, metadata[original_key])
            return obj

        return resolve(resolved_params)
    
    
    

    def search_database(self, query_text: Optional[str] = None, query: Optional[str] = None, 
                        filters: Optional[dict] = None, document_filter: Optional[dict] = None, 
                        collection_filter: Optional[str] = None, **kwargs) -> List[dict]:
        """
        Searches the database with your original, robust normalization logic, now correctly
        handling pre-built complex filters from smart tools.
        """
        qt = query or query_text
        final_query_texts: Optional[List[str]] = None
        if isinstance(qt, list):
            final_query_texts = qt
        elif isinstance(qt, str):
            final_query_texts = [qt]

        self.debug(f"🔎 search_database | query(s)='{final_query_texts}' | filters={filters} | doc_filter={document_filter} | coll_filter='{collection_filter}'")
        all_hits: List[dict] = []

        where_clause: Optional[dict] = None
        if filters:
            # --- ✨ THIS IS THE FIX, INTEGRATED WITH YOUR CODE ✨ ---
            if '$or' in filters and isinstance(filters.get('$or'), list):
                where_clause = filters
            else:
                # --- YOUR ORIGINAL, POWERFUL NORMALIZATION LOGIC IS PRESERVED BELOW ---
                COURSE_ALIASES = {
                    "BSCS": ["BSCS", "BS COMPUTER SCIENCE", "BS Computer Science"],
                    "BSTM": ["BSTM", "BS TOURISM MANAGEMENT", "BS Tourism Management"],
                    "BSOA": ["BSOA", "BS OFFICE ADMINISTRATION", "BS Office Administration"],
                    "BECED": ["BECED", "BACHELOR OF EARLY CHILDHOOD EDUCATION", "Bachelor of Early Childhood Education"],
                    "BSIT": ["BSIT", "BS INFORMATION TECHNOLOGY", "BS Information Technology"],
                    "BSHM": ["BSHM", "BS HOSPITALITY MANAGEMENT", "BS Hospitality Management"],
                    "BTLE": ["BTLE", "BACHELOR OF TECHNOLOGY AND LIVELIHOOD EDUCATION", "Bachelor of Technology and Livelihood Education"]
                }
                
                and_conditions: List[dict] = []
                for k, v in filters.items():
                    standard_key = self.REVERSE_SCHEMA_MAP.get(k, k)
                    possible_keys = list(set([standard_key] + [orig for orig, std in self.REVERSE_SCHEMA_MAP.items() if std == standard_key]))
                    
                    or_conditions = []

                    if standard_key == "program":
                        value_from_placeholder = v.get('$in') if isinstance(v, dict) else [v]
                        all_aliases = set(value_from_placeholder)
                        for item in value_from_placeholder:
                            item_upper = str(item).upper()
                            for key, aliases in COURSE_ALIASES.items():
                                if item_upper == key or item_upper in [a.upper() for a in aliases]:
                                    all_aliases.update(aliases)
                                    break
                        for key in possible_keys:
                            or_conditions.append({key: {"$in": list(all_aliases)}})

                    elif standard_key == "year_level":
                        year_str = str(v)
                        year_variations_str = {year_str, f"Year {year_str}"}
                        for key in possible_keys:
                            or_conditions.append({key: {"$in": list(year_variations_str)}})
                            try:
                                year_int = int(v)
                                or_conditions.append({key: {"$eq": year_int}})
                            except (ValueError, TypeError):
                                pass
                    
                    else: # For any other filter, like 'position' or 'section'
                        if isinstance(v, str):
                            value_variations = list(set([v.lower(), v.upper(), v.title()]))
                            for key in possible_keys:
                                or_conditions.append({key: {"$in": value_variations}})
                        elif isinstance(v, dict):
                            for key in possible_keys:
                                or_conditions.append({key: v})
                        else:
                            for key in possible_keys:
                                or_conditions.append({key: {"$eq": v}})
                    
                    if len(or_conditions) > 1:
                        and_conditions.append({"$or": or_conditions})
                    elif or_conditions:
                        and_conditions.append(or_conditions[0])

                if len(and_conditions) > 1:
                    where_clause = {"$and": and_conditions}
                elif and_conditions:
                    where_clause = and_conditions[0]
                    
                    
            # --- ✨ FINAL WILDCARD FIX START ✨ ---
        # If there is no text search and no filters, we must use a wildcard to get all documents.
        if not final_query_texts and not where_clause and not document_filter:
            final_query_texts = ["*"]
            self.debug("⚠️ No query or filters provided. Using wildcard '*' to retrieve all documents.")
        elif (where_clause or document_filter) and not final_query_texts:
            final_query_texts = ["*"]
            self.debug("⚠️ No query text provided with filters. Using wildcard '*' search.")
        # --- ✨ FINAL WILDCARD FIX END ✨ ---
        
        if self.debug_mode:
            try: self.debug("🧩 Final where_clause:", json.dumps(where_clause, ensure_ascii=False))
            except Exception: self.debug("🧩 Final where_clause (non-serializable):", where_clause)
        
        if (where_clause or document_filter) and not final_query_texts:
            final_query_texts = ["*"]
            self.debug("⚠️ No query text provided with filters. Using wildcard '*' search.")
        
        for name, coll in self.collections.items():
            if collection_filter and isinstance(collection_filter, str) and collection_filter not in name:
                continue
            try:
                res = coll.query(
                    query_texts=final_query_texts, n_results=50,
                    where=where_clause, where_document=document_filter
                )
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                for i, doc in enumerate(docs):
                    all_hits.append({
                        "source_collection": name, "content": doc,
                        "metadata": metas[i] if i < len(metas) else {}
                    })
            except Exception as e:
                self.debug(f"⚠️ Query error in {name}: {e}")

        return all_hits
        
    
    
    
    def _validate_plan(self, plan_json: Optional[dict]) -> tuple[bool, Optional[str]]:
        """
        Validates the planner's output before execution.
        Returns a tuple: (is_valid: bool, error_message: Optional[str]).
        If unsupported operators like $gt/$lt slip through, they are rewritten into a safe form.
        """
        # 1. Check if the overall plan object is a dictionary
        if not isinstance(plan_json, dict):
            return False, "The plan is not a valid JSON object (expected a dictionary)."

        # 2. Check for the 'plan' key and if its value is a list
        plan_list = plan_json.get("plan")
        if not isinstance(plan_list, list):
            return False, "The plan is missing a 'plan' key with a list of steps."
            
        # 3. Check if the plan is empty
        if not plan_list:
            return False, "The plan is empty and contains no steps."

        # 4. Iterate and validate each step
        for i, step in enumerate(plan_list):
            step_num = i + 1

            # 4a. Check if the step is a dictionary
            if not isinstance(step, dict):
                return False, f"Step {step_num} is not a valid object (expected a dictionary)."

            # 4b. Check for 'tool_call'
            tool_call = step.get("tool_call")
            if not isinstance(tool_call, dict):
                return False, f"Step {step_num} is missing or has an invalid 'tool_call' section."

            # 4c. Check for 'tool_name'
            tool_name = tool_call.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                return False, f"Step {step_num} is missing a 'tool_name'."

            # 4d. If it's a search tool, validate its parameters
            if tool_name == "search_database":
                params = tool_call.get("parameters")
                if not isinstance(params, dict):
                    if params is not None:
                        return False, f"Step {step_num} has invalid 'parameters' (expected a dictionary)."
                    continue 

                filters = params.get("filters")
                if filters is not None and not isinstance(filters, dict):
                    return False, f"Step {step_num} has an invalid 'filters' parameter (expected a dictionary)."

                if isinstance(filters, dict) and "$or" in filters:
                    or_conditions = filters.get("$or")
                    if isinstance(or_conditions, list):
                        for condition_index, condition in enumerate(or_conditions):
                            if isinstance(condition, dict) and len(condition) > 1:
                                return False, (f"Step {step_num} contains an invalid complex '$or' filter. "
                                               f"The condition at index {condition_index} has multiple keys. "
                                               f"Each condition inside '$or' must have only one key.")
                # 🆕 END OF NEW BLOCK

                doc_filter = params.get("document_filter")
                if doc_filter is not None and not isinstance(doc_filter, dict):
                    return False, f"Step {step_num} has an invalid 'document_filter' parameter (expected a dictionary)."
                
                if isinstance(doc_filter, dict) and "$contains" in doc_filter:
                    if not isinstance(doc_filter["$contains"], str):
                        return False, f"Step {step_num} has an invalid value for '$contains' (expected a string)."

                # 🔥 NEW PATCH: auto-rewrite unsupported operators
                if isinstance(filters, dict):
                    unsupported_ops = {"$gt", "$lt", "$gte", "$lte"}
                    bad_keys = [k for k, v in filters.items() if isinstance(v, dict) and any(op in v for op in unsupported_ops)]
                    if bad_keys:
                        for key in bad_keys:
                            # Instead of $in: [], just drop the invalid filter entirely
                            filters.pop(key, None)
                        # also strip sort/limit if present
                        if "sort" in params: params.pop("sort")
                        if "limit" in params: params.pop("limit")
                        self.debug(f"⚠️ Step {step_num}: Removed unsupported operators ($gt/$lt) from filters, fallback to all records.")


            elif tool_name not in self.available_tools and tool_name != "finish_plan":
                return False, f"Step {step_num} uses an unknown tool: '{tool_name}'."
        
        # 5. Check that the plan ends with 'finish_plan'
        last_step = plan_list[-1]
        if not (isinstance(last_step, dict) and last_step.get("tool_call", {}).get("tool_name") == "finish_plan"):
            return False, "The plan must conclude with a 'finish_plan' step."

        return True, None





    def execute_reasoning_plan(self, query: str, history: Optional[List[dict]] = None) -> tuple[str, Optional[dict]]:
        self.debug("🤖 Smart Tool Planner with Fallback starting...")
        start_time = time.time()
        
        plan_json = None
        final_context = {}
        error_msg = None
        success = False
        results_count = 0
        
        try:
            # 1. Generate the single tool call from the planner.
            sys_prompt = PROMPT_TEMPLATES["planner_agent"].format(schema=self.db_schema_summary)
            planner_history = history if self.llm.api_mode == 'online' else None
            
            plan_raw = self.llm.execute(
                system_prompt=sys_prompt, user_prompt=f"User Query: {query}",
                json_mode=True, phase="planner", history=planner_history
            )
            
            tool_call_json = self._repair_json(plan_raw)
            plan_json = {"plan": [{"step": 1, "thought": "AI selected the best tool.", "tool_call": tool_call_json}]}
            
            if not tool_call_json or "tool_name" not in tool_call_json:
                raise ValueError("AI failed to select a valid tool.")

            # 2. Execute the single, precise smart tool call.
            tool_name = tool_call_json["tool_name"]
            params = tool_call_json.get("parameters", {})
            
            collected_docs = []
            if tool_name in self.available_tools:
                tool_function = self.available_tools[tool_name]
                self.debug(f"   -> Executing primary tool: {tool_name} with params: {params}")
                results = tool_function(**params)
                collected_docs = results if isinstance(results, list) else [results]
            else:
                raise ValueError(f"AI selected an unknown tool: '{tool_name}'")

            # --- ✨ NEW FALLBACK LOGIC START ✨ ---
            # Check if the primary tool call failed or returned an empty/error result.
            primary_tool_failed = not collected_docs or "error" in collected_docs[0].get("status", "") or "empty" in collected_docs[0].get("status", "")

            if primary_tool_failed:
                self.debug(f"⚠️ Primary tool '{tool_name}' failed or found nothing. Attempting fallback semantic search.")
                # Perform a broad, general-purpose semantic search as a fallback.
                fallback_docs = self.search_database(query_text=query)
                if fallback_docs:
                    self.debug(f"✅ Fallback search found {len(fallback_docs)} documents.")
                    summary_doc = {
                        "source_collection": "system_note",
                        "content": f"Note: The initial targeted search for tool '{tool_name}' failed. The following are broader, semantically related results for your query.",
                        "metadata": {}
                    }
                    collected_docs = [summary_doc] + fallback_docs
                else:
                    self.debug("❌ Fallback search also found nothing.")
            # --- ✨ NEW FALLBACK LOGIC END ✨ ---

            # 3. Build the final context for the synthesizer
            if not collected_docs or "error" in collected_docs[0].get("status", "") or "empty" in collected_docs[0].get("status", ""):
                final_context = {"status": "empty", "summary": "I tried a precise search and a broad search, but could not find any relevant documents."}
            else:
                results_count = len(collected_docs)
                final_context = {
                    "status": "success",
                    "summary": f"Found {results_count} relevant document(s).",
                    "data": collected_docs[:100]
                }
                success = True

        except Exception as e:
            self.debug(f"❌ An unexpected error occurred during execution: {e}")
            error_msg = str(e)
            final_context = {"status": "error", "summary": f"I ran into a technical problem: {e}"}

        # 4. Synthesize the final answer
        self.debug("🧠 Synthesizing final answer...")
        context_for_llm = json.dumps(final_context, indent=2, ensure_ascii=False)
        synth_prompt = PROMPT_TEMPLATES["final_synthesizer"].format(context=context_for_llm, query=query)
        
        final_answer = self.llm.execute(
            system_prompt="You are a careful AI analyst who provides conversational answers based only on the provided facts.",
            user_prompt=synth_prompt, 
            history=history or [], 
            phase="synth"
        )

        # Record the results
        execution_time = time.time() - start_time
        self.training_system.record_query_result(
            query=query, plan=plan_json, results_count=results_count,
            success=success, execution_time=execution_time, error_msg=error_msg
        )
        
        return final_answer, plan_json

    def start_ai_analyst(self):
        print("\n" + "="*70)
        print("🤖 AI SCHOOL ANALYST (Retrieve → Analyze)")
        print("   Type 'exit' to quit or 'train' to save the last plan.")
        print("="*70)

        last_query = None
        last_plan_for_training = None
        chat_history: List[dict] = []

        while True:
            q = input("\n👤 You: ").strip()
            if not q: continue
            
            if q.lower() == "exit":
                break
            
            if q.lower() == "train":
                if last_query and last_plan_for_training:
                    self._save_dynamic_example(last_query, last_plan_for_training)
                    self.dynamic_examples = self._load_dynamic_examples()
                    print("✅ Plan saved as a new training example.")
                else:
                    print("⚠️ No plan to save. Please run a query first.")
                continue

            # This single call now handles everything correctly
            final_answer, plan_json = self.execute_reasoning_plan(q, history=chat_history)
            
            print("\n🧠 Analyst:", final_answer)
            
            # Store the plan for the 'train' command
            if plan_json and "plan" in plan_json:
                last_query = q
                last_plan_for_training = plan_json

            # Update the history
            chat_history.append({"role": "user", "content": q})
            chat_history.append({"role": "assistant", "content": final_answer})

# -------------------------------
# Helper to load config.json
# -------------------------------
def load_llm_config(mode: str, config_path: str = "config.json") -> dict:
    """
    Loads config with extreme debugging to diagnose file path or content issues.
    """
    # This default config is only used if the function fails entirely.
    default_config = {
        "api_mode": mode, "debug_mode": True, "mistral_api_key": "YOUR_MISTRAL_API_KEY",
        "mistral_api_url": "https://api.mistral.ai/v1/chat/completions",
        "ollama_api_url": "http://localhost:11434/api/chat",
        "planner_model": None, "synth_model": None
    }

    print("\n--- CONFIG LOADER DIAGNOSTICS ---")
    print(f"[1] Function received request for mode: '{mode}'")
    print(f"[2] Using config file path: '{config_path}'")

    # Check if the file actually exists at that path before we try to open it.
    if not os.path.exists(config_path):
        print(f"[3] ❌ FATAL: File does NOT exist at the path above.")
        print(f"    Please verify the file is in the correct directory and the name is spelled correctly.")
        print("--- END DIAGNOSTICS ---\n")
        print(f"⚠️ Could not find '{config_path}'. Using default settings.")
        return default_config

    print(f"[3] ✅ SUCCESS: File found at the specified path.")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            # First, read the raw text of the file to see its exact content.
            raw_content = f.read()
            print("[4] Raw content of the file being read:")
            print("<<<<<<<<<<<<<<<<<<<<")
            # We print repr(raw_content) to see hidden characters like extra spaces or newlines
            print(repr(raw_content))
            print(">>>>>>>>>>>>>>>>>>>>")

            if not raw_content.strip():
                print("[5] ❌ FATAL: The config file is empty.")
                print("--- END DIAGNOSTICS ---\n")
                print(f"⚠️ Config file '{config_path}' is empty. Using default settings.")
                return default_config

            # IMPORTANT: We must reset the file reader's cursor to the beginning
            # before trying to parse the JSON.
            f.seek(0)

            # Now, try to parse the content as JSON.
            all_config = json.load(f)
            print(f"[5] JSON parsed. Top-level keys found are: {list(all_config.keys())}")

        if mode in all_config:
            print(f"[6] ✅ SUCCESS: Mode '{mode}' was found in the keys.")
            cfg = all_config[mode]
            cfg["api_mode"] = mode
            print("--- END DIAGNOSTICS ---\n")
            print(f"✅ Loaded {mode.upper()} configuration from {config_path}")
            return cfg
        else:
            print(f"[6] ❌ FAILURE: Mode '{mode}' was NOT found in the keys {list(all_config.keys())}.")
            print("--- END DIAGNOSTICS ---\n")
            print(f"⚠️ Mode '{mode}' not found in {config_path}, using defaults.")
            return default_config

    except Exception as e:
        print(f"[!] An unexpected error occurred during file processing: {e}")
        print("--- END DIAGNOSTICS ---\n")
        print(f"⚠️ An error occurred reading {config_path}. Using default settings.")
        return default_config
    
