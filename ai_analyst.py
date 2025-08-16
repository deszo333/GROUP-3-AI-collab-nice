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
            if json_mode:
                payload["format"] = "json"
        return api_url, headers, payload

    def execute(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False,
                history: Optional[List[dict]] = None, retries: int = 2, phase: str = "planner") -> str:
        messages = list(history) if history else []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        api_url, headers, payload = self._prepare_request(messages, json_mode, phase=phase)
        if not api_url:
            return "Configuration Error: API URL is not set."

        if self.debug_mode:
            print(f"🧠 LLMService → {self.api_mode.upper()} | phase={phase} | json={json_mode}")

        last_err = None
        for attempt in range(retries + 1):
            try:
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
        You are a highly intelligent AI data analyst for a school-wide database. Your goal is to create a research plan to answer user questions.

        DATABASE SCHEMA (collections & metadata fields):
        {schema}

        AVAILABLE TOOLS:
        1. `search_database(query_text: str, filters: dict = None, document_filter: dict = None, collection_filter: str = None)`
        - Searches the database. Use for finding any type of information.
        2. `resolve_person_entity(name: str)`
        - Takes a person's name, finds all possible matches in the database, and returns a single profile with their primary full name and a list of all known aliases (e.g., "Dr. Deborah", "Lewis, Deborah K.", "Deborah").
        - **This tool MUST be used first for any query about a specific person.**
        INSTRUCTIONS:
        1)  **Analyze the Goal:** Understand if the user wants simple information about a person OR if they want information related to that person (like a schedule, adviser, etc.).
        2.  **Formulate a Plan:**
            - For a simple lookup ("who is Lee"), a single search step followed by "finish_plan" is sufficient.
            - For related information ("schedule of Lee"), you MUST first find the person, then use placeholders to find the related data in a second step.

        3)  **PERSON-ENTITY RESOLUTION WORKFLOW:**
            - **CRITICAL RULE**: If the user's query is about a specific person (e.g., "who teaches BSCS?", "what is Deborah's schedule?", "find Dr. Lewis"), your **FIRST step MUST be** to use the `resolve_person_entity` tool with the person's name.
            - After resolving the entity, you **MUST use the `aliases` list** returned by the tool to perform subsequent database searches. Use the `$in` operator in the `filters` parameter for this.
        4)  **PARAMETER RULES:**
            - To find specific data like a program (e.g., "BSCS students"), use the `filters` parameter like `{{"program": "BSCS"}}`. Do not try to guess a full collection name in the `collection_filter`.
            - `collection_filter`: Use ONLY for general types like "students" or "schedules". For broad searches, do not use it at all.
            - `filters`: Use for PRECISE METADATA matching. Valid operators are `$eq`, `$ne`, and `$in`. The `$contains` operator is NOT valid here.
            - `document_filter`: Use for TEXT SEARCH inside a document's content. The `$contains` operator is valid here.
            - **IMPORTANT**: The 'subjects_by_year' field is complex and should NOT be used for filtering. Find the curriculum first, then analyze its content.
            - **CRITICAL RULE for `collection_filter`**: This parameter must ONLY contain a single, generic keyword like `students` or `schedules`.
                - **CORRECT USAGE**: `"collection_filter": "schedules"`
                - **INCORRECT USAGE**: `"collection_filter": "schedules_ccs_newcourse"`
                - NEVER try to construct a full or partial collection name. Let the system find the exact collection based on the generic keyword.
            - Do NOT use $gt, $lt, $gte, or $lte in filters. 
            - If the query involves "highest", "lowest", "top", or "maximum/minimum", 
            instead return a normal query for all records in that collection, 
            and let the system compute the max/min.
            - Always prefer exact $eq or $in filters.
            
        5)  **CRITICAL RULE:** To use information from a previous step, you MUST use a placeholder variable like '$year_level_from_step_1'.
        6)  When you have all the facts, add a final step: {{"tool_name": "finish_plan"}}
        7)  Output a SINGLE JSON object.

        RELATIONSHIPS ACROSS COLLECTIONS:
        - Faculty ↔ Schedules:
        * Faculty collections store teacher/adviser details. Schedules (COR) store adviser names.
        * Adviser names can be inconsistent (e.g., "Dr. Deborah" vs. "Deborah K. Lewis").
        * **To find who a faculty member teaches:**
            1. Use `resolve_person_entity` with the faculty member's name.
            2. Search the `schedules` collections using the returned `aliases` list in the `filters` with the `$in` operator (e.g., `"filters": {{"adviser": {{"$in": ["Lewis, Deborah K.", "Dr. Deborah"]}}}}`).
        - Students ↔ Schedules:
        * Students are linked to schedules via program + year + section.
        **COMPREHENSIVE EXAMPLE 1 (Connecting a Person to Related Data):**
        User Query: "what is the schedule of lee pace"
        Your JSON Response:
        {{
        "plan": [
            {{
            "step": 1,
            "thought": "The user is asking for the schedule of a person named Lee Pace. First, I must find the record for 'Lee Pace' to get their details like program, year, and section. I will search all collections to be thorough.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                "query_text": "Lee Pace",
                "document_filter": {{"$and": [{{"$contains": "Lee"}}, {{"$contains": "Pace"}}]}}
                }}
            }}
            }},
            {{
            "step": 2,
            "thought": "Now that I have the student's details from Step 1, I must use placeholders to find their specific schedule in the 'schedules' collections.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                "collection_filter": "schedules",
                "query_text": "class schedule",
                "filters": {{
                    "year_level": "$year_level_from_step_1",
                    "section": "$section_from_step_1",
                    "program": "$program_from_step_1"
                }}
                }}
            }}
            }},
            {{
            "step": 3,
            "thought": "I have found the student and their matching schedule. I have all the information needed to answer the user's question.",
            "tool_call": {{ "tool_name": "finish_plan" }}
            }}
        ]
        }}

        **COMPREHENSIVE EXAMPLE 2 (Using the New Entity Resolution Workflow):**
        User Query: "who does deborah teach"
        Your JSON Response:
        {{
        "plan": [
            {{
            "step": 1,
            "thought": "The user is asking about a person named 'Deborah'. My first step must be to use the resolve_person_entity tool to find all known names and aliases for this person.",
            "tool_call": {{
                "tool_name": "resolve_person_entity",
                "parameters": {{
                "name": "Deborah"
                }}
            }}
            }},
            {{
            "step": 2,
            "thought": "Now that I have resolved the entity and have a list of aliases from Step 1, I will search the 'schedules' collections to find any classes where she is the adviser. I must use the placeholder for the aliases list with the `$in` operator.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                "collection_filter": "schedules",
                "query_text": "class schedule",
                "filters": {{
                    "adviser": {{"$in": "$aliases_from_step_1"}}
                }}
                }}
            }}
            }},
            {{
            "step": 3,
            "thought": "I have found all schedules associated with all known aliases for 'Deborah'. I now have all the information needed to answer the user's question.",
            "tool_call": {{ "tool_name": "finish_plan" }}
            }}
        ]
        }}
        {dynamic_examples}
        
        
    
    
    
    
    
    
    
    """,
    "final_synthesizer": r"""
        You are an AI Analyst. Your answer must be based ONLY on the "Factual Documents" provided.

        INSTRUCTIONS:
        - Synthesize information from all documents to create a complete answer.
        - Infer logical connections. For example, if a student document and a class schedule share the same program, year, and section, you MUST state that the schedule applies to that student.
        - **Name Interpretation Rule:** When a user asks about a person using a single name (e.g., "who is Lee"), you must summarize information for all individuals where that name appears as a first OR last name. If you find a match on a last name (e.g., "Michelle Lee"), you MUST include this person in your summary and clarify their role. Do not restrict your answer to only first-name matches.
        - If data is truly missing, state that clearly.
        - Cite the source_collection for key facts using [source_collection_name].

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
        "search_database": self.search_database,
        "resolve_person_entity": self.resolve_person_entity
        }
        
        
    def _fuzzy_name_match(self, name1, name2, threshold=0.8):
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
    def resolve_person_entity(self, name: str) -> dict:
        """
        Finds all documents related to a person's name and resolves their various aliases.
        Returns the primary full name and a list of all known aliases.
        """
        print(f"🕵️  Resolving entity for: '{name}'")
        
        # Step 1: Broad search to gather all potential documents
        # We reuse the existing search_database tool for this.
        initial_results = self.search_database(query=name)
        
        if not initial_results:
            return {"primary_name": name.title(), "aliases": [name.title()]}

        # Step 2: Gather all name variations from the results
        potential_names = {name.title()} # Start with the original query name
        primary_name = name.title()
        
        for result in initial_results:
            meta = result.get('metadata', {})
            # Extract names from all relevant metadata fields
            name_fields = ['full_name', 'adviser', 'adviser_name', 'staff_name', 'student_name']
            for field in name_fields:
                if meta.get(field):
                    potential_names.add(str(meta[field]).strip().title())

        # Step 3: Link names that are fuzzy matches to the primary name
        resolved_aliases = {primary_name}
        for potential_name in potential_names:
            if self._fuzzy_name_match(primary_name, potential_name):
                resolved_aliases.add(potential_name)
                # Heuristic: The longest name is likely the primary one
                if len(potential_name) > len(primary_name):
                    primary_name = potential_name

        print(f"✅ Entity resolved: Primary='{primary_name}', Aliases={list(resolved_aliases)}")
        return {
            "primary_name": primary_name,
            "aliases": list(resolved_aliases)
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
                    # Only use the simple numeric year and a canonical string form
                    out.append(vs)
                    out.append(f"Year {vs}")
                return {"$in": [str(x) for x in list(dict.fromkeys(out))]}

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
        Searches the database. Accepts 'query' as an alias for 'query_text'
        and ignores any other extra parameters via **kwargs to prevent crashes.
        """
        # FIX: Gracefully handle both 'query' and 'query_text' arguments.
        final_query = query or query_text
        
        self.debug(f"🔎 search_database | query='{final_query}' | filters={filters} | doc_filter={document_filter} | coll_filter='{collection_filter}'")
        all_hits: List[dict] = []

        if isinstance(collection_filter, str) and collection_filter.strip().startswith("$or:"):
            try:
                list_str = collection_filter.split(":", 1)[1].strip()
                collection_list = json.loads(list_str.replace("'", '"'))
                collection_filter = {"$or": collection_list}
            except Exception:
                collection_filter = None
                
        # --- BUILDING CHROMA-COMPLIANT WHERE CLAUSE ---
        where_clause: Optional[dict] = None
        if filters:
            if '$or' in filters and isinstance(filters.get('$or'), list):
                where_clause = filters
            else:
                and_conditions: List[dict] = []
                for k, v in filters.items():
                    standard_key = self.REVERSE_SCHEMA_MAP.get(k, k)
                    possible_keys = list(set(
                        [standard_key] + [orig for orig, std in self.REVERSE_SCHEMA_MAP.items() if std == standard_key]
                    ))

                    # Handle year_level with special mixed-type logic
                    if standard_key == "year_level":
                        if isinstance(v, dict) and "$in" in v:
                            year_conditions = []
                            for key in possible_keys:
                                year_conditions.append({key: v})
                            if year_conditions:
                                and_conditions.append({"$or": year_conditions})
                        else:
                            year_val = str(v).strip()
                            year_conditions = []
                            string_values = [year_val, f"Year {year_val}"]
                            for key in possible_keys:
                                year_conditions.append({key: {"$in": string_values}})
                            try:
                                int_value = int(year_val)
                                for key in possible_keys:
                                    year_conditions.append({key: {"$in": [int_value]}})
                            except (ValueError, TypeError):
                                pass
                            if year_conditions:
                                and_conditions.append({"$or": year_conditions})
                    
                    # Handle program with aliases
                    elif standard_key == "program":
                        if isinstance(v, dict) and "$in" in v:
                            program_conditions = [{key: v} for key in possible_keys]
                            if program_conditions:
                                and_conditions.append({"$or": program_conditions})
                        else:
                            and_conditions.append({"$or": [{key: {"$eq": v}} for key in possible_keys]})
                    
                    # All other fields get a simple filter
                    else:
                        if isinstance(v, dict) and any(op in v for op in ("$in", "$eq", "$ne", "$gt", "$gte", "$lt", "$lte")):
                            conditions = [{key: v} for key in possible_keys]
                            if len(conditions) > 1:
                                and_conditions.append({"$or": conditions})
                            elif len(conditions) == 1:
                                and_conditions.append(conditions[0])
                        else:
                            conditions = [{key: {"$eq": v}} for key in possible_keys]
                            if len(conditions) > 1:
                                and_conditions.append({"$or": conditions})
                            elif len(conditions) == 1:
                                and_conditions.append(conditions[0])

                if len(and_conditions) == 1:
                    where_clause = and_conditions[0]
                elif len(and_conditions) > 1:
                    where_clause = {"$and": and_conditions}

        if self.debug_mode:
            try:
                self.debug("🧩 Final where_clause:", json.dumps(where_clause, ensure_ascii=False))
            except Exception:
                self.debug("🧩 Final where_clause (non-serializable):", where_clause)
        
        # --- ✨ CORRECTED LOGIC BLOCK TO PREPARE QUERY ✨ ---
        final_query_texts = [final_query] if final_query else None
        
        # FIX: If we are only filtering (no text query), provide a wildcard query to ChromaDB.
        if where_clause and not final_query_texts:
            final_query_texts = ["*"] 
            self.debug("⚠️ No query text provided with filters. Using wildcard '*' search to apply metadata filters.")
        # FIX: If we have both, prioritize the metadata filter by using a wildcard search.
        elif where_clause and final_query_texts:
            self.debug("⚠️ Both filters and query text present. Prioritizing filters by using wildcard search.")
            final_query_texts = ["*"]
            
        # ---- Execute across collections ----
        for name, coll in self.collections.items():
            if collection_filter:
                if isinstance(collection_filter, str):
                    if collection_filter not in name:
                        continue
                elif isinstance(collection_filter, dict):
                    allowed = False
                    if "$or" in collection_filter and isinstance(collection_filter.get("$or"), list):
                        if name in collection_filter["$or"]:
                            allowed = True
                    if not allowed:
                        continue

            try:
                # This call now uses the corrected 'final_query_texts'
                res = coll.query(
                    query_texts=final_query_texts,
                    n_results=50,
                    where=where_clause,
                    where_document=document_filter
                )
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0] # Corrected typo: metadatas
                for i, doc in enumerate(docs):
                    all_hits.append({
                        "source_collection": name,
                        "content": doc,
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





    def execute_reasoning_plan(self, query: str, history: Optional[List[dict]] = None) -> str:
        self.debug("🤖 Planner starting...")
        start_time = time.time()
        
        plan_json = None
        error_msg = None
        is_valid = False
        
        # --- Self-correction loop to generate and validate a plan ---
        for attempt in range(2):
            sys_prompt = PROMPT_TEMPLATES["planner_agent"].format(
                schema=self.db_schema_summary,
                dynamic_examples=self.dynamic_examples
            )
            user_prompt = f"User Query: {query}"
            
            # On a retry, provide a more explicit correction prompt
            if attempt > 0:
                self.debug(f"Attempting self-correction. Reason: {error_msg}")
                correction_message = (
                    f"INVALID PLAN. Your last plan for '{query}' was rejected because: {error_msg}. "
                    "Please regenerate the plan using ONLY simple key-value filters with `$eq` or `$in`. "
                    "Do NOT use `$gt`, `$lt`, `$gte`, `$lte`, `sort`, or `limit`. "
                    "If the user asks for 'highest', 'lowest', 'smartest', or similar, "
                    "just retrieve ALL relevant records and let the analyst determine the ranking."
                    "Please generate a new, valid plan. New User Query: {query}"
                )
                plan_raw = self.llm.execute(system_prompt=sys_prompt, user_prompt=correction_message, json_mode=True, phase="planner", history=history)
            else:
                plan_raw = self.llm.execute(system_prompt=sys_prompt, user_prompt=user_prompt, json_mode=True, phase="planner")
            
            plan_json = self._repair_json(plan_raw)
            
            if not plan_json:
                error_msg = "Could not generate a valid JSON plan."
                continue # Try again

            is_valid, error_msg = self._validate_plan(plan_json)
            
            if is_valid:
                self.debug("✅ Plan validation successful. Proceeding with execution.")
                break 
        
        # --- Handle final validation failure ---
        if not is_valid:
            final_answer = f"I couldn't generate a valid research plan. Details: {error_msg or 'Unknown validation error.'}"
            execution_time = time.time() - start_time
            self.training_system.record_query_result(
                query=query, plan=plan_json, results_count=0,
                success=False, execution_time=execution_time, error_msg=error_msg
            )
            return final_answer
        
        # --- Execute the validated plan ---
        final_answer = ""
        results_count = 0
        success = False

        try:
            plan = plan_json["plan"]
            self.debug(f"📝 Executing {len(plan)} steps.")
            collected = []
            step_results = {}

            for i, step in enumerate(plan):
                step_num = int(step.get("step", i + 1))
                tool_call = step["tool_call"]
                tool = tool_call["tool_name"]

                if tool == "finish_plan":
                    self.debug("✅ Reached finish_plan. Moving to synthesis.")
                    break
                
                # --- ✨ CORRECTED LOGIC BLOCK START ✨ ---
                # This block now correctly handles all registered tools.
                if tool in self.available_tools:
                    # FIX 1: Use 'tool' everywhere in this block, not 'tool_name'.
                    tool_function = self.available_tools[tool]
                    params = tool_call.get("parameters", {})
                    
                    resolved_params = self._resolve_placeholders(params, step_results)
                    self.debug(f"   -> Resolved params: {resolved_params}")

                    hits = tool_function(**resolved_params)
                    
                    step_results[step_num] = hits
                    if isinstance(hits, list):
                        collected.extend(hits)
                    self.debug(f"👀 Step {step_num} ({tool}): Result -> {hits}")

                else:
                    raise ValueError(f"Execution failed: Unknown tool '{tool}'")
                # --- ✨ CORRECTED LOGIC BLOCK END ✨ ---

                # --- FIX 2: The old logic that started here has been REMOVED ---
                # (The block that started with `params = (step.get("tool_call") or {})...` is gone)

            # Your ambiguity resolution and synthesis logic follows...
            if collected and isinstance(collected[0], dict) and "metadata" in collected[0]:
                unique = {f"{d['source_collection']}::{hash(d['content'])}": d for d in collected}
                docs = list(unique.values())
            else:
                docs = collected

            results_count = len(docs)
            if results_count > 100:
                docs = docs[:100]
                self.debug("⚠️ Evidence capped at 100 docs.")
            if not docs:
                final_answer = "I couldn't find any relevant documents to answer that. Try a more specific query."
            else:
                context = json.dumps(docs, indent=2, ensure_ascii=False)
                synth_user = PROMPT_TEMPLATES["final_synthesizer"].format(context=context, query=query)
                answer = self.llm.execute(system_prompt="You are a careful analyst who uses only provided facts.",
                                        user_prompt=synth_user, history=history or [], phase="synth")
                final_answer = answer
            success = True

        except (ValueError, StopIteration) as e:
            if not error_msg:
                error_msg = str(e)
            final_answer = str(e) # Pass clarification prompt to user
        except Exception as e:
            self.debug(f"❌ An unexpected error occurred: {e}")
            error_msg = f"An unexpected error occurred: {e}"
            final_answer = f"An unexpected error occurred: {e}"
        finally:
            execution_time = time.time() - start_time
            self.training_system.record_query_result(
                query=query,
                plan=plan_json,
                results_count=results_count,
                success=success,
                execution_time=execution_time,
                error_msg=error_msg
            )
        
        return final_answer

    def start_ai_analyst(self):
        print("\n" + "="*70)
        print("🤖 AI SCHOOL ANALYST (Retrieve → Analyze)")
        print("   Type 'exit' to quit or 'train' to save the last plan.")
        print("="*70)

        last_query = None
        last_plan = None
        
        while True:
            q = input("\n👤 You: ").strip()
            if not q: continue
            
            if q.lower() == "exit":
                break
            
            if q.lower() == "train":
                if last_query and last_plan:
                    self._save_dynamic_example(last_query, last_plan)
                    # Reload examples to ensure the new one is used immediately
                    self.dynamic_examples = self._load_dynamic_examples()
                    print("✅ Plan saved as a new training example.")
                else:
                    print("⚠️ No plan to save. Please run a query first.")
                continue

            # This is the key change: format the prompt with dynamic examples
            sys_prompt = PROMPT_TEMPLATES["planner_agent"].format(
                schema=self.db_schema_summary, 
                dynamic_examples=self.dynamic_examples
            )
            user_prompt = f"User Query: {q}"
            
            # The rest of the logic remains the same
            plan_raw = self.llm.execute(system_prompt=sys_prompt, user_prompt=user_prompt, json_mode=True, phase="planner")
            plan_json = self._repair_json(plan_raw)
            
            if plan_json and "plan" in plan_json:
                last_query = q
                last_plan = plan_json
                
            ans = self.execute_reasoning_plan(q)
            print("\n🧠 Analyst:", ans)

# -------------------------------
# Helper to load config.json
# -------------------------------
def load_llm_config(config_path: str = "config.json") -> dict:
    default_config = {
        "api_mode": "online", "debug_mode": True,
        "mistral_api_key": "YOUR_MISTRAL_API_KEY",
        "mistral_api_url": "https://api.mistral.ai/v1/chat/completions",
        "ollama_api_url": "http://localhost:11434/api/chat",
        "planner_model": None, "synth_model": None
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            print(f"✅ Successfully loaded configuration from {config_path}")
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ WARNING: Configuration file '{config_path}' not found.")
        print("   -> Using default settings with a placeholder API key.")
        return default_config
    except json.JSONDecodeError:
        print(f"❌ ERROR: The configuration file '{config_path}' contains a JSON syntax error.")
        print("   -> Using default settings with a placeholder API key.")
        return default_config
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading '{config_path}': {e}")
        return default_config