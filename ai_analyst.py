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
        You are a highly intelligent AI data analyst for a school wide database. Your goal is to create the simplest possible research plan to answer user questions.

        DATABASE SCHEMA (collections & metadata fields):
        {schema}

        AVAILABLE TOOLS:
        1. `search_database(...)`
        2. `resolve_person_entity(...)`
        3. `get_distinct_combinations(collection_filter: str, fields: list, filters: dict)`
        - Finds all unique combinations of values for given fields.
        - **Use this for broad queries about groups**, like "all BSCS schedules", to first find all `year_level` and `section` combinations.

        INSTRUCTIONS:
        1)  **Analyze the Goal:** Understand if the user wants simple information about a person OR if they want information related to that person (like a schedule, adviser, etc.).
        2.  **Formulate a Plan:**
            - For a simple lookup ("who is Lee"), a single search step followed by "finish_plan" is sufficient.
            - For related information ("schedule of Lee"), you MUST first find the person, then use placeholders to find the related data in a second step.
        3)  **PERSON-ENTITY RESOLUTION WORKFLOW:**
            - **CRITICAL RULE**: If the user's query is about a specific person (e.g., "who teaches BSCS?", "what is Deborah's schedule?", "find Dr. Lewis"), your **FIRST step MUST be** to use the `resolve_person_entity` tool with the person's name.
            - After resolving the entity, you **MUST use the `aliases` list** returned by the tool to perform subsequent database searches. Use the `$in` operator in the `filters` parameter for this.
        4)  **SCHEDULE RETRIEVAL WORKFLOW (CRITICAL):**
            - The user may ask about the schedule of a student, a faculty member, or a group (e.g., "BSCS 1st year").
            - You MUST always begin by finding a person or representative record in the `students` or `faculty` collections.
            - Once you have that record:
                - For students: extract their `program`, `year_level`, and `section`.
                - For teaching faculty/staff: resolve aliases for their name and match by adviser/staff name, then use those courses to find their teaching schedules in the `schedules` collections.
                - For non-teaching faculty/staff (e.g., Librarian, Registrar, Guidance Counselor), do NOT use program/year/section filters. Instead, look for their schedule directly in the `faculty_*_non_teaching_schedule` collections.**
            - Only THEN use this metadata to query the relevant schedules collections.
            - **Never jump directly to schedules using only raw filters.
            - For group queries (like "all 1st year BSCS schedules"), you may take any representative student record that matches the description, and then use their metadata to get the schedule.
            - For broad program-level queries (like "BSCS schedules"), you MUST first use `get_distinct_combinations` to gather all unique `year_level` and `section` pairs. Then, perform a single, broad `search_database` call for all schedules in that program. The final synthesizer will connect the results.
        5)  **MULTI-ENTITY & CONFLICT QUERIES:**
            - If the user asks to compare two or more people or entities (e.g., "schedule conflict between A and B"), you MUST create a separate `search_database` step to find the information for EACH entity individually.
            - DO NOT combine searches for different entities into a single step with a complex `$or` filter.
        6)  **PARAMETER RULES:**
            - `collection_filter`: Use ONLY for general types like "students" or "schedules". For broad searches, do not use it at all. **NEVER** try to construct a full collection name like "schedules_ccs_newcourse".
            - `filters`: Use for PRECISE METADATA matching. Valid operators are `$eq`, `$ne`, and `$in`.
            - `document_filter`: Use for TEXT SEARCH inside a document's content. The `$contains` operator is valid here.
            - Do NOT use comparison operators like `$gt`, `$lt`, etc. If asked for "highest" or "lowest", retrieve all relevant records and let the system compute the result.
        7)  **CRITICAL RULE (Placeholders):** To use information from a previous step, you MUST use a placeholder variable like '$year_level_from_step_1'.
        8)  When you have all the facts, add a final step: `{{"tool_name": "finish_plan"}}`
        9)  Output a SINGLE JSON object.

        RELATIONSHIPS ACROSS COLLECTIONS:
        - Faculty profiles are linked to schedules by the faculty member's name.
        - Student profiles are linked to schedules by the combination of program, year, and section.

        **COMPREHENSIVE EXAMPLE 1 (Connecting a Student to their Schedule):**
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

        **COMPREHENSIVE EXAMPLE 2 (Connecting a Faculty Member to their Schedule):**
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

        **COMPREHENSIVE EXAMPLE 3 (Indirect Faculty Schedule Lookup):**
        User Query: "what is the schedule of our librarian"
        Your JSON Response:
        {{
        "plan": [
            {{
            "step": 1,
            "thought": "The user is asking for the schedule of a person identified by their role ('librarian'). My first step is to use `search_database` to find the actual person who holds this position.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                    "collection_filter": "faculty",
                    "filters": {{
                        "position": "Librarian"
                    }}
                }}
            }}
            }},
            {{
            "step": 2,
            "thought": "Now that Step 1 will give me the librarian's profile, I have their name. It is CRITICAL that I now use `resolve_person_entity` with a placeholder for their name. This step is mandatory to find all possible name variations before searching for a schedule, preventing mismatches.",
            "tool_call": {{
                "tool_name": "resolve_person_entity",
                "parameters": {{
                    "name": "$full_name_from_step_1"
                }}
            }}
            }},
            {{
            "step": 3,
            "thought": "Now that I have the person's primary name from Step 2, I will perform a simple but powerful text search for that name within the 'schedules' collections. This is more reliable than filtering specific metadata fields.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                    "collection_filter": "schedules",
                    "query_text": "$primary_name_from_step_2"
                }}
            }}
            }},
            {{
            "step": 4,
            "thought": "I have found the person and searched for their schedule using their resolved primary name. I now have all the information required to answer the user.",
            "tool_call": {{ "tool_name": "finish_plan" }}
            }}
        ]
        }}
        
        
         **COMPREHENSIVE EXAMPLE 4 (Indirect Information Retrieval via Bridge):**
        User Query: "Who advises the 2nd year BSCS students?"
        Your JSON Response:
        {{
        "plan": [
            {{
            "step": 1,
            "thought": "The user wants an adviser for a group of students. The adviser's name is on the schedule document, not the student record itself. First, I need to find a sample student from that group to get the linking information: their program, year, and section.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                    "collection_filter": "students",
                    "filters": {{
                        "program": "BSCS",
                        "year_level": "2"
                    }}
                }}
            }}
            }},
            {{
            "step": 2,
            "thought": "Now that I have a sample student from Step 1, I will use placeholders for their program, year, and section to find the specific schedule that applies to their entire group. This schedule document contains the adviser's name.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                    "collection_filter": "schedules",
                    "filters": {{
                        "program": "$program_from_step_1",
                        "year_level": "$year_level_from_step_1",
                        "section": "$section_from_step_1"
                    }}
                }}
            }}
            }},
            {{
            "step": 3,
            "thought": "I have successfully used the student record as a bridge to find the correct schedule. The synthesizer will now be able to extract the adviser's name from that document.",
            "tool_call": {{ "tool_name": "finish_plan" }}
            }}
        ]
        }}
        
          **COMPREHENSIVE EXAMPLE 6 (Verification of a Relationship):**
        User Query: "My friend Lee Pace is in Dr. Lewis's advisory class. How many total units is he taking this semester?"
        Your JSON Response:
        {{
        "plan": [
            {{
            "step": 1,
            "thought": "The query is about 'Lee Pace'. My first step must be to find his official student record to get his precise program, year, and section.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                    "query_text": "Lee Pace"
                }}
            }}
            }},
            {{
            "step": 2,
            "thought": "Now that I have Lee Pace's details from Step 1, I will use them as precise metadata filters to find the one and only schedule document that applies to him. This is more reliable than a broad text search.",
            "tool_call": {{
                "tool_name": "search_database",
                "parameters": {{
                    "collection_filter": "schedules",
                    "filters": {{
                        "program": "$program_from_step_1",
                        "year_level": "$year_level_from_step_1",
                        "section": "$section_from_step_1"
                    }}
                }}
            }}
            }},
            {{
            "step": 3,
            "thought": "I also need to verify the 'Dr. Lewis' part of the query. I will resolve this name to get all her aliases so I can compare it to the adviser listed on the schedule I found in Step 2.",
            "tool_call": {{
                "tool_name": "resolve_person_entity",
                "parameters": {{
                    "name": "Dr. Lewis"
                }}
            }}
            }},
            {{
            "step": 4,
            "thought": "I have retrieved Lee Pace's record, his exact schedule, and the aliases for Dr. Lewis. The synthesizer now has all the facts to verify the relationship and answer the question about units.",
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
        - **Entity Linking Rule (CRITICAL):** You must actively try to link entities across documents. If one document mentions 'Dr. Deborah' as an adviser and another document lists a faculty member named 'Deborah K. Lewis', you MUST assume they are the same person. Synthesize their information into a single, coherent description and state the connection clearly (e.g., "The adviser, Dr. Deborah, is Professor Deborah K. Lewis."). Do not present them as two different people unless the documents give conflicting information.
        - Infer logical connections. For example, if a student document and a class schedule share the same program, year, and section, you MUST state that the schedule applies to that student.
        - **Name Interpretation Rule:** When a user asks about a person using a single name (e.g., "who is Lee"), you must summarize information for all individuals where that name appears as a first OR last name. If you find a match on a last name (e.g., "Michelle Lee"), you MUST include this person in your summary and clarify their role. Do not restrict your answer to only first-name matches.
        - If data is truly missing, state that clearly.
        - Cite the source_collection for key facts using [source_collection_name].
        - If status is 'empty': Do NOT say "status empty". Instead, use the 'summary' to inform the user conversationally that you couldn't find information. You can suggest an alternative query.
        - If status is 'error': Do NOT show the technical error message. Instead, use the 'summary' to apologize for the technical difficulty in a simple, user-friendly way.
        - Be conversational and natural in your response.

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
            "resolve_person_entity": self.resolve_person_entity,
            "get_distinct_combinations": self.get_distinct_combinations,
        }
        
        
    # In ai_analyst.py, inside the AIAnalyst class

    # In ai_analyst.py, inside the AIAnalyst class
    
    
    # ✨ ADD THIS NEW METHOD TO THE AIANALYST CLASS
    # ✨ REPLACE THE OLD FUNCTION WITH THIS CORRECTED VERSION
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
        Finds all documents related to a person's name and resolves their various aliases.
        Returns the primary full name and a list of all known aliases.
        """
        print(f"🕵️  Resolving entity for: '{name}'")
        
         # --- ✨ FINAL CORRECTED LOGIC START ✨ ---
        # The 'who is deborah' query proved that a combined query_text and document_filter
        # is the most effective search strategy. This tool will now replicate that.

        # 1. Create a simple text query. e.g., "dr. lewis"
        search_query = name.lower()
        
        # 2. Create a document filter for the most significant part of the name.
        # We clean the name and assume the last word is the surname.
        name_clean = re.sub(r'^(DR|PROF|MR|MS|MRS)\.?\s*', '', name.upper()).replace(',', '')
        name_parts = name_clean.split()
        doc_filter = None
        if name_parts:
            most_significant_name = name_parts[-1].lower()
            doc_filter = {"$contains": most_significant_name}
        
        print(f"   -> Performing combined search for query='{search_query}' and filter={doc_filter}")
        
        # 3. Perform the search using BOTH query_text AND document_filter
        initial_results = self.search_database(query=search_query, document_filter=doc_filter)
        # --- ✨ FINAL CORRECTED LOGIC END ✨ ---
        
        if not initial_results:
            return {"primary_name": name.title(), "aliases": [name.title()]}

        # Step 3: Gather all name variations from the results found.
        potential_names = {name.title()}
        primary_name = name.title()
        
        for result in initial_results:
            meta = result.get('metadata', {})
            name_fields = ['full_name', 'adviser', 'adviser_name', 'staff_name', 'student_name']
            for field in name_fields:
                if meta.get(field):
                    potential_names.add(str(meta[field]).strip().title())

        # Step 4: Link names that are fuzzy matches to the primary name.
        resolved_aliases = {primary_name}
        for potential_name in potential_names:
            if self._fuzzy_name_match(primary_name, potential_name):
                resolved_aliases.add(potential_name)
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
        Searches the database. Accepts 'query' as an alias for 'query_text'.
        Can handle a string or a list of strings for query_text.
        """
        # --- ✨ NEW LOGIC to handle list of queries ---
        qt = query or query_text
        final_query_texts: Optional[List[str]] = None
        if isinstance(qt, list):
            final_query_texts = qt
        elif isinstance(qt, str):
            final_query_texts = [qt]
        # --- ✨ END NEW LOGIC ---

        self.debug(f"🔎 search_database | query(s)='{final_query_texts}' | filters={filters} | doc_filter={document_filter} | coll_filter='{collection_filter}'")
        all_hits: List[dict] = []

        if isinstance(collection_filter, str) and collection_filter.strip().startswith("$or:"):
            try:
                list_str = collection_filter.split(":", 1)[1].strip()
                collection_list = json.loads(list_str.replace("'", '"'))
                collection_filter = {"$or": collection_list}
            except Exception:
                collection_filter = None
                
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

                    if standard_key == "year_level":
                        if isinstance(v, dict) and "$in" in v:
                            year_conditions = [{key: v} for key in possible_keys]
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
                    
                    elif standard_key == "program":
                        if isinstance(v, dict) and "$in" in v:
                            program_conditions = [{key: v} for key in possible_keys]
                            if program_conditions:
                                and_conditions.append({"$or": program_conditions})
                        else:
                            and_conditions.append({"$or": [{key: {"$eq": v}} for key in possible_keys]})
                    
                    else:
                        if isinstance(v, dict) and any(op in v for op in ("$in", "$eq", "$ne", "$gt", "$gte", "$lt", "$lte")):
                            conditions = [{key: v} for key in possible_keys]
                            if len(conditions) > 1: and_conditions.append({"$or": conditions})
                            elif len(conditions) == 1: and_conditions.append(conditions[0])
                        else:
                            conditions = [{key: {"$eq": v}} for key in possible_keys]
                            if len(conditions) > 1: and_conditions.append({"$or": conditions})
                            elif len(conditions) == 1: and_conditions.append(conditions[0])

                if len(and_conditions) == 1: where_clause = and_conditions[0]
                elif len(and_conditions) > 1: where_clause = {"$and": and_conditions}

        if self.debug_mode:
            try: self.debug("🧩 Final where_clause:", json.dumps(where_clause, ensure_ascii=False))
            except Exception: self.debug("🧩 Final where_clause (non-serializable):", where_clause)
        
        if (where_clause or document_filter) and not final_query_texts:
            final_query_texts = ["*"] 
            self.debug("⚠️ No query text provided with filters. Using wildcard '*' search to apply filters.")
        
        elif (where_clause or document_filter) and final_query_texts:
            self.debug("⚠️ Both filters and query text present. Prioritizing filters by using wildcard search.")
            final_query_texts = ["*"]
            
        for name, coll in self.collections.items():
            if collection_filter:
                if isinstance(collection_filter, str):
                    if collection_filter not in name: continue
                elif isinstance(collection_filter, dict):
                    allowed = False
                    if "$or" in collection_filter and isinstance(collection_filter.get("$or"), list):
                        if name in collection_filter["$or"]: allowed = True
                    if not allowed: continue

            try:
                res = coll.query(
                    query_texts=final_query_texts,
                    n_results=50,
                    where=where_clause,
                    where_document=document_filter
                )
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                for i, doc in enumerate(docs):
                    all_hits.append({
                        "source_collection": name,
                        "content": doc,
                        "metadata": metas[i] if i < len(metas) else {}
                    })
            except Exception as e:
                self.debug(f"⚠️ Query error in {name}: {e}")
                
                
        if not all_hits and filters:
            self.debug("⚠️ Filtered search returned no results. Retrying with text search.")
            if query_text and query_text not in ("*", None):
                for name, coll in self.collections.items():
                    if collection_filter and isinstance(collection_filter, str):
                        if collection_filter not in name:
                            continue
                    try:
                        res = coll.query(
                            query_texts=[query_text],
                            n_results=50
                        )
                        docs = (res.get("documents") or [[]])[0]
                        metas = (res.get("metadatas") or [[]])[0]
                        for i, doc in enumerate(docs):
                            all_hits.append({
                                "source_collection": name,
                                "content": doc,
                                "metadata": metas[i] if i < len(metas) else {}
                            })
                    except Exception as e:
                        self.debug(f"⚠️ Fallback query error in {name}: {e}")

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
        self.debug("🤖 Planner starting...")
        start_time = time.time()
        
        plan_json = None
        error_msg = None
        is_valid = False
        
        # --- Plan Generation and Validation ---
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
                plan_raw = self.llm.execute(system_prompt=sys_prompt, user_prompt=user_prompt, json_mode=True, phase="planner", history=history)
            
            plan_json = self._repair_json(plan_raw)
            
            if not plan_json:
                error_msg = "Could not generate a valid JSON plan."
                continue # Try again

            is_valid, error_msg = self._validate_plan(plan_json)
            
            if is_valid:
                self.debug("✅ Plan validation successful. Proceeding with execution.")
                break 
        
        # --- Centralized Execution and Synthesis Logic ---
        final_context = {}
        results_count = 0
        success = False

        if not is_valid:
            final_context = {
                "status": "error",
                "summary": f"I couldn't generate a valid research plan to answer your question. The error was: {error_msg or 'Unknown validation error.'}"
            }
            error_msg = error_msg or "Plan validation failed."
        else:
            try:
                plan = plan_json["plan"]
                self.debug(f"📝 Executing {len(plan)} steps.")
                collected_docs = []
                step_results = {}

                for i, step in enumerate(plan):
                    step_num = int(step.get("step", i + 1))
                    tool_call = step["tool_call"]
                    tool = tool_call["tool_name"]

                    if tool == "finish_plan":
                        self.debug("✅ Reached finish_plan. Moving to synthesis.")
                        break
                    
                    if tool in self.available_tools:
                        tool_function = self.available_tools[tool]
                        params = tool_call.get("parameters", {})
                        
                        resolved_params = self._resolve_placeholders(params, step_results)
                        self.debug(f"   -> Resolved params: {resolved_params}")

                        hits = tool_function(**resolved_params)
                        
                        step_results[step_num] = hits
                        if isinstance(hits, list):
                            collected_docs.extend(hits)
                        self.debug(f"👀 Step {step_num} ({tool}): Result -> {str(hits)[:200]}...")

                    else:
                        raise ValueError(f"Execution failed: Unknown tool '{tool}'")

                if not collected_docs:
                    final_context = {
                        "status": "empty",
                        "summary": "My search of the database did not return any relevant documents for your query."
                    }
                else:
                    unique = {f"{d['source_collection']}::{hash(d['content'])}": d for d in collected_docs}
                    docs = list(unique.values())
                    results_count = len(docs)
                    final_context = {
                        "status": "success",
                        "summary": f"Found {results_count} relevant document(s).",
                        "data": docs[:100] # Cap evidence at 100 docs
                    }
                    success = True

            except Exception as e:
                self.debug(f"❌ An unexpected error occurred during plan execution: {e}")
                error_msg = f"An unexpected error occurred during plan execution: {e}"
                final_context = {
                    "status": "error",
                    "summary": f"I ran into a technical problem while trying to execute the research plan. The error was: {e}"
                }

        # --- Final Synthesis (This block now runs every time) ---
        self.debug("🧠 Synthesizing final answer...")
        
        context_for_llm = json.dumps(final_context, indent=2, ensure_ascii=False)
        
        synth_prompt = PROMPT_TEMPLATES["final_synthesizer"].format(context=context_for_llm, query=query)
        
        final_answer = self.llm.execute(
            system_prompt="You are a careful AI analyst who provides conversational answers based only on the provided facts.",
            user_prompt=synth_prompt, 
            history=history or [], 
            phase="synth"
        )

        # --- Record results in the training system ---
        execution_time = time.time() - start_time
        self.training_system.record_query_result(
            query=query,
            plan=plan_json,
            results_count=results_count,
            success=success,
            execution_time=execution_time,
            error_msg=error_msg
        )
        
        return final_answer , plan_json

    def start_ai_analyst(self):
        print("\n" + "="*70)
        print("🤖 AI SCHOOL ANALYST (Retrieve → Analyze)")
        print("   Type 'exit' to quit or 'train' to save the last plan.")
        print("="*70)

        last_query = None
        last_plan_for_training = None
        chat_history: List[dict] = [] # Initialize an empty history list before the loop

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

            # This single call now handles everything and passes the current history
            final_answer, plan_json = self.execute_reasoning_plan(q, history=chat_history)
            
            print("\n🧠 Analyst:", final_answer)
            
            # Store the plan so the 'train' command can use it
            if plan_json and "plan" in plan_json:
                last_query = q
                last_plan_for_training = plan_json

            # Update the history with the latest exchange to give the AI a memory
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
    
