# smart_data_api.py
"""
UI-Friendly API Interface for Smart Student Data System
This module provides clean, simple methods for the UI team to interact with the data system
"""

from typing import List, Dict, Optional, Tuple, Any
import json
from datetime import datetime
import logging
from enum import Enum

# Import your existing system
from g1_data_process import SmartStudentDataSystem

class DataType(Enum):
    """Enum for different data types"""
    STUDENT = "student"
    FACULTY = "faculty"
    SCHEDULE = "schedule"
    CURRICULUM = "curriculum"
    GRADES = "grades"

class SearchType(Enum):
    """Enum for search types"""
    SMART = "smart"
    EXACT = "exact"
    FUZZY = "fuzzy"

class SmartDataAPI:
    """
    Clean, UI-friendly interface for the Smart Student Data System
    This class wraps the complex functionality into simple methods
    """
    
    def __init__(self):
        """Initialize the API with the data system"""
        self.system = SmartStudentDataSystem()
        self.logger = self._setup_logging()
        self._initialize_system()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for API operations"""
        logger = logging.getLogger('SmartDataAPI')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_system(self) -> None:
        """Initialize the data system and load existing data"""
        try:
            success = self.system.quick_setup()
            self.logger.info(f"System initialized successfully: {success}")
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    # ===================== FILE OPERATIONS =====================
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload and process a file
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dict with success status, message, and details
        """
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # Process the file using your existing system
            success = self.system.process_file(file_path)
            
            if success:
                # Get file info
                file_type = self.system.detect_file_type(file_path)
                
                return {
                    "success": True,
                    "message": f"File processed successfully",
                    "file_name": file_path,
                    "file_type": file_type,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to process file",
                    "file_name": file_path,
                    "error": "File processing failed"
                }
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {
                "success": False,
                "message": f"Error processing file: {str(e)}",
                "file_name": file_path,
                "error": str(e)
            }
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types"""
        return [".xlsx", ".pdf"]
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate if a file can be processed
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with validation results
        """
        try:
            import os
            
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File does not exist"}
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.get_supported_file_types():
                return {"valid": False, "error": f"Unsupported file type: {ext}"}
            
            file_type = self.system.detect_file_type(file_path)
            
            return {
                "valid": True,
                "file_type": file_type,
                "extension": ext,
                "size": os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    # ===================== SEARCH OPERATIONS =====================
    
    def search(self, query: str, search_type: SearchType = SearchType.SMART, 
               max_results: int = 15, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform a search across all data
        
        Args:
            query: Search query string
            search_type: Type of search to perform
            max_results: Maximum number of results to return
            filters: Optional filters to apply
            
        Returns:
            Dict with search results and metadata
        """
        try:
            self.logger.info(f"Searching for: '{query}' with type: {search_type.value}")
            
            if search_type == SearchType.SMART:
                results = self.system.smart_search_with_ai_reasoning(query, max_results)
            elif search_type == SearchType.EXACT:
                results = self._exact_search(query, max_results)
            else:
                results = self._fuzzy_search(query, max_results)
            
            # Apply filters if provided
            if filters:
                results = self._apply_filters(results, filters)
            
            # Format results for UI
            formatted_results = self._format_search_results(results)
            
            return {
                "success": True,
                "query": query,
                "search_type": search_type.value,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }
    
    def _exact_search(self, query: str, max_results: int) -> List[Dict]:
        """Perform exact text search"""
        matches = []
        for name, collection in self.system.collections.items():
            try:
                all_docs = collection.get()
                for i, doc in enumerate(all_docs["documents"]):
                    if query.lower() in doc.lower():
                        metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                        matches.append({
                            "content": doc,
                            "metadata": metadata,
                            "source": self.system.get_collection_type(name),
                            "relevance": 100  # Exact match
                        })
                        
                        if len(matches) >= max_results:
                            break
            except Exception as e:
                self.logger.error(f"Error in exact search for {name}: {e}")
        
        return matches[:max_results]
    
    def _fuzzy_search(self, query: str, max_results: int) -> List[Dict]:
        """Perform fuzzy search"""
        # Use the existing search_all_collections method
        return self.system.search_all_collections(query, max_results)
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            include_result = True
            
            # Apply department filter
            if filters.get("department"):
                if metadata.get("department") != filters["department"]:
                    include_result = False
            
            # Apply course filter
            if filters.get("course"):
                if metadata.get("course") != filters["course"]:
                    include_result = False
            
            # Apply year level filter
            if filters.get("year_level"):
                if str(metadata.get("year_level")) != str(filters["year_level"]):
                    include_result = False
            
            # Apply data type filter
            if filters.get("data_type"):
                if metadata.get("data_type") != filters["data_type"]:
                    include_result = False
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def _format_search_results(self, results: List[Dict]) -> List[Dict]:
        """Format search results for UI consumption"""
        formatted = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            formatted_result = {
                "id": self._generate_result_id(result),
                "title": self._extract_title(result),
                "content": result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""),
                "source": result.get("source", "Unknown"),
                "relevance": result.get("relevance", 0),
                "metadata": {
                    "department": metadata.get("department", "Unknown"),
                    "course": metadata.get("course", ""),
                    "year_level": metadata.get("year_level", ""),
                    "section": metadata.get("section", ""),
                    "data_type": metadata.get("data_type", ""),
                    "full_name": metadata.get("full_name", ""),
                    "student_id": metadata.get("student_id", "")
                },
                "hierarchy": result.get("hierarchy", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            formatted.append(formatted_result)
        
        return formatted
    
    def _generate_result_id(self, result: Dict) -> str:
        """Generate a unique ID for a search result"""
        content_hash = hash(result.get("content", ""))
        return f"result_{abs(content_hash)}"
    
    def _extract_title(self, result: Dict) -> str:
        """Extract a meaningful title from the result"""
        metadata = result.get("metadata", {})
        
        # Try different title sources
        if metadata.get("full_name"):
            return metadata["full_name"]
        elif metadata.get("student_id"):
            return f"Student {metadata['student_id']}"
        elif metadata.get("course"):
            return f"{metadata['course']} - {metadata.get('section', 'Unknown Section')}"
        else:
            content = result.get("content", "")
            # Extract first line as title
            first_line = content.split('\n')[0] if content else "Unknown"
            return first_line[:100] + "..." if len(first_line) > 100 else first_line
    
    # ===================== DATA RETRIEVAL =====================
    
    def get_collections_summary(self) -> Dict[str, Any]:
        """Get summary of all data collections"""
        try:
            collections_info = []
            total_records = 0
            
            for name, collection in self.system.collections.items():
                count = collection.count()
                collection_type = self.system.get_collection_type(name)
                
                collections_info.append({
                    "name": name,
                    "type": collection_type,
                    "count": count,
                    "category": self._categorize_collection(name)
                })
                
                total_records += count
            
            return {
                "success": True,
                "total_collections": len(collections_info),
                "total_records": total_records,
                "collections": collections_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collections summary: {e}")
            return {
                "success": False,
                "error": str(e),
                "collections": []
            }
    
    def _categorize_collection(self, collection_name: str) -> str:
        """Categorize collection by type"""
        name_lower = collection_name.lower()
        
        if "student" in name_lower:
            if "grade" in name_lower:
                return "Student Grades"
            else:
                return "Student Data"
        elif "faculty" in name_lower:
            if "schedule" in name_lower:
                return "Faculty Schedules"
            else:
                return "Faculty Data"
        elif "schedule" in name_lower or "cor" in name_lower:
            return "Class Schedules"
        elif "curriculum" in name_lower:
            return "Curriculum"
        else:
            return "Other"
    
    def get_departments(self) -> List[str]:
        """Get list of all departments in the system"""
        departments = set()
        
        for collection in self.system.collections.values():
            try:
                all_docs = collection.get()
                for metadata in all_docs.get("metadatas", []):
                    if metadata and metadata.get("department"):
                        departments.add(metadata["department"])
            except Exception as e:
                self.logger.error(f"Error extracting departments: {e}")
        
        return sorted(list(departments))
    
    def get_courses(self, department: Optional[str] = None) -> List[str]:
        """Get list of courses, optionally filtered by department"""
        courses = set()
        
        for collection in self.system.collections.values():
            try:
                all_docs = collection.get()
                for metadata in all_docs.get("metadatas", []):
                    if metadata and metadata.get("course"):
                        # Filter by department if specified
                        if department is None or metadata.get("department") == department:
                            courses.add(metadata["course"])
            except Exception as e:
                self.logger.error(f"Error extracting courses: {e}")
        
        return sorted(list(courses))
    
    def get_students_by_course(self, course: str, year_level: Optional[str] = None) -> List[Dict]:
        """Get students filtered by course and optionally by year level"""
        students = []
        
        for collection in self.system.collections.values():
            try:
                all_docs = collection.get()
                for i, metadata in enumerate(all_docs.get("metadatas", [])):
                    if (metadata and 
                        metadata.get("course") == course and
                        metadata.get("data_type", "").startswith("student") and
                        (year_level is None or str(metadata.get("year_level")) == str(year_level))):
                        
                        student_data = {
                            "student_id": metadata.get("student_id", ""),
                            "full_name": metadata.get("full_name", ""),
                            "course": metadata.get("course", ""),
                            "year_level": metadata.get("year_level", ""),
                            "section": metadata.get("section", ""),
                            "department": metadata.get("department", "")
                        }
                        students.append(student_data)
                        
            except Exception as e:
                self.logger.error(f"Error extracting students: {e}")
        
        return students
    
    def get_faculty_by_department(self, department: str) -> List[Dict]:
        """Get faculty members by department"""
        faculty = []
        
        for collection in self.system.collections.values():
            try:
                all_docs = collection.get()
                for metadata in all_docs.get("metadatas", []):
                    if (metadata and 
                        metadata.get("department") == department and
                        metadata.get("data_type", "").endswith("faculty")):
                        
                        faculty_data = {
                            "full_name": metadata.get("full_name", ""),
                            "position": metadata.get("position", ""),
                            "department": metadata.get("department", ""),
                            "email": metadata.get("email", ""),
                            "faculty_type": metadata.get("faculty_type", "")
                        }
                        faculty.append(faculty_data)
                        
            except Exception as e:
                self.logger.error(f"Error extracting faculty: {e}")
        
        return faculty
    
    # ===================== SYSTEM STATUS =====================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            collections_count = len(self.system.collections)
            total_records = sum(collection.count() for collection in self.system.collections.values())
            
            return {
                "status": "healthy" if collections_count > 0 else "no_data",
                "collections_loaded": collections_count,
                "total_records": total_records,
                "data_loaded": self.system.data_loaded,
                "timestamp": datetime.now().isoformat(),
                "system_ready": True
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "system_ready": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        try:
            # Test collection access
            collections_accessible = True
            error_collections = []
            
            for name, collection in self.system.collections.items():
                try:
                    collection.count()
                except Exception as e:
                    collections_accessible = False
                    error_collections.append({"name": name, "error": str(e)})
            
            return {
                "healthy": collections_accessible,
                "collections_accessible": collections_accessible,
                "error_collections": error_collections,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # ===================== UTILITY METHODS =====================
    
    def get_filters(self) -> Dict[str, List[str]]:
        """Get available filter options for the UI"""
        return {
            "departments": self.get_departments(),
            "courses": self.get_courses(),
            "data_types": ["student", "faculty", "schedule", "curriculum", "grades"],
            "year_levels": ["1", "2", "3", "4"]
        }
    
    def export_search_results(self, results: List[Dict], format: str = "json") -> str:
        """Export search results in specified format"""
        if format.lower() == "json":
            return json.dumps(results, indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            # Simple CSV export
            if not results:
                return ""
            
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            if results:
                headers = list(results[0].keys())
                writer.writerow(headers)
                
                # Write data
                for result in results:
                    writer.writerow([str(result.get(header, "")) for header in headers])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")