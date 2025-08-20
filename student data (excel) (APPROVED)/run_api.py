# run_api.py

from flask import Flask, request, jsonify
from api import SmartDataAPI, SearchType

# Initialize the Flask application
app = Flask(__name__)

# Initialize your API class
# This will also initialize your main data system
api_service = SmartDataAPI()

# Define the API endpoint for system status
@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Endpoint to check the health and status of the system.
    Returns: a JSON object with system status details.
    """
    status_info = api_service.get_system_status()
    # Your method already returns a dict, which Flask can jsonify
    return jsonify(status_info)

# Define the API endpoint for searching
@app.route('/api/search', methods=['GET'])
def search_data():
    """
    Endpoint to perform a search.
    Expects a 'query' parameter in the URL.
    Returns: a JSON object with search results.
    """
    query = request.args.get('query', default="", type=str)
    search_type_str = request.args.get('search_type', default="smart", type=str)
    
    # Map the string from the URL to the SearchType enum
    try:
        search_type = SearchType[search_type_str.upper()]
    except KeyError:
        return jsonify({"error": f"Invalid search_type: {search_type_str}. Valid types are smart, exact, fuzzy."}), 400

    results = api_service.search(query=query, search_type=search_type)
    return jsonify(results)

if __name__ == '__main__':
    # This will start the web server
    print("Starting Flask API server...")
    print("To test, open http://127.0.0.1:5000/api/status in your browser.")
    print("To search, open http://127.0.0.1:5000/api/search?query=Daniel Gomez in your browser.")
    app.run(debug=True)