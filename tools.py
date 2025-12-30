from google.adk.tools import ToolContext

from neo4j_for_adk import tool_success, tool_error

def get_approved_user_goal(tool_context: ToolContext):
    """Returns the user's goal, which is a dictionary containing the kind of graph and its description."""
    if "approved_user_goal" not in tool_context.state:
        return tool_error("approved_user_goal not set. Ask the user to clarify their goal (kind of graph and description).")  
    
    user_goal_data = tool_context.state["approved_user_goal"]

    return tool_success("approved_user_goal", user_goal_data)

def get_approved_files(tool_context: ToolContext):
    """Returns the files that have been approved for import."""
    if "approved_files" not in tool_context.state:
        return tool_error("approved_files not set. Ask the user to approve the file suggestions.")
    
    files = tool_context.state["approved_files"]
    
    return tool_success("approved_files", files)
    
# Tool: Sample File
def sample_file(file_path: str, tool_context: ToolContext) -> dict:
    """Samples a file by reading its content as text.
    
    Treats any file as text and reads up to a maximum of 100 lines.
    
    Args:
      file_path: file to sample, relative to the import directory
      tool_context: ToolContext object
      
    Returns:
        dict: A dictionary containing metadata about the content,
              along with a sampling of the file.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'content' key with textual file content.
              If 'error', includes an 'error_message' key.
    """
    import_dir_result = graphdb.get_import_directory() # use the helper available in Neo4jForADK
    if import_dir_result["status"] == "error": return import_dir_result
    import_dir = Path(import_dir_result["neo4j_import_dir"])

    full_path_to_file = import_dir / file_path
    
    if not full_path_to_file.exists():
        return tool_error(f"File does not exist in import directory: {file_path}")
    
    try:
        # Treat all files as text
        with open(full_path_to_file, 'r', encoding='utf-8') as file:
            # Read up to 100 lines
            lines = list(islice(file, 100))
            content = ''.join(lines)
            return tool_success("content", content)
    
    except Exception as e:
        return tool_error(f"Error reading or processing file {file_path}: {e}")

