import os

def is_ignored(pattern_to_check, gitignore_path='.gitignore'):
    """
    Checks if a pattern is present in the .gitignore file.
    
    Args:
        pattern_to_check (str): The string (e.g., 'logs/', '*.csv') to look for.
        gitignore_path (str): The path to the .gitignore file.
        
    Returns:
        bool: True if the pattern is found, False otherwise.
    """
    # Check if the .gitignore file exists
    if not os.path.exists(gitignore_path):
        print(f"⚠️ .gitignore file not found at {gitignore_path}")
        return False
        
    try:
        with open(gitignore_path, 'r') as f:
            # Read all lines and strip whitespace/newline characters from each line
            lines = [line.strip() for line in f if line.strip()]
            
            # Check for the pattern in the list of lines
            if pattern_to_check in lines:
                return True
            else:
                return False
                
    except IOError as e:
        print(f"❌ Error reading .gitignore file: {e}")
        return False