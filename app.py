"""
app.py — Root-level entry point for Streamlit Community Cloud deployment.
Delegates to src/main.py via runpy.
"""
import os
import sys
import runpy

# Ensure src/ and project root are on the path
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

# Load .env for local dev (on Streamlit Cloud, secrets are injected automatically)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_dir, ".env"))
    load_dotenv(os.path.join(src_dir, ".env"))
except ImportError:
    pass

# Run the main app
runpy.run_path(os.path.join(src_dir, "main.py"), run_name="__main__")
