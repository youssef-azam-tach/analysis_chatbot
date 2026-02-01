"""
Streamlit entrypoint that avoids conflicting with top-level `app` package name.
Run with: `streamlit run ui/streamlit/streamlit_app.py`
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so top-level packages are importable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import runpy

# Execute the original app.py from this entrypoint to avoid naming conflict
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
