"""
Root Entry Point for Streamlit Application
Allows running either:
    python -m streamlit run app.py
or
    streamlit run app/streamlit_app.py
"""

import sys
import runpy
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

target_script = PROJECT_ROOT / "app" / "streamlit_app.py"

if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(str(target_script), run_name="__main__")
