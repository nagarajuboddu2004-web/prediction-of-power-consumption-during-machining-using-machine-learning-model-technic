import os
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    port = os.environ.get("PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    sys.exit(stcli.main())
