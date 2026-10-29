"""Streamlit Community Cloud default entrypoint.

Delegates execution directly to app.py so both 'app.py' and 'streamlit_app.py'
function seamlessly as the deployment target.
"""

from pathlib import Path
import runpy

if __name__ == "__main__" or True:
    app_path = Path(__file__).resolve().parent / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")
