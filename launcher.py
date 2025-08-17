# main_app.py
import subprocess
import time
import webbrowser
import threading
import requests

def run_fastapi():
    """Run FastAPI backend on port 8000"""
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

def wait_for_fastapi():
    """Wait until FastAPI is available before starting Streamlit"""
    url = "http://localhost:8000/docs"  # FastAPI docs page
    for _ in range(60):  # wait up to ~60 seconds
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("✅ FastAPI is up!")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("❌ FastAPI failed to start in time.")
    return False

def run_streamlit():
    """Run Streamlit frontend on port 8501"""
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])

if __name__ == "__main__":
    # Start FastAPI in background thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Wait until FastAPI responds before starting Streamlit
    if wait_for_fastapi():
        # Start Streamlit
        streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
        streamlit_thread.start()

        # Open browser automatically
        time.sleep(5)
        webbrowser.open("http://localhost:8501")

    # Keep the script alive
    while True:
        time.sleep(1)
