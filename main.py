from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
from apscheduler.schedulers.background import BackgroundScheduler

from chatbot import UniversityChatbot
from Webscrape import scrape_and_update  # updated import

from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Config
# -------------------------

app = FastAPI(title="University Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JSON_FILE_PATH = "C:\\Nawal\\IT Department\\Practical Training\\Final Chatbot\\data_backups\\menu_hierarchy.json"

# Multi-user sessions
chatbot_sessions: Dict[str, UniversityChatbot] = {}

# -------------------------
# Models
# -------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

class ScrapeResponse(BaseModel):
    status: str
    file_path: str

# -------------------------
# Helpers
# -------------------------
def load_chatbot_session(user_id: str) -> UniversityChatbot:
    """Load a chatbot session for a specific user."""
    if user_id not in chatbot_sessions:
        if not os.path.exists(JSON_FILE_PATH):
            raise FileNotFoundError(f"JSON file not found: {JSON_FILE_PATH}")
        chatbot_sessions[user_id] = UniversityChatbot(JSON_FILE_PATH)
    return chatbot_sessions[user_id]

def reload_all_sessions(new_json_path: str):
    """Reload all chatbot sessions after data update."""
    for user_id in chatbot_sessions.keys():
        chatbot_sessions[user_id] = UniversityChatbot(new_json_path)

# -------------------------
# API Routes
# -------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Handles chat messages per user session."""
    try:
        bot = load_chatbot_session(request.user_id)
        response_text = bot.chat(request.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape", response_model=ScrapeResponse)
def run_scraper():
    """
    Runs your scraper from Webscrape.py, updates JSON, and reloads all sessions.
    """
    try:
        output_path = scrape_and_update()

        # Reload chatbot sessions with new JSON
        reload_all_sessions(output_path)

        return ScrapeResponse(status="completed and reloaded", file_path=output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/welcome")
async def welcome_message():
    return {
        "response": (
            "🎓 Welcome! I can help with admissions, academics, libraries, housing, faculty, fees, research, and more. "
            "Type your question or 'help' for examples.\n"
            "💡 Tip: Ask natural questions like 'How do I apply for undergraduate admission?' or 'What are the library hours?'\n"
            "Type 'help' for examples, or 'exit' to quit."
        )
    }

# -------------------------
# Auto-scraping scheduler
# -------------------------
scheduler = BackgroundScheduler()

def scheduled_scrape():
    """Automatically run scraper and reload sessions."""
    try:
        output_path = scrape_and_update()
        reload_all_sessions(output_path)
        print(f"✅ Auto-scrape completed and reloaded: {output_path}")
    except Exception as e:
        print(f"❌ Auto-scrape failed: {e}")

@app.on_event("startup")
def startup_event():
    """Run on server startup."""
    if not os.path.exists(JSON_FILE_PATH):
        raise FileNotFoundError(f"JSON file not found: {JSON_FILE_PATH}")
    print("✅ Server startup: Ready to create chatbot sessions as users connect.")

    # Schedule scraping every 24 hours (change as needed)
    scheduler.add_job(scheduled_scrape, "interval", hours=24)
    scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    """Shutdown scheduler when API stops."""
    scheduler.shutdown()

