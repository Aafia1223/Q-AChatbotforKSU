from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Union, List
import os
import json
from apscheduler.schedulers.background import BackgroundScheduler

from chatbot import UniversityChatbot
from Webscrape import scrape_and_update

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="KSU Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JSON_FILE_PATH = "data_backups/menu_hierarchy.json"
chatbot_sessions: Dict[str, UniversityChatbot] = {}

# -------------------------
# Models
# -------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    type: str
    message: str
    options: Union[List[str], None] = None

class ScrapeResponse(BaseModel):
    status: str
    file_path: str

# -------------------------
# Helpers
# -------------------------
def load_chatbot_session(user_id: str) -> UniversityChatbot:
    if user_id not in chatbot_sessions:
        if not os.path.exists(JSON_FILE_PATH):
            raise FileNotFoundError(f"JSON file not found: {JSON_FILE_PATH}")
        chatbot_sessions[user_id] = UniversityChatbot(JSON_FILE_PATH)
    return chatbot_sessions[user_id]

def reload_all_sessions(new_json_path: str):
    for user_id in chatbot_sessions.keys():
        chatbot_sessions[user_id] = UniversityChatbot(new_json_path)

# -------------------------
# API Endpoints
# -------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        bot = load_chatbot_session(request.user_id)
        response_text = bot.chat(request.message)
        return {"type": "text", "message": response_text, "options": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/welcome")
async def welcome_message():
    return {
        "response": (
            "Welcome! I can answer questions about:\n"
            "Admissions, Academic Calendar, College info, Student services,\n"
            "Policies, FAQs, etc."
        )
    }

@app.post("/scrape", response_model=ScrapeResponse)
def run_scraper():
    try:
        output_path = scrape_and_update()
        reload_all_sessions(output_path)
        return ScrapeResponse(status="completed and reloaded", file_path=output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Scheduler
# -------------------------
scheduler = BackgroundScheduler()

def scheduled_scrape():
    try:
        output_path = scrape_and_update()
        reload_all_sessions(output_path)
        print(f"Auto-scrape completed and reloaded: {output_path}")
    except Exception as e:
        print(f"Auto-scrape failed: {e}")

@app.on_event("startup")
def startup_event():
    if not os.path.exists(JSON_FILE_PATH):
        raise FileNotFoundError(f"JSON file not found: {JSON_FILE_PATH}")
    print("Server startup: Ready.")
    scheduler.add_job(scheduled_scrape, "interval", hours=24)
    scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

# -------------------------
# Serve Frontend Chat GUI
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def chat_gui():
    # Load menu hierarchy from JSON
    with open("data_backups/menu_hierarchy.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>KSU Chatbot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a237e, #6a1b9a);
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            #chat-container {{
                width: 95%;
                max-width: 420px;
                height: 90vh;
                background: #1e1e2f;
                border-radius: 20px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.4);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}
            #messages {{
                flex: 1;
                padding: 15px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }}
            .message {{
                padding: 12px 16px;
                border-radius: 16px;
                max-width: 80%;
                word-wrap: break-word;
                white-space: pre-wrap;
                font-size: 15px;
            }}
            .user {{
                align-self: flex-end;
                background: #4fc3f7;
                color: #000;
            }}
            .bot {{
                align-self: flex-start;
                background: #7e57c2;
                color: #fff;
            }}
            #input-container {{
                display: flex;
                border-top: 1px solid #444;
                background: #2c2c3c;
            }}
            #user-input {{
                flex: 1;
                padding: 12px;
                border: none;
                outline: none;
                font-size: 15px;
                background: #2c2c3c;
                color: #fff;
            }}
            #send-btn {{
                padding: 0 20px;
                background: #512da8;
                color: white;
                border: none;
                cursor: pointer;
                font-weight: bold;
                transition: 0.3s;
            }}
            #send-btn:hover {{
                background: #673ab7;
            }}
            #options {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                padding: 12px;
                border-top: 1px solid #444;
                background: #2c2c3c;
                justify-content: center;
                max-height: 160px;
                overflow-y: auto;
            }}
            .option-btn {{
                padding: 8px 14px;
                border: none;
                border-radius: 12px;
                background: #512da8;
                color: white;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: 0.3s;
            }}
            .option-btn:hover {{
                background: #7e57c2;
            }}
            #welcome-robot {{
                width: 60px;
                height: 60px;
                margin: 0 auto 10px auto;
                display: block;
                animation: bounce 2s infinite;
            }}
            @keyframes bounce {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-10px); }}
            }}
        </style>
    </head>
    <body>
        <div id="chat-container">
            <img id="welcome-robot" src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" alt="robot" />
            <div id="messages"></div>
            <div id="options"></div>
            <div id="input-container">
                <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off">
                <button id="send-btn">Send</button>
            </div>
        </div>

        <script>
            const messagesDiv = document.getElementById("messages");
            const optionsDiv = document.getElementById("options");
            const input = document.getElementById("user-input");
            const sendBtn = document.getElementById("send-btn");

            const menuData = {json.dumps(data)};
            const allowedTopButtons = [
                "Study at KSU",
                "Regulations and Policies",
                "FAQs",
                "Research",
                "Libraries",
                "Academic Calendar",
                "It Helpdesk",
                "Housing"
            ];
            let currentMenu = menuData.filter(item =>
                allowedTopButtons.includes(item.title)
            );
            let menuStack = [];

            async function fetchWelcome() {{
                const res = await fetch("/welcome");
                const data = await res.json();
                appendMessage(data.response, "bot");
                renderOptions(currentMenu);
            }}

            function appendMessage(text, sender) {{
                const div = document.createElement("div");
                div.className = "message " + sender;
                div.innerHTML = text.replace(/\\n/g, "<br>");
                messagesDiv.appendChild(div);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}

            async function sendMessage(text=null) {{
                const msg = text || input.value.trim();
                if (!msg) return;
                appendMessage(msg, "user");
                input.value = "";

                const userId = "default-user";
                const res = await fetch("/chat", {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify({{ user_id: userId, message: msg }})
                }});

                const data = await res.json();
                appendMessage(data.message, "bot");

                if (data.type === "options" && data.options) {{
                    renderOptions(data.options.map(opt => ({{title: opt}})));
                }} else {{
                    renderOptions(currentMenu);
                }}
            }}

            function renderOptions(list) {{
                optionsDiv.innerHTML = "";

                list.forEach(item => {{
                    const btn = document.createElement("button");
                    btn.className = "option-btn";
                    btn.textContent = item.title;
                    btn.onclick = () => {{
                        // Special Undergraduate handling
                        if(item.title.toLowerCase() === "undergraduate") {{
                            appendMessage(
                                "To apply for undergraduate program, you need a high school diploma with a strong GPA (usually 90%+). " +
                                "Some programs require specific subjects, like math and physics for engineering. " +
                                "Saudi students take the GAT, and some programs need the SAAT. " +
                                "English programs require TOEFL or IELTS. You must pass a medical check and be under 25. " +
                                "Some programs may also ask for a personal statement, recommendation letters, or an interview.",
                                "bot"
                            );
                            return;
                        }}

                        sendMessage(item.title);

                        if(item.title.toLowerCase() === "admission requirements") {{
                            let children = [...(item.children || [])];
                            children.push({{ "title": "Undergraduate" }});
                            menuStack.push(currentMenu);
                            currentMenu = children;
                            renderOptions(children);
                        }}
                        else if(item.children && item.children.length > 0) {{
                            menuStack.push(currentMenu);
                            currentMenu = item.children;
                            renderOptions(item.children);
                        }} else {{
                            currentMenu = list;
                        }}
                    }};
                    optionsDiv.appendChild(btn);
                }});

                if(menuStack.length > 0) {{
                    const backBtn = document.createElement("button");
                    backBtn.className = "option-btn";
                    backBtn.textContent = "⬅ Back";
                    backBtn.onclick = () => {{
                        currentMenu = menuStack.pop() || menuData;
                        renderOptions(currentMenu);
                    }};
                    optionsDiv.appendChild(backBtn);
                }}
            }}

            sendBtn.onclick = () => sendMessage();
            input.addEventListener("keypress", e => {{
                if(e.key === "Enter") sendMessage();
            }});

            fetchWelcome();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
