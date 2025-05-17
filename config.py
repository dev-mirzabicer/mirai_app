# mirai_app/config.py
import os
from pathlib import Path

# --- Project Root ---
BASE_DIR = Path(__file__).resolve().parent

# --- Data Directories ---
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
CHATS_DIR = DATA_DIR / "chats"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Files ---
TASKS_FILE = DATA_DIR / "tasks.json"
NOTES_FILE = DATA_DIR / "notes.json"
REMINDERS_FILE = DATA_DIR / "reminders.json"
HABITS_FILE = DATA_DIR / "habits.md"
ABOUT_MIRZA_FILE = DATA_DIR / "about_mirza.md"

# --- API Keys & Credentials (Replace with your actual keys or use environment variables) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GOOGLE_CREDENTIALS_FILE = DATA_DIR / "google_credentials.json"  # For Google Calendar

# CallMeBot (or other communication API) credentials
CALLMEBOT_PHONE_NUMBER = os.getenv(
    "CALLMEBOT_PHONE_NUMBER", "YOUR_CALLMEBOT_PHONE_NUMBER"
)  # Your phone number for WhatsApp
CALLMEBOT_API_KEY_WHATSAPP = os.getenv(
    "CALLMEBOT_API_KEY_WHATSAPP", "YOUR_CALLMEBOT_WHATSAPP_API_KEY"
)
CALLMEBOT_USERNAME_TELEGRAM = os.getenv(
    "CALLMEBOT_USERNAME_TELEGRAM", "YOUR_TELEGRAM_USERNAME"
)
# Add other API keys as needed (e.g., for weather, news)

# --- User Specifics ---
MIRZA_LOCATION_DEFAULT = "Istanbul, Turkey"  # Default location
MIRZA_TIMEZONE = "Europe/Istanbul"  # Important for date/time operations

# --- LLM Settings ---
LLM_MODEL_NAME = "gemini-1.5-flash-latest"  # Or your preferred Gemini model

# --- Chat Settings ---
DAILY_CHAT_FILENAME = "daily_chat.md"
CHAT_INSTANCE_PREFIX = "instance"

# --- System Behavior ---
END_OF_DAY_LOG_TIME = "23:40"  # Time to trigger end-of-day logging
PERIODIC_PROMPT_INTERVAL_MINUTES = 30

if __name__ == "__main__":
    # Quick test to see if paths are resolved correctly
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Tasks File Path: {TASKS_FILE}")
    # Create dummy files if they don't exist for initial setup
    for f_path in [
        TASKS_FILE,
        NOTES_FILE,
        REMINDERS_FILE,
        HABITS_FILE,
        ABOUT_MIRZA_FILE,
    ]:
        if not f_path.exists():
            if f_path.suffix == ".json":
                f_path.write_text("[]")  # Initialize JSON files as empty lists
            else:
                f_path.write_text("")  # Initialize MD files as empty
            print(f"Created dummy file: {f_path}")
