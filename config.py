# mirai_app/config.py
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# --- Project Root ---
BASE_DIR = Path(__file__).resolve().parent

# --- Data Directories ---
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
CHATS_DIR = DATA_DIR / "chats"  # For daily chat logs

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure chats base directory exists

# --- Data Files ---
TASKS_FILE = DATA_DIR / "tasks.json"
NOTES_FILE = DATA_DIR / "notes.json"
REMINDERS_FILE = DATA_DIR / "reminders.json"
HABITS_FILE = DATA_DIR / "habits.md"
ABOUT_MIRZA_FILE = DATA_DIR / "about_mirza.md"
CALENDAR_ICS_FILE = DATA_DIR / "calendar.ics"  # Added for CalendarManager
SYSTEM_PROMPT_FILE = DATA_DIR / "system_prompt.md"  # For LLM system prompt

# --- API Keys & Credentials ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCkcXPRpHnhr0e8QPqsyliJvok_jIjJdmo")
GOOGLE_CREDENTIALS_FILE = (
    DATA_DIR / "google_credentials.json"
)  # Kept for potential future use

# --- Telegram Bot Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "7734348087:AAEmMCaUXV3f53zXWlywN3iCAapOBLc0hTk"
)
# Your numerical Telegram User ID. The bot will primarily interact with this user.
# For private chat, this is also the chat_id where the bot sends messages to you.
MIRZA_TELEGRAM_USER_ID = 7192698056
# Ensure MIRZA_TELEGRAM_USER_ID is an integer if loaded from env and used directly as int
try:
    MIRZA_TELEGRAM_CHAT_ID = int(
        os.getenv("MIRZA_TELEGRAM_CHAT_ID", str(MIRZA_TELEGRAM_USER_ID))
    )
except ValueError:
    logger.warning(
        "MIRZA_TELEGRAM_CHAT_ID or MIRZA_TELEGRAM_USER_ID from env is not a valid integer. Using 0 as placeholder."
    )
    MIRZA_TELEGRAM_CHAT_ID = 0  # Placeholder if conversion fails or not set


# --- User Specifics ---
MIRZA_LOCATION_DEFAULT = "Istanbul, Turkey"
MIRZA_TIMEZONE = "Europe/Istanbul"

GEMINI_CANDIDATE_COUNT = 1
GEMINI_MAX_OUTPUT_TOKENS = 65536
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95

# --- LLM Settings ---
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
LLM_MAX_FUNCTION_CALL_TURNS = (
    8  # Maximum number of consecutive function calls the LLM can make
)

# --- Chat Settings ---
DAILY_CHAT_FILENAME = (
    "daily_chat.md"  # Filename for daily chats within YYYY-MM-DD subdirs
)
CHAT_INSTANCE_PREFIX = "instance"  # Kept for now, though functionality is deferred

# --- System Behavior ---
END_OF_DAY_HOUR = 23  # Hour for end-of-day logging (24-hour format)
END_OF_DAY_MINUTE = 40  # Minute for end-of-day logging
END_OF_DAY_CHECK_INTERVAL_SECONDS = (
    600  # How often to check if it's time for EOD logging (e.g., 10 minutes)
)
REMINDER_CHECK_INTERVAL_SECONDS = (
    300  # How often to check for reminders (e.g., 5 minutes)
)
PERIODIC_LLM_PROMPT_INTERVAL_SECONDS = (
    1800  # Interval for periodic LLM prompts (e.g., 30 minutes)
)

# --- Logging Setup (basic, can be expanded) ---
LOG_LEVEL = "DEBUG"  # Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
import logging

logger = logging.getLogger(__name__)  # For config-specific logging if needed
# Basic configuration if no handlers are set by the main app yet
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"Chats Directory: {CHATS_DIR}")
    print(f"Tasks File Path: {TASKS_FILE}")
    print(f"Calendar ICS File Path: {CALENDAR_ICS_FILE}")
    print(
        f"Telegram Bot Token: {'SET' if TELEGRAM_BOT_TOKEN != 'YOUR_TELEGRAM_BOT_TOKEN_HERE' else 'NOT SET (Placeholder)'}"
    )
    print(
        f"Mirza Telegram User ID: {MIRZA_TELEGRAM_USER_ID if MIRZA_TELEGRAM_USER_ID != 'YOUR_TELEGRAM_USER_ID_HERE' else 'NOT SET (Placeholder)'}"
    )
    print(
        f"Mirza Telegram Chat ID (for bot to send messages): {MIRZA_TELEGRAM_CHAT_ID if MIRZA_TELEGRAM_CHAT_ID != 0 else 'NOT SET / Invalid'}"
    )

    # Create dummy files if they don't exist for initial setup
    initial_files_to_check = [
        TASKS_FILE,
        NOTES_FILE,
        REMINDERS_FILE,
        HABITS_FILE,
        ABOUT_MIRZA_FILE,
        CALENDAR_ICS_FILE,
    ]
    for f_path in initial_files_to_check:
        if not f_path.exists():
            if f_path.suffix == ".json":
                f_path.write_text("[]")
            elif f_path.suffix == ".ics":
                # A minimal valid ICS file
                from mirai_app.core.calendar_manager import (
                    CalendarManager,
                )  # Temp import for this

                temp_cal_manager = CalendarManager(
                    calendar_file_path=f_path
                )  # This will create it
                logger.info(
                    f"Created dummy calendar file via CalendarManager: {f_path}"
                )
                continue  # CalendarManager handles its own creation
            else:
                f_path.write_text("")
            print(f"Created dummy file: {f_path}")
