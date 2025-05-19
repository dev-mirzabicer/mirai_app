import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent


DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
CHATS_DIR = DATA_DIR / "chats"


DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DIR.mkdir(parents=True, exist_ok=True)


TASKS_FILE = DATA_DIR / "tasks.json"
NOTES_FILE = DATA_DIR / "notes.json"
REMINDERS_FILE = DATA_DIR / "reminders.json"
HABITS_FILE = DATA_DIR / "habits.md"
ABOUT_MIRZA_FILE = DATA_DIR / "about_mirza.md"
CALENDAR_ICS_FILE = DATA_DIR / "calendar.ics"
SYSTEM_PROMPT_FILE = DATA_DIR / "system_prompt.md"


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_FILE = DATA_DIR / "google_credentials.json"


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


MIRZA_TELEGRAM_USER_ID = os.getenv("MIRZA_TELEGRAM_USER_ID")

try:
    MIRZA_TELEGRAM_CHAT_ID = int(
        os.getenv("MIRZA_TELEGRAM_CHAT_ID", str(MIRZA_TELEGRAM_USER_ID))
    )
except ValueError:
    logger.warning(
        "MIRZA_TELEGRAM_CHAT_ID or MIRZA_TELEGRAM_USER_ID from env is not a valid integer. Using 0 as placeholder."
    )
    MIRZA_TELEGRAM_CHAT_ID = 0


MIRZA_LOCATION_DEFAULT = "Istanbul, Turkey"
MIRZA_TIMEZONE = "Europe/Istanbul"

GEMINI_CANDIDATE_COUNT = 1
GEMINI_MAX_OUTPUT_TOKENS = 65536
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95


GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-05-06"
LLM_MAX_FUNCTION_CALL_TURNS = 8


DAILY_CHAT_FILENAME = "daily_chat.md"
CHAT_INSTANCE_PREFIX = "instance"


END_OF_DAY_HOUR = 23
END_OF_DAY_MINUTE = 40
END_OF_DAY_CHECK_INTERVAL_SECONDS = 600
REMINDER_CHECK_INTERVAL_SECONDS = 300
PERIODIC_LLM_PROMPT_INTERVAL_SECONDS = 1800


LOG_LEVEL = "DEBUG"
import logging

logger = logging.getLogger(__name__)

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

                from mirai_app.core.calendar_manager import (
                    CalendarManager,
                )

                temp_cal_manager = CalendarManager(calendar_file_path=f_path)
                logger.info(
                    f"Created dummy calendar file via CalendarManager: {f_path}"
                )
                continue
            else:
                f_path.write_text("")
            print(f"Created dummy file: {f_path}")
