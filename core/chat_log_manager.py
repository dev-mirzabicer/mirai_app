# mirai_app/core/chat_log_manager.py

import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Literal, List, Union, Dict
import re
from google.genai.types import Content, Part

from mirai_app import config
from mirai_app.core import utils

logger = logging.getLogger(__name__)
# BasicConfig will be set by main_orchestrator or if run standalone


class ChatLogManager:
    """
    Manages daily chat logs between Mirza and MIRAI.
    Each day's chat is stored in a separate Markdown file within a dated subdirectory.
    """

    def __init__(self, chats_dir_path: Optional[Union[str, Path]] = None):
        """
        Initializes the ChatLogManager.

        Args:
            chats_dir_path: Path to the base directory where chat subdirectories are stored.
                            Defaults to config.CHATS_DIR.
        """
        self.chats_dir = Path(chats_dir_path) if chats_dir_path else config.CHATS_DIR
        self.chats_dir.mkdir(
            parents=True, exist_ok=True
        )  # Ensure base chats dir exists
        logger.info(
            f"ChatLogManager initialized. Using base chats directory: {self.chats_dir}"
        )

    def _get_daily_chat_filepath(self, chat_date: date) -> Path:
        """
        Constructs the full path to a daily chat log file.
        Ensures the YYYY-MM-DD subdirectory exists.

        Args:
            chat_date: The date for the chat log.

        Returns:
            A Path object to the daily_chat.md file.
        """
        date_subdir_name = chat_date.strftime("%Y-%m-%d")
        date_subdir_path = self.chats_dir / date_subdir_name
        date_subdir_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure YYYY-MM-DD subdir exists
        return date_subdir_path / config.DAILY_CHAT_FILENAME

    async def append_to_daily_chat(
        self,
        chat_date: date,
        speaker: Literal["Mirza", "MIRAI"],
        message: str,
        message_dt_utc: Optional[datetime] = None,
    ) -> bool:
        """
        Appends a message to the daily chat log.

        Args:
            chat_date: The logical date of the conversation.
            speaker: Who sent the message ("Mirza" or "MIRAI").
            message: The content of the message.
            message_dt_utc: The UTC timestamp of when the message was sent/received.
                            If None, current UTC time is used.

        Returns:
            True if appending was successful, False otherwise.
        """
        if not message:
            logger.warning("Attempted to append an empty message to chat log. Skipped.")
            return True  # Or False, depending on desired strictness

        filepath = self._get_daily_chat_filepath(chat_date)

        timestamp_to_log = (
            message_dt_utc if message_dt_utc else utils.get_current_datetime_utc()
        )
        # Format timestamp for readability in the log, including timezone
        formatted_timestamp = utils.format_datetime_for_llm(timestamp_to_log)

        log_entry = f"[{formatted_timestamp}] {speaker}: {message.strip()}\n"

        try:
            with filepath.open("a", encoding="utf-8") as f:
                f.write(log_entry)
            logger.debug(
                f"Appended to daily chat for {chat_date.strftime('%Y-%m-%d')}: [{speaker}] {message[:50]}..."
            )
            return True
        except IOError as e:
            logger.error(f"Failed to append to daily chat file {filepath}: {e}")
            return False

    async def get_daily_chat_content(self, chat_date: date) -> str:
        """
        Retrieves the content of the daily chat log for a specific date.

        Args:
            chat_date: The date of the chat log to retrieve.

        Returns:
            The chat log content as a string, or an empty string if the log doesn't exist or an error occurs.
        """
        filepath = self._get_daily_chat_filepath(chat_date)
        if not filepath.exists():
            logger.debug(
                f"Daily chat file for {chat_date.strftime('%Y-%m-%d')} not found at {filepath}."
            )
            return ""

        try:
            content = filepath.read_text(encoding="utf-8")
            logger.debug(
                f"Retrieved daily chat content for {chat_date.strftime('%Y-%m-%d')} (length: {len(content)})."
            )
            return content
        except IOError as e:
            logger.error(f"Failed to read daily chat file {filepath}: {e}")
            return ""

    async def get_chat_history_for_prompt(
        self, end_date: date, num_days: int = 1
    ) -> str:
        """
        Retrieves a consolidated chat history string for the LLM prompt.

        Args:
            end_date: The most recent date of chat to include (e.g., today's date).
            num_days: The total number of days of chat history to retrieve, ending with end_date.

        Returns:
            A string containing the concatenated chat history, with date separators.
        """
        if num_days < 1:
            return ""

        all_history_parts = []
        for i in range(num_days):
            current_date = end_date - timedelta(days=i)
            daily_content = await self.get_daily_chat_content(current_date)
            if daily_content:
                # Prepend older messages first
                all_history_parts.insert(
                    0,
                    f"--- Chat from {current_date.strftime('%Y-%m-%d')} ---\n{daily_content.strip()}\n",
                )

        if not all_history_parts:
            return ""

        return "\n".join(all_history_parts).strip()

    async def get_daily_chat_history_as_contents(
        self, chat_date: date
    ) -> List[Content]:
        raw_content = await self.get_daily_chat_content(chat_date)
        if not raw_content:
            logger.info(
                f"No chat log content found for {chat_date.strftime('%Y-%m-%d')} to parse into Contents."
            )
            return []

        entries: List[Dict[str, str]] = []
        log_entry_pattern = re.compile(r"^\[(.*?)\]\s+(Mirza|MIRAI):\s+(.*)$")

        for line in raw_content.splitlines():
            m = log_entry_pattern.match(line)
            if m:
                # start a fresh entry
                entries.append({"speaker": m.group(2), "message": m.group(3)})
            else:
                # continuation of the last entry
                if entries:
                    entries[-1]["message"] += "\n" + line.strip()

        history_contents: List[Content] = []
        for e in entries:
            text = e["message"]
            if e["speaker"] == "Mirza":
                text = f"[!Mirza!] {text}"
                history_contents.append(
                    Content(role="user", parts=[Part.from_text(text=text)])
                )
            else:
                history_contents.append(
                    Content(role="model", parts=[Part.from_text(text=text)])
                )

        logger.info(
            f"Parsed {len(history_contents)} messages from chat log for {chat_date.strftime('%Y-%m-%d')} into Content objects."
        )
        return history_contents


if __name__ == "__main__":
    import asyncio

    # Configure logger for standalone testing
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )

    async def main_test():
        test_chats_dir = config.DATA_DIR / "test_chat_manager_chats"
        # Clean up previous test directory
        if test_chats_dir.exists():
            import shutil

            shutil.rmtree(test_chats_dir)
        test_chats_dir.mkdir(parents=True, exist_ok=True)

        manager = ChatLogManager(chats_dir_path=test_chats_dir)
        print(f"--- ChatLogManager Test Initialized ---")
        print(f"Using test chats directory: {manager.chats_dir}")

        today = utils.get_current_datetime_local().date()
        yesterday = today - timedelta(days=1)

        # --- Test Append and Get for Today ---
        print(f"\n--- Testing Appending for Today ({today.strftime('%Y-%m-%d')}) ---")
        await manager.append_to_daily_chat(
            today, "Mirza", "Hello MIRAI, how are you today?"
        )
        # Simulate a slight delay for timestamp difference
        await asyncio.sleep(0.01)
        utc_now_for_mirai = utils.get_current_datetime_utc()
        await manager.append_to_daily_chat(
            today,
            "MIRAI",
            "I'm doing well, Mirza! Ready to assist.",
            message_dt_utc=utc_now_for_mirai,
        )
        await manager.append_to_daily_chat(
            today, "Mirza", "Great! Can you check my tasks?"
        )

        today_chat_content = await manager.get_daily_chat_content(today)
        print(f"\nToday's chat content:\n{today_chat_content.strip()}")
        assert "Hello MIRAI, how are you today?" in today_chat_content
        assert "I'm doing well, Mirza!" in today_chat_content
        assert "Can you check my tasks?" in today_chat_content
        assert today_chat_content.count("Mirza:") == 2
        assert today_chat_content.count("MIRAI:") == 1

        # --- Test Append and Get for Yesterday ---
        print(
            f"\n--- Testing Appending for Yesterday ({yesterday.strftime('%Y-%m-%d')}) ---"
        )
        await manager.append_to_daily_chat(
            yesterday, "Mirza", "Yesterday's first message."
        )
        await manager.append_to_daily_chat(
            yesterday, "MIRAI", "Yesterday's MIRAI response."
        )

        yesterday_chat_content = await manager.get_daily_chat_content(yesterday)
        print(f"\nYesterday's chat content:\n{yesterday_chat_content.strip()}")
        assert "Yesterday's first message." in yesterday_chat_content
        assert "Yesterday's MIRAI response." in yesterday_chat_content

        # --- Test Get Non-Existent Chat ---
        two_days_ago = today - timedelta(days=2)
        non_existent_chat = await manager.get_daily_chat_content(two_days_ago)
        assert non_existent_chat == ""
        print(
            f"\nContent for non-existent chat ({two_days_ago.strftime('%Y-%m-%d')}): Empty as expected."
        )

        # --- Test Get Chat History for Prompt ---
        print("\n--- Testing Get Chat History for Prompt ---")
        # History for today only
        history_today_only = await manager.get_chat_history_for_prompt(
            end_date=today, num_days=1
        )
        print(f"\nChat history (today only):\n{history_today_only}")
        assert f"--- Chat from {today.strftime('%Y-%m-%d')} ---" in history_today_only
        assert "Can you check my tasks?" in history_today_only
        assert (
            f"--- Chat from {yesterday.strftime('%Y-%m-%d')} ---"
            not in history_today_only
        )

        # History for today and yesterday
        history_two_days = await manager.get_chat_history_for_prompt(
            end_date=today, num_days=2
        )
        print(f"\nChat history (today and yesterday):\n{history_two_days}")
        assert f"--- Chat from {today.strftime('%Y-%m-%d')} ---" in history_two_days
        assert f"--- Chat from {yesterday.strftime('%Y-%m-%d')} ---" in history_two_days
        assert "Yesterday's MIRAI response." in history_two_days
        assert "Can you check my tasks?" in history_two_days
        # Check order (yesterday should appear before today in the concatenated string)
        assert history_two_days.find(
            yesterday.strftime("%Y-%m-%d")
        ) < history_two_days.find(today.strftime("%Y-%m-%d"))

        # History for 3 days (including one non-existent)
        history_three_days = await manager.get_chat_history_for_prompt(
            end_date=today, num_days=3
        )
        assert (
            f"--- Chat from {two_days_ago.strftime('%Y-%m-%d')} ---"
            not in history_three_days
        )  # Non-existent
        assert (
            len(history_three_days.split("--- Chat from ")) - 1 == 2
        )  # Should contain 2 day headers

        print("\n--- ChatLogManager Testing Complete ---")
        # Optional: Clean up test directory after tests
        # import shutil
        # shutil.rmtree(test_chats_dir)
        # print(f"Cleaned up test directory: {test_chats_dir}")

    if __name__ == "__main__":
        asyncio.run(main_test())
