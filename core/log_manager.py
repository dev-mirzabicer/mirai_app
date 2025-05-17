# mirai_app/core/log_manager.py

import logging
from pathlib import Path
from typing import Optional, Union

from mirai_app import config
from mirai_app.core import utils

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class LogManager:
    """
    Manages daily logs for MIRAI.
    Each day's log is stored as a separate Markdown file.
    """

    def __init__(self, logs_dir_path: Optional[Union[str, Path]] = None):
        """
        Initializes the LogManager.

        Args:
            logs_dir_path: Path to the directory where log files are stored.
                           Defaults to config.LOGS_DIR.
        """
        self.logs_dir = Path(logs_dir_path) if logs_dir_path else config.LOGS_DIR
        # config.py already ensures LOGS_DIR exists, but an explicit check here is fine.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LogManager initialized. Using logs directory: {self.logs_dir}")

    def _get_log_filepath(self, date_str: str) -> Optional[Path]:
        """
        Constructs the full path to a log file for a given date string.
        Validates and normalizes the date_str to 'YYYY-MM-DD' format for the filename.

        Args:
            date_str: The date for the log (e.g., "2023-05-17", "today", "yesterday").
                      "today" and "yesterday" will be resolved.

        Returns:
            A Path object to the log file, or None if date_str is invalid.
        """
        parsed_date = utils.parse_datetime_flexible(
            date_str, tz_aware=False
        )  # We only need the date part
        if not parsed_date:
            logger.error(f"Invalid date_str for log file: '{date_str}'")
            return None

        filename = f"{parsed_date.strftime('%Y-%m-%d')}.md"
        return self.logs_dir / filename

    def save_daily_log(self, date_str: str, content: str) -> bool:
        """
        Saves or overwrites the entire log content for a specific date.
        Typically used for the LLM's end-of-day summary.

        Args:
            date_str: The date for the log (e.g., "YYYY-MM-DD").
            content: The full Markdown content for the log.

        Returns:
            True if saving was successful, False otherwise.
        """
        log_filepath = self._get_log_filepath(date_str)
        if not log_filepath:
            return False

        if utils.write_md_file(log_filepath, content):
            logger.info(
                f"Successfully saved daily log for {date_str} to {log_filepath} (length: {len(content)})."
            )
            return True
        else:
            logger.error(f"Failed to save daily log for {date_str} to {log_filepath}.")
            return False

    def get_daily_log_content(self, date_str: str) -> Optional[str]:
        """
        Retrieves the content of the daily log for a specific date.

        Args:
            date_str: The date for the log (e.g., "YYYY-MM-DD").

        Returns:
            The log content as a string, or an empty string if the log doesn't exist or an error occurs.
            Returns None if the date_str itself is invalid.
        """
        log_filepath = self._get_log_filepath(date_str)
        if not log_filepath:
            return None  # Invalid date_str

        # utils.read_md_file returns default_content (empty string) if file not found
        content = utils.read_md_file(log_filepath, default_content="")
        if log_filepath.exists():
            logger.debug(
                f"Retrieved log content for {date_str} from {log_filepath} (length: {len(content)})."
            )
        else:
            logger.debug(
                f"No log file found for {date_str} at {log_filepath}. Returning empty content."
            )
        return content

    def append_to_daily_log(
        self,
        date_str: str,
        text_to_append: str,
        section_title: Optional[str] = None,
        add_timestamp: bool = True,
    ) -> bool:
        """
        Appends a new entry to the daily log for a specific date.
        If the log file doesn't exist, it will be created.

        Args:
            date_str: The date for the log (e.g., "YYYY-MM-DD").
            text_to_append: The text content for the new entry.
            section_title: Optional title for this appended section.
            add_timestamp: Whether to prefix the appended entry with a timestamp.

        Returns:
            True if appending was successful, False otherwise.
        """
        if not text_to_append:
            logger.warning("Attempted to append empty text to log. Operation skipped.")
            return False  # Or True, depending on desired behavior for empty appends

        log_filepath = self._get_log_filepath(date_str)
        if not log_filepath:
            return False

        current_content = utils.read_md_file(log_filepath, default_content="")

        entry_parts = []
        if current_content:  # Add a separator if there's existing content
            entry_parts.append("\n\n---\n")  # Markdown horizontal rule for separation

        if add_timestamp:
            timestamp_str = utils.format_datetime_for_llm(
                utils.get_current_datetime_local()
            )
            entry_parts.append(f"**Timestamp:** {timestamp_str}\n")

        if section_title:
            entry_parts.append(f"**Section:** {section_title}\n")

        entry_parts.append("\n")  # Ensure a newline before the actual content
        entry_parts.append(text_to_append)

        new_entry_str = "".join(entry_parts)
        updated_content = current_content + new_entry_str

        if utils.write_md_file(log_filepath, updated_content):
            logger.info(
                f"Successfully appended to daily log for {date_str} at {log_filepath}."
            )
            return True
        else:
            logger.error(
                f"Failed to append to daily log for {date_str} at {log_filepath}."
            )
            return False


if __name__ == "__main__":
    # --- Setup for Testing ---
    # Use a subdirectory within LOGS_DIR for tests to keep it clean
    test_logs_subdir = config.LOGS_DIR / "test_log_manager_logs"
    test_logs_subdir.mkdir(parents=True, exist_ok=True)

    # Clean up any pre-existing test files in that subdir
    for f in test_logs_subdir.glob("*.md"):
        f.unlink()

    manager = LogManager(logs_dir_path=test_logs_subdir)
    print(f"\n--- LogManager Test Initialized ---")
    print(f"Using test logs directory: {manager.logs_dir}")

    today_str = utils.get_current_datetime_local().strftime("%Y-%m-%d")
    yesterday_dt = utils.get_current_datetime_local() - utils.timedelta(days=1)
    yesterday_str = yesterday_dt.strftime("%Y-%m-%d")

    # --- Test Get Non-Existent Log ---
    print("\n--- Testing Get Non-Existent Log ---")
    non_existent_content = manager.get_daily_log_content(today_str)
    print(
        f"Content for non-existent log ({today_str}): '{non_existent_content}' (length: {len(non_existent_content)})"
    )
    assert (
        non_existent_content == ""
    ), "Content for non-existent log should be empty string."

    # --- Test Save Daily Log (Create New) ---
    print("\n--- Testing Save Daily Log (Create New) ---")
    log_content_v1 = f"# Daily Log for {today_str}\n\n- Attended morning meeting.\n- Worked on Project Phoenix."
    save_success_v1 = manager.save_daily_log(today_str, log_content_v1)
    assert save_success_v1, f"Failed to save new log for {today_str}."

    retrieved_content_v1 = manager.get_daily_log_content(today_str)
    assert (
        retrieved_content_v1 == log_content_v1
    ), "Retrieved content (v1) does not match saved content."
    print(f"Saved and retrieved log for {today_str} successfully.")

    # --- Test Save Daily Log (Overwrite Existing) ---
    print("\n--- Testing Save Daily Log (Overwrite Existing) ---")
    log_content_v2 = f"# Daily Log for {today_str} (Revised)\n\n- Morning meeting was productive.\n- Made significant progress on Project Phoenix.\n- Afternoon: Code review session."
    save_success_v2 = manager.save_daily_log(today_str, log_content_v2)
    assert save_success_v2, f"Failed to overwrite log for {today_str}."

    retrieved_content_v2 = manager.get_daily_log_content(today_str)
    assert (
        retrieved_content_v2 == log_content_v2
    ), "Retrieved content (v2) does not match overwritten content."
    print(f"Overwritten and retrieved log for {today_str} successfully.")

    # --- Test Append to Daily Log (Existing Log) ---
    print("\n--- Testing Append to Daily Log (Existing Log) ---")
    append_text_1 = "Quick thought: Need to follow up with Jane on the report."
    append_success_1 = manager.append_to_daily_log(
        today_str, append_text_1, section_title="Evening Note"
    )
    assert append_success_1, f"Failed to append to log for {today_str} (1st append)."

    appended_content_1 = manager.get_daily_log_content(today_str)
    assert append_text_1 in appended_content_1, "Appended text (1) not found in log."
    assert "Evening Note" in appended_content_1, "Appended section title (1) not found."
    assert (
        log_content_v2 in appended_content_1
    ), "Original content (v2) lost after append (1)."
    print(f"Appended to log for {today_str} (1st append) successfully.")
    # print(f"Content after 1st append:\n'''\n{appended_content_1}\n'''")

    # --- Test Append to Daily Log (New Log for Yesterday) ---
    print("\n--- Testing Append to Daily Log (New Log) ---")
    append_text_2 = "Late night idea for the UI design."
    append_success_2 = manager.append_to_daily_log(
        yesterday_str, append_text_2, section_title="Late Idea", add_timestamp=True
    )
    assert (
        append_success_2
    ), f"Failed to append to new log for {yesterday_str} (2nd append)."

    appended_content_2 = manager.get_daily_log_content(yesterday_str)
    assert (
        append_text_2 in appended_content_2
    ), "Appended text (2) not found in new log."
    assert "Late Idea" in appended_content_2, "Appended section title (2) not found."
    assert "**Timestamp:**" in appended_content_2, "Timestamp missing in append (2)."
    print(f"Appended to new log for {yesterday_str} (2nd append) successfully.")
    # print(f"Content of new log for {yesterday_str}:\n'''\n{appended_content_2}\n'''")

    # --- Test Append Multiple Times ---
    append_text_3 = "Another quick note for today."
    manager.append_to_daily_log(
        today_str, append_text_3, add_timestamp=False
    )  # No section, no timestamp for variety
    final_content_today = manager.get_daily_log_content(today_str)
    assert append_text_3 in final_content_today
    assert "---" in final_content_today.split(append_text_1)[1]  # Check for separator
    print("Appended multiple times successfully.")

    # --- Test Invalid Date String ---
    print("\n--- Testing Invalid Date String ---")
    invalid_date_content = manager.get_daily_log_content("invalid-date-string")
    assert (
        invalid_date_content is None
    ), "Getting log for invalid date should return None."
    invalid_save = manager.save_daily_log("invalid-date-string", "test")
    assert not invalid_save, "Saving log for invalid date should fail."
    invalid_append = manager.append_to_daily_log("invalid-date-string", "test")
    assert not invalid_append, "Appending to log for invalid date should fail."
    print("Handled invalid date strings correctly.")

    # --- Clean up test files/directory ---
    # print(f"\nCleaning up test log directory: {test_logs_subdir}")
    # for f in test_logs_subdir.glob("*.md"):
    #     f.unlink()
    # test_logs_subdir.rmdir() # Remove the test subdir itself

    print("\n--- LogManager Testing Complete ---")
