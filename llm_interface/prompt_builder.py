# mirai_app/llm_interface/prompt_builder.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Literal, Optional
from datetime import date, timedelta

from mirai_app import config
from mirai_app.core import utils
from mirai_app.core.tasks_manager import TasksManager
from mirai_app.core.notes_manager import NotesManager
from mirai_app.core.calendar_manager import CalendarManager
from mirai_app.core.log_manager import LogManager
from mirai_app.core.reminders_manager import RemindersManager
from mirai_app.core.habits_manager import HabitsManager

# from mirai_app.core.about_mirza_manager import AboutMirzaManager
from mirai_app.core.chat_log_manager import ChatLogManager

# Import ALL_TOOLS to list available functions for the LLM
from mirai_app.llm_interface.function_declarations import ALL_TOOLS

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Constructs the system prompt for MIRAI by fetching dynamic data
    from various managers and injecting it into a template.
    """

    def __init__(
        self,
        tasks_mgr: TasksManager,
        notes_mgr: NotesManager,
        calendar_mgr: CalendarManager,
        log_mgr: LogManager,
        reminders_mgr: RemindersManager,
        habits_mgr: HabitsManager,
        # about_mirza_mgr: AboutMirzaManager,
        chat_log_mgr: ChatLogManager,
    ):
        self.tasks_mgr = tasks_mgr
        self.notes_mgr = notes_mgr
        self.calendar_mgr = calendar_mgr
        self.log_mgr = log_mgr
        self.reminders_mgr = reminders_mgr
        self.habits_mgr = habits_mgr
        # self.about_mirza_mgr = about_mirza_mgr
        self.chat_log_mgr = chat_log_mgr

        try:
            self.system_prompt_template = config.SYSTEM_PROMPT_FILE.read_text(
                encoding="utf-8"
            )
            logger.info(
                f"System prompt template loaded from {config.SYSTEM_PROMPT_FILE}"
            )
        except FileNotFoundError:
            logger.error(
                f"CRITICAL: System prompt template file not found at {config.SYSTEM_PROMPT_FILE}"
            )
            self.system_prompt_template = "ERROR: System prompt template missing."
        except IOError as e:
            logger.error(
                f"CRITICAL: IOError reading system prompt template {config.SYSTEM_PROMPT_FILE}: {e}"
            )
            self.system_prompt_template = (
                f"ERROR: Could not read system prompt template: {e}"
            )

    def _format_task_item(self, task: Dict[str, Any]) -> str:
        due_date_str = "No due date"
        if task.get("due_date"):
            parsed_due = utils.parse_datetime_flexible(task["due_date"])  # Already UTC
            if parsed_due:
                # Convert to local for display to LLM, as LLM thinks in Mirza's local time
                local_due = parsed_due.astimezone(
                    utils.pytz.timezone(config.MIRZA_TIMEZONE)
                )
                due_date_str = f"Due: {utils.format_datetime_for_llm(local_due)}"
        return (
            f"- ID: {task['id']}, Desc: {task['description'][:100]}... ({due_date_str})"
        )

    def _format_note_item(self, note: Dict[str, Any]) -> str:
        expiry_str = "Expires: Never"
        if note.get("expiry_date"):
            parsed_expiry = utils.parse_datetime_flexible(
                note["expiry_date"]
            )  # Already UTC
            if parsed_expiry:
                local_expiry = parsed_expiry.astimezone(
                    utils.pytz.timezone(config.MIRZA_TIMEZONE)
                )
                expiry_str = f"Expires: {utils.format_datetime_for_llm(local_expiry)}"
        title_str = f"Title: {note['title']} - " if note.get("title") else ""
        tags_str = (
            f"Tags: {', '.join(note.get('tags', []))}"
            if note.get("tags")
            else "No tags"
        )
        return f"- ID: {note['id']}, {title_str}Content: {note['content'][:100]}... ({expiry_str}, {tags_str})"

    def _format_calendar_event_item(self, event: Dict[str, Any]) -> str:
        start_dt = event.get("start")
        end_dt = event.get("end")
        start_str = utils.format_datetime_for_llm(start_dt) if start_dt else "N/A"
        end_str = utils.format_datetime_for_llm(end_dt) if end_dt else "N/A"
        summary = event.get("summary", "No Summary")
        location = f", Location: {event['location']}" if event.get("location") else ""
        all_day_str = " (All-day)" if event.get("all_day") else ""
        return f"- {summary}{all_day_str}: From {start_str} to {end_str}{location} (UID: {event.get('uid', 'N/A')}), Description: {event.get('description', 'No Description')[:150]}..."

    def _format_reminder_item(self, reminder: Dict[str, Any]) -> str:
        due_dt_utc_str = reminder.get("due_datetime_utc")
        due_str = "N/A"
        if due_dt_utc_str:
            parsed_due = utils.parse_datetime_flexible(due_dt_utc_str)  # Already UTC
            if parsed_due:
                local_due = parsed_due.astimezone(
                    utils.pytz.timezone(config.MIRZA_TIMEZONE)
                )
                due_str = utils.format_datetime_for_llm(local_due)

        # Format notification points for brevity
        np_count = len(reminder.get("notification_points", []))
        pending_np_count = sum(
            1
            for p in reminder.get("notification_points", [])
            if p.get("status") == "pending"
        )
        np_summary = f"{pending_np_count}/{np_count} pending notifications"

        return f"- ID: {reminder['id']}, Desc: {reminder['description'][:100]}..., Due: {due_str}, Status: {reminder.get('status', 'N/A')}, Notifications: {np_summary}"

    def _format_list_data(
        self,
        items: Optional[List[Dict[str, Any]]],
        item_formatter_func: Callable[[Dict[str, Any]], str],
        empty_message: str,
        max_items: int = 15,  # Limit number of items to prevent overly long prompts
    ) -> str:
        if not items:
            return empty_message
        if len(items) > max_items:
            formatted_items = [item_formatter_func(item) for item in items[:max_items]]
            formatted_items.append(f"... and {len(items) - max_items} more.")
        else:
            formatted_items = [item_formatter_func(item) for item in items]
        return "\n".join(formatted_items) if formatted_items else empty_message

    # def _format_external_abilities(self) -> str:
    #     if not ALL_TOOLS:
    #         return "No external abilities (tools) are currently defined."
    #     abilities_list = []
    #     for tool in ALL_TOOLS:
    #         abilities_list.append(f"- {tool.name}: {tool.description}")
    #     return "\n".join(abilities_list)

    async def build_system_instruction(
        self, current_chat_type: Literal["Day Chat", "Chat Instance"]
    ) -> str:
        """
        Builds the complete system instruction string for the LLM.
        """
        placeholders = {}
        current_local_dt = utils.get_current_datetime_local()
        today_date = current_local_dt.date()
        yesterday_date = today_date - timedelta(days=1)
        today_date_str = today_date.isoformat()
        yesterday_date_str = yesterday_date.isoformat()

        # --- Static & Simple Dynamic Data ---
        try:
            placeholders["{{ABOUT_MIRZA}}"] = (
                # self.about_mirza_mgr.get_about_mirza_content()
                config.ABOUT_MIRZA_FILE.read_text(encoding="utf-8")
                or "No 'About Mirza' content available."
            )
        except Exception as e:
            logger.error(f"Error fetching About Mirza content: {e}")
            placeholders["{{ABOUT_MIRZA}}"] = "Error fetching 'About Mirza' content."

        placeholders["{{CHAT_TYPE}}"] = current_chat_type

        try:
            placeholders["{{MIRZAS_HABITS}}"] = (
                self.habits_mgr.get_habits_content() or "No habits defined."
            )
        except Exception as e:
            logger.error(f"Error fetching habits content: {e}")
            placeholders["{{MIRZAS_HABITS}}"] = "Error fetching habits content."

        placeholders["{{EXTERNAL_ABILITIES}}"] = "None yet."
        placeholders["{{CURRENT_DATETIME}}"] = utils.format_datetime_for_llm(
            current_local_dt
        )
        placeholders["{{LOCATION}}"] = utils.get_mirza_location()  # Assumed synchronous

        # --- Data from Managers (potentially async) ---
        try:
            todays_schedule_items = self.calendar_mgr.get_todays_schedule()
            placeholders["{{TODAYS_SCHEDULE}}"] = self._format_list_data(
                todays_schedule_items,
                self._format_calendar_event_item,
                "No events scheduled for today.",
            )
        except Exception as e:
            logger.error(f"Error fetching today's schedule: {e}")
            logger.error(f"Stack trace: {e.__traceback__}")
            placeholders["{{TODAYS_SCHEDULE}}"] = "Error fetching today's schedule."

        try:
            tomorrows_schedule_items = self.calendar_mgr.get_tomorrows_schedule()
            placeholders["{{TOMORROWS_SCHEDULE}}"] = self._format_list_data(
                tomorrows_schedule_items,
                self._format_calendar_event_item,
                "No events scheduled for tomorrow.",
            )
        except Exception as e:
            logger.error(f"Error fetching tomorrow's schedule: {e}")
            placeholders["{{TOMORROWS_SCHEDULE}}"] = (
                "Error fetching tomorrow's schedule."
            )

        try:
            active_notes_items = (
                self.notes_mgr.get_active_notes()
            )  # Assumed synchronous
            placeholders["{{ACTIVE_NOTES}}"] = self._format_list_data(
                active_notes_items, self._format_note_item, "No active notes."
            )
        except Exception as e:
            logger.error(f"Error fetching active notes: {e}")
            placeholders["{{ACTIVE_NOTES}}"] = "Error fetching active notes."

        try:
            placeholders["{{YESTERDAYS_LOG}}"] = (
                self.log_mgr.get_daily_log_content(yesterday_date_str)
                or "No log entries for yesterday."
            )
        except Exception as e:
            logger.error(f"Error fetching yesterday's log: {e}")
            placeholders["{{YESTERDAYS_LOG}}"] = "Error fetching yesterday's log."

        try:
            placeholders["{{TODAYS_LOG}}"] = (
                self.log_mgr.get_daily_log_content(today_date_str)
                or "No log entries for today yet."
            )
        except Exception as e:
            logger.error(f"Error fetching today's log: {e}")
            placeholders["{{TODAYS_LOG}}"] = "Error fetching today's log."

        try:
            active_tasks_items = (
                self.tasks_mgr.get_active_tasks()
            )  # Assumed synchronous
            placeholders["{{ACTIVE_TASKS}}"] = self._format_list_data(
                active_tasks_items, self._format_task_item, "No active tasks."
            )
        except Exception as e:
            logger.error(f"Error fetching active tasks: {e}")
            placeholders["{{ACTIVE_TASKS}}"] = "Error fetching active tasks."

        try:
            # Combine pending and active reminders for the prompt
            pending_reminders = self.reminders_mgr.query_reminders(status="pending")
            active_reminders_status = self.reminders_mgr.query_reminders(
                status="active"
            )
            all_active_reminders = (pending_reminders or []) + (
                active_reminders_status or []
            )
            # Sort them by due date for consistent display
            all_active_reminders.sort(key=lambda r: r.get("due_datetime_utc") or "")

            placeholders["{{ACTIVE_REMINDERS}}"] = self._format_list_data(
                all_active_reminders, self._format_reminder_item, "No active reminders."
            )
        except Exception as e:
            logger.error(f"Error fetching active reminders: {e}")
            placeholders["{{ACTIVE_REMINDERS}}"] = "Error fetching active reminders."

        # Chat context placeholders
        try:
            chat_instance_refs = self.chat_log_mgr.get_chat_instance_references_for_day(
                today_date
            )
            if chat_instance_refs:
                placeholders["{{CHAT_INSTANCES}}"] = "\n".join(
                    [f"- {ref}" for ref in chat_instance_refs]
                )
            else:
                placeholders["{{CHAT_INSTANCES}}"] = "No other chat instances today."
        except Exception as e:
            logger.error(f"Error fetching chat instance references: {e}")
            placeholders["{{CHAT_INSTANCES}}"] = (
                "Error fetching chat instance references."
            )

        if current_chat_type == "Chat Instance":
            try:
                todays_day_chat_content = self.chat_log_mgr.get_daily_chat_content(
                    today_date
                )
                if todays_day_chat_content:
                    placeholders["{{TODAYS_CHAT_IF_NOT_DAY_CHAT}}"] = (
                        f"\n--- Today's Main Day Chat (for context) ---\n{todays_day_chat_content.strip()}"
                    )
                else:
                    placeholders["{{TODAYS_CHAT_IF_NOT_DAY_CHAT}}"] = (
                        "\n--- Today's Main Day Chat (for context) ---\n(No entries yet in Day Chat)"
                    )
            except Exception as e:
                logger.error(f"Error fetching today's Day Chat content: {e}")
                placeholders["{{TODAYS_CHAT_IF_NOT_DAY_CHAT}}"] = (
                    "Error fetching today's Day Chat content."
                )
        else:  # Current chat is Day Chat
            placeholders["{{TODAYS_CHAT_IF_NOT_DAY_CHAT}}"] = (
                ""  # Or "N/A (This is the Day Chat)"
            )

        # Substitute all placeholders
        prompt_string = self.system_prompt_template
        for placeholder, value in placeholders.items():
            prompt_string = prompt_string.replace(placeholder, str(value))

        # Check for any missed placeholders (should not happen if all are covered)
        if "{{" in prompt_string and "}}" in prompt_string:
            logger.warning(
                f"Unfilled placeholders might remain in the system prompt: {prompt_string[:500]}..."
            )

        return prompt_string


if __name__ == "__main__":
    import asyncio

    # --- Setup for Standalone Testing ---
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create dummy data files and directories if they don't exist for testing
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHATS_DIR.mkdir(parents=True, exist_ok=True)

    # Dummy system_prompt.md
    dummy_system_prompt_content = """
System Prompt for MIRAI
About Mirza: {{ABOUT_MIRZA}}
Current Chat Type: {{CHAT_TYPE}}
Today's Other Chat Instances:
{{CHAT_INSTANCES}}
External Abilities:
{{EXTERNAL_ABILITIES}}
Current Date & Time: {{CURRENT_DATETIME}}
Location: {{LOCATION}}
Today's Schedule:
{{TODAYS_SCHEDULE}}
Mirza's Habits:
{{MIRZAS_HABITS}}
Tomorrow's Schedule:
{{TOMORROWS_SCHEDULE}}
Active Notes:
{{ACTIVE_NOTES}}
Yesterday's Log:
{{YESTERDAYS_LOG}}
Today's Log:
{{TODAYS_LOG}}
Active Tasks:
{{ACTIVE_TASKS}}
Active Reminders:
{{ACTIVE_REMINDERS}}
--- Context from Day Chat (if applicable) ---
{{TODAYS_CHAT_IF_NOT_DAY_CHAT}}
    """
    if not config.SYSTEM_PROMPT_FILE.exists():
        config.SYSTEM_PROMPT_FILE.write_text(
            dummy_system_prompt_content.strip(), encoding="utf-8"
        )

    if not config.ABOUT_MIRZA_FILE.exists():
        config.ABOUT_MIRZA_FILE.write_text(
            "Mirza is developing MIRAI.", encoding="utf-8"
        )
    if not config.HABITS_FILE.exists():
        config.HABITS_FILE.write_text(
            "- Meditate daily\n- Exercise 3x a week", encoding="utf-8"
        )

    # Mock ChatLogManager's new method for testing
    def mock_get_chat_instance_references_for_day(self, chat_date: date) -> List[str]:
        if chat_date == utils.get_current_datetime_local().date():
            return [
                "Instance: 'Test Chat 1' at 10:00 AM",
                "Instance: 'Another Idea' at 02:30 PM",
            ]
        return []

    ChatLogManager.get_chat_instance_references_for_day = mock_get_chat_instance_references_for_day  # type: ignore

    async def test_prompt_builder():
        # Initialize managers (they will create their default files if not present)
        tasks_mgr = TasksManager()
        notes_mgr = NotesManager()
        calendar_mgr = CalendarManager()  # Will create an empty calendar.ics
        log_mgr = LogManager()
        reminders_mgr = RemindersManager()
        habits_mgr = HabitsManager()
        # about_mirza_mgr = AboutMirzaManager()
        chat_log_mgr = ChatLogManager()  # Will create chats dir

        # Add some sample data for more realistic prompt
        tasks_mgr.create_task("Test Task 1", due_date_str="May 19, 2025 10:00 AM")
        notes_mgr.create_note(
            "Test Note 1",
            title="Important Idea",
            expiry_duration_str="2 days",
            tags=["testing"],
        )
        calendar_mgr.create_event(
            "Today's Meeting",
            start_dt_str="May 18, 2025 10:00 AM",
            duration_str="1h",
            location="Office",
        )
        calendar_mgr.create_event(
            "Tomorrow's Plan",
            start_dt_str="May 19, 2025 09:00 AM",
            end_dt_str="May 19, 2025 05:00 PM",
            description="Work on MIRAI",
        )
        log_mgr.save_daily_log(
            (utils.get_current_datetime_local().date() - timedelta(days=1)).isoformat(),
            "# Yesterday's Log\n- Did some work.",
        )
        reminders_mgr.create_reminder(
            "Test Reminder",
            due_datetime_str="May 18, 2025 06:00 PM",
            notify_before_list=["15m"],
        )

        chat_log_mgr.append_to_daily_chat(
            utils.get_current_datetime_local().date(),
            "Mirza",
            "This is the main day chat content.",
        )

        builder = PromptBuilder(
            tasks_mgr,
            notes_mgr,
            calendar_mgr,
            log_mgr,
            reminders_mgr,
            habits_mgr,
            # about_mirza_mgr,
            chat_log_mgr,
        )

        print("\n--- Building System Instruction for 'Day Chat' ---")
        day_chat_prompt = await builder.build_system_instruction(
            current_chat_type="Day Chat"
        )
        print(day_chat_prompt)
        assert "{{ABOUT_MIRZA}}" not in day_chat_prompt  # Check one placeholder
        # assert "Mirza is developing MIRAI." in day_chat_prompt
        assert "This chat is a Day Chat" in day_chat_prompt
        assert "Test Task 1" in day_chat_prompt
        assert "Today's Meeting" in day_chat_prompt
        assert (
            "No other chat instances today." not in day_chat_prompt
        )  # Should list them
        assert "Instance: 'Test Chat 1' at 10:00 AM" in day_chat_prompt
        assert (
            "--- Today's Main Day Chat (for context) ---" not in day_chat_prompt
        )  # Because it IS day chat

        print("\n--- Building System Instruction for 'Chat Instance' ---")
        instance_chat_prompt = await builder.build_system_instruction(
            current_chat_type="Chat Instance"
        )
        print(instance_chat_prompt)
        assert "This chat is a Chat Instance" in instance_chat_prompt
        # assert "--- Today's Main Day Chat (for context) ---" in instance_chat_prompt
        # assert "This is the main day chat content." in instance_chat_prompt

        # Basic cleanup (optional, as files are small and useful for inspection)
        # config.SYSTEM_PROMPT_FILE.unlink(missing_ok=True)
        # config.ABOUT_MIRZA_FILE.unlink(missing_ok=True)
        # config.HABITS_FILE.unlink(missing_ok=True)
        # tasks_mgr.tasks_file.unlink(missing_ok=True)
        # notes_mgr.notes_file.unlink(missing_ok=True)
        # calendar_mgr.calendar_file.unlink(missing_ok=True)
        # reminders_mgr.reminders_file.unlink(missing_ok=True)
        # For logs and chats, might need to remove directories or specific files

    asyncio.run(test_prompt_builder())
