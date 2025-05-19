# mirai_app/llm_interface/function_declarations.py

"""
This module defines the function declarations (tools) that MIRAI (the LLM)
can use. Each declaration follows the Google GenAI SDK's schema for tools.
"""

from google.genai import types as genai_types

from mirai_app import config


# Helper to create a basic STRING schema for parameters
def _string_param_schema(description: str) -> genai_types.Schema:
    return genai_types.Schema(type=genai_types.Type.STRING, description=description)


def _boolean_param_schema(description: str) -> genai_types.Schema:
    return genai_types.Schema(type=genai_types.Type.BOOLEAN, description=description)


def _integer_param_schema(description: str) -> genai_types.Schema:
    return genai_types.Schema(type=genai_types.Type.INTEGER, description=description)


def _array_string_param_schema(description: str) -> genai_types.Schema:
    return genai_types.Schema(
        type=genai_types.Type.ARRAY,
        items=genai_types.Schema(type=genai_types.Type.STRING),
        description=description,
    )


TOOL_DECLARATIONS = [
    # --- TasksManager Functions ---
    genai_types.FunctionDeclaration(
        name="create_task",
        description="Creates a new task for Mirza.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "description": _string_param_schema(
                    "The detailed description of the task."
                ),
                "due_date_str": _string_param_schema(
                    "Optional. The due date as a string (e.g., '2025-12-31 14:30', 'tomorrow at 5 PM', 'next Monday'). If the time is omitted for a date, it defaults to end of that day. It's assumed to be in Mirza's local timezone."
                ),
            },
            required=["description"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="get_task",
        description="Retrieves a specific task by its ID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "task_id": _string_param_schema("The ID of the task to retrieve."),
            },
            required=["task_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="update_task",
        description="Updates an existing task. Allowed fields for update: 'description', 'due_date_str', 'completed'. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "task_id": _string_param_schema("The ID of the task to update."),
                "description": _string_param_schema(
                    "Optional. New description for the task."
                ),
                "due_date_str": _string_param_schema(
                    "Optional. New due date string. Can be set to null/empty to remove due date."
                ),
                "completed": _boolean_param_schema(
                    "Optional. New completion status (true or false)."
                ),
            },
            required=["task_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="delete_task",
        description="Deletes a task by its ID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "task_id": _string_param_schema("The ID of the task to delete."),
            },
            required=["task_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="query_tasks",
        description="Queries tasks based on criteria like completion status, due dates, creation dates, or overdue status. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "completed": _boolean_param_schema(
                    "Optional. Filter by completion status (true for completed, false for active)."
                ),
                "due_before_str": _string_param_schema(
                    "Optional. Filter tasks due before this date/time string."
                ),
                "due_after_str": _string_param_schema(
                    "Optional. Filter tasks due after this date/time string."
                ),
                "created_before_str": _string_param_schema(
                    "Optional. Filter tasks created before this date/time string."
                ),
                "created_after_str": _string_param_schema(
                    "Optional. Filter tasks created after this date/time string."
                ),
                "overdue": _boolean_param_schema(
                    "Optional. If true, filter for tasks that are overdue (not completed and past due date)."
                ),
            },
            required=[],  # All parameters are optional for querying
        ),
    ),
    # --- NotesManager Functions ---
    genai_types.FunctionDeclaration(
        name="create_note",
        description="Creates a new note. Expiry can be set with a duration (e.g., '1 day', '2 weeks', 'forever') or a specific date/time. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "content": _string_param_schema("The main content of the note."),
                "title": _string_param_schema("Optional. Title for the note."),
                "expiry_duration_str": _string_param_schema(
                    "Optional. Duration for expiry (e.g., '1 day', 'forever'). Calculated from creation time."
                ),
                "expiry_date_str": _string_param_schema(
                    "Optional. Specific expiry date/time string. Takes precedence over duration."
                ),
                "tags": _array_string_param_schema(
                    "Optional. List of tags for the note."
                ),
            },
            required=["content"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="get_note",
        description="Retrieves a specific note by its ID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "note_id": _string_param_schema("The ID of the note to retrieve."),
            },
            required=["note_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="update_note",
        description="Updates an existing note. Allowed fields: 'title', 'content', 'tags', 'expiry_date_str', 'expiry_duration_str'. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "note_id": _string_param_schema("The ID of the note to update."),
                "title": _string_param_schema("Optional. New title for the note."),
                "content": _string_param_schema("Optional. New content for the note."),
                "tags": _array_string_param_schema(
                    "Optional. New list of tags for the note."
                ),
                "expiry_date_str": _string_param_schema(
                    "Optional. New specific expiry date/time string. Takes precedence over duration."
                ),
                "expiry_duration_str": _string_param_schema(
                    "Optional. New duration for expiry (e.g., '1 day', 'forever'). Calculated from original creation date if expiry_date_str is not provided."
                ),
            },
            required=["note_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="delete_note",
        description="Deletes a note by its ID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "note_id": _string_param_schema("The ID of the note to delete."),
            },
            required=["note_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="query_notes",
        description="Queries notes based on criteria like active/expired status, creation/expiry dates, tags, or content. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "active": _boolean_param_schema(
                    "Optional. True for active notes, False for expired notes."
                ),
                "created_before_str": _string_param_schema(
                    "Optional. Filter notes created before this date/time string."
                ),
                "created_after_str": _string_param_schema(
                    "Optional. Filter notes created after this date/time string."
                ),
                "expiry_before_str": _string_param_schema(
                    "Optional. Filter notes expiring before this date/time string."
                ),
                "expiry_after_str": _string_param_schema(
                    "Optional. Filter notes expiring after this date/time string."
                ),
                "tags": _array_string_param_schema(
                    "Optional. List of tags to filter by (OR logic)."
                ),
                "content_contains": _string_param_schema(
                    "Optional. Search for notes where title or content contains this text (case-insensitive)."
                ),
            },
            required=[],
        ),
    ),
    # --- CalendarManager Functions ---
    genai_types.FunctionDeclaration(
        name="create_calendar_event",
        description="Creates a new calendar event. Start and end/duration are required. Dates/times should be in Mirza's local timezone. For all-day events, provide dates only (e.g., '2025-12-25').",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "summary": _string_param_schema("The summary or title of the event."),
                "start_dt_str": _string_param_schema(
                    "Start date/time string of the event."
                ),
                "end_dt_str": _string_param_schema(
                    "Optional. End date/time string of the event. Required if duration_str is not provided."
                ),
                "duration_str": _string_param_schema(
                    "Optional. Duration of the event (e.g., '1h', '30min', '2 days'). Required if end_dt_str is not provided. For all-day events, use '1d' for a single day event if end_dt_str is not given."
                ),
                "description": _string_param_schema(
                    "Optional. Description of the event."
                ),
                "location": _string_param_schema("Optional. Location of the event."),
                "rrule_dict_str": _string_param_schema(
                    'Optional. A JSON string representing the recurrence rule (RRULE) dictionary (e.g., \'{"FREQ": "WEEKLY", "BYDAY": "MO", "COUNT": 5}\'). See icalendar vRecur documentation for format. UNTIL dates within RRULE should also be in Mirza\'s local timezone.'
                ),
                "timezone_str": _string_param_schema(
                    f"Optional. Timezone for the event (e.g., 'Europe/Istanbul'). Defaults to Mirza's configured timezone: {config.MIRZA_TIMEZONE} if not specified and times are naive."
                ),
            },
            required=["summary", "start_dt_str"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="get_calendar_event_details",
        description="Retrieves the details of a calendar event by its UID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "uid": _string_param_schema(
                    "The UID of the calendar event to retrieve."
                ),
            },
            required=["uid"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="update_calendar_event",
        description="Updates an existing calendar event. Provide only the fields to be changed. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "uid": _string_param_schema("The UID of the event to update."),
                "summary": _string_param_schema("Optional. New summary for the event."),
                "start_dt_str": _string_param_schema(
                    "Optional. New start date/time string."
                ),
                "end_dt_str": _string_param_schema(
                    "Optional. New end date/time string. If provided, duration_str will be ignored for this update."
                ),
                "duration_str": _string_param_schema(
                    "Optional. New duration string. Ignored if end_dt_str is provided."
                ),
                "description": _string_param_schema(
                    "Optional. New description for the event."
                ),
                "location": _string_param_schema(
                    "Optional. New location for the event."
                ),
                "rrule_dict_str": _string_param_schema(
                    "Optional. New JSON string for RRULE dictionary. Pass null or empty string to remove recurrence."
                ),
                "timezone_str": _string_param_schema(
                    f"Optional. Timezone for updated date/time strings if they are naive. Defaults to Mirza's configured timezone: {config.MIRZA_TIMEZONE}."
                ),
            },
            required=["uid"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="delete_calendar_event",
        description="Deletes a calendar event by its UID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "uid": _string_param_schema("The UID of the calendar event to delete."),
            },
            required=["uid"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="list_calendar_events_in_range",
        description="Lists all event occurrences within a specified date/time range. Dates/times should be in Mirza's local timezone. Results will be in the target_timezone_str.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "start_range_str": _string_param_schema(
                    "Start date/time string of the range."
                ),
                "end_range_str": _string_param_schema(
                    "End date/time string of the range."
                ),
                "target_timezone_str": _string_param_schema(
                    f"Optional. Timezone for the results (e.g., 'America/New_York'). Defaults to Mirza's configured timezone: {config.MIRZA_TIMEZONE}."
                ),
            },
            required=["start_range_str", "end_range_str"],
        ),
    ),
    # --- RemindersManager Functions ---
    genai_types.FunctionDeclaration(
        name="create_reminder",
        description="Creates a new reminder for Mirza. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "description": _string_param_schema(
                    "Description of what to be reminded of."
                ),
                "due_datetime_str": _string_param_schema(
                    "The specific date/time for the reminder."
                ),
                "notify_before_list": _array_string_param_schema(
                    "Optional. List of durations before the due time to also send notifications (e.g., ['5m', '1h', '1d'])."
                ),
                "related_item_id": _string_param_schema(
                    "Optional. ID of a related task, note, or event."
                ),
                "action_on_trigger": _string_param_schema(
                    "Optional. Action to take when reminder triggers. Default is 'prompt_llm'. Can be 'text_mirza', 'call_mirza' (for main due time)."
                ),
            },
            required=["description", "due_datetime_str"],
        ),
    ),
    # get_reminder, update_reminder, delete_reminder, query_reminders can be added similarly if LLM needs to manage them directly.
    genai_types.FunctionDeclaration(
        name="get_reminder",
        description="Retrieves a specific reminder by its ID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "reminder_id": _string_param_schema(
                    "The ID of the reminder to retrieve."
                ),
            },
            required=["reminder_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="update_reminder",
        description="Updates an existing reminder. Provide only the fields to be changed. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "reminder_id": _string_param_schema(
                    "The ID of the reminder to update."
                ),
                "description": _string_param_schema(
                    "Optional. New description for the reminder."
                ),
                "due_datetime_str": _string_param_schema(
                    "Optional. New due date/time string for the reminder."
                ),
                "notify_before_list": _array_string_param_schema(
                    "Optional. New list of durations for pre-notifications (e.g., ['5m', '1h']). Set to empty list to remove pre-notifications."
                ),
                "status": _string_param_schema(
                    "Optional. New status for the reminder (e.g., 'pending', 'active', 'completed', 'dismissed')."
                ),
                "related_item_id": _string_param_schema(
                    "Optional. New ID of a related task, note, or event. Set to null/empty to remove."
                ),
                "action_on_trigger": _string_param_schema(
                    "Optional. New action to take when reminder triggers."
                ),
            },
            required=["reminder_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="delete_reminder",
        description="Deletes a reminder by its ID.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "reminder_id": _string_param_schema(
                    "The ID of the reminder to delete."
                ),
            },
            required=["reminder_id"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="query_reminders",
        description="Queries reminders based on criteria like status or due date range. Dates/times should be in Mirza's local timezone.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "status": _string_param_schema(
                    "Optional. Filter reminders by status (e.g., 'pending', 'active', 'completed', 'dismissed')."
                ),
                "due_before_str": _string_param_schema(
                    "Optional. Filter reminders due before this date/time string."
                ),
                "due_after_str": _string_param_schema(
                    "Optional. Filter reminders due after this date/time string."
                ),
            },
            required=[],  # All parameters are optional
        ),
    ),
    # --- HabitsManager Functions ---
    genai_types.FunctionDeclaration(
        name="get_habits_content",
        description="Retrieves the current list of Mirza's habits as a Markdown string.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT, properties={}, required=[]
        ),
    ),
    genai_types.FunctionDeclaration(
        name="update_habits_content",
        description="Updates Mirza's habits. The entire habits content (Markdown string) should be provided.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "new_content": _string_param_schema(
                    "The new, complete Markdown content for Mirza's habits."
                ),
            },
            required=["new_content"],
        ),
    ),
    # --- LogManager Functions ---
    genai_types.FunctionDeclaration(
        name="get_daily_log_content",
        description="Retrieves the content of the daily log for a specific date (e.g., 'YYYY-MM-DD', 'today', 'yesterday').",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "date_str": _string_param_schema("The date for the log to retrieve."),
            },
            required=["date_str"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="save_daily_log",
        description="Saves or overwrites the entire log content for a specific date. Used for end-of-day summaries.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "date_str": _string_param_schema(
                    "The date for the log (e.g., 'YYYY-MM-DD', 'today')."
                ),
                "content": _string_param_schema(
                    "The full Markdown content for the log."
                ),
            },
            required=["date_str", "content"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="append_to_daily_log",
        description="Appends a new entry to the daily log for a specific date. Useful for adding quick notes or observations during the day.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "date_str": _string_param_schema(
                    "The date for the log (e.g., 'YYYY-MM-DD', 'today')."
                ),
                "text_to_append": _string_param_schema(
                    "The text content for the new log entry."
                ),
                "section_title": _string_param_schema(
                    "Optional. A title for this appended section within the log."
                ),
                "add_timestamp": _boolean_param_schema(
                    "Optional. Whether to prefix the appended entry with a timestamp. Defaults to true."
                ),
            },
            required=["date_str", "text_to_append"],
        ),
    ),
    # --- CommunicationManager Functions ---
    genai_types.FunctionDeclaration(
        name="send_text_message_to_mirza",  # Renamed for clarity vs internal system messages
        description="Sends a text message to Mirza via Telegram.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "text": _string_param_schema(
                    "The text content of the message to send to Mirza."
                ),
                "parse_mode": _string_param_schema(
                    "Optional. Send 'MarkdownV2', 'HTML'. If not set, plain text."
                ),
            },
            required=["text"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="send_alert_to_mirza",  # Renamed for clarity
        description="Sends an alert message (high-priority notification) to Mirza via Telegram. Simulates a 'call'.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "alert_text": _string_param_schema("The core text of the alert."),
                "add_alert_prefix": _boolean_param_schema(
                    "Optional. If true (default), prepends an alert emoji/text like '🚨 MIRAI ALERT 🚨'."
                ),
            },
            required=["alert_text"],
        ),
    ),
    # --- ChatLogManager Functions (for LLM to access past chats if needed beyond prompt context) ---
    genai_types.FunctionDeclaration(
        name="get_specific_daily_chat_content",  # Renamed to avoid conflict with LogManager's get_daily_log_content
        description="Retrieves the full chat content for a specific date (e.g., 'YYYY-MM-DD', 'yesterday').",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "chat_date_str": _string_param_schema(
                    "The date of the chat log to retrieve."
                ),
            },
            required=["chat_date_str"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="get_chat_history_for_reference",  # Renamed
        description="Retrieves a consolidated chat history string for a specified number of past days, ending with a specific date. Useful for broader context than the immediate prompt history.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "end_date_str": _string_param_schema(
                    "The most recent date of chat to include (e.g., 'today', 'yesterday')."
                ),
                "num_days": _integer_param_schema(
                    "The total number of days of chat history to retrieve, ending with end_date_str. E.g., 3 for last 3 days."
                ),
            },
            required=["end_date_str", "num_days"],
        ),
    ),
    # TODO: Add external tools like Google Search, News, Weather, Fitbit later.
    # For now, focus on internal MIRAI functions.
]

# To be imported by gemini_client.py for model initialization
# and by function_handler.py for dispatching.

# -- NEW -----------------------------------------------------------------
# One tool that bundles every declaration.
ALL_TOOLS: list[genai_types.Tool] = [
    genai_types.Tool(function_declarations=TOOL_DECLARATIONS)
]
# ------------------------------------------------------------------------
