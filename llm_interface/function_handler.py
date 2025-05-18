# mirai_app/llm_interface/function_handler.py

import json
import logging
import asyncio
from typing import Any, Callable, Coroutine

from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# Type hint for the manager methods
ManagerMethod = Callable[..., Any] | Callable[..., Coroutine[Any, Any, Any]]


class FunctionHandler:
    def __init__(
        self,
        tasks_mgr,  # tasks_manager.TasksManager
        notes_mgr,  # notes_manager.NotesManager
        calendar_mgr,  # calendar_manager.CalendarManager
        reminders_mgr,  # reminders_manager.RemindersManager
        habits_mgr,  # habits_manager.HabitsManager
        log_mgr,  # log_manager.LogManager
        comm_mgr,  # communication_manager.CommunicationManager
        chat_log_mgr,  # chat_log_manager.ChatLogManager
    ):
        self.FUNCTION_MAP: dict[str, ManagerMethod] = {
            # TasksManager
            "create_task": tasks_mgr.create_task,
            "get_task": tasks_mgr.get_task,
            "update_task": tasks_mgr.update_task,
            "delete_task": tasks_mgr.delete_task,
            "query_tasks": tasks_mgr.query_tasks,
            # NotesManager
            "create_note": notes_mgr.create_note,
            "get_note": notes_mgr.get_note,
            "update_note": notes_mgr.update_note,
            "delete_note": notes_mgr.delete_note,
            "query_notes": notes_mgr.query_notes,
            # CalendarManager
            "create_calendar_event": calendar_mgr.create_event,
            "get_calendar_event_details": calendar_mgr.get_event_details,
            "update_calendar_event": calendar_mgr.update_event,
            "delete_calendar_event": calendar_mgr.delete_event,
            "list_calendar_events_in_range": calendar_mgr.list_events_in_range,
            # RemindersManager
            "create_reminder": reminders_mgr.create_reminder,
            # HabitsManager
            "get_habits_content": habits_mgr.get_habits_content,
            "update_habits_content": habits_mgr.update_habits_content,
            # LogManager
            "get_daily_log_content": log_mgr.get_daily_log_content,
            "save_daily_log": log_mgr.save_daily_log,
            "append_to_daily_log": log_mgr.append_to_daily_log,
            # CommunicationManager (async methods)
            "send_text_message_to_mirza": comm_mgr.send_text_message,
            "send_alert_to_mirza": comm_mgr.send_alert_message,
            # ChatLogManager (async methods)
            # Ensure the key matches function_declarations.py
            "get_specific_daily_chat_content": chat_log_mgr.get_daily_chat_content,
            "get_chat_history_for_reference": chat_log_mgr.get_chat_history_for_prompt,
        }
        logger.info("FunctionHandler initialized with provided manager instances.")

    async def dispatch_function_call(
        self,
        function_call: genai_types.FunctionCall,
    ) -> genai_types.Part:
        function_name = function_call.name
        args = dict(function_call.args)

        logger.info(f"Dispatching function call: {function_name} with args: {args}")

        method_to_call = self.FUNCTION_MAP.get(function_name)

        if not method_to_call:
            logger.error(f"Unknown function called: {function_name}")
            return genai_types.Part.from_function_response(
                name=function_name,
                response={"error": f"Unknown function: {function_name}"},
            )

        if function_name in ["create_calendar_event", "update_calendar_event"]:
            rrule_dict_str = args.pop("rrule_dict_str", None)
            if rrule_dict_str:
                try:
                    args["rrule_dict"] = json.loads(rrule_dict_str)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid JSON for rrule_dict_str: {rrule_dict_str}. Error: {e}"
                    )
                    return genai_types.Part.from_function_response(
                        name=function_name,
                        response={
                            "error": f"Invalid JSON format for rrule_dict_str: {e}"
                        },
                    )
            # else:
            # If rrule_dict_str is not provided or empty, ensure rrule_dict is not in args
            # or explicitly set to None if the manager method expects it.
            # The calendar_manager.create_event expects rrule_dict, so if not provided, it's fine.
            # For update, if rrule_dict_str is None/empty, it means no change or remove recurrence.
            # The manager handles rrule_dict=None correctly.
            # pass # No specific action needed here, args["rrule_dict"] will not be set if not parsed

        try:
            if asyncio.iscoroutinefunction(method_to_call):
                result = await method_to_call(**args)
            else:
                result = method_to_call(**args)

            logger.info(
                f"Function {function_name} executed successfully. Result: {str(result)[:200]}..."
            )
            return genai_types.Part.from_function_response(
                name=function_name,
                response={"output": result},
            )
        except ValueError as e:
            logger.error(f"ValueError executing {function_name}: {e}")
            return genai_types.Part.from_function_response(
                name=function_name,
                response={"error": f"Invalid arguments for {function_name}: {e}"},
            )
        except TypeError as e:
            logger.error(f"TypeError executing {function_name} with args {args}: {e}")
            return genai_types.Part.from_function_response(
                name=function_name,
                response={"error": f"Type error in arguments for {function_name}: {e}"},
            )
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError executing {function_name}: {e}")
            return genai_types.Part.from_function_response(
                name=function_name,
                response={"error": f"File not found during {function_name}: {e}"},
            )
        except IOError as e:
            logger.error(f"IOError executing {function_name}: {e}")
            return genai_types.Part.from_function_response(
                name=function_name,
                response={"error": f"I/O error during {function_name}: {e}"},
            )
        except Exception as e:
            logger.exception(f"Unexpected error executing {function_name}: {e}")
            return genai_types.Part.from_function_response(
                name=function_name,
                response={
                    "error": f"An unexpected error occurred: {type(e).__name__} - {e}"
                },
            )
