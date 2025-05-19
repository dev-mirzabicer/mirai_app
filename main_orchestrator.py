# mirai_app/main_orchestrator.py

# Standard Library
import asyncio
import logging
import signal
import json
from datetime import datetime, date, timezone, timedelta
from typing import List, Optional, Tuple

# Third-Party
import pytz
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
    Defaults,
)
from google.genai.types import Content, Part, FunctionCall

# MIRAI Application Modules
from mirai_app import config
from mirai_app.core import utils
from mirai_app.core.tasks_manager import TasksManager
from mirai_app.core.notes_manager import NotesManager
from mirai_app.core.calendar_manager import CalendarManager
from mirai_app.core.reminders_manager import RemindersManager
from mirai_app.core.habits_manager import HabitsManager
from mirai_app.core.log_manager import LogManager
from mirai_app.core.communication_manager import CommunicationManager
from mirai_app.core.chat_log_manager import ChatLogManager

from mirai_app.llm_interface.gemini_client import GeminiClient
from mirai_app.llm_interface.prompt_builder import PromptBuilder
from mirai_app.llm_interface.function_handler import FunctionHandler

logger = logging.getLogger(__name__)


class MiraiOrchestrator:
    def __init__(self):
        self._configure_logging()
        logger.info("Initializing MIRAI Orchestrator...")

        self.tasks_mgr = TasksManager()
        self.notes_mgr = NotesManager()
        self.calendar_mgr = CalendarManager()
        self.reminders_mgr = RemindersManager()
        self.habits_mgr = HabitsManager()
        self.log_mgr = LogManager()
        self.chat_log_mgr = ChatLogManager()
        self.comm_mgr = CommunicationManager()

        self.gemini_client = GeminiClient()
        if not self.gemini_client.is_ready():
            logger.critical("GeminiClient failed to initialize. MIRAI cannot function.")
            raise RuntimeError("GeminiClient initialization failed.")

        self.prompt_builder = PromptBuilder(
            tasks_mgr=self.tasks_mgr,
            notes_mgr=self.notes_mgr,
            calendar_mgr=self.calendar_mgr,
            log_mgr=self.log_mgr,
            reminders_mgr=self.reminders_mgr,
            habits_mgr=self.habits_mgr,
            chat_log_mgr=self.chat_log_mgr,
        )

        self.function_handler_instance = FunctionHandler(
            tasks_mgr=self.tasks_mgr,
            notes_mgr=self.notes_mgr,
            calendar_mgr=self.calendar_mgr,
            reminders_mgr=self.reminders_mgr,
            habits_mgr=self.habits_mgr,
            log_mgr=self.log_mgr,
            comm_mgr=self.comm_mgr,
            chat_log_mgr=self.chat_log_mgr,
        )

        if not self.comm_mgr.bot or self.comm_mgr.mirza_chat_id is None:
            logger.warning(
                "CommunicationManager may not be fully functional (missing Telegram Bot Token or Mirza Chat ID)."
            )

        ptb_defaults = Defaults(
            tzinfo=pytz.timezone(config.MIRZA_TIMEZONE),
            parse_mode=None,
        )
        self.ptb_application = (
            ApplicationBuilder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .defaults(ptb_defaults)
            .build()
        )
        self.bot = self.ptb_application.bot

        self.stop_event = asyncio.Event()
        self.periodic_tasks: List[asyncio.Task] = []
        self._shutdown_called_flag = False

        logger.info("MIRAI Orchestrator initialized components.")

    def _configure_logging(self):
        log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
        numeric_log_level = getattr(logging, log_level_str, logging.INFO)

        logging.basicConfig(
            level=numeric_log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        # Quieter Google Auth logs unless debugging auth specifically
        logging.getLogger("google.auth.transport.requests").setLevel(logging.INFO)
        logging.getLogger("google.auth.compute_engine._metadata").setLevel(logging.INFO)
        # PTB logs can be noisy on DEBUG, set to INFO or WARNING for normal operation
        # logging.getLogger("telegram.ext").setLevel(logging.INFO)
        # logging.getLogger("telegram.bot").setLevel(logging.INFO)

    async def _setup_telegram_handlers(self):
        if not config.MIRZA_TELEGRAM_USER_ID or config.MIRZA_TELEGRAM_USER_ID == 0:
            logger.critical(
                "MIRZA_TELEGRAM_USER_ID is not configured. Cannot set up message handlers."
            )
            return

        mirza_text_handler = MessageHandler(
            filters.TEXT
            & filters.User(user_id=config.MIRZA_TELEGRAM_USER_ID)
            & (~filters.COMMAND),
            self._handle_mirza_message,
        )
        self.ptb_application.add_handler(mirza_text_handler)
        self.ptb_application.add_error_handler(self._handle_telegram_error)
        logger.info("Telegram handlers configured.")

    async def _handle_mirza_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or not update.message.text:
            logger.warning("Received an update without message text. Ignoring.")
            return

        message_text = update.message.text
        message_dt_utc = update.message.date
        if message_dt_utc.tzinfo is None:
            message_dt_utc = message_dt_utc.replace(tzinfo=timezone.utc)

        current_chat_date = message_dt_utc.astimezone(
            pytz.timezone(config.MIRZA_TIMEZONE)
        ).date()
        logger.info(
            f"Received message from Mirza (UID: {update.message.from_user.id}): '{message_text[:70]}...'"
        )

        # 1. Log the current user's message
        await self.chat_log_mgr.append_to_daily_chat(
            chat_date=current_chat_date,
            speaker="Mirza",
            message=message_text,
            message_dt_utc=message_dt_utc,
        )

        # 2. Fetch the entire day's chat history, which now includes the message just logged.
        # This list will be in chronological order.
        llm_conversation_history: List[Content] = (
            await self.chat_log_mgr.get_daily_chat_history_as_contents(
                current_chat_date
            )
        )

        # Safety net: if parsing failed or file was empty before this message,
        # ensure the current message is at least present in the history.
        # This situation implies the log file was empty or unparseable before this message.
        if (
            not llm_conversation_history and message_text
        ):  # If history is empty but we have a message
            logger.warning(
                f"Chat history for {current_chat_date} was empty after parsing. Initializing history with current user message."
            )
            llm_conversation_history = [
                Content(role="user", parts=[Part.from_text(text=message_text)])
            ]
        elif (
            not message_text and not llm_conversation_history
        ):  # If message is empty and history is empty
            logger.warning(
                f"Received empty message text and no prior chat history for {current_chat_date}. LLM history will be empty."
            )
            llm_conversation_history = []

        logger.debug(
            f"LLM conversation history initialized with {len(llm_conversation_history)} messages from chat log for {current_chat_date}."
        )
        if llm_conversation_history:
            # Log the last message for verification
            last_msg_in_hist = llm_conversation_history[-1]
            logger.debug(
                f"Last message in history: role='{last_msg_in_hist.role}', text='{last_msg_in_hist.parts[0].text[:70]}...'"
            )

        current_chat_type = "Day Chat"
        system_instruction = await self.prompt_builder.build_system_instruction(
            current_chat_type=current_chat_type,
            current_chat_date_for_history=current_chat_date,  # Pass the date here
        )

        for turn_count in range(config.LLM_MAX_FUNCTION_CALL_TURNS):
            logger.debug(f"LLM Interaction Turn: {turn_count + 1} for Mirza's message.")

            text_response, function_calls, error_msg = (
                await self.gemini_client.generate_response(
                    prompt_history_contents=llm_conversation_history,
                    system_instruction_text=system_instruction,
                )
            )

            if error_msg:
                logger.error(f"LLM API Error processing Mirza's message: {error_msg}")
                await self.comm_mgr.send_text_message(
                    f"MIRAI encountered an LLM error: {error_msg}"
                )
                return

            if function_calls:
                logger.info(
                    f"LLM requested function call(s): {[fc.name for fc in function_calls]}"
                )
                # Add the model's request for function calls to history
                llm_conversation_history.append(
                    Content(
                        role="model",
                        parts=[
                            Part.from_function_call(name=fc.name, args=fc.args)
                            for fc in function_calls
                        ],
                    )
                )

                function_response_parts_for_history = []
                all_fc_successful = True

                for fc_item in function_calls:
                    args_str = json.dumps(dict(fc_item.args))
                    await self.chat_log_mgr.append_to_daily_chat(
                        chat_date=current_chat_date,
                        speaker="MIRAI",
                        message=f"[Function Call Request: {fc_item.name}({args_str})]",
                        message_dt_utc=utils.get_current_datetime_utc(),
                    )
                    response_part_from_handler = (
                        await self.function_handler_instance.dispatch_function_call(
                            fc_item
                        )
                    )
                    function_response_parts_for_history.append(
                        response_part_from_handler
                    )

                    func_resp_content_dict = (
                        response_part_from_handler.function_response.response
                    )
                    func_resp_content_str = json.dumps(
                        dict(func_resp_content_dict), default=str
                    )

                    await self.chat_log_mgr.append_to_daily_chat(
                        chat_date=current_chat_date,
                        speaker="MIRAI",
                        message=f"[Function Call Result: {fc_item.name} -> {func_resp_content_str[:200]}]",
                        message_dt_utc=utils.get_current_datetime_utc(),
                    )
                    if "error" in func_resp_content_dict:
                        all_fc_successful = False
                        logger.warning(
                            f"Function call {fc_item.name} reported an error: {func_resp_content_dict['error']}"
                        )

                llm_conversation_history.append(
                    Content(role="function", parts=function_response_parts_for_history)
                )

                # If the LLM's action was *only* to send messages and all were successful, end interaction.
                is_only_message_sending_actions = all(
                    fc.name in ["send_text_message_to_mirza", "send_alert_to_mirza"]
                    for fc in function_calls
                )

                if is_only_message_sending_actions and all_fc_successful:
                    logger.info(
                        "LLM's primary action was to send message(s). Ending interaction for this user input."
                    )
                    return  # Exit _handle_mirza_message

                # Otherwise, continue the loop for the LLM to process function results

            elif text_response:  # No function calls, but there is a text response
                logger.info(
                    f"LLM final text response to Mirza: '{text_response[:70]}...'"
                )
                await self.comm_mgr.send_text_message(text_response)
                await self.chat_log_mgr.append_to_daily_chat(
                    chat_date=current_chat_date,
                    speaker="MIRAI",
                    message=text_response,
                    message_dt_utc=utils.get_current_datetime_utc(),
                )
                return

            else:
                logger.warning("LLM returned no actionable output for Mirza's message.")
                await self.comm_mgr.send_text_message(
                    "MIRAI processed your request but had no specific action or textual response."
                )
                return

        logger.warning(
            f"LLM interaction for Mirza's message exceeded max turns ({config.LLM_MAX_FUNCTION_CALL_TURNS})."
        )
        await self.comm_mgr.send_text_message(
            "MIRAI seems to be in a processing loop. Please try rephrasing or a different request."
        )

    async def _handle_telegram_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.error(
            f"PTB Error: Exception while handling an update '{update}': {context.error}",
            exc_info=context.error,
        )

    async def _run_single_llm_interaction_for_system(
        self, system_event_description: str, current_date_for_log: date
    ) -> None:
        logger.info(f"System Event for LLM: {system_event_description[:100]}...")
        system_instruction = await self.prompt_builder.build_system_instruction(
            "Day Chat"
        )

        llm_history = [
            Content(role="user", parts=[Part.from_text(text=system_event_description)])
        ]

        for turn_count in range(config.LLM_MAX_FUNCTION_CALL_TURNS):
            logger.debug(f"LLM Interaction Turn: {turn_count + 1} for System Event.")
            text_res, fc_res, err_res = await self.gemini_client.generate_response(
                prompt_history_contents=llm_history,
                system_instruction_text=system_instruction,
            )

            if err_res:
                logger.error(
                    f"LLM error during system event '{system_event_description[:50]}...': {err_res}"
                )
                break

            if fc_res:
                logger.info(
                    f"System Event LLM requested function call(s): {[fc.name for fc in fc_res]}"
                )
                llm_history.append(
                    Content(
                        role="model",
                        parts=[
                            Part.from_function_call(name=f.name, args=f.args)
                            for f in fc_res
                        ],
                    )
                )

                fc_responses_for_history = []
                all_fc_successful = True
                for fc_item in fc_res:
                    args_str = json.dumps(dict(fc_item.args))
                    await self.chat_log_mgr.append_to_daily_chat(
                        current_date_for_log,
                        "MIRAI",
                        f"[System Task Function Call: {fc_item.name}({args_str})]",
                        utils.get_current_datetime_utc(),
                    )
                    response_part = (
                        await self.function_handler_instance.dispatch_function_call(
                            fc_item
                        )
                    )
                    fc_responses_for_history.append(response_part)

                    func_resp_content_dict = response_part.function_response.response
                    func_resp_content_str = json.dumps(dict(func_resp_content_dict))
                    await self.chat_log_mgr.append_to_daily_chat(
                        current_date_for_log,
                        "MIRAI",
                        f"[System Task Function Result: {fc_item.name} -> {func_resp_content_str[:100]}]",
                        utils.get_current_datetime_utc(),
                    )
                    if "error" in func_resp_content_dict:
                        all_fc_successful = False

                llm_history.append(
                    Content(role="function", parts=fc_responses_for_history)
                )

                is_only_message_sending_actions = all(
                    f.name in ["send_text_message_to_mirza", "send_alert_to_mirza"]
                    for f in fc_res
                )
                if is_only_message_sending_actions and all_fc_successful:
                    logger.info(
                        "System event LLM's action was to send message(s). Ending interaction."
                    )
                    break

            elif text_res:
                logger.info(
                    f"LLM response to system event '{system_event_description[:50]}...': {text_res[:100]}..."
                )
                await self.chat_log_mgr.append_to_daily_chat(
                    current_date_for_log,
                    "MIRAI",
                    f"[System Event LLM Response: {text_res}]",
                    utils.get_current_datetime_utc(),
                )
                break
            else:
                logger.warning(
                    f"LLM no output for system event '{system_event_description[:50]}...'."
                )
                break
        else:
            logger.warning(
                f"Max turns reached for system event '{system_event_description[:50]}...'."
            )

    async def _task_reminder_checks(self) -> None:
        logger.info("Reminder check task started.")
        while not self.stop_event.is_set():
            try:
                now_utc = utils.get_current_datetime_utc()
                current_local_date_for_log = now_utc.astimezone(
                    pytz.timezone(config.MIRZA_TIMEZONE)
                ).date()
                triggerable_notifications = (
                    self.reminders_mgr.get_triggerable_notifications(
                        check_until_datetime_utc=now_utc
                    )
                )

                for notification in triggerable_notifications:
                    logger.info(
                        f"Processing triggered reminder: {notification['reminder_id']} - {notification['reminder_description']}"
                    )
                    system_event_desc = (
                        f"[!SYSTEM!] A reminder is due: '{notification['reminder_description']}'. "
                        f"Due time (UTC): {notification['notify_at_utc_str']}. "
                        f"This is {'the main due time' if notification['is_main_due_time'] else 'a pre-notification'}. "
                        f"Configured action: {notification['action_on_trigger']}. "
                        f"Please take the appropriate action."
                    )
                    await self._run_single_llm_interaction_for_system(
                        system_event_desc, current_local_date_for_log
                    )
                    self.reminders_mgr.process_triggered_notification(
                        notification["reminder_id"],
                        notification["notification_point_id"],
                    )
            except Exception as e:
                logger.exception("Error in _task_reminder_checks.")
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=config.REMINDER_CHECK_INTERVAL_SECONDS,
                )
                if self.stop_event.is_set():
                    break
            except asyncio.TimeoutError:
                continue
        logger.info("Reminder check task stopped.")

    async def _task_end_of_day_logging(self) -> None:
        logger.info("End-of-day logging task started.")
        while not self.stop_event.is_set():
            try:
                now_local = utils.get_current_datetime_local()
                current_local_date_for_log = now_local.date()
                if (
                    now_local.hour == config.END_OF_DAY_HOUR
                    and now_local.minute == config.END_OF_DAY_MINUTE
                ):
                    logger.info("Initiating end-of-day log generation.")
                    log_generation_date = now_local.date()
                    todays_chat_for_log = (
                        await self.chat_log_mgr.get_daily_chat_content(
                            log_generation_date
                        )
                    )
                    if not todays_chat_for_log.strip():
                        logger.info(
                            f"No chat content for {log_generation_date.isoformat()} to generate log from."
                        )
                    else:
                        system_event_desc = (
                            f"[!SYSTEM!] The day ({log_generation_date.isoformat()}) is ending. "
                            f"Please review the day's chat log provided below and generate a detailed daily log summary. "
                            f"Use the 'save_daily_log' function with date_str='{log_generation_date.isoformat()}' and the generated content. "
                            f"\n\n--- Chat Log for {log_generation_date.isoformat()} ---\n{todays_chat_for_log}"
                        )
                        await self._run_single_llm_interaction_for_system(
                            system_event_desc, current_local_date_for_log
                        )
                    await asyncio.sleep(61)
                    continue
            except Exception as e:
                logger.exception("Error in _task_end_of_day_logging.")
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=config.END_OF_DAY_CHECK_INTERVAL_SECONDS,
                )
                if self.stop_event.is_set():
                    break
            except asyncio.TimeoutError:
                continue
        logger.info("End-of-day logging task stopped.")

    async def _task_periodic_llm_prompts(self) -> None:
        logger.info("Periodic LLM prompt task started.")
        while not self.stop_event.is_set():
            try:
                # Wait for stop_event or timeout
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=config.PERIODIC_LLM_PROMPT_INTERVAL_SECONDS,
                )
                # If wait_for completed without TimeoutError, stop_event was set.
                logger.info(
                    "Periodic LLM prompt task: stop_event set during wait. Exiting."
                )
                break

            except asyncio.TimeoutError:
                # Interval elapsed. Time to do the work.
                if (
                    self.stop_event.is_set()
                ):  # Check again in case it was set concurrently
                    logger.info(
                        "Periodic LLM prompt task: stop_event set just after timeout. Exiting."
                    )
                    break

                logger.info("Initiating periodic LLM proactive check.")
                try:
                    current_local_dt_for_log = utils.get_current_datetime_local()
                    current_local_date_for_log = current_local_dt_for_log.date()
                    current_time_str = utils.format_datetime_for_llm(
                        current_local_dt_for_log
                    )
                    system_event_desc = (
                        f"[!SYSTEM!] This is a periodic proactive check-in. Current time is {current_time_str}. "
                        f"Review current active tasks, today's/tomorrow's schedule, and Mirza's habits. "
                        f"Are there any proactive actions, reminders, or suggestions you should make? "
                        f"Use appropriate tools to communicate if needed."
                    )
                    await self._run_single_llm_interaction_for_system(
                        system_event_desc, current_local_date_for_log
                    )
                except asyncio.CancelledError:
                    logger.info("Periodic LLM prompt task: Action cancelled.")
                    # Propagate to outer CancelledError handler to stop the task
                    raise
                except Exception as e_action:
                    logger.exception(
                        "Error during periodic LLM proactive check action."
                    )
                    # Action failed, but the task loop continues for the next interval.

            except asyncio.CancelledError:
                logger.info("Periodic LLM prompt task cancelled (likely during wait).")
                break  # Exit the while loop

            except (
                Exception
            ) as e_wait:  # Catch other unexpected errors from wait_for itself
                logger.exception(
                    f"Unexpected error in periodic LLM prompt task's wait logic: {e_wait}"
                )
                break  # Exit the while loop to be safe

        logger.info("Periodic LLM prompt task stopped.")

    async def start_periodic_tasks(self):
        logger.info("Starting periodic tasks...")
        if (
            not hasattr(config, "REMINDER_CHECK_INTERVAL_SECONDS")
            or not hasattr(config, "END_OF_DAY_HOUR")
            or not hasattr(config, "END_OF_DAY_MINUTE")
            or not hasattr(config, "END_OF_DAY_CHECK_INTERVAL_SECONDS")
            or not hasattr(config, "PERIODIC_LLM_PROMPT_INTERVAL_SECONDS")
        ):
            logger.error("Periodic task configurations missing. Cannot start.")
            return

        self.periodic_tasks.append(
            asyncio.create_task(self._task_reminder_checks(), name="ReminderChecks")
        )
        self.periodic_tasks.append(
            asyncio.create_task(self._task_end_of_day_logging(), name="EndOfDayLogging")
        )
        self.periodic_tasks.append(
            asyncio.create_task(
                self._task_periodic_llm_prompts(), name="PeriodicLLMPrompts"
            )
        )
        logger.info(f"{len(self.periodic_tasks)} periodic tasks scheduled.")

    async def stop_periodic_tasks(self):
        logger.info("Stopping periodic tasks...")
        for task in self.periodic_tasks:
            if task and not task.done():
                task.cancel()
        results = await asyncio.gather(*self.periodic_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            task_name = self.periodic_tasks[i].get_name()
            if isinstance(result, asyncio.CancelledError):
                logger.info(f"Periodic task '{task_name}' cancelled successfully.")
            elif isinstance(result, Exception):
                logger.error(
                    f"Error during cancellation/shutdown of task '{task_name}': {result}"
                )
        self.periodic_tasks.clear()
        logger.info("Periodic tasks stopped and cleared.")

    async def run(self):
        await self._setup_telegram_handlers()
        await self.start_periodic_tasks()
        logger.info("Starting Telegram bot components...")
        try:
            if not self.ptb_application:
                logger.error("PTB Application not initialized. Cannot run.")
                return
            await self.ptb_application.initialize()
            logger.info("PTB Application initialized.")
            if not self.ptb_application.updater:
                logger.error("PTB Updater not available. Cannot start polling.")
                return
            await self.ptb_application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
            )
            logger.info("PTB Updater started polling.")
            await self.ptb_application.start()
            logger.info("PTB Application started processing updates.")
            await self.stop_event.wait()
            logger.info(
                "Stop event received, MiraiOrchestrator.run proceeding to shutdown."
            )
        except Exception as e:
            logger.critical(f"Error in MiraiOrchestrator.run: {e}", exc_info=True)
        finally:
            logger.info("MiraiOrchestrator.run finally block reached.")
            await self.shutdown_orchestrator()

    async def shutdown_orchestrator(self):
        logger.info("Initiating MIRAI Orchestrator shutdown sequence...")
        if self.stop_event.is_set() and self._shutdown_called_flag:
            logger.info("Shutdown already in progress or completed.")
            return
        self.stop_event.set()
        self._shutdown_called_flag = True

        if (
            hasattr(self.ptb_application, "updater")
            and self.ptb_application.updater
            and self.ptb_application.updater.running
        ):
            logger.info("Stopping PTB Updater...")
            await self.ptb_application.updater.stop()
        if hasattr(self.ptb_application, "running") and self.ptb_application.running:
            logger.info("Stopping PTB Application...")
            await self.ptb_application.stop()
        await self.stop_periodic_tasks()
        if hasattr(self.ptb_application, "shutdown"):
            logger.info("Shutting down PTB Application resources...")
            await self.ptb_application.shutdown()
        logger.info("MIRAI Orchestrator shutdown complete.")


async def main_orchestrator_entry():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHATS_DIR.mkdir(parents=True, exist_ok=True)
    orchestrator = None
    try:
        orchestrator = MiraiOrchestrator()
    except RuntimeError as e:
        logger.critical(f"Failed to initialize MiraiOrchestrator: {e}")
        return
    except Exception as e:
        logger.critical(
            f"Unexpected error during MiraiOrchestrator initialization: {e}",
            exc_info=True,
        )
        return

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(signal, sig_name),
                lambda s=sig_name: asyncio.create_task(
                    handle_shutdown_signal(s, orchestrator)
                ),
            )
        except (NotImplementedError, RuntimeError) as e:
            logger.warning(
                f"Signal handling for {sig_name} not fully supported or failed: {e}"
            )

    try:
        await orchestrator.run()
    except Exception as e:
        logger.critical(f"Critical error in orchestrator run: {e}", exc_info=True)
    finally:
        logger.info("Orchestrator run loop exited or was interrupted.")
        if orchestrator and not orchestrator.stop_event.is_set():
            logger.info(
                "Ensuring orchestrator shutdown from main_orchestrator_entry finally block..."
            )
            await orchestrator.shutdown_orchestrator()


async def handle_shutdown_signal(signal_name: str, orchestrator: MiraiOrchestrator):
    logger.info(f"Received signal {signal_name}. Initiating graceful shutdown...")
    if (
        orchestrator and not orchestrator.stop_event.is_set()
    ):  # Check orchestrator existence
        await orchestrator.shutdown_orchestrator()


if __name__ == "__main__":
    try:
        asyncio.run(main_orchestrator_entry())
    except KeyboardInterrupt:
        logger.info("Application terminated by KeyboardInterrupt in asyncio.run.")
    except Exception as e:
        logger.critical(
            f"Unhandled exception at top level of __main__: {e}", exc_info=True
        )
    finally:
        logger.info("Application exiting from __main__.")
