# mirai_app/main_orchestrator.py

# Standard Library
import asyncio
import logging
import signal
import json  # Added json for serializing args in logging
from datetime import datetime, date, timezone, timedelta  # timezone from datetime
from typing import List, Optional, Tuple  # Added Tuple

# Third-Party
import pytz  # Keep pytz for now, as per user request
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,  # For potential future commands
    ContextTypes,
    filters,
    # PicklePersistence, # Optional: for PTB persistence
    Defaults,
)
from google.genai.types import Content, Part  # For constructing LLM history

# MIRAI Application Modules
from mirai_app import config  # Assumes config.py is in mirai_app directory
from mirai_app.core import utils
from mirai_app.core.tasks_manager import TasksManager
from mirai_app.core.notes_manager import NotesManager
from mirai_app.core.calendar_manager import CalendarManager
from mirai_app.core.reminders_manager import RemindersManager
from mirai_app.core.habits_manager import HabitsManager
from mirai_app.core.log_manager import LogManager
from mirai_app.core.communication_manager import CommunicationManager
from mirai_app.core.chat_log_manager import ChatLogManager

# from mirai_app.core.about_mirza_manager import AboutMirzaManager # If it becomes a class

from mirai_app.llm_interface.gemini_client import GeminiClient
from mirai_app.llm_interface.prompt_builder import PromptBuilder
from mirai_app.llm_interface.function_handler import FunctionHandler  # Import the class

# Global logger instance for this module
logger = logging.getLogger(__name__)


class MiraiOrchestrator:
    def __init__(self):
        # 1. Configuration & Logging Setup (early)
        self._configure_logging()
        logger.info("Initializing MIRAI Orchestrator...")

        # 2. Initialize Core Managers (SINGLETONS for the orchestrator)
        self.tasks_mgr = TasksManager()
        self.notes_mgr = NotesManager()
        self.calendar_mgr = CalendarManager()
        self.reminders_mgr = RemindersManager()
        self.habits_mgr = HabitsManager()
        self.log_mgr = LogManager()
        self.chat_log_mgr = ChatLogManager()
        self.comm_mgr = (
            CommunicationManager()
        )  # Initialize before FunctionHandler if it needs it
        # self.about_mirza_mgr = AboutMirzaManager() # If it becomes a class

        # 3. Initialize LLM Interface Components
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

        # Initialize FunctionHandler with orchestrator's manager instances
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

        # 4. Initialize Communication Manager (for outgoing messages) - already done above
        if not self.comm_mgr.bot or self.comm_mgr.mirza_chat_id is None:
            logger.warning(
                "CommunicationManager may not be fully functional (missing Telegram Bot Token or Mirza Chat ID)."
            )

        # 5. Initialize Telegram Bot Application (PTB)
        # Using pytz as per user request to not change this part yet
        ptb_defaults = Defaults(
            tzinfo=pytz.timezone(config.MIRZA_TIMEZONE),
            parse_mode=None,  # Default to plain text; LLM can specify via tool
        )
        # self.persistence = PicklePersistence(filepath=config.DATA_DIR / "ptb_persistence.pickle") # Optional

        self.ptb_application = (
            ApplicationBuilder()
            .token(config.TELEGRAM_BOT_TOKEN)  # Assumes TELEGRAM_BOT_TOKEN in config
            # .persistence(self.persistence) # Optional
            .defaults(ptb_defaults)
            .build()
        )
        self.bot = self.ptb_application.bot  # Convenience accessor

        # 6. State for graceful shutdown and periodic tasks
        self.stop_event = asyncio.Event()
        self.periodic_tasks: List[asyncio.Task] = []

        logger.info("MIRAI Orchestrator initialized components.")

    def _configure_logging(self):
        log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
        numeric_log_level = getattr(logging, log_level_str, logging.INFO)

        logging.basicConfig(
            level=numeric_log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler()
            ],  # Add FileHandler for production if needed
        )
        # Silence overly verbose loggers from libraries if necessary
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("google.auth.compute_engine._metadata").setLevel(
            logging.DEBUG
        )

        # In _configure_logging or at the top of main_orchestrator.py
        logging.getLogger("telegram.ext").setLevel(logging.DEBUG)
        logging.getLogger("telegram.bot").setLevel(logging.DEBUG)

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
        if message_dt_utc.tzinfo is None:  # Should be aware from PTB
            message_dt_utc = message_dt_utc.replace(tzinfo=timezone.utc)

        # Using pytz as per user request to not change this part yet
        current_chat_date = message_dt_utc.astimezone(
            pytz.timezone(config.MIRZA_TIMEZONE)
        ).date()
        logger.info(
            f"Received message from Mirza (UID: {update.message.from_user.id}): '{message_text[:70]}...'"
        )

        await self.chat_log_mgr.append_to_daily_chat(
            chat_date=current_chat_date,
            speaker="Mirza",
            message=message_text,
            message_dt_utc=message_dt_utc,
        )

        current_chat_type = "Day Chat"  # Current simplification
        system_instruction = await self.prompt_builder.build_system_instruction(
            current_chat_type
        )

        llm_conversation_history: List[Content] = [
            Content(role="user", parts=[Part.from_text(text=message_text)])
        ]

        for turn_count in range(
            config.LLM_MAX_FUNCTION_CALL_TURNS
        ):  # Assumes LLM_MAX_FUNCTION_CALL_TURNS in config
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
                return  # Exit this handler on LLM error

            if function_calls:
                logger.info(
                    f"LLM requested function call(s): {[fc.name for fc in function_calls]}"
                )
                fc_parts_for_history = [
                    Part.from_function_call(name=fc.name, args=fc.args)
                    for fc in function_calls
                ]
                llm_conversation_history.append(
                    Content(role="model", parts=fc_parts_for_history)
                )

                function_response_parts_for_history = []
                for fc in function_calls:
                    args_str = json.dumps(
                        dict(fc.args)
                    )  # fc.args is a Struct, convert to dict
                    await self.chat_log_mgr.append_to_daily_chat(
                        chat_date=current_chat_date,
                        speaker="MIRAI",
                        message=f"[Function Call Request: {fc.name}({args_str})]",
                        message_dt_utc=utils.get_current_datetime_utc(),  # Log with current UTC
                    )
                    # USE THE INSTANCE HERE
                    response_part_from_handler = (
                        await self.function_handler_instance.dispatch_function_call(fc)
                    )
                    function_response_parts_for_history.append(
                        response_part_from_handler
                    )

                    # Log the function response (or error from it)
                    func_resp_content = str(
                        response_part_from_handler.function_response.response
                    )
                    await self.chat_log_mgr.append_to_daily_chat(
                        chat_date=current_chat_date,
                        speaker="MIRAI",
                        message=f"[Function Call Result: {fc.name} -> {func_resp_content[:200]}]",  # Log snippet
                        message_dt_utc=utils.get_current_datetime_utc(),
                    )
                llm_conversation_history.append(
                    Content(role="function", parts=function_response_parts_for_history)
                )

            elif text_response:
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
                return  # End of successful interaction

            else:  # No text response and no function calls
                logger.warning("LLM returned no actionable output for Mirza's message.")
                await self.comm_mgr.send_text_message(
                    "MIRAI processed your request but had no specific action or textual response."
                )
                return  # End of interaction

        # This part is reached if the loop completes without returning (max turns exceeded)
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
            "Day Chat"  # Assuming system events also use "Day Chat" context for now
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
                break  # Stop processing this event on LLM error

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
                for fc_item in fc_res:
                    args_str = json.dumps(
                        dict(fc_item.args)
                    )  # fc_item.args is a Struct
                    await self.chat_log_mgr.append_to_daily_chat(
                        current_date_for_log,
                        "MIRAI",
                        f"[System Task Function Call: {fc_item.name}({args_str})]",
                        utils.get_current_datetime_utc(),
                    )
                    # USE THE INSTANCE HERE
                    response_part = (
                        await self.function_handler_instance.dispatch_function_call(
                            fc_item
                        )
                    )
                    fc_responses_for_history.append(response_part)
                    func_resp_content = str(response_part.function_response.response)
                    await self.chat_log_mgr.append_to_daily_chat(
                        current_date_for_log,
                        "MIRAI",
                        f"[System Task Function Result: {fc_item.name} -> {func_resp_content[:100]}]",
                        utils.get_current_datetime_utc(),
                    )
                llm_history.append(
                    Content(role="function", parts=fc_responses_for_history)
                )

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
                break  # Successful text response, end interaction

            else:  # No text response and no function calls
                logger.warning(
                    f"LLM no output for system event '{system_event_description[:50]}...'."
                )
                break  # End interaction
        else:  # Loop completed without break (max turns exceeded)
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
                )  # Assumes in config
                if self.stop_event.is_set():
                    break  # Exit if stop event is set during wait
            except asyncio.TimeoutError:
                continue  # Timeout means it's time for the next check
        logger.info("Reminder check task stopped.")

    async def _task_end_of_day_logging(self) -> None:
        logger.info("End-of-day logging task started.")
        while not self.stop_event.is_set():
            try:
                now_local = utils.get_current_datetime_local()
                current_local_date_for_log = now_local.date()

                # Assumes END_OF_DAY_HOUR and END_OF_DAY_MINUTE in config
                if (
                    now_local.hour == config.END_OF_DAY_HOUR
                    and now_local.minute == config.END_OF_DAY_MINUTE
                ):
                    logger.info("Initiating end-of-day log generation.")
                    log_generation_date = now_local.date()  # Log for "today"
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

                    await asyncio.sleep(
                        61
                    )  # Sleep past the minute to avoid re-triggering
                    continue  # Go to the start of the loop to check stop_event and then sleep for interval

            except Exception as e:
                logger.exception("Error in _task_end_of_day_logging.")

            try:
                # Assumes END_OF_DAY_CHECK_INTERVAL_SECONDS in config
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
                # Assumes PERIODIC_LLM_PROMPT_INTERVAL_SECONDS in config
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=config.PERIODIC_LLM_PROMPT_INTERVAL_SECONDS,
                )
                if self.stop_event.is_set():
                    break

                logger.info("Initiating periodic LLM proactive check.")
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

            except Exception as e:
                logger.exception("Error in _task_periodic_llm_prompts.")
                # If an error occurs, we still wait for the next interval
                # The wait_for at the beginning of the loop handles this.
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
            logger.error(
                "One or more periodic task interval/time configurations are missing. Cannot start periodic tasks."
            )
            return

        self.periodic_tasks.append(asyncio.create_task(self._task_reminder_checks()))
        self.periodic_tasks.append(asyncio.create_task(self._task_end_of_day_logging()))
        self.periodic_tasks.append(
            asyncio.create_task(self._task_periodic_llm_prompts())
        )
        logger.info(f"{len(self.periodic_tasks)} periodic tasks scheduled.")

    async def stop_periodic_tasks(self):
        logger.info("Stopping periodic tasks...")
        for task in self.periodic_tasks:
            if task and not task.done():
                task.cancel()

        # Wait for all tasks to actually finish or be cancelled
        # return_exceptions=True ensures that if a task raised an exception (other than CancelledError),
        # it doesn't stop the gather.
        results = await asyncio.gather(*self.periodic_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            task_name = (
                self.periodic_tasks[i].get_name()
                if hasattr(self.periodic_tasks[i], "get_name")
                else f"Task {i}"
            )
            if isinstance(result, asyncio.CancelledError):
                logger.info(f"Periodic task '{task_name}' cancelled successfully.")
            elif isinstance(result, Exception):
                logger.error(
                    f"Error during cancellation/shutdown of periodic task '{task_name}': {result}"
                )

        self.periodic_tasks.clear()
        logger.info("Periodic tasks stopped and cleared.")

    async def run(self):
        await self._setup_telegram_handlers()  # Setup handlers before starting tasks or polling
        await self.start_periodic_tasks()

        logger.info("Starting Telegram bot components...")
        try:
            if not self.ptb_application:
                logger.error("PTB Application not initialized. Cannot run.")
                return

            await self.ptb_application.initialize()
            logger.info("PTB Application initialized.")

            if not self.ptb_application.updater:
                logger.error(
                    "PTB Updater not available after initialize. Cannot start polling."
                )
                return

            await self.ptb_application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,  # Good for production
                # stop_signals are not handled by updater.start_polling directly
            )
            logger.info("PTB Updater started polling.")

            await self.ptb_application.start()
            logger.info("PTB Application started processing updates.")

            # Keep the run method alive until stop_event is set
            await self.stop_event.wait()
            logger.info(
                "Stop event received, MiraiOrchestrator.run proceeding to shutdown."
            )

        except Exception as e:
            logger.critical(
                f"Error in MiraiOrchestrator.run main try block: {e}", exc_info=True
            )
        finally:
            logger.info("MiraiOrchestrator.run finally block reached.")
            # Ensure shutdown is called, stop_event might have been set by a signal or error
            # The shutdown_orchestrator method is idempotent.
            await self.shutdown_orchestrator()

    async def shutdown_orchestrator(self):
        logger.info("Initiating MIRAI Orchestrator shutdown sequence...")
        if (
            self.stop_event.is_set()
            and hasattr(self, "_shutdown_called_flag")
            and self._shutdown_called_flag
        ):
            logger.info("Shutdown already effectively in progress or completed.")
            return

        self.stop_event.set()
        self._shutdown_called_flag = (
            True  # To prevent re-entry issues if called rapidly
        )

        # Stop PTB components
        if (
            hasattr(self.ptb_application, "updater")
            and self.ptb_application.updater
            and self.ptb_application.updater.running
        ):
            logger.info("Stopping PTB Updater...")
            try:
                await self.ptb_application.updater.stop()
            except Exception as e:
                logger.error(f"Error stopping PTB Updater: {e}", exc_info=True)

        if hasattr(self.ptb_application, "running") and self.ptb_application.running:
            logger.info("Stopping PTB Application...")
            try:
                await self.ptb_application.stop()
            except Exception as e:
                logger.error(f"Error stopping PTB Application: {e}", exc_info=True)

        await self.stop_periodic_tasks()

        if hasattr(self.ptb_application, "shutdown"):
            logger.info("Shutting down PTB Application resources...")
            try:
                await self.ptb_application.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down PTB Application: {e}", exc_info=True)

        logger.info("MIRAI Orchestrator shutdown complete.")


async def main_orchestrator_entry():  # Renamed to avoid conflict if imported
    # Ensure necessary directories exist (config.py should ideally handle this on import or first access)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHATS_DIR.mkdir(parents=True, exist_ok=True)
    # Add other directories like config.TASKS_FILE.parent.mkdir(parents=True, exist_ok=True) if needed

    orchestrator = None
    try:
        orchestrator = MiraiOrchestrator()
    except RuntimeError as e:  # Catch initialization errors (e.g., GeminiClient failed)
        logger.critical(f"Failed to initialize MiraiOrchestrator: {e}")
        return  # Exit if orchestrator can't be created
    except Exception as e:  # Catch any other unexpected init error
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
        except NotImplementedError:  # pragma: no cover
            logger.warning(
                f"Signal handling for {sig_name} not supported on this platform. Manual stop (Ctrl+C) might be abrupt."
            )
        except RuntimeError as e:  # pragma: no cover
            logger.warning(
                f"Could not set signal handler for {sig_name} (perhaps not in main thread?): {e}"
            )

    try:
        await orchestrator.run()
    except Exception as e:  # pragma: no cover
        logger.critical(f"Critical error in orchestrator run: {e}", exc_info=True)
    finally:
        logger.info("Orchestrator run loop exited or was interrupted.")
        if orchestrator and not orchestrator.stop_event.is_set():  # pragma: no cover
            # This ensures shutdown is called if orchestrator.run() exits cleanly
            # without a signal (e.g. if run_polling was to ever return None without error
            # and without stop_event being set - though unlikely with close_loop=False)
            logger.info(
                "Ensuring orchestrator shutdown is called from main_orchestrator_entry finally block..."
            )
            await orchestrator.shutdown_orchestrator()


async def handle_shutdown_signal(signal_name: str, orchestrator: MiraiOrchestrator):
    logger.info(f"Received signal {signal_name}. Initiating graceful shutdown...")
    if not orchestrator.stop_event.is_set():
        await orchestrator.shutdown_orchestrator()

    # After orchestrator shutdown, we might want to cancel any other lingering tasks
    # and stop the event loop if this is the main entry point.
    # This part is tricky as run_polling might still be blocking.
    # The shutdown_orchestrator should handle stopping run_polling.
    # If run_polling has truly stopped, the loop might exit naturally.


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
