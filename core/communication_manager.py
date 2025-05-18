# mirai_app/core/communication_manager.py

import logging
from typing import Optional, Union

import telegram  # From python-telegram-bot
from telegram.error import TelegramError, BadRequest, NetworkError, Forbidden
from telegram.helpers import escape_markdown

from mirai_app.core import utils
from mirai_app import config

logger = logging.getLogger(__name__)
# BasicConfig will be set by main_orchestrator or if run standalone


class CommunicationManager:
    """
    Manages sending outgoing messages to Mirza via Telegram.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        mirza_chat_id: Optional[Union[str, int]] = None,
    ):
        """
        Initializes the CommunicationManager.

        Args:
            bot_token: The Telegram Bot Token. Defaults to config.TELEGRAM_BOT_TOKEN.
            mirza_chat_id: Mirza's Telegram Chat ID. Defaults to config.MIRZA_TELEGRAM_CHAT_ID.
        """
        _bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        _mirza_chat_id = mirza_chat_id or config.MIRZA_TELEGRAM_CHAT_ID

        if "YOUR_TELEGRAM_BOT_TOKEN_HERE" in _bot_token or not _bot_token:
            logger.error(
                "Telegram Bot Token is not configured or is a placeholder. CommunicationManager will not function."
            )
            # Raise an error or handle this more gracefully depending on application requirements
            # For now, we'll allow initialization but sending will fail.
            self.bot: Optional[telegram.Bot] = None
        else:
            self.bot = telegram.Bot(token=_bot_token)

        if (
            isinstance(_mirza_chat_id, str)
            and "YOUR_TELEGRAM_USER_ID_HERE" in _mirza_chat_id
        ):
            logger.error(
                "Mirza Telegram Chat ID is not configured or is a placeholder."
            )
            self.mirza_chat_id: Optional[Union[str, int]] = None
        elif (
            isinstance(_mirza_chat_id, int) and _mirza_chat_id == 0
        ):  # Default placeholder from config
            logger.error(
                "Mirza Telegram Chat ID is not configured (placeholder value 0)."
            )
            self.mirza_chat_id: Optional[Union[str, int]] = None
        else:
            try:
                self.mirza_chat_id = int(_mirza_chat_id)
            except ValueError:
                logger.error(
                    f"MIRZA_TELEGRAM_CHAT_ID '{_mirza_chat_id}' is not a valid integer."
                )
                self.mirza_chat_id = None

        if self.bot and self.mirza_chat_id is not None:
            logger.info(
                f"CommunicationManager initialized for Telegram. Target Chat ID: {self.mirza_chat_id}"
            )
        else:
            logger.warning(
                "CommunicationManager initialized but may not be fully functional due to missing Bot Token or Chat ID."
            )

    async def send_text_message(
        self,
        text: str,
        parse_mode: Optional[
            str
        ] = None,  # e.g., telegram.constants.ParseMode.MARKDOWN_V2
        disable_web_page_preview: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> bool:
        """
        Sends a text message to Mirza via Telegram.

        Args:
            text: The text content of the message.
            parse_mode: Optional. Send 'MarkdownV2', 'HTML' or 'Markdown' (legacy).
            disable_web_page_preview: Optional. Disables link previews for links in this message.
            reply_to_message_id: Optional. If the message is a reply, ID of the original message.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if not self.bot or self.mirza_chat_id is None:
            logger.error(
                "Telegram Bot or Mirza Chat ID not configured. Cannot send message."
            )
            return False
        if not text:
            logger.warning("Attempted to send an empty text message. Skipped.")
            return True  # Or False, depending on desired behavior

        try:
            # Escape text for MarkdownV2 if parse_mode is set to MarkdownV2
            if parse_mode == telegram.constants.ParseMode.MARKDOWN_V2:
                text = escape_markdown(text, version=2)
            elif parse_mode == telegram.constants.ParseMode.MARKDOWN:
                text = escape_markdown(text, version=1)

            await self.bot.send_message(
                chat_id=self.mirza_chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                reply_to_message_id=reply_to_message_id,
            )
            logger.info(
                f"Sent Telegram message to {self.mirza_chat_id}: '{text[:70]}...'"
            )
            return True
        except Forbidden:
            logger.error(
                f"Unauthorized: Bot token might be invalid or bot blocked by user {self.mirza_chat_id}."
            )
            return False
        except BadRequest as e:
            logger.error(
                f"BadRequest sending Telegram message to {self.mirza_chat_id}: {e}"
            )
            return False
        except NetworkError as e:
            logger.error(f"NetworkError sending Telegram message: {e}")
            # Consider retry logic for network errors in a more advanced implementation
            return False
        except TelegramError as e:
            logger.error(f"TelegramError sending message to {self.mirza_chat_id}: {e}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error sending Telegram message: {type(e).__name__} - {e}"
            )
            return False

    async def send_alert_message(
        self, alert_text: str, add_alert_prefix: bool = True
    ) -> bool:
        """
        Sends an alert message to Mirza, typically for high-priority notifications.
        This effectively simulates a "call" by sending a distinct message.

        Args:
            alert_text: The core text of the alert.
            add_alert_prefix: If True, prepends an alert emoji/text.

        Returns:
            True if the alert message was sent successfully, False otherwise.
        """
        if not alert_text:
            logger.warning("Attempted to send an empty alert message. Skipped.")
            return True

        formatted_text = alert_text
        if add_alert_prefix:
            formatted_text = f"🚨 MIRAI ALERT 🚨\n\n{alert_text}"

        return await self.send_text_message(text=formatted_text)


if __name__ == "__main__":
    import asyncio

    # Configure logger for standalone testing
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )

    async def main_test_comm_manager():
        print("\n--- CommunicationManager Test Initialized (Live Telegram Test) ---")

        # Ensure token and chat_id are set in config.py or environment variables
        # For this test to actually send, replace placeholders in config.py or set env vars
        if (
            "YOUR_TELEGRAM_BOT_TOKEN_HERE" in config.TELEGRAM_BOT_TOKEN
            or config.MIRZA_TELEGRAM_CHAT_ID == 0
            or (
                isinstance(config.MIRZA_TELEGRAM_USER_ID, str)
                and "YOUR_TELEGRAM_USER_ID_HERE" in config.MIRZA_TELEGRAM_USER_ID
            )
        ):
            print(
                "Skipping live Telegram tests: TELEGRAM_BOT_TOKEN or MIRZA_TELEGRAM_CHAT_ID not configured."
            )
            print(
                "Please configure them in mirai_app/config.py or via environment variables to run live tests."
            )
            return

        manager = CommunicationManager()
        if not manager.bot or manager.mirza_chat_id is None:
            print(
                "CommunicationManager could not be initialized properly with bot/chat_id. Aborting test."
            )
            return

        # --- Test Send Simple Text Message ---
        print("\n--- Testing Send Simple Text Message ---")
        test_message_simple = f"Hello Mirza, this is a test message from MIRAI's CommunicationManager at {utils.get_current_datetime_local().strftime('%Y-%m-%d %H:%M:%S')}."
        success_simple = await manager.send_text_message(test_message_simple)
        print(f"Send simple text result: {'Success' if success_simple else 'Failed'}")
        assert (
            success_simple  # This will fail if token/chat_id is wrong or bot is blocked
        )

        await asyncio.sleep(1)  # Small delay between messages

        # --- Test Send Alert Message ---
        print("\n--- Testing Send Alert Message ---")
        test_message_alert = (
            "This is a critical alert test. Please acknowledge via chat."
        )
        success_alert = await manager.send_alert_message(test_message_alert)
        print(f"Send alert message result: {'Success' if success_alert else 'Failed'}")
        assert success_alert

        await asyncio.sleep(1)

        # --- Test Send Message with MarkdownV2 ---
        print("\n--- Testing Send Message with MarkdownV2 ---")
        markdown_text = (
            "*Hello* _Mirza_! This is a ~test~ message with `MarkdownV2` formatting.\n"
            "Visit [python-telegram-bot](https://python-telegram-bot.org) for more info."  # Escaped .
        )
        success_markdown = await manager.send_text_message(
            markdown_text, parse_mode=telegram.constants.ParseMode.MARKDOWN_V2
        )
        print(
            f"Send MarkdownV2 text result: {'Success' if success_markdown else 'Failed'}"
        )
        assert success_markdown

        print("\n--- CommunicationManager Live Testing Complete ---")
        print(
            f"Check your Telegram chat with the bot (Chat ID: {manager.mirza_chat_id}) for messages."
        )

    if __name__ == "__main__":
        # This will run the async main_test function
        # Ensure you have a running asyncio event loop if importing and calling elsewhere
        asyncio.run(main_test_comm_manager())
