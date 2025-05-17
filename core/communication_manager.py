# mirai_app/core/communication_manager.py

import logging
from typing import Optional, Literal

import requests  # We'll need this for actual implementation later

from mirai_app import config

# from mirai_app.core import utils # Not strictly needed for interface, but good practice

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Define supported platforms as literals for type hinting and validation
TextPlatform = Literal["whatsapp", "telegram", "sms"]  # Add more as needed
CallPlatform = Literal[
    "telegram_call", "whatsapp_call", "standard_phone_call"
]  # Add more as needed


class CommunicationManager:
    """
    Manages sending text messages and initiating voice calls to Mirza.
    This class acts as an abstraction layer over specific communication APIs.
    Actual API interactions are placeholder for now.
    """

    def __init__(self):
        """
        Initializes the CommunicationManager.
        API keys and user identifiers are expected to be in config.py.
        """
        # In a full implementation, you might initialize API clients here.
        logger.info(
            "CommunicationManager initialized. (Implementations are placeholders)"
        )
        self.default_text_platform: TextPlatform = (
            "whatsapp"  # Could be set from config
        )
        self.default_call_platform: CallPlatform = (
            "telegram_call"  # Could be set from config
        )

    def _send_whatsapp_callmebot_text(
        self, phone_number: str, api_key: str, message: str
    ) -> bool:
        """
        Placeholder for sending a WhatsApp message via CallMeBot API.
        """
        logger.info(
            f"[PLACEHOLDER] Attempting to send WhatsApp (CallMeBot) text to {phone_number}: '{message[:30]}...'"
        )
        # --- Actual implementation would go here ---
        # Example (do not run without actual keys and understanding CallMeBot API):
        # try:
        #     url = f"https://api.callmebot.com/whatsapp.php?phone={phone_number}&text={requests.utils.quote(message)}&apikey={api_key}"
        #     response = requests.get(url, timeout=10)
        #     response.raise_for_status() # Raise an exception for HTTP errors
        #     if "ERROR" in response.text.upper(): # CallMeBot specific error check
        #         logger.error(f"CallMeBot API error for WhatsApp: {response.text}")
        #         return False
        #     logger.info("WhatsApp (CallMeBot) text sent successfully (placeholder check).")
        #     return True
        # except requests.exceptions.RequestException as e:
        #     logger.error(f"Error sending WhatsApp (CallMeBot) text: {e}")
        #     return False
        # except Exception as e:
        #     logger.error(f"Unexpected error in _send_whatsapp_callmebot_text: {e}")
        #     return False
        return False  # Placeholder return

    def _send_telegram_callmebot_text(self, username: str, message: str) -> bool:
        """
        Placeholder for sending a Telegram message via CallMeBot API.
        Note: CallMeBot Telegram usually requires prior user interaction with the bot.
        """
        logger.info(
            f"[PLACEHOLDER] Attempting to send Telegram (CallMeBot) text to @{username}: '{message[:30]}...'"
        )
        # --- Actual implementation would go here ---
        return False  # Placeholder return

    def _make_telegram_callmebot_call(
        self, username: str, message_to_speak: str
    ) -> bool:
        """
        Placeholder for making a Telegram call via CallMeBot API.
        """
        logger.info(
            f"[PLACEHOLDER] Attempting to make Telegram (CallMeBot) call to @{username} with message: '{message_to_speak[:30]}...'"
        )
        # --- Actual implementation would go here ---
        return False  # Placeholder return

    # --- Public Interface Methods ---

    def send_text_message(
        self, message: str, platform: Optional[TextPlatform] = None
    ) -> bool:
        """
        Sends a text message to Mirza.

        The LLM's direct textual responses will be routed through this function
        by the main_orchestrator. The LLM can also request this function explicitly.

        Args:
            message: The text content of the message.
            platform: The desired platform ("whatsapp", "telegram", "sms").
                      If None, uses the default text platform.

        Returns:
            True if the message was sent (or queued) successfully, False otherwise.
        """
        if not message:
            logger.warning(
                "Attempted to send an empty text message. Operation skipped."
            )
            return False

        chosen_platform = platform or self.default_text_platform
        logger.info(
            f"Preparing to send text message via {chosen_platform}: '{message[:50]}...'"
        )

        if chosen_platform == "whatsapp":
            # These would come from config.py
            phone_num = config.CALLMEBOT_PHONE_NUMBER
            api_key = config.CALLMEBOT_API_KEY_WHATSAPP
            if (
                not phone_num
                or not api_key
                or "YOUR_" in phone_num
                or "YOUR_" in api_key
            ):
                logger.error(
                    "WhatsApp (CallMeBot) credentials not configured properly."
                )
                return False
            return self._send_whatsapp_callmebot_text(phone_num, api_key, message)

        elif chosen_platform == "telegram":
            username = config.CALLMEBOT_USERNAME_TELEGRAM
            if not username or "YOUR_" in username:
                logger.error("Telegram username not configured properly.")
                return False
            return self._send_telegram_callmebot_text(username, message)

        # elif chosen_platform == "sms":
        #     logger.warning("SMS platform not yet implemented (placeholder).")
        #     return False # Placeholder

        else:
            logger.error(f"Unsupported text platform: {chosen_platform}")
            return False

    def make_voice_call(
        self, message_to_speak: str, platform: Optional[CallPlatform] = None
    ) -> bool:
        """
        Initiates a voice call to Mirza, which acts as an audible alert.
        Mirza is not expected to "answer" this call in a traditional way but
        may respond via text message after being alerted.

        The LLM can request this function explicitly (e.g., for critical alerts or wake-ups).

        Args:
            message_to_speak: The message that the automated system will speak during the call.
            platform: The desired call platform ("telegram_call", "whatsapp_call", "standard_phone_call").
                      If None, uses the default call platform.

        Returns:
            True if the call was initiated successfully, False otherwise.
        """
        if not message_to_speak:
            logger.warning(
                "Attempted to make a voice call with an empty message. Operation skipped."
            )
            return False

        chosen_platform = platform or self.default_call_platform
        logger.info(
            f"Preparing to make voice call via {chosen_platform} with message: '{message_to_speak[:50]}...'"
        )

        if chosen_platform == "telegram_call":
            username = config.CALLMEBOT_USERNAME_TELEGRAM
            if not username or "YOUR_" in username:
                logger.error("Telegram username for calls not configured properly.")
                return False
            return self._make_telegram_callmebot_call(username, message_to_speak)

        # elif chosen_platform == "whatsapp_call":
        #     logger.warning("WhatsApp call platform not yet implemented (placeholder).")
        #     return False # Placeholder
        # elif chosen_platform == "standard_phone_call":
        #     logger.warning("Standard phone call platform not yet implemented (placeholder).")
        #     return False # Placeholder

        else:
            logger.error(f"Unsupported call platform: {chosen_platform}")
            return False


if __name__ == "__main__":
    print("\n--- CommunicationManager Test Initialized (Placeholders) ---")
    manager = CommunicationManager()

    # --- Test Send Text Message ---
    print("\n--- Testing Send Text Message (Placeholder) ---")
    test_message = "Hello Mirza, this is a test message from MIRAI!"

    # Test with default platform (WhatsApp)
    print(
        f"Attempting to send via default text platform ({manager.default_text_platform})..."
    )
    success_text_default = manager.send_text_message(test_message)
    print(
        f"Send text (default platform) result: {'Success (Placeholder)' if success_text_default else 'Failed (Placeholder)'}"
    )
    # In a real test, you'd check your phone or API logs. For now, we expect False due to placeholders.
    # assert not success_text_default, "Placeholder should return False or be mocked for True"

    # Test with explicit platform (Telegram)
    print(f"\nAttempting to send via Telegram...")
    success_text_telegram = manager.send_text_message(test_message, platform="telegram")
    print(
        f"Send text (Telegram) result: {'Success (Placeholder)' if success_text_telegram else 'Failed (Placeholder)'}"
    )
    # assert not success_text_telegram

    # Test with unsupported platform
    print(f"\nAttempting to send via unsupported platform (email)...")
    # success_text_unsupported = manager.send_text_message(test_message, platform="email") # This will cause a type error with Literal
    # print(f"Send text (email) result: {'Success (Placeholder)' if success_text_unsupported else 'Failed (Placeholder)'}")
    # For now, we can't directly pass an invalid Literal. The type checker would catch it.
    # We can test the else branch by temporarily modifying the Literal or the chosen_platform.
    # This test is more about the internal logic if somehow an invalid platform string got through.
    manager.default_text_platform = (
        "non_existent_platform"  # Temporarily override for test
    )
    success_text_invalid_internal = manager.send_text_message(
        test_message, platform="non_existent_platform"
    )
    print(
        f"Send text (internally invalid platform) result: {'Success (Placeholder)' if success_text_invalid_internal else 'Failed (Placeholder)'}"
    )
    assert not success_text_invalid_internal
    manager.default_text_platform = "whatsapp"  # Reset to valid default

    # --- Test Make Voice Call ---
    print("\n--- Testing Make Voice Call (Placeholder) ---")
    call_message = "Good morning Mirza! Time to wake up."

    # Test with default platform (Telegram Call)
    print(
        f"Attempting to call via default call platform ({manager.default_call_platform})..."
    )
    success_call_default = manager.make_voice_call(call_message)
    print(
        f"Make call (default platform) result: {'Success (Placeholder)' if success_call_default else 'Failed (Placeholder)'}"
    )
    # assert not success_call_default

    # Test with explicit platform (if we had another placeholder)
    # print(f"\nAttempting to call via WhatsApp Call...")
    # success_call_whatsapp = manager.make_voice_call(call_message, platform="whatsapp_call")
    # print(f"Make call (WhatsApp Call) result: {'Success (Placeholder)' if success_call_whatsapp else 'Failed (Placeholder)'}")
    # assert not success_call_whatsapp

    print("\n--- CommunicationManager Testing Complete (Placeholders) ---")
    print(
        "NOTE: Actual sending depends on configured API keys and implemented private methods."
    )
    print(
        "For now, functions will likely return False or log errors if keys are placeholders."
    )
