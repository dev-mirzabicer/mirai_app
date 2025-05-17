# mirai_app/core/habits_manager.py

import logging
from pathlib import Path
from typing import Optional

from mirai_app import config
from mirai_app.core import utils

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class HabitsManager:
    """
    Manages Mirza's habits, stored in a Markdown file.
    The LLM is responsible for interpreting and acting upon these habits.
    This manager provides the interface to read and update the habits content.
    """

    def __init__(self, habits_file_path: Optional[str] = None):
        """
        Initializes the HabitsManager.

        Args:
            habits_file_path: Path to the Markdown file storing habits.
                              Defaults to config.HABITS_FILE.
        """
        self.habits_file = (
            Path(habits_file_path) if habits_file_path else config.HABITS_FILE
        )
        # Ensure the file exists with default content if it's missing,
        # utils.read_md_file handles this if default_content is provided.
        # For habits, an empty string is a fine default if the file is truly new.
        initial_content = utils.read_md_file(self.habits_file, default_content="")
        if (
            not self.habits_file.exists()
        ):  # If read_md_file didn't create it because default was empty
            utils.write_md_file(self.habits_file, "")  # Ensure it exists

        logger.info(f"HabitsManager initialized. Using habits file: {self.habits_file}")
        if not initial_content:
            logger.info("Habits file is currently empty or was just created.")
        else:
            logger.info(
                f"Habits file loaded with existing content (length: {len(initial_content)})."
            )

    def get_habits_content(self) -> str:
        """
        Retrieves the current content of the habits Markdown file.

        Returns:
            The string content of the habits file.
        """
        content = utils.read_md_file(self.habits_file, default_content="")
        logger.debug(
            f"Retrieved habits content (length: {len(content)}) from {self.habits_file}"
        )
        return content

    def update_habits_content(self, new_content: str) -> bool:
        """
        Updates the content of the habits Markdown file.
        This will overwrite the existing content.

        Args:
            new_content: The new string content for the habits file.

        Returns:
            True if the update was successful, False otherwise.
        """
        if utils.write_md_file(self.habits_file, new_content):
            logger.info(
                f"Successfully updated habits content in {self.habits_file} (new length: {len(new_content)})."
            )
            return True
        else:
            logger.error(f"Failed to update habits content in {self.habits_file}.")
            return False


if __name__ == "__main__":
    # --- Setup for Testing ---
    test_habits_file = config.DATA_DIR / "test_habits.md"
    # Clean up from previous test if it exists
    if test_habits_file.exists():
        test_habits_file.unlink()

    manager = HabitsManager(habits_file_path=str(test_habits_file))
    print(f"\n--- HabitsManager Test Initialized ---")
    print(f"Using test file: {manager.habits_file}")

    # --- Test Get Initial Content (should be empty or default) ---
    print("\n--- Testing Get Initial Content ---")
    initial_content = manager.get_habits_content()
    print(f"Initial habits content:\n'''\n{initial_content}\n'''")
    assert initial_content == "", "Initial content should be empty for a new test file."

    # --- Test Update Content ---
    print("\n--- Testing Update Content ---")
    sample_habits_v1 = """
# Mirza's Habits - Version 1

## Morning Routine
- Wake up at 7:00 AM
- Meditate for 10 minutes
- Drink a glass of water

## Evening Routine
- Read for 30 minutes before bed
- Plan tasks for the next day
    """.strip()

    update_success_v1 = manager.update_habits_content(sample_habits_v1)
    assert update_success_v1, "Failed to update habits content (v1)."
    print("Updated habits with Version 1 content.")

    # --- Test Get Updated Content ---
    retrieved_content_v1 = manager.get_habits_content()
    print(f"\nRetrieved habits content (Version 1):\n'''\n{retrieved_content_v1}\n'''")
    assert (
        retrieved_content_v1 == sample_habits_v1
    ), "Retrieved content (v1) does not match updated content."

    # --- Test Overwrite Content ---
    print("\n--- Testing Overwrite Content ---")
    sample_habits_v2 = """
# Mirza's Habits - Version 2 (Revised)

## Daily Focus
- **Mindfulness:** 2x 10-minute sessions (morning, afternoon)
- **Hydration:** Target 3 liters of water.
- **Learning:** Dedicate 1 hour to skill development.

## Weekly Goals
- Exercise: 3x strength, 2x cardio sessions.
    """.strip()

    update_success_v2 = manager.update_habits_content(sample_habits_v2)
    assert update_success_v2, "Failed to update habits content (v2)."
    print("Overwritten habits with Version 2 content.")

    retrieved_content_v2 = manager.get_habits_content()
    print(f"\nRetrieved habits content (Version 2):\n'''\n{retrieved_content_v2}\n'''")
    assert (
        retrieved_content_v2 == sample_habits_v2
    ), "Retrieved content (v2) does not match overwritten content."

    # --- Test with Empty Content Update ---
    print("\n--- Testing Update with Empty Content ---")
    update_success_empty = manager.update_habits_content("")
    assert update_success_empty, "Failed to update habits with empty content."
    retrieved_content_empty = manager.get_habits_content()
    print(
        f"Retrieved habits content (after empty update):\n'''\n{retrieved_content_empty}\n'''"
    )
    assert (
        retrieved_content_empty == ""
    ), "Content should be empty after updating with an empty string."

    # --- Clean up test file ---
    if test_habits_file.exists():
        print(f"\nCleaning up test file: {test_habits_file}")
        # test_habits_file.unlink() # Uncomment to auto-delete after test
    else:
        print(f"\nTest file {test_habits_file} not found for cleanup.")

    print("\n--- HabitsManager Testing Complete ---")
