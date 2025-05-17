# mirai_app/core/tasks_manager.py

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

from mirai_app import config
from mirai_app.core import utils

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TasksManager:
    """Manages tasks for MIRAI."""

    def __init__(self, tasks_file_path: Optional[str] = None):
        """
        Initializes the TasksManager.

        Args:
            tasks_file_path: Path to the JSON file storing tasks.
                             Defaults to config.TASKS_FILE.
        """
        self.tasks_file = (
            Path(tasks_file_path) if tasks_file_path else config.TASKS_FILE
        )
        self.tasks: List[Dict[str, Any]] = self._load_tasks()
        logger.info(
            f"TasksManager initialized. Loaded {len(self.tasks)} tasks from {self.tasks_file}"
        )

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Loads tasks from the JSON file."""
        return utils.read_json_file(self.tasks_file, default_content=[])

    def _save_tasks(self) -> bool:
        """Saves the current list of tasks to the JSON file."""
        if utils.write_json_file(self.tasks_file, self.tasks):
            logger.debug(
                f"Successfully saved {len(self.tasks)} tasks to {self.tasks_file}"
            )
            return True
        logger.error(f"Failed to save tasks to {self.tasks_file}")
        return False

    def _find_task_index(self, task_id: str) -> Optional[int]:
        """Finds the index of a task by its ID."""
        for i, task in enumerate(self.tasks):
            if task.get("id") == task_id:
                return i
        return None

    def create_task(
        self, description: str, due_date_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a new task.

        Args:
            description: The description of the task.
            due_date_str: Optional. The due date as a string (e.g., "YYYY-MM-DD HH:MM").
                          If naive, it's assumed to be in MIRZA_TIMEZONE.

        Returns:
            The created task dictionary.
        """
        if not description:
            raise ValueError("Task description cannot be empty.")

        now_utc = utils.get_current_datetime_utc()

        # Generate ID based on the first few words of the description
        desc_slug_elements = description.split()[:5]  # Use first 5 words for slug
        task_id = utils.generate_unique_id(
            prefix="task", name_elements=desc_slug_elements
        )

        task_data: Dict[str, Any] = {
            "id": task_id,
            "description": description,
            "date_created": utils.format_datetime_iso(now_utc),
            "due_date": None,
            "completed": False,
            "date_completed": None,
        }

        if due_date_str:
            parsed_due_date = utils.parse_datetime_flexible(due_date_str, tz_aware=True)
            if parsed_due_date:
                # Convert to UTC for storage
                utc_due_date = parsed_due_date.astimezone(timezone.utc)
                task_data["due_date"] = utils.format_datetime_iso(utc_due_date)
            else:
                logger.warning(
                    f"Could not parse due_date_str: '{due_date_str}' for task '{description}'. Due date not set."
                )

        self.tasks.append(task_data)
        self._save_tasks()
        logger.info(f"Created task '{task_id}': {description[:50]}...")
        return task_data

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a task by its ID."""
        index = self._find_task_index(task_id)
        if index is not None:
            return self.tasks[index]
        logger.warning(f"Task with ID '{task_id}' not found.")
        return None

    def update_task(
        self, task_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Updates an existing task.

        Args:
            task_id: The ID of the task to update.
            updates: A dictionary of fields to update.
                     Allowed fields: "description", "due_date", "completed".
                     "due_date" should be a string.
                     "date_completed" is handled automatically based on "completed".

        Returns:
            The updated task dictionary, or None if the task is not found.
        """
        index = self._find_task_index(task_id)
        if index is None:
            logger.warning(f"Cannot update. Task with ID '{task_id}' not found.")
            return None

        task = self.tasks[index]
        updated_fields = []

        if "description" in updates:
            new_description = updates["description"]
            if not new_description:
                raise ValueError(
                    "Task description cannot be set to empty during update."
                )
            task["description"] = new_description
            updated_fields.append("description")

        if "due_date" in updates:
            due_date_str = updates["due_date"]
            if due_date_str is None:  # Allow setting due_date to None
                task["due_date"] = None
            else:
                parsed_due_date = utils.parse_datetime_flexible(
                    due_date_str, tz_aware=True
                )
                if parsed_due_date:
                    utc_due_date = parsed_due_date.astimezone(timezone.utc)
                    task["due_date"] = utils.format_datetime_iso(utc_due_date)
                else:
                    logger.warning(
                        f"Could not parse due_date_str: '{due_date_str}' for task update '{task_id}'. Due date not changed."
                    )
            updated_fields.append("due_date")

        if "completed" in updates:
            was_completed = task["completed"]
            task["completed"] = bool(updates["completed"])
            if task["completed"] and not was_completed:  # Task marked as completed
                task["date_completed"] = utils.format_datetime_iso(
                    utils.get_current_datetime_utc()
                )
            elif (
                not task["completed"] and was_completed
            ):  # Task marked as not completed (reopened)
                task["date_completed"] = None
            updated_fields.append("completed")
            if (
                "date_completed" not in updated_fields
                and task.get("date_completed") is not None
                and not task["completed"]
            ):
                updated_fields.append("date_completed (cleared)")  # for logging clarity

        if not updated_fields:
            logger.info(f"No valid fields to update for task '{task_id}'.")
            return task  # Return original task if no valid updates

        self.tasks[index] = task
        self._save_tasks()
        logger.info(
            f"Updated task '{task_id}'. Changed fields: {', '.join(updated_fields)}"
        )
        return task

    def delete_task(self, task_id: str) -> bool:
        """Deletes a task by its ID."""
        index = self._find_task_index(task_id)
        if index is not None:
            deleted_task_desc = self.tasks[index].get("description", "N/A")[:50]
            del self.tasks[index]
            self._save_tasks()
            logger.info(f"Deleted task '{task_id}': {deleted_task_desc}...")
            return True
        logger.warning(f"Cannot delete. Task with ID '{task_id}' not found.")
        return False

    def query_tasks(
        self,
        completed: Optional[bool] = None,
        due_before_str: Optional[str] = None,
        due_after_str: Optional[str] = None,
        created_before_str: Optional[str] = None,
        created_after_str: Optional[str] = None,
        overdue: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries tasks based on various criteria.
        All date strings, if naive, are assumed to be in MIRZA_TIMEZONE.
        """
        results = self.tasks

        # Filter by completion status
        if completed is not None:
            results = [task for task in results if task.get("completed") == completed]

        # Prepare UTC datetime objects for date comparisons
        due_before_dt_utc: Optional[datetime] = None
        if due_before_str:
            parsed = utils.parse_datetime_flexible(due_before_str, tz_aware=True)
            if parsed:
                due_before_dt_utc = parsed.astimezone(timezone.utc)

        due_after_dt_utc: Optional[datetime] = None
        if due_after_str:
            parsed = utils.parse_datetime_flexible(due_after_str, tz_aware=True)
            if parsed:
                due_after_dt_utc = parsed.astimezone(timezone.utc)

        created_before_dt_utc: Optional[datetime] = None
        if created_before_str:
            parsed = utils.parse_datetime_flexible(created_before_str, tz_aware=True)
            if parsed:
                created_before_dt_utc = parsed.astimezone(timezone.utc)

        created_after_dt_utc: Optional[datetime] = None
        if created_after_str:
            parsed = utils.parse_datetime_flexible(created_after_str, tz_aware=True)
            if parsed:
                created_after_dt_utc = parsed.astimezone(timezone.utc)

        # Filter by dates
        filtered_results = []
        now_utc = utils.get_current_datetime_utc()

        for task in results:
            passes_filters = True

            # Due date filters
            task_due_date_str = task.get("due_date")
            task_due_date_dt_utc: Optional[datetime] = None
            if task_due_date_str:
                # Stored due_date is already UTC, parse it directly
                task_due_date_dt_utc = utils.parse_datetime_flexible(task_due_date_str)
                if (
                    task_due_date_dt_utc and task_due_date_dt_utc.tzinfo is None
                ):  # Should not happen if stored correctly
                    task_due_date_dt_utc = task_due_date_dt_utc.replace(
                        tzinfo=timezone.utc
                    )

            if due_before_dt_utc and (
                not task_due_date_dt_utc or task_due_date_dt_utc >= due_before_dt_utc
            ):
                passes_filters = False
            if due_after_dt_utc and (
                not task_due_date_dt_utc or task_due_date_dt_utc <= due_after_dt_utc
            ):
                passes_filters = False

            # Created date filters
            task_created_date_str = task.get("date_created")
            task_created_date_dt_utc: Optional[datetime] = None
            if task_created_date_str:
                task_created_date_dt_utc = utils.parse_datetime_flexible(
                    task_created_date_str
                )  # Already UTC
                if task_created_date_dt_utc and task_created_date_dt_utc.tzinfo is None:
                    task_created_date_dt_utc = task_created_date_dt_utc.replace(
                        tzinfo=timezone.utc
                    )

            if created_before_dt_utc and (
                not task_created_date_dt_utc
                or task_created_date_dt_utc >= created_before_dt_utc
            ):
                passes_filters = False
            if created_after_dt_utc and (
                not task_created_date_dt_utc
                or task_created_date_dt_utc <= created_after_dt_utc
            ):
                passes_filters = False

            # Overdue filter
            if overdue is True and passes_filters:  # Only apply if other filters pass
                if task.get("completed") is True:  # Completed tasks cannot be overdue
                    passes_filters = False
                elif (
                    not task_due_date_dt_utc
                ):  # Tasks without due date cannot be overdue
                    passes_filters = False
                elif (
                    task_due_date_dt_utc >= now_utc
                ):  # Tasks due in future are not overdue
                    passes_filters = False

            if passes_filters:
                filtered_results.append(task)

        return filtered_results

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Returns all tasks that are not completed."""
        return self.query_tasks(completed=False)

    def get_completed_tasks(self) -> List[Dict[str, Any]]:
        """Returns all tasks that are completed."""
        return self.query_tasks(completed=True)

    def get_overdue_tasks(self) -> List[Dict[str, Any]]:
        """Returns all tasks that are not completed and past their due date."""
        return self.query_tasks(completed=False, overdue=True)

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Returns all tasks."""
        return self.tasks


if __name__ == "__main__":
    # --- Setup for Testing ---
    # Create a temporary tasks file for testing
    test_tasks_file = config.DATA_DIR / "test_tasks.json"
    if test_tasks_file.exists():
        test_tasks_file.unlink()  # Clean up from previous test

    manager = TasksManager(tasks_file_path=str(test_tasks_file))
    print(f"\n--- Initial state: {len(manager.get_all_tasks())} tasks ---")

    # --- Test Create ---
    print("\n--- Testing Create Task ---")
    task1_desc = "Buy groceries: milk, eggs, bread"
    task1_due = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() + utils.timedelta(days=2)
    )  # Due in 2 days
    task1 = manager.create_task(description=task1_desc, due_date_str=task1_due)
    print(f"Created Task 1: {task1['id']} - Due: {task1.get('due_date')}")

    task2_desc = "Schedule dentist appointment"
    task2 = manager.create_task(description=task2_desc)  # No due date
    print(f"Created Task 2: {task2['id']}")

    task3_desc = "Finish project report"
    task3_due_past = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() - utils.timedelta(days=1)
    )  # Due yesterday
    task3 = manager.create_task(description=task3_desc, due_date_str=task3_due_past)
    print(
        f"Created Task 3 (due yesterday): {task3['id']} - Due: {task3.get('due_date')}"
    )

    print(f"Total tasks after creation: {len(manager.get_all_tasks())}")
    assert len(manager.get_all_tasks()) == 3

    # --- Test Get ---
    print("\n--- Testing Get Task ---")
    retrieved_task1 = manager.get_task(task1["id"])
    assert retrieved_task1 is not None and retrieved_task1["description"] == task1_desc
    print(f"Retrieved Task 1: {retrieved_task1['id']}")
    non_existent_task = manager.get_task("non_existent_id")
    assert non_existent_task is None
    print(
        f"Attempt to retrieve non_existent_id: {'Not found' if non_existent_task is None else 'Found!'}"
    )

    # --- Test Update ---
    print("\n--- Testing Update Task ---")
    updated_desc = "Buy groceries: milk, eggs, bread, and cheese"
    update1_result = manager.update_task(
        task1["id"], {"description": updated_desc, "completed": False}
    )
    assert update1_result is not None and update1_result["description"] == updated_desc
    print(f"Updated Task 1 description: {update1_result['description']}")

    new_due_date_str = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() + utils.timedelta(days=5)
    )
    update2_result = manager.update_task(task1["id"], {"due_date": new_due_date_str})
    assert update2_result is not None and update2_result["due_date"] is not None
    print(f"Updated Task 1 due date to: {update2_result['due_date']}")

    # Mark task2 as completed
    update3_result = manager.update_task(task2["id"], {"completed": True})
    assert (
        update3_result is not None
        and update3_result["completed"] is True
        and update3_result["date_completed"] is not None
    )
    print(
        f"Marked Task 2 as completed. Date completed: {update3_result['date_completed']}"
    )

    # Reopen task2
    update4_result = manager.update_task(task2["id"], {"completed": False})
    assert (
        update4_result is not None
        and update4_result["completed"] is False
        and update4_result["date_completed"] is None
    )
    print(
        f"Reopened Task 2. Completed: {update4_result['completed']}, Date completed: {update4_result['date_completed']}"
    )

    # --- Test Querying ---
    print("\n--- Testing Query Tasks ---")
    active_tasks = manager.get_active_tasks()
    print(f"Active tasks ({len(active_tasks)}):")
    for t in active_tasks:
        print(
            f"  - {t['id']}: {t['description'][:30]}... (Due: {t.get('due_date', 'N/A')})"
        )
    assert len(active_tasks) == 3  # task1, task2 (reopened), task3

    manager.update_task(task1["id"], {"completed": True})  # Complete task1
    completed_tasks = manager.get_completed_tasks()
    print(f"Completed tasks ({len(completed_tasks)}):")
    for t in completed_tasks:
        print(f"  - {t['id']}: {t['description'][:30]}...")
    assert len(completed_tasks) == 1 and completed_tasks[0]["id"] == task1["id"]

    active_tasks_after_complete = manager.get_active_tasks()
    assert len(active_tasks_after_complete) == 2  # task2, task3

    overdue_tasks = manager.get_overdue_tasks()
    print(f"Overdue tasks ({len(overdue_tasks)}):")  # Should be task3
    for t in overdue_tasks:
        print(f"  - {t['id']}: {t['description'][:30]}... (Due: {t.get('due_date')})")
    assert len(overdue_tasks) == 1 and overdue_tasks[0]["id"] == task3["id"]

    # Test query with due_before
    due_in_3_days_str = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() + utils.timedelta(days=3)
    )
    tasks_due_soon = manager.query_tasks(
        completed=False, due_before_str=due_in_3_days_str
    )
    print(
        f"Tasks due before {due_in_3_days_str} ({len(tasks_due_soon)}):"
    )  # Should be task3 (if not completed)
    # Task 3 is due yesterday, so it's before "in 3 days"
    # Task 1 is completed. Task 2 has no due date.
    assert len(tasks_due_soon) == 1 and tasks_due_soon[0]["id"] == task3["id"]

    # --- Test Delete ---
    print("\n--- Testing Delete Task ---")
    delete_result_task2 = manager.delete_task(task2["id"])
    assert delete_result_task2 is True
    print(f"Deleted Task 2: {'Success' if delete_result_task2 else 'Failed'}")
    assert manager.get_task(task2["id"]) is None
    assert len(manager.get_all_tasks()) == 2

    delete_non_existent = manager.delete_task("non_existent_id")
    assert delete_non_existent is False
    print(
        f"Attempt to delete non_existent_id: {'Success' if delete_non_existent else 'Failed (expected)'}"
    )

    print(f"\n--- Final state: {len(manager.get_all_tasks())} tasks ---")
    for t in manager.get_all_tasks():
        print(f"  - {t['id']}: {t['description'][:30]}... Completed: {t['completed']}")

    # --- Clean up test file ---
    if test_tasks_file.exists():
        print(f"\nCleaning up test file: {test_tasks_file}")
        # test_tasks_file.unlink() # Uncomment to auto-delete after test
    else:
        print(f"\nTest file {test_tasks_file} not found for cleanup.")

    print("\n--- TasksManager Testing Complete ---")
