# mirai_app/core/reminders_manager.py

import logging
from pathlib import Path
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from mirai_app import config
from mirai_app.core import utils

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class RemindersManager:
    """Manages reminders for MIRAI."""

    def __init__(self, reminders_file_path: Optional[str] = None):
        self.reminders_file = (
            Path(reminders_file_path) if reminders_file_path else config.REMINDERS_FILE
        )
        self.reminders: List[Dict[str, Any]] = self._load_reminders()
        logger.info(
            f"RemindersManager initialized. Loaded {len(self.reminders)} reminders from {self.reminders_file}"
        )

    def _load_reminders(self) -> List[Dict[str, Any]]:
        return utils.read_json_file(self.reminders_file, default_content=[])

    def _save_reminders(self) -> bool:
        if utils.write_json_file(self.reminders_file, self.reminders):
            logger.debug(
                f"Successfully saved {len(self.reminders)} reminders to {self.reminders_file}"
            )
            return True
        logger.error(f"Failed to save reminders to {self.reminders_file}")
        return False

    def _find_reminder_index(self, reminder_id: str) -> Optional[int]:
        for i, reminder in enumerate(self.reminders):
            if reminder.get("id") == reminder_id:
                return i
        return None

    def _generate_notification_points(
        self, due_datetime_utc: datetime, notify_before_strings: List[str]
    ) -> List[Dict[str, Any]]:
        """Generates sorted notification points for a reminder."""
        points = []
        # Add pre-notifications
        if notify_before_strings:
            # utils.calculate_notification_times returns sorted list of datetimes
            calculated_times_utc = utils.calculate_notification_times(
                due_datetime_utc, notify_before_strings
            )
            for dt_utc in calculated_times_utc:
                # Avoid duplicate notification points if a notify_before string results in the due_datetime_utc itself
                if dt_utc == due_datetime_utc:
                    continue
                points.append(
                    {
                        "id": f"np_{uuid.uuid4().hex[:8]}",  # Unique ID for this notification point
                        "notify_at_utc": utils.format_datetime_iso(dt_utc),
                        "is_main_due_time": False,
                        "status": "pending",  # "pending", "triggered", "skipped"
                    }
                )

        # Add main due time notification
        points.append(
            {
                "id": f"np_{uuid.uuid4().hex[:8]}",
                "notify_at_utc": utils.format_datetime_iso(due_datetime_utc),
                "is_main_due_time": True,
                "status": "pending",
            }
        )

        # Remove duplicates that might arise from specific notify_before strings
        # (e.g. "0 minutes before" could be same as main due time)
        # A more robust way is to ensure uniqueness based on notify_at_utc and is_main_due_time
        unique_points_dict = {}
        for p in points:
            key = (p["notify_at_utc"], p["is_main_due_time"])
            if key not in unique_points_dict:
                unique_points_dict[key] = p

        # Sort by notification time
        return sorted(
            list(unique_points_dict.values()),
            key=lambda p_item: p_item["notify_at_utc"],
        )

    def create_reminder(
        self,
        description: str,
        due_datetime_str: str,  # Must be provided
        notify_before_list: Optional[List[str]] = None,  # e.g., ["5m", "1h"]
        related_item_id: Optional[str] = None,
        action_on_trigger: str = "prompt_llm",  # Default action
    ) -> Dict[str, Any]:
        """
        Creates a new reminder.
        due_datetime_str, if naive, is assumed to be in MIRZA_TIMEZONE.
        """
        if not description:
            raise ValueError("Reminder description cannot be empty.")
        if not due_datetime_str:
            raise ValueError("Reminder due_datetime_str must be provided.")

        parsed_due_dt = utils.parse_datetime_flexible(due_datetime_str, tz_aware=True)
        if not parsed_due_dt:
            raise ValueError(f"Invalid due_datetime_str: {due_datetime_str}")

        due_datetime_utc = parsed_due_dt.astimezone(timezone.utc)
        now_utc = utils.get_current_datetime_utc()

        name_elements_for_id = description.split()[:5]
        reminder_id = utils.generate_unique_id(
            prefix="rem", name_elements=name_elements_for_id
        )

        actual_notify_before_list = (
            notify_before_list if notify_before_list is not None else []
        )
        notification_points = self._generate_notification_points(
            due_datetime_utc, actual_notify_before_list
        )

        reminder_data: Dict[str, Any] = {
            "id": reminder_id,
            "description": description,
            "due_datetime_utc": utils.format_datetime_iso(due_datetime_utc),
            "original_notify_before_strings": actual_notify_before_list,
            "status": "pending",  # "pending", "active", "triggered", "completed", "dismissed"
            "date_created_utc": utils.format_datetime_iso(now_utc),
            "date_updated_utc": utils.format_datetime_iso(now_utc),
            "related_item_id": related_item_id,
            "action_on_trigger": action_on_trigger,
            "notification_points": notification_points,
        }

        self.reminders.append(reminder_data)
        self._save_reminders()
        logger.info(f"Created reminder '{reminder_id}': {description[:50]}...")
        return reminder_data

    def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        index = self._find_reminder_index(reminder_id)
        if index is not None:
            return self.reminders[index]
        logger.warning(f"Reminder with ID '{reminder_id}' not found.")
        return None

    def update_reminder(
        self, reminder_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        index = self._find_reminder_index(reminder_id)
        if index is None:
            logger.warning(
                f"Cannot update. Reminder with ID '{reminder_id}' not found."
            )
            return None

        reminder = self.reminders[index]
        updated_fields = []
        needs_notification_points_recalc = False

        if "description" in updates:
            new_description = updates["description"]
            if not new_description:
                raise ValueError("Reminder description cannot be empty during update.")
            reminder["description"] = new_description
            updated_fields.append("description")

        if "due_datetime_str" in updates:
            parsed_due_dt = utils.parse_datetime_flexible(
                updates["due_datetime_str"], tz_aware=True
            )
            if not parsed_due_dt:
                raise ValueError(
                    f"Invalid due_datetime_str for update: {updates['due_datetime_str']}"
                )
            reminder["due_datetime_utc"] = utils.format_datetime_iso(
                parsed_due_dt.astimezone(timezone.utc)
            )
            updated_fields.append("due_datetime_utc")
            needs_notification_points_recalc = True

        if "notify_before_list" in updates:
            reminder["original_notify_before_strings"] = (
                updates["notify_before_list"]
                if updates["notify_before_list"] is not None
                else []
            )
            updated_fields.append("original_notify_before_strings")
            needs_notification_points_recalc = True

        if needs_notification_points_recalc:
            current_due_dt_utc = utils.parse_datetime_flexible(
                reminder["due_datetime_utc"]
            )  # Already UTC
            reminder["notification_points"] = self._generate_notification_points(
                current_due_dt_utc, reminder["original_notify_before_strings"]
            )
            # After recalculating points, if reminder was 'active' or 'triggered',
            # it might revert to 'pending' if all new points are in the future.
            # For simplicity, we can reset to 'pending' if points are recalc'd,
            # or implement more complex status preservation if needed.
            # Let's reset to 'pending' for now, assuming a schedule change implies a fresh start for notifications.
            if reminder["status"] not in ["completed", "dismissed"]:
                reminder["status"] = "pending"
                updated_fields.append(
                    "status (reset to pending due to schedule change)"
                )

        if "related_item_id" in updates:
            reminder["related_item_id"] = updates["related_item_id"]
            updated_fields.append("related_item_id")

        if "action_on_trigger" in updates:
            reminder["action_on_trigger"] = updates["action_on_trigger"]
            updated_fields.append("action_on_trigger")

        if (
            "status" in updates and not needs_notification_points_recalc
        ):  # Status update not due to recalc
            new_status = updates["status"]
            if new_status not in [
                "pending",
                "active",
                "triggered",
                "completed",
                "dismissed",
            ]:
                raise ValueError(f"Invalid status for update: {new_status}")

            if reminder["status"] != new_status:
                reminder["status"] = new_status
                updated_fields.append("status")
                if new_status in ["completed", "dismissed"]:
                    # Mark all pending notification points as "skipped"
                    for point in reminder.get("notification_points", []):
                        if point["status"] == "pending":
                            point["status"] = "skipped"
                    updated_fields.append("notification_points (pending ones skipped)")

        if not updated_fields:
            logger.info(f"No valid fields to update for reminder '{reminder_id}'.")
            return reminder

        reminder["date_updated_utc"] = utils.format_datetime_iso(
            utils.get_current_datetime_utc()
        )
        self.reminders[index] = reminder
        self._save_reminders()
        logger.info(
            f"Updated reminder '{reminder_id}'. Changed fields: {', '.join(updated_fields)}"
        )
        return reminder

    def delete_reminder(self, reminder_id: str) -> bool:
        index = self._find_reminder_index(reminder_id)
        if index is not None:
            deleted_desc = self.reminders[index].get("description", "N/A")[:50]
            del self.reminders[index]
            self._save_reminders()
            logger.info(f"Deleted reminder '{reminder_id}': {deleted_desc}...")
            return True
        logger.warning(f"Cannot delete. Reminder with ID '{reminder_id}' not found.")
        return False

    def get_triggerable_notifications(
        self, check_until_datetime_utc: datetime
    ) -> List[Dict[str, Any]]:
        """
        Finds all notification points that are due and pending.
        Returns a list of dicts, each representing a notification to be triggered.
        """
        triggerable = []
        active_reminder_statuses = [
            "pending",
            "active",
        ]  # Only process notifications for these overall reminder statuses

        for reminder in self.reminders:
            if reminder.get("status") not in active_reminder_statuses:
                continue

            for point in reminder.get("notification_points", []):
                if point.get("status") == "pending":
                    notify_at_dt_utc = utils.parse_datetime_flexible(
                        point["notify_at_utc"]
                    )  # Already UTC
                    if (
                        notify_at_dt_utc
                        and notify_at_dt_utc <= check_until_datetime_utc
                    ):
                        triggerable.append(
                            {
                                "reminder_id": reminder["id"],
                                "reminder_description": reminder["description"],
                                "notification_point_id": point["id"],
                                "notify_at_utc_str": point["notify_at_utc"],
                                "is_main_due_time": point["is_main_due_time"],
                                "action_on_trigger": reminder["action_on_trigger"],
                            }
                        )
        return sorted(triggerable, key=lambda x: x["notify_at_utc_str"])

    def process_triggered_notification(
        self, reminder_id: str, notification_point_id: str
    ) -> bool:
        """
        Marks a specific notification point as 'triggered' and updates reminder status if needed.
        """
        index = self._find_reminder_index(reminder_id)
        if index is None:
            logger.error(f"Cannot process trigger. Reminder '{reminder_id}' not found.")
            return False

        reminder = self.reminders[index]
        point_processed = False

        for point in reminder.get("notification_points", []):
            if (
                point.get("id") == notification_point_id
                and point.get("status") == "pending"
            ):
                point["status"] = "triggered"
                point_processed = True

                # Update overall reminder status
                if point["is_main_due_time"]:
                    if (
                        reminder["status"] != "completed"
                        and reminder["status"] != "dismissed"
                    ):
                        reminder["status"] = "triggered"
                else:  # It's a pre-notification
                    if reminder["status"] == "pending":
                        reminder["status"] = "active"
                break

        if point_processed:
            reminder["date_updated_utc"] = utils.format_datetime_iso(
                utils.get_current_datetime_utc()
            )
            self.reminders[index] = reminder
            self._save_reminders()
            logger.info(
                f"Processed notification point '{notification_point_id}' for reminder '{reminder_id}'. Reminder status: {reminder['status']}"
            )
            return True
        else:
            logger.warning(
                f"Notification point '{notification_point_id}' for reminder '{reminder_id}' not found or not pending."
            )
            return False

    def query_reminders(
        self,
        status: Optional[str] = None,
        due_before_str: Optional[str] = None,
        due_after_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = self.reminders
        if status:
            results = [r for r in results if r.get("status") == status]

        due_before_dt_utc = (
            utils.parse_datetime_flexible(due_before_str, tz_aware=True).astimezone(
                timezone.utc
            )
            if due_before_str
            else None
        )
        due_after_dt_utc = (
            utils.parse_datetime_flexible(due_after_str, tz_aware=True).astimezone(
                timezone.utc
            )
            if due_after_str
            else None
        )

        final_results = []
        for r in results:
            passes = True
            reminder_due_dt_utc = utils.parse_datetime_flexible(
                r["due_datetime_utc"]
            )  # Already UTC
            if due_before_dt_utc and (
                not reminder_due_dt_utc or reminder_due_dt_utc >= due_before_dt_utc
            ):
                passes = False
            if due_after_dt_utc and (
                not reminder_due_dt_utc or reminder_due_dt_utc <= due_after_dt_utc
            ):
                passes = False
            if passes:
                final_results.append(r)
        return final_results

    def get_all_reminders(self) -> List[Dict[str, Any]]:
        return self.reminders


if __name__ == "__main__":
    test_reminders_file = config.DATA_DIR / "test_reminders.json"
    if test_reminders_file.exists():
        test_reminders_file.unlink()

    manager = RemindersManager(reminders_file_path=str(test_reminders_file))
    print(f"\n--- Initial state: {len(manager.get_all_reminders())} reminders ---")

    # --- Test Create ---
    print("\n--- Testing Create Reminder ---")
    now_local_dt = utils.get_current_datetime_local()

    # Reminder 1: Due in 10 minutes, notify 5m and 1m before
    rem1_due_dt = now_local_dt + utils.timedelta(minutes=10)
    rem1_due_str = utils.format_datetime_for_llm(rem1_due_dt)
    rem1 = manager.create_reminder(
        description="Short meeting in 10 mins",
        due_datetime_str=rem1_due_str,
        notify_before_list=["5m", "1m"],
    )
    print(f"Created Reminder 1 ('{rem1['id']}'), due: {rem1['due_datetime_utc']}")
    print(f"  Notification points ({len(rem1['notification_points'])}):")
    for p in rem1["notification_points"]:
        print(
            f"    - ID: {p['id']}, At: {p['notify_at_utc']}, Main: {p['is_main_due_time']}, Status: {p['status']}"
        )
    assert len(rem1["notification_points"]) == 3  # 5m, 1m, main

    # Reminder 2: Due in 2 hours, no pre-notifications
    rem2_due_dt = now_local_dt + utils.timedelta(hours=2)
    rem2_due_str = utils.format_datetime_for_llm(rem2_due_dt)
    rem2 = manager.create_reminder(
        description="Project deadline", due_datetime_str=rem2_due_str
    )
    print(f"Created Reminder 2 ('{rem2['id']}'), due: {rem2['due_datetime_utc']}")
    assert (
        len(rem2["notification_points"]) == 1
        and rem2["notification_points"][0]["is_main_due_time"]
    )

    assert len(manager.get_all_reminders()) == 2

    # --- Test Get Triggerable Notifications ---
    print("\n--- Testing Get Triggerable Notifications ---")
    # Check for notifications due in the next 6 minutes (should catch rem1's 5m pre-notification)
    check_time_utc = utils.get_current_datetime_utc() + utils.timedelta(minutes=6)
    triggerable = manager.get_triggerable_notifications(check_time_utc)
    print(
        f"Triggerable notifications by {utils.format_datetime_iso(check_time_utc)} ({len(triggerable)}):"
    )
    assert len(triggerable) == 1
    assert triggerable[0]["reminder_id"] == rem1["id"]
    assert not triggerable[0]["is_main_due_time"]  # Should be the 5m pre-notification

    first_triggerable_np_id = triggerable[0]["notification_point_id"]
    print(
        f"  - Rem ID: {triggerable[0]['reminder_id']}, NP ID: {first_triggerable_np_id}, Main: {triggerable[0]['is_main_due_time']}"
    )

    # --- Test Process Triggered Notification ---
    print("\n--- Testing Process Triggered Notification ---")
    process_result = manager.process_triggered_notification(
        rem1["id"], first_triggerable_np_id
    )
    assert process_result is True
    updated_rem1 = manager.get_reminder(rem1["id"])
    assert updated_rem1["status"] == "active"  # Changed from pending
    assert (
        updated_rem1["notification_points"][0]["status"] == "triggered"
    )  # The 5m pre-notification
    assert (
        updated_rem1["notification_points"][1]["status"] == "pending"
    )  # The 1m pre-notification
    print(f"Reminder 1 status after processing 1st NP: {updated_rem1['status']}")

    # Try to trigger again (should fail as it's not pending)
    process_again_result = manager.process_triggered_notification(
        rem1["id"], first_triggerable_np_id
    )
    assert process_again_result is False

    # Check triggerable again (should be none now for that specific point)
    triggerable_after_proc = manager.get_triggerable_notifications(check_time_utc)
    assert (
        len(triggerable_after_proc) == 0
    )  # The 5m one is processed, 1m one is > 6 mins from now_utc

    # Check for notifications due in the next 11 minutes (should catch rem1's 1m and main)
    check_time_utc_later = utils.get_current_datetime_utc() + utils.timedelta(
        minutes=11
    )
    triggerable_later = manager.get_triggerable_notifications(check_time_utc_later)
    print(
        f"Triggerable notifications by {utils.format_datetime_iso(check_time_utc_later)} ({len(triggerable_later)}):"
    )
    # Should be 2: rem1's 1m pre-notification and rem1's main due time.
    assert len(triggerable_later) == 2

    # Process the next one (1m pre-notification)
    rem1_np2_id = updated_rem1["notification_points"][1][
        "id"
    ]  # Assuming it's the 1m one
    manager.process_triggered_notification(rem1["id"], rem1_np2_id)
    updated_rem1_again = manager.get_reminder(rem1["id"])
    assert updated_rem1_again["status"] == "active"  # Still active, main not triggered
    assert updated_rem1_again["notification_points"][1]["status"] == "triggered"

    # Process the main one
    rem1_np_main_id = updated_rem1_again["notification_points"][2][
        "id"
    ]  # Assuming it's the main one
    manager.process_triggered_notification(rem1["id"], rem1_np_main_id)
    updated_rem1_final = manager.get_reminder(rem1["id"])
    assert updated_rem1_final["status"] == "triggered"  # Main due time triggered
    assert updated_rem1_final["notification_points"][2]["status"] == "triggered"

    # --- Test Update ---
    print("\n--- Testing Update Reminder ---")
    # Update rem2: change due time and add pre-notifications
    new_rem2_due_dt = now_local_dt + utils.timedelta(hours=3)
    new_rem2_due_str = utils.format_datetime_for_llm(new_rem2_due_dt)
    update_rem2_result = manager.update_reminder(
        rem2["id"],
        {"due_datetime_str": new_rem2_due_str, "notify_before_list": ["15m", "5m"]},
    )
    assert update_rem2_result is not None
    assert len(update_rem2_result["notification_points"]) == 3
    assert update_rem2_result["status"] == "pending"  # Reset due to schedule change
    print(
        f"Updated Reminder 2. New due: {update_rem2_result['due_datetime_utc']}, NP count: {len(update_rem2_result['notification_points'])}"
    )

    # Mark Reminder 1 as completed
    manager.update_reminder(rem1["id"], {"status": "completed"})
    completed_rem1 = manager.get_reminder(rem1["id"])
    assert completed_rem1["status"] == "completed"
    # Check if its pending notification points became "skipped" (all were triggered in this test though)
    # Let's create a new one to test this specific skipping logic
    rem_for_skip_due = utils.format_datetime_for_llm(
        now_local_dt + utils.timedelta(days=1)
    )
    rem_for_skip = manager.create_reminder("Test Skip", rem_for_skip_due, ["1h"])
    manager.update_reminder(rem_for_skip["id"], {"status": "completed"})
    completed_rem_for_skip = manager.get_reminder(rem_for_skip["id"])
    assert all(
        p["status"] == "skipped" for p in completed_rem_for_skip["notification_points"]
    )
    print(
        f"Reminder for skipping test status: {completed_rem_for_skip['status']}, NP statuses: {[p['status'] for p in completed_rem_for_skip['notification_points']]}"
    )

    # --- Test Delete ---
    print("\n--- Testing Delete Reminder ---")
    manager.delete_reminder(rem2["id"])
    assert manager.get_reminder(rem2["id"]) is None
    assert len(manager.get_all_reminders()) == 2  # rem1, rem_for_skip

    print(f"\n--- Final state: {len(manager.get_all_reminders())} reminders ---")

    # --- Clean up test file ---
    if test_reminders_file.exists():
        print(f"\nCleaning up test file: {test_reminders_file}")
        # test_reminders_file.unlink()
    else:
        print(f"\nTest file {test_reminders_file} not found for cleanup.")

    print("\n--- RemindersManager Testing Complete ---")
