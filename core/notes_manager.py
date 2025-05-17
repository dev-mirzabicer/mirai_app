# mirai_app/core/notes_manager.py

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from mirai_app import config
from mirai_app.core import utils

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class NotesManager:
    """Manages notes for MIRAI."""

    def __init__(self, notes_file_path: Optional[str] = None):
        """
        Initializes the NotesManager.

        Args:
            notes_file_path: Path to the JSON file storing notes.
                             Defaults to config.NOTES_FILE.
        """
        self.notes_file = (
            Path(notes_file_path) if notes_file_path else config.NOTES_FILE
        )
        self.notes: List[Dict[str, Any]] = self._load_notes()
        logger.info(
            f"NotesManager initialized. Loaded {len(self.notes)} notes from {self.notes_file}"
        )

    def _load_notes(self) -> List[Dict[str, Any]]:
        """Loads notes from the JSON file."""
        return utils.read_json_file(self.notes_file, default_content=[])

    def _save_notes(self) -> bool:
        """Saves the current list of notes to the JSON file."""
        if utils.write_json_file(self.notes_file, self.notes):
            logger.debug(
                f"Successfully saved {len(self.notes)} notes to {self.notes_file}"
            )
            return True
        logger.error(f"Failed to save notes to {self.notes_file}")
        return False

    def _find_note_index(self, note_id: str) -> Optional[int]:
        """Finds the index of a note by its ID."""
        for i, note in enumerate(self.notes):
            if note.get("id") == note_id:
                return i
        return None

    def create_note(
        self,
        content: str,
        title: Optional[str] = None,
        expiry_duration_str: Optional[
            str
        ] = None,  # e.g., "1 day", "2 weeks", "forever"
        expiry_date_str: Optional[str] = None,  # e.g., "YYYY-MM-DD HH:MM"
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Creates a new note.

        Args:
            content: The main content of the note.
            title: Optional title for the note.
            expiry_duration_str: Duration for expiry (e.g., "1 day", "forever").
                                 Calculated from creation time.
            expiry_date_str: Specific expiry date/time. Takes precedence over duration if valid.
                             If naive, assumed to be in MIRZA_TIMEZONE.
            tags: Optional list of tags.

        Returns:
            The created note dictionary.
        """
        if not content:
            raise ValueError("Note content cannot be empty.")

        now_utc = utils.get_current_datetime_utc()

        name_elements_for_id = [title] if title else content.split()[:5]
        note_id = utils.generate_unique_id(
            prefix="note", name_elements=name_elements_for_id
        )

        final_expiry_date_iso: Optional[str] = None

        if expiry_date_str:
            parsed_expiry_dt = utils.parse_datetime_flexible(
                expiry_date_str, tz_aware=True
            )
            if parsed_expiry_dt:
                final_expiry_date_iso = utils.format_datetime_iso(
                    parsed_expiry_dt.astimezone(timezone.utc)
                )
            else:
                logger.warning(
                    f"Could not parse expiry_date_str: '{expiry_date_str}'. Trying duration."
                )

        if final_expiry_date_iso is None and expiry_duration_str:
            if expiry_duration_str.lower() == "forever":
                final_expiry_date_iso = None
            else:
                calculated_expiry_dt = utils.calculate_expiry_date(
                    now_utc, expiry_duration_str
                )
                if calculated_expiry_dt:
                    final_expiry_date_iso = utils.format_datetime_iso(
                        calculated_expiry_dt
                    )  # Already UTC
                else:
                    logger.warning(
                        f"Could not parse expiry_duration_str: '{expiry_duration_str}'. Note will not expire automatically."
                    )

        note_data: Dict[str, Any] = {
            "id": note_id,
            "title": title,
            "content": content,
            "tags": sorted(list(set(tag.lower() for tag in tags))) if tags else [],
            "date_created": utils.format_datetime_iso(now_utc),
            "expiry_date": final_expiry_date_iso,
        }

        self.notes.append(note_data)
        self._save_notes()
        logger.info(f"Created note '{note_id}': {title or content[:30]}...")
        return note_data

    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a note by its ID."""
        index = self._find_note_index(note_id)
        if index is not None:
            return self.notes[index]
        logger.warning(f"Note with ID '{note_id}' not found.")
        return None

    def update_note(
        self, note_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Updates an existing note.

        Args:
            note_id: The ID of the note to update.
            updates: A dictionary of fields to update.
                     Allowed fields: "title", "content", "tags",
                                     "expiry_date" (string or None),
                                     "expiry_duration" (string or None, e.g. "1 day", "forever").
                     If "expiry_date" is provided, it takes precedence.
                     If "expiry_duration" is provided, it's calculated from the note's original creation date.

        Returns:
            The updated note dictionary, or None if the note is not found.
        """
        index = self._find_note_index(note_id)
        if index is None:
            logger.warning(f"Cannot update. Note with ID '{note_id}' not found.")
            return None

        note = self.notes[index]
        updated_fields = []

        if "title" in updates:
            note["title"] = updates["title"]  # Allow None for title
            updated_fields.append("title")

        if "content" in updates:
            new_content = updates["content"]
            if not new_content:
                raise ValueError("Note content cannot be set to empty during update.")
            note["content"] = new_content
            updated_fields.append("content")

        if "tags" in updates:
            new_tags = updates["tags"]
            note["tags"] = (
                sorted(list(set(tag.lower() for tag in new_tags)))
                if isinstance(new_tags, list)
                else []
            )
            updated_fields.append("tags")

        # Expiry update logic: expiry_date takes precedence
        expiry_updated = False
        if "expiry_date" in updates:  # Explicit expiry date string or None
            expiry_date_str_update = updates["expiry_date"]
            if expiry_date_str_update is None:
                note["expiry_date"] = None
            else:
                parsed_expiry_dt = utils.parse_datetime_flexible(
                    str(expiry_date_str_update), tz_aware=True
                )
                if parsed_expiry_dt:
                    note["expiry_date"] = utils.format_datetime_iso(
                        parsed_expiry_dt.astimezone(timezone.utc)
                    )
                else:
                    logger.warning(
                        f"Could not parse expiry_date: '{expiry_date_str_update}' for note update '{note_id}'. Expiry not changed by this field."
                    )
            updated_fields.append("expiry_date")
            expiry_updated = True

        if (
            not expiry_updated and "expiry_duration" in updates
        ):  # Duration string (e.g., "1 day", "forever")
            expiry_duration_str_update = updates["expiry_duration"]
            if (
                expiry_duration_str_update is None
                or str(expiry_duration_str_update).lower() == "forever"
            ):
                note["expiry_date"] = None
            else:
                # Calculate from original creation date
                date_created_dt = utils.parse_datetime_flexible(
                    note["date_created"]
                )  # Already UTC
                if date_created_dt:
                    calculated_expiry_dt = utils.calculate_expiry_date(
                        date_created_dt, str(expiry_duration_str_update)
                    )
                    if calculated_expiry_dt:
                        note["expiry_date"] = utils.format_datetime_iso(
                            calculated_expiry_dt
                        )  # Already UTC
                    else:
                        logger.warning(
                            f"Could not parse expiry_duration: '{expiry_duration_str_update}' for note update '{note_id}'. Expiry not changed by this field."
                        )
                else:  # Should not happen if date_created is always set
                    logger.error(
                        f"Could not parse original date_created for note '{note_id}' during expiry_duration update."
                    )
            updated_fields.append("expiry_duration (applied as expiry_date)")

        if not updated_fields:
            logger.info(f"No valid fields to update for note '{note_id}'.")
            return note

        self.notes[index] = note
        self._save_notes()
        logger.info(
            f"Updated note '{note_id}'. Changed fields: {', '.join(updated_fields)}"
        )
        return note

    def delete_note(self, note_id: str) -> bool:
        """Deletes a note by its ID."""
        index = self._find_note_index(note_id)
        if index is not None:
            deleted_note_title = self.notes[index].get(
                "title", self.notes[index].get("content", "N/A")[:30]
            )
            del self.notes[index]
            self._save_notes()
            logger.info(f"Deleted note '{note_id}': {deleted_note_title}...")
            return True
        logger.warning(f"Cannot delete. Note with ID '{note_id}' not found.")
        return False

    def query_notes(
        self,
        active: Optional[bool] = None,  # True for active, False for inactive (expired)
        created_before_str: Optional[str] = None,
        created_after_str: Optional[str] = None,
        expiry_before_str: Optional[str] = None,
        expiry_after_str: Optional[str] = None,
        tags: Optional[List[str]] = None,  # List of tags to filter by (OR logic)
        content_contains: Optional[
            str
        ] = None,  # Simple case-insensitive content search
    ) -> List[Dict[str, Any]]:
        """
        Queries notes based on various criteria.
        Date strings, if naive, are assumed to be in MIRZA_TIMEZONE.
        """
        results = self.notes
        now_utc = utils.get_current_datetime_utc()

        # Prepare UTC datetime objects for date comparisons
        created_before_dt_utc = (
            utils.parse_datetime_flexible(created_before_str, tz_aware=True).astimezone(
                timezone.utc
            )
            if created_before_str
            else None
        )
        created_after_dt_utc = (
            utils.parse_datetime_flexible(created_after_str, tz_aware=True).astimezone(
                timezone.utc
            )
            if created_after_str
            else None
        )
        expiry_before_dt_utc = (
            utils.parse_datetime_flexible(expiry_before_str, tz_aware=True).astimezone(
                timezone.utc
            )
            if expiry_before_str
            else None
        )
        expiry_after_dt_utc = (
            utils.parse_datetime_flexible(expiry_after_str, tz_aware=True).astimezone(
                timezone.utc
            )
            if expiry_after_str
            else None
        )

        normalized_query_tags = [tag.lower() for tag in tags] if tags else None

        filtered_results = []
        for note in results:
            passes_filters = True

            # Active/Expired filter
            note_expiry_date_str = note.get("expiry_date")
            is_currently_active = True
            if note_expiry_date_str:
                note_expiry_dt_utc = utils.parse_datetime_flexible(
                    note_expiry_date_str
                )  # Already UTC
                if note_expiry_dt_utc and note_expiry_dt_utc <= now_utc:
                    is_currently_active = False

            if active is True and not is_currently_active:
                passes_filters = False
            if (
                active is False and is_currently_active
            ):  # active=False means we want expired notes
                passes_filters = False

            # Created date filters
            note_created_dt_utc = utils.parse_datetime_flexible(
                note["date_created"]
            )  # Already UTC
            if created_before_dt_utc and (
                not note_created_dt_utc or note_created_dt_utc >= created_before_dt_utc
            ):
                passes_filters = False
            if created_after_dt_utc and (
                not note_created_dt_utc or note_created_dt_utc <= created_after_dt_utc
            ):
                passes_filters = False

            # Expiry date filters (only if note has an expiry date)
            if (
                note_expiry_date_str
            ):  # Only apply these if the note actually has an expiry date
                note_expiry_dt_utc_for_filter = utils.parse_datetime_flexible(
                    note_expiry_date_str
                )  # Already UTC
                if expiry_before_dt_utc and (
                    not note_expiry_dt_utc_for_filter
                    or note_expiry_dt_utc_for_filter >= expiry_before_dt_utc
                ):
                    passes_filters = False
                if expiry_after_dt_utc and (
                    not note_expiry_dt_utc_for_filter
                    or note_expiry_dt_utc_for_filter <= expiry_after_dt_utc
                ):
                    passes_filters = False
            elif (
                expiry_before_str or expiry_after_str
            ):  # If filtering by expiry but note doesn't expire, it fails these filters
                passes_filters = False

            # Tags filter (OR logic: if any of the note's tags match any of the query tags)
            if normalized_query_tags and passes_filters:
                note_tags = note.get("tags", [])
                if not any(qt in note_tags for qt in normalized_query_tags):
                    passes_filters = False

            # Content contains filter
            if content_contains and passes_filters:
                if content_contains.lower() not in note.get("content", "").lower() and (
                    not note.get("title")
                    or content_contains.lower() not in note.get("title", "").lower()
                ):
                    passes_filters = False

            if passes_filters:
                filtered_results.append(note)

        return filtered_results

    def get_active_notes(self) -> List[Dict[str, Any]]:
        """Returns all notes that are currently active (not expired or set to 'forever')."""
        return self.query_notes(active=True)

    def get_expired_notes(self) -> List[Dict[str, Any]]:
        """Returns all notes that have expired."""
        return self.query_notes(active=False)

    def get_all_notes(self) -> List[Dict[str, Any]]:
        """Returns all notes."""
        return self.notes


if __name__ == "__main__":
    # --- Setup for Testing ---
    test_notes_file = config.DATA_DIR / "test_notes.json"
    if test_notes_file.exists():
        test_notes_file.unlink()

    manager = NotesManager(notes_file_path=str(test_notes_file))
    print(f"\n--- Initial state: {len(manager.get_all_notes())} notes ---")

    # --- Test Create ---
    print("\n--- Testing Create Note ---")
    note1_content = "Remember to buy a gift for Mom's birthday."
    note1_title = "Mom's Birthday Gift"
    note1_tags = ["personal", "reminder", "family"]
    # Expires in 7 days from now
    note1_expiry_duration = "7 days"
    note1 = manager.create_note(
        content=note1_content,
        title=note1_title,
        expiry_duration_str=note1_expiry_duration,
        tags=note1_tags,
    )
    print(
        f"Created Note 1 ('{note1['id']}') with duration '{note1_expiry_duration}', expires: {note1.get('expiry_date')}"
    )

    note2_content = (
        "Project Alpha research links: example.com/research, anothersite.org/data"
    )
    note2_title = "Project Alpha Links"
    # Explicit expiry date
    note2_expiry_date = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() + utils.timedelta(days=30)
    )
    note2 = manager.create_note(
        content=note2_content,
        title=note2_title,
        expiry_date_str=note2_expiry_date,
        tags=["work", "research"],
    )
    print(
        f"Created Note 2 ('{note2['id']}') with explicit expiry date '{note2_expiry_date}', stored as: {note2.get('expiry_date')}"
    )

    note3_content = "Core philosophical principles to live by."
    note3_title = "Life Principles"
    # Expires "forever" (i.e., expiry_date is None)
    note3 = manager.create_note(
        content=note3_content,
        title=note3_title,
        expiry_duration_str="forever",
        tags=["philosophy", "core"],
    )
    print(
        f"Created Note 3 ('{note3['id']}') with 'forever' expiry, expires: {note3.get('expiry_date')}"
    )

    note4_content = "This note should expire yesterday."
    note4_title = "Expired Note Test"
    note4_expiry_past = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() - utils.timedelta(days=1)
    )
    note4 = manager.create_note(
        content=note4_content,
        title=note4_title,
        expiry_date_str=note4_expiry_past,
        tags=["test"],
    )
    print(
        f"Created Note 4 ('{note4['id']}') to be expired, expires: {note4.get('expiry_date')}"
    )

    assert len(manager.get_all_notes()) == 4

    # --- Test Get ---
    print("\n--- Testing Get Note ---")
    retrieved_note1 = manager.get_note(note1["id"])
    assert retrieved_note1 is not None and retrieved_note1["title"] == note1_title
    print(f"Retrieved Note 1: {retrieved_note1['id']}")

    # --- Test Update ---
    print("\n--- Testing Update Note ---")
    update1_result = manager.update_note(
        note1["id"],
        {
            "content": "UPDATED: " + note1_content,
            "tags": ["personal", "family", "important"],
        },
    )
    assert update1_result is not None and "UPDATED" in update1_result["content"]
    assert (
        "important" in update1_result["tags"]
        and "reminder" not in update1_result["tags"]
    )
    print(f"Updated Note 1 content and tags. New tags: {update1_result['tags']}")

    # Update expiry of note2 to be "forever" using expiry_duration
    update2_result = manager.update_note(note2["id"], {"expiry_duration": "forever"})
    assert update2_result is not None and update2_result["expiry_date"] is None
    print(
        f"Updated Note 2 expiry to forever. New expiry_date: {update2_result['expiry_date']}"
    )

    # Update expiry of note3 to a specific date (was forever)
    new_expiry_for_note3_str = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() + utils.timedelta(days=10)
    )
    update3_result = manager.update_note(
        note3["id"], {"expiry_date": new_expiry_for_note3_str}
    )
    assert update3_result is not None and update3_result["expiry_date"] is not None
    print(
        f"Updated Note 3 expiry to '{new_expiry_for_note3_str}'. Stored as: {update3_result['expiry_date']}"
    )

    # --- Test Querying ---
    print("\n--- Testing Query Notes ---")
    active_notes = manager.get_active_notes()
    print(
        f"Active notes ({len(active_notes)}):"
    )  # note1, note3 (updated), note2 (updated to forever)
    for n in active_notes:
        print(
            f"  - {n['id']}: {n.get('title', n['content'][:20])}... (Expires: {n.get('expiry_date', 'Forever')})"
        )
    assert len(active_notes) == 3
    assert note4["id"] not in [n["id"] for n in active_notes]

    expired_notes = manager.get_expired_notes()
    print(f"Expired notes ({len(expired_notes)}):")  # note4
    for n in expired_notes:
        print(f"  - {n['id']}: {n.get('title', n['content'][:20])}...")
    assert len(expired_notes) == 1 and expired_notes[0]["id"] == note4["id"]

    research_notes = manager.query_notes(
        tags=["research"]
    )  # note2 was, but expiry changed.
    # Note2's original tags were ["work", "research"]. It's now active (forever).
    print(f"Notes with tag 'research' ({len(research_notes)}):")
    for n in research_notes:
        print(f"  - {n['id']}: {n.get('title')}")
    assert len(research_notes) == 1 and research_notes[0]["id"] == note2["id"]

    philosophy_notes = manager.query_notes(tags=["philosophy", "core"])  # note3
    assert len(philosophy_notes) == 1 and philosophy_notes[0]["id"] == note3["id"]

    notes_containing_gift = manager.query_notes(content_contains="gift")  # note1
    print(f"Notes containing 'gift' ({len(notes_containing_gift)}):")
    assert (
        len(notes_containing_gift) == 1
        and notes_containing_gift[0]["id"] == note1["id"]
    )

    # Test expiry_before_str
    # note1 expires in 7 days, note3 in 10 days. note4 is past. note2 is forever.
    expires_in_8_days_str = utils.format_datetime_for_llm(
        utils.get_current_datetime_local() + utils.timedelta(days=8)
    )
    notes_expiring_soon = manager.query_notes(
        active=True, expiry_before_str=expires_in_8_days_str
    )
    print(
        f"Active notes expiring before {expires_in_8_days_str} ({len(notes_expiring_soon)}):"
    )  # Should be note1
    for n in notes_expiring_soon:
        print(f"  - {n['id']}: {n.get('title')} (Expires: {n.get('expiry_date')})")
    assert len(notes_expiring_soon) == 1 and notes_expiring_soon[0]["id"] == note1["id"]

    # --- Test Delete ---
    print("\n--- Testing Delete Note ---")
    delete_result_note4 = manager.delete_note(note4["id"])
    assert delete_result_note4 is True
    print(f"Deleted Note 4: {'Success' if delete_result_note4 else 'Failed'}")
    assert manager.get_note(note4["id"]) is None
    assert len(manager.get_all_notes()) == 3

    print(f"\n--- Final state: {len(manager.get_all_notes())} notes ---")

    # --- Clean up test file ---
    if test_notes_file.exists():
        print(f"\nCleaning up test file: {test_notes_file}")
        # test_notes_file.unlink() # Uncomment to auto-delete
    else:
        print(f"\nTest file {test_notes_file} not found for cleanup.")

    print("\n--- NotesManager Testing Complete ---")
