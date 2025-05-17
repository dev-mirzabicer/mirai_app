# mirai_app/core/utils.py

import json
import uuid
import re
import logging
from datetime import datetime, timezone, timedelta, date  # Added date here
from pathlib import Path
from typing import Any, Optional, List, Union  # Added Union and date

import pytz
from dateutil import parser as dateutil_parser
from dateutil.parser import isoparse  # For strict ISO date parsing

from mirai_app import config  # Import our config

logger = logging.getLogger(__name__)

# --- Date & Time Utilities ---


def get_current_datetime_utc() -> datetime:
    """Returns the current datetime in UTC."""
    return datetime.now(timezone.utc)


def get_current_datetime_local() -> datetime:
    """Returns the current datetime in Mirza's local timezone."""
    local_tz = pytz.timezone(config.MIRZA_TIMEZONE)
    return datetime.now(local_tz)


def format_datetime_iso(dt_obj: Union[datetime, date]) -> str:  # Accepts date too
    """Formats a datetime or date object to ISO 8601 string (UTC if naive datetime)."""
    if isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None:  # Assume UTC if naive
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    # For date objects, isoformat() is YYYY-MM-DD, which is fine.
    return dt_obj.isoformat()


def format_datetime_for_llm(dt_obj: Union[datetime, date]) -> str:  # Accepts date too
    """
    Formats a datetime or date object into a human-readable string for the LLM.
    For datetimes, includes timezone name. For dates, just YYYY-MM-DD.
    """
    if isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            local_tz = pytz.timezone(config.MIRZA_TIMEZONE)
            dt_obj = (
                local_tz.localize(dt_obj)
                if dt_obj.tzinfo is None
                else dt_obj.astimezone(local_tz)
            )
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")  # e.g., 2025-05-17 19:09:51 EEST
    elif isinstance(dt_obj, date):
        return dt_obj.strftime("%Y-%m-%d")
    return str(dt_obj)  # Fallback


def parse_datetime_flexible(
    datetime_str: str, tz_aware: bool = True
) -> Optional[Union[datetime, date]]:
    """
    Parses a datetime string using dateutil.parser.
    If the string represents only a date (e.g., "YYYY-MM-DD" or "May 20, 2024"),
    it attempts to return a datetime.date object.
    Otherwise, it returns a datetime.datetime object.
    If tz_aware is True and the parsed result is a datetime.datetime object and naive,
    it localizes to MIRZA_TIMEZONE.
    """
    if not datetime_str:
        return None
    try:
        # Attempt to parse as a strict ISO date first
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", datetime_str):
            try:
                # isoparse will raise ValueError if not a valid ISO date string
                # .date() directly gives datetime.date
                return isoparse(datetime_str).date()
            except ValueError:
                # If isoparse fails (e.g. "2023-13-01"), let general parser try or fail
                pass

        # General parsing using dateutil
        dt_obj_parsed = dateutil_parser.parse(datetime_str)

        if isinstance(dt_obj_parsed, datetime):
            # Heuristic: if the parsed datetime is midnight AND the original string
            # doesn't contain explicit time indicators, treat it as a date.
            if dt_obj_parsed.time() == datetime.min.time():
                has_time_indicators = any(
                    indicator in datetime_str.upper()
                    for indicator in [
                        ":",
                        "AM",
                        "PM",
                        "H",
                        "T",
                    ]  # 'T' for ISO8601 T separator
                )
                # Also check for numeric time patterns like HHMMSS or HHMM
                has_numeric_time = re.search(
                    r"\b\d{4,6}\b", datetime_str
                ) and not re.fullmatch(
                    r"\d{4}", datetime_str
                )  # Avoid matching year as time

                if not has_time_indicators and not has_numeric_time:
                    return dt_obj_parsed.date()  # Convert to pure date

            # It's a datetime object, ensure timezone awareness if requested
            if tz_aware and dt_obj_parsed.tzinfo is None:
                local_tz = pytz.timezone(config.MIRZA_TIMEZONE)
                dt_obj_parsed = local_tz.localize(dt_obj_parsed)
            return dt_obj_parsed
        elif isinstance(
            dt_obj_parsed, date
        ):  # Should be rare from dateutil_parser.parse directly
            return dt_obj_parsed

        # Fallback if somehow not datetime or date (shouldn't happen with dateutil)
        return dt_obj_parsed

    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(
            f"Failed to parse datetime string '{datetime_str}': {type(e).__name__} - {e}"
        )
        return None


def _parse_duration_string(duration_str: str) -> Optional[tuple[int, str]]:
    """Helper to parse duration string into value and unit type."""
    if not duration_str:
        return None
    # Regex to capture value and unit.
    match = re.match(
        r"(\d+)\s*(days?|weeks?|months?|hours?|minutes?|d|w|mon|h|m)",
        duration_str,
        re.IGNORECASE,
    )
    if not match:
        # Try ISO 8601 duration format like P1D, PT1H (simplified)
        iso_match = re.match(
            r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
            duration_str.upper(),
        )
        if iso_match:
            years, months, weeks, days, hours, minutes, seconds = [
                int(g) if g else 0 for g in iso_match.groups()
            ]
            if years > 0:
                return years * 12, "months"  # Approximate years to months
            if months > 0:
                return months, "months"
            if weeks > 0:
                return weeks, "weeks"
            if days > 0:
                return days, "days"
            if hours > 0:
                return hours, "hours"
            if minutes > 0:
                return minutes, "minutes"
            # Seconds duration not directly used for event.add('duration', timedelta) in this simplified parser
            logger.warning(
                f"Seconds from ISO duration '{duration_str}' not directly converted by _parse_duration_string."
            )
            return None  # Or handle seconds if needed by returning timedelta directly
        return None

    value = int(match.group(1))
    unit_str_matched = match.group(2).lower()

    if unit_str_matched in ["minute", "minutes", "min", "m"]:
        unit_type = "minutes"
    elif unit_str_matched in ["hour", "hours", "h"]:
        unit_type = "hours"
    elif unit_str_matched in ["day", "days", "d"]:
        unit_type = "days"
    elif unit_str_matched in ["week", "weeks", "w"]:
        unit_type = "weeks"
    elif unit_str_matched in ["month", "months", "mon"]:
        unit_type = "months"
    else:
        logger.warning(
            f"Unmatched unit string after regex: '{unit_str_matched}' from '{duration_str}'"
        )
        return None
    return value, unit_type


def calculate_expiry_date(
    base_dt: datetime, duration_str: Optional[str]
) -> Optional[datetime]:
    """
    Calculates an expiry date based on a duration string (e.g., "1 day", "2 weeks", "30min").
    Returns None if duration_str is None, "forever", or invalid.
    """
    if not duration_str or duration_str.lower() == "forever":
        return None

    parsed_duration = _parse_duration_string(duration_str)
    if not parsed_duration:
        return None

    value, unit_type = parsed_duration

    if unit_type == "minutes":
        return base_dt + timedelta(minutes=value)
    elif unit_type == "hours":
        return base_dt + timedelta(hours=value)
    elif unit_type == "days":
        return base_dt + timedelta(days=value)
    elif unit_type == "weeks":
        return base_dt + timedelta(weeks=value)
    elif unit_type == "months":
        return base_dt + timedelta(days=value * 30)  # Approximation

    return None


def calculate_notification_times(
    due_datetime: datetime, notify_before_list: List[str]
) -> List[datetime]:
    """
    Calculates specific notification datetimes based on a due_datetime and a list of "notify before" durations.
    """
    notification_datetimes = []
    if not notify_before_list:
        return []

    for duration_str in notify_before_list:
        parsed_duration = _parse_duration_string(duration_str)
        if not parsed_duration:
            continue

        value, unit_type = parsed_duration
        delta = None

        if unit_type == "minutes":
            delta = timedelta(minutes=value)
        elif unit_type == "hours":
            delta = timedelta(hours=value)
        elif unit_type == "days":
            delta = timedelta(days=value)
        elif unit_type == "weeks":
            delta = timedelta(weeks=value)
        elif unit_type == "months":
            delta = timedelta(days=value * 30)  # Approximation

        if delta:
            notification_datetimes.append(due_datetime - delta)

    return sorted(list(set(notification_datetimes)))


# --- String & ID Utilities ---


def slugify(text: str) -> str:
    """
    Convert a string to a URL-friendly slug.
    Example: "My Awesome Task!" -> "my_awesome_task"
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(
        r"[^\w\s-]", "", text
    )  # Remove non-alphanumeric characters except spaces and hyphens
    text = re.sub(r"[\s_-]+", "_", text).strip(
        "_"
    )  # Replace spaces/hyphens with single underscore
    return text


def generate_unique_id(
    prefix: str = "", name_elements: Optional[List[str]] = None
) -> str:
    """
    Generates a unique ID.
    Format: {prefix}_{YYYYMMDDHHMMSS}_{slugified_name_elements}_{random_suffix}
    If name_elements are provided, they are slugified and included.
    """
    timestamp = get_current_datetime_utc().strftime("%Y%m%d%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]  # Short random suffix

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(timestamp)

    if name_elements:
        slugified_name = slugify("_".join(filter(None, name_elements)))
        if slugified_name:
            parts.append(slugified_name[:30])  # Limit length of slug part

    parts.append(random_suffix)
    return "_".join(parts)


# --- File I/O Utilities ---


def read_json_file(file_path: Path, default_content: Any = None) -> Any:
    """Reads a JSON file. Returns default_content if file doesn't exist or is invalid."""
    if not file_path.exists():
        if default_content is not None and (isinstance(default_content, (list, dict))):
            write_json_file(
                file_path, default_content
            )  # Create with default if specified
        return (
            default_content if default_content is not None else []
        )  # Default to empty list for typical use
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default_content if default_content is not None else []


def write_json_file(file_path: Path, data: Any) -> bool:
    """Writes data to a JSON file."""
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except IOError:
        return False


def read_md_file(file_path: Path, default_content: str = "") -> str:
    """Reads a Markdown file. Returns default_content if file doesn't exist."""
    if not file_path.exists():
        if (
            default_content
        ):  # Only write if default_content is not empty, to avoid creating empty files unnecessarily
            write_md_file(file_path, default_content)
        return default_content
    try:
        return file_path.read_text(encoding="utf-8")
    except IOError:
        return default_content


def write_md_file(file_path: Path, content: str) -> bool:
    """Writes content to a Markdown file."""
    try:
        file_path.write_text(content, encoding="utf-8")
        return True
    except IOError:
        return False


# --- Other Utilities ---
def get_mirza_location() -> str:
    """
    Gets Mirza's current location.
    Placeholder: For now, returns default. Could be dynamic later.
    """
    return config.MIRZA_LOCATION_DEFAULT


if __name__ == "__main__":

    # Configure logger here
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("--- Testing Date/Time Utilities ---")
    now_utc = get_current_datetime_utc()
    now_local = get_current_datetime_local()
    print(f"Current UTC: {format_datetime_iso(now_utc)}")
    print(f"Current Local ({config.MIRZA_TIMEZONE}): {format_datetime_iso(now_local)}")
    print(f"Current Local for LLM: {format_datetime_for_llm(now_local)}")

    parsed_dt_str = "2025-06-15 10:30 PM"
    parsed_dt = parse_datetime_flexible(parsed_dt_str)
    if parsed_dt:
        print(
            f"Parsed '{parsed_dt_str}': {format_datetime_iso(parsed_dt)} ({parsed_dt.tzinfo})"
        )
    else:
        print(f"Failed to parse '{parsed_dt_str}'")

    parsed_dt_naive_str = "2025-07-01 09:00"
    parsed_dt_naive = parse_datetime_flexible(parsed_dt_naive_str, tz_aware=True)
    if parsed_dt_naive:
        print(
            f"Parsed naive '{parsed_dt_naive_str}' (tz_aware=True): {format_datetime_iso(parsed_dt_naive)} ({parsed_dt_naive.tzinfo})"
        )

    expiry_test_dt = get_current_datetime_local()
    print(
        f"Expiry from '{format_datetime_for_llm(expiry_test_dt)}' + '2 days': {format_datetime_for_llm(calculate_expiry_date(expiry_test_dt, '2 days')) if calculate_expiry_date(expiry_test_dt, '2 days') else 'None'}"
    )
    print(
        f"Expiry from '{format_datetime_for_llm(expiry_test_dt)}' + '30min': {format_datetime_for_llm(calculate_expiry_date(expiry_test_dt, '30min')) if calculate_expiry_date(expiry_test_dt, '30min') else 'None'}"
    )
    print(
        f"Expiry from '{format_datetime_for_llm(expiry_test_dt)}' + '1mon': {format_datetime_for_llm(calculate_expiry_date(expiry_test_dt, '1mon')) if calculate_expiry_date(expiry_test_dt, '1mon') else 'None'}"
    )  # Test for month
    print(
        f"Expiry from '{format_datetime_for_llm(expiry_test_dt)}' + 'forever': {calculate_expiry_date(expiry_test_dt, 'forever')}"
    )

    due_time = parse_datetime_flexible("2025-12-25 14:00:00")
    notify_prefs = ["1d", "2h", "15min", "1w"]  # Added 1 week
    notif_times = calculate_notification_times(due_time, notify_prefs)
    print(
        f"Notification times for due {format_datetime_for_llm(due_time)} with prefs {notify_prefs}:"
    )
    for nt in notif_times:
        print(f"  - {format_datetime_for_llm(nt)}")

    print("\n--- Testing String & ID Utilities ---")
    print(f"Slugify 'My Awesome Task!': {slugify('My Awesome Task!')}")
    print(
        f"Slugify '  Another  Example--Project  ': {slugify('  Another  Example--Project  ')}"
    )
    print(f"Generated ID (no prefix, no name): {generate_unique_id()}")
    print(f"Generated ID (prefix='task'): {generate_unique_id(prefix='task')}")
    print(
        f"Generated ID (name_elements=['Buy Groceries', 'Milk and Eggs']): {generate_unique_id(name_elements=['Buy Groceries', 'Milk and Eggs'])}"
    )
    print(
        f"Generated ID (prefix='note', name_elements=['Important Idea']): {generate_unique_id(prefix='note', name_elements=['Important Idea'])}"
    )

    print("\n--- Testing File I/O Utilities (dummy operations) ---")
    dummy_json_path = config.DATA_DIR / "dummy_test.json"
    dummy_md_path = config.DATA_DIR / "dummy_test.md"

    # Test with default content creation
    non_existent_json = config.DATA_DIR / "new_default.json"
    if non_existent_json.exists():
        non_existent_json.unlink()  # ensure it doesn't exist
    print(
        f"Read non-existent JSON with default list: {read_json_file(non_existent_json, default_content=[])}"
    )
    print(f"Does {non_existent_json.name} exist now? {non_existent_json.exists()}")

    write_json_file(dummy_json_path, {"test": "data", "value": 123})
    print(f"Read from {dummy_json_path.name}: {read_json_file(dummy_json_path)}")
    if dummy_json_path.exists():
        dummy_json_path.unlink()  # Clean up

    write_md_file(dummy_md_path, "# Test Markdown\nThis is a test.")
    print(f"Read from {dummy_md_path.name}: \n{read_md_file(dummy_md_path)}")
    if dummy_md_path.exists():
        dummy_md_path.unlink()  # Clean up

    print(f"\nMirza's default location: {get_mirza_location()}")
