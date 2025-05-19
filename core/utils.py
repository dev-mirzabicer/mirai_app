# mirai_app/core/utils.py

import json
import uuid
import logging
import re
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, List, Union

import pytz
import dateparser
from dateutil import parser as dateutil_parser
from dateutil.parser import isoparse, ParserError as DateutilParserError
from dateutil.relativedelta import relativedelta

from mirai_app import config

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# --- Date & Time Utilities ---


def get_current_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_current_datetime_local() -> datetime:
    try:
        local_tz = pytz.timezone(config.MIRZA_TIMEZONE)
        return datetime.now(local_tz)
    except pytz.UnknownTimeZoneError:
        logger.error(
            f"Unknown timezone '{config.MIRZA_TIMEZONE}', falling back to UTC."
        )
        return datetime.now(timezone.utc)


def format_datetime_iso(dt_obj: Union[datetime, date]) -> str:
    if isinstance(dt_obj, datetime) and dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.isoformat()


def format_datetime_for_llm(dt_obj: Union[datetime, date]) -> str:
    if isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            try:
                tz = pytz.timezone(config.MIRZA_TIMEZONE)
                dt_obj = tz.localize(dt_obj)
            except Exception:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif isinstance(dt_obj, date):
        return dt_obj.strftime("%Y-%m-%d")
    return str(dt_obj)


def parse_datetime_flexible(
    datetime_str: str,
    tz_aware: bool = True,
    default_localize_tz_str: Optional[str] = None,
) -> Optional[Union[datetime, date]]:
    """
    Parse absolute or natural language date/time strings using dateparser with fallback to dateutil.
    Maintains same signature for compatibility.
    """
    if not datetime_str or not datetime_str.strip():
        return None
    settings = {
        "RETURN_AS_TIMEZONE_AWARE": tz_aware,
        "TIMEZONE": default_localize_tz_str or config.MIRZA_TIMEZONE,
        "TO_TIMEZONE": default_localize_tz_str or config.MIRZA_TIMEZONE,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": get_current_datetime_local(),
    }
    try:
        parsed = dateparser.parse(datetime_str, settings=settings)
        if parsed:
            # If time part is midnight and no explicit time in string, return date
            if parsed.time() == datetime.min.time():
                # detect explicit time tokens
                if not re.search(
                    r"\d{1,2}:\d{2}|AM|PM|at ", datetime_str, re.IGNORECASE
                ):
                    return parsed.date()
            return parsed
    except Exception as e:
        logger.warning(f"dateparser failed for '{datetime_str}': {e}")
    # Fallback to dateutil
    try:
        # Handle strict YYYY-MM-DD
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", datetime_str):
            return isoparse(datetime_str).date()
        dt = dateutil_parser.parse(datetime_str)
        if (
            tz_aware
            and isinstance(dt, datetime)
            and (dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None)
        ):
            tz = pytz.timezone(default_localize_tz_str or config.MIRZA_TIMEZONE)
            dt = tz.localize(dt)
        return dt
    except (DateutilParserError, ValueError) as e:
        logger.warning(f"dateutil failed for '{datetime_str}': {e}")
    return None


def _parse_duration_string(duration_str: str) -> Optional[tuple[int, str]]:
    if not duration_str:
        return None
    match = re.match(
        r"(\d+)\s*(years?|months?|weeks?|days?|hours?|minutes?|secs?|s|m|h|d|w)",
        duration_str,
        re.IGNORECASE,
    )
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("year"):
            return value * 12, "months"
        if unit.startswith("month"):
            return value, "months"
        if unit.startswith("week"):
            return value, "weeks"
        if unit.startswith("day"):
            return value, "days"
        if unit.startswith("hour") or unit == "h":
            return value, "hours"
        if unit.startswith("min") or unit == "m":
            return value, "minutes"
        if unit.startswith("sec") or unit == "s":
            return value, "seconds"
    # ISO8601 fallback
    iso = re.match(
        r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        duration_str.upper(),
    )
    if iso:
        parts = [int(p) if p else 0 for p in iso.groups()]
        years, months, weeks, days, hours, minutes, seconds = parts
        if years:
            return years * 12, "months"
        if months:
            return months, "months"
        if weeks:
            return weeks, "weeks"
        if days:
            return days, "days"
        if hours:
            return hours, "hours"
        if minutes:
            return minutes, "minutes"
    return None


def calculate_expiry_date(
    base_dt: datetime, duration_str: Optional[str]
) -> Optional[datetime]:
    if not duration_str or duration_str.lower() == "forever":
        return None
    parsed = _parse_duration_string(duration_str)
    if not parsed:
        return None
    value, unit = parsed
    if unit == "seconds":
        return base_dt + timedelta(seconds=value)
    if unit == "minutes":
        return base_dt + timedelta(minutes=value)
    if unit == "hours":
        return base_dt + timedelta(hours=value)
    if unit == "days":
        return base_dt + timedelta(days=value)
    if unit == "weeks":
        return base_dt + timedelta(weeks=value)
    if unit == "months":
        return base_dt + relativedelta(months=value)
    return None


def calculate_notification_times(
    due_datetime: datetime, notify_before_list: List[str]
) -> List[datetime]:
    times = []
    for ds in set(notify_before_list or []):
        parsed = _parse_duration_string(ds)
        if not parsed:
            continue
        val, unit = parsed
        delta = None
        if unit == "seconds":
            delta = timedelta(seconds=val)
        elif unit == "minutes":
            delta = timedelta(minutes=val)
        elif unit == "hours":
            delta = timedelta(hours=val)
        elif unit == "days":
            delta = timedelta(days=val)
        elif unit == "weeks":
            delta = timedelta(weeks=val)
        elif unit == "months":
            delta = relativedelta(months=val)
        if delta:
            times.append(due_datetime - delta)
    return sorted(times)


def slugify(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text


def generate_unique_id(
    prefix: str = "", name_elements: Optional[List[str]] = None
) -> str:
    ts = get_current_datetime_utc().strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    parts = [prefix] if prefix else []
    parts.append(ts)
    if name_elements:
        slug = slugify("_".join(filter(None, name_elements)))[:30]
        if slug:
            parts.append(slug)
    parts.append(suffix)
    return "_".join(parts)


def read_json_file(file_path: Path, default_content: Any = None) -> Any:
    if not file_path.exists():
        if default_content is not None and isinstance(default_content, (dict, list)):
            write_json_file(file_path, default_content)
        return default_content if default_content is not None else []
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return default_content if default_content is not None else []


def write_json_file(file_path: Path, data: Any) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def read_md_file(file_path: Path, default_content: str = "") -> str:
    if not file_path.exists():
        if default_content:
            write_md_file(file_path, default_content)
        return default_content
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        return default_content


def write_md_file(file_path: Path, content: str) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def get_mirza_location() -> str:
    return config.MIRZA_LOCATION_DEFAULT
