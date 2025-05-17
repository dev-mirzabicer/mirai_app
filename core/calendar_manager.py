# mirai_app/core/calendar_manager.py

import logging
import uuid
from datetime import (
    datetime,
    date,
    timedelta,
)  # Removed timezone, use dt_timezone or pytz
from datetime import timezone as dt_timezone  # Alias for datetime.timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from icalendar import (
    Calendar as iCalCalendar,
    Event as iCalEvent,
    vRecur,
    vText,
)
from icalendar import cal as icalendar_cal_module
from icalevents.icalevents import events as fetch_events
from icalevents import icalparser  # For type hints
import pytz  # For robust timezone handling
from dateutil.relativedelta import relativedelta  # For accurate month arithmetic
from dateutil.tz import tzoffset  # To check for this specific tzinfo type

from mirai_app import config  # Assuming this is in the parent directory or PYTHONPATH
from mirai_app.core import utils  # Assuming this is in the same directory or PYTHONPATH

# Configure logging - ensure this is configured by the main application
# If this module is run standalone for testing, then configure it.
logger = logging.getLogger(__name__)
if (
    not logger.hasHandlers()
):  # Configure only if not already configured by a higher-level module
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class CalendarManager:
    """
    Manages calendar events using an .ics file.
    Uses 'icalendar' for creating and manipulating iCalendar objects,
    and 'icalevents' for querying events within date ranges (handles recurrence).
    """

    def __init__(self, calendar_file_path: Optional[Union[str, Path]] = None):
        self.calendar_file = (
            Path(calendar_file_path)
            if calendar_file_path
            else config.DATA_DIR / "calendar.ics"
        )
        self.calendar: iCalCalendar = self._load_calendar()
        logger.info(
            f"CalendarManager initialized. Using calendar file: {self.calendar_file}"
        )

    def _load_calendar(self) -> iCalCalendar:
        """Loads the calendar from the .ics file, or creates a new one."""
        if self.calendar_file.exists() and self.calendar_file.stat().st_size > 0:
            try:
                with self.calendar_file.open("rb") as f:
                    cal = iCalCalendar.from_ical(f.read())
                # Ensure essential properties exist on the loaded calendar
                if not cal.get("prodid"):
                    cal.add("prodid", "-//MIRAI Personal Assistant//mirzabicer.dev//")
                if not cal.get("version"):
                    cal.add("version", "2.0")
                logger.info(
                    f"Loaded {len(list(cal.walk('VEVENT')))} events from {self.calendar_file}"
                )
                return cal
            except Exception as e:  # Catch broad exceptions from icalendar parsing
                logger.error(
                    f"Error parsing existing calendar file {self.calendar_file}: {type(e).__name__} - {e}. Creating a new calendar."
                )

        # Create a new calendar if file doesn't exist, is empty, or parsing failed
        cal = iCalCalendar()
        cal.add("prodid", "-//MIRAI Personal Assistant//mirzabicer.dev//")
        cal.add("version", "2.0")
        if not self._save_calendar(cal):  # Attempt to save immediately
            logger.error(
                f"CRITICAL: Failed to create and save initial calendar file at {self.calendar_file}. Calendar operations may fail."
            )
            # Depending on requirements, you might want to raise an exception here
        else:
            logger.info(f"Created a new calendar at {self.calendar_file}")
        return cal

    def _save_calendar(self, calendar_obj: Optional[iCalCalendar] = None) -> bool:
        """Saves the provided calendar object (or self.calendar) to the .ics file."""
        cal_to_save = calendar_obj if calendar_obj is not None else self.calendar
        try:
            # This method ensures VTIMEZONE components are added for TZIDs used in the calendar
            # if they are missing. It uses dateutil.tz.gettz() internally.
            cal_to_save.add_missing_timezones()

            with self.calendar_file.open("wb") as f:  # Open in binary write mode
                f.write(cal_to_save.to_ical())
            logger.debug(f"Successfully saved calendar to {self.calendar_file}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to save calendar to {self.calendar_file}: {type(e).__name__} - {e}"
            )
            return False

    def _generate_event_uid(self) -> str:
        """Generates a unique ID for an event."""
        return str(uuid.uuid4())

    def _ensure_timezone_aware(
        self,
        dt_input: Union[datetime, date],
        default_tz_str: str,  # This should be a valid IANA name for pytz
    ) -> Union[datetime, date]:
        """Ensures datetime is timezone-aware. Dates are returned as is."""
        if isinstance(dt_input, datetime):
            if (
                dt_input.tzinfo is None or dt_input.tzinfo.utcoffset(dt_input) is None
            ):  # Naive
                try:
                    # Basic check for IANA format; pytz.timezone will do the real validation
                    if (
                        not isinstance(default_tz_str, str)
                        or "/" not in default_tz_str
                        and default_tz_str.upper() != "UTC"
                    ):
                        raise pytz.UnknownTimeZoneError(
                            f"Invalid IANA name format for pytz: {default_tz_str}"
                        )

                    target_tz = (
                        pytz.timezone(default_tz_str)
                        if default_tz_str.upper() != "UTC"
                        else pytz.utc
                    )
                    return target_tz.localize(dt_input)
                except pytz.UnknownTimeZoneError:
                    logger.warning(
                        f"Unknown default timezone '{default_tz_str}' in _ensure_timezone_aware. Falling back to UTC."
                    )
                    return pytz.utc.localize(dt_input)  # Use pytz.utc for consistency
            return dt_input  # Already aware
        return dt_input  # It's a date object

    def _event_to_dict(self, event: iCalEvent) -> Dict[str, Any]:
        """Converts an icalendar.Event object to a dictionary."""
        data = {}

        def get_prop_val(prop_name: str):
            return event.get(prop_name)

        def get_prop_str(prop_name: str, default_val: str = "") -> str:
            prop = get_prop_val(prop_name)
            if prop is None:
                return default_val
            if isinstance(prop, list):  # e.g. CATEGORIES
                return ", ".join(
                    (
                        p.to_ical().decode()
                        if hasattr(p, "to_ical") and isinstance(p.to_ical(), bytes)
                        else str(p)
                    )
                    for p in prop
                )

            # Handle UID specifically as it might be a plain string or vUID
            if prop_name.upper() == "UID":
                return str(prop)  # UID is often just a string from icalendar.Event.add

            ical_repr = prop.to_ical() if hasattr(prop, "to_ical") else str(prop)
            return (
                ical_repr.decode() if isinstance(ical_repr, bytes) else str(ical_repr)
            )

        data["uid"] = get_prop_str("uid")
        data["summary"] = get_prop_str("summary")
        data["description"] = get_prop_str("description")
        data["location"] = get_prop_str("location")

        for prop_name_upper in [
            "DTSTART",
            "DTEND",
            "CREATED",
            "LAST-MODIFIED",
            "DTSTAMP",
        ]:
            prop_val = get_prop_val(prop_name_upper)
            key = prop_name_upper.lower().replace("-", "_")  # e.g. last_modified
            data[key] = prop_val.dt if prop_val and hasattr(prop_val, "dt") else None

        duration_prop = get_prop_val("DURATION")
        data["duration"] = (
            duration_prop.dt if duration_prop and hasattr(duration_prop, "dt") else None
        )

        rrule_prop = get_prop_val("rrule")
        if rrule_prop:
            if isinstance(rrule_prop, list):  # Though rare, RRULE can be a list
                data["rrule"] = [
                    r.to_ical().decode("utf-8")
                    for r in rrule_prop
                    if hasattr(r, "to_ical")
                ]
            elif hasattr(rrule_prop, "to_ical"):  # Should be vRecur
                data["rrule"] = rrule_prop.to_ical().decode("utf-8")
            else:  # Fallback, should not happen for valid RRULE
                data["rrule"] = str(rrule_prop)
        else:
            data["rrule"] = None

        return data

    def _icalevents_event_to_dict(
        self, icalevents_event: icalparser.Event
    ) -> Dict[str, Any]:
        """Converts an icalevents.icalparser.Event to our standard dictionary format."""
        calculated_duration = None
        # icalevents.Event.start/end are already timezone-aware if tzinfo was passed to fetch_events,
        # or they are date objects for all-day events.
        if isinstance(icalevents_event.start, datetime) and isinstance(
            icalevents_event.end, datetime
        ):
            calculated_duration = icalevents_event.end - icalevents_event.start
        elif (
            isinstance(icalevents_event.start, date)
            and isinstance(icalevents_event.end, date)
            and icalevents_event.all_day
        ):
            # For all-day events, icalevents.end is exclusive.
            # So, (end - start) gives the number of days of the occurrence.
            calculated_duration = icalevents_event.end - icalevents_event.start

        return {
            "uid": str(icalevents_event.uid),
            "summary": str(icalevents_event.summary),
            "start": icalevents_event.start,
            "end": icalevents_event.end,
            "duration": calculated_duration,  # Use calculated duration
            "description": str(icalevents_event.description or ""),
            "location": str(icalevents_event.location or ""),
            "all_day": icalevents_event.all_day,
        }

    def _add_datetime_property_with_tzid(
        self,
        event_component: iCalEvent,
        prop_name: str,
        dt_obj: Union[datetime, date],
        # effective_tz_str: Optional[str] # Not directly needed if dt_obj is already correctly aware
        default_tz_str: str,  # <— new required arg
    ):
        """Helper to add DTSTART, DTEND with explicit TZID if applicable."""
        params = {}
        if isinstance(dt_obj, datetime) and dt_obj.tzinfo is not None:
            # Check if it's not UTC. UTC datetimes are written with 'Z' and don't need TZID.
            if dt_obj.tzinfo.utcoffset(dt_obj) != timedelta(0):
                # Try to get tzid from pytz-like (.zone) or zoneinfo-like (.key) objects
                tzid = (
                    getattr(dt_obj.tzinfo, "zone", None)
                    or getattr(dt_obj.tzinfo, "key", None)
                    or default_tz_str
                )  # <— fallback
                if tzid:
                    params["TZID"] = tzid

        event_component.add(prop_name, dt_obj, parameters=params if params else None)

    def create_event(
        self,
        summary: str,
        start_dt_str: str,
        end_dt_str: Optional[str] = None,
        duration_str: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        uid: Optional[str] = None,
        rrule_dict: Optional[Dict[str, Any]] = None,
        timezone_str: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:

        if not summary or not start_dt_str:
            logger.error("Summary and start datetime are required to create an event.")
            return None

        effective_input_tz_str = timezone_str or config.MIRZA_TIMEZONE

        start_dt_obj_parsed = utils.parse_datetime_flexible(
            start_dt_str, tz_aware=True, default_localize_tz_str=effective_input_tz_str
        )
        if not start_dt_obj_parsed:
            logger.error(f"Invalid start datetime string: {start_dt_str}")
            return None
        start_dt_obj = self._ensure_timezone_aware(
            start_dt_obj_parsed, effective_input_tz_str
        )
        is_start_pure_date = isinstance(start_dt_obj, date) and not isinstance(
            start_dt_obj, datetime
        )

        if not end_dt_str and not duration_str:
            if not is_start_pure_date:
                logger.error(
                    "Either end datetime or duration is required for timed events."
                )
                return None
            duration_str = "1d"

        end_dt_obj = None
        if end_dt_str:
            end_dt_obj_parsed = utils.parse_datetime_flexible(
                end_dt_str,
                tz_aware=True,
                default_localize_tz_str=effective_input_tz_str,
            )
            if not end_dt_obj_parsed:
                logger.error(f"Invalid end datetime string: {end_dt_str}")
                return None
            end_dt_obj = self._ensure_timezone_aware(
                end_dt_obj_parsed, effective_input_tz_str
            )

        if end_dt_obj:  # Validation block from your original code
            is_end_pure_date = isinstance(end_dt_obj, date) and not isinstance(
                end_dt_obj, datetime
            )
            if is_start_pure_date != is_end_pure_date:
                logger.error(
                    "Start and End must both be dates (all-day) or datetimes (timed). Mixed types not allowed."
                )
                return None
            comp_start, comp_end = start_dt_obj, end_dt_obj
            if isinstance(comp_start, datetime) and isinstance(comp_end, datetime):
                utc = pytz.utc
                s_utc = (
                    comp_start.astimezone(utc)
                    if comp_start.tzinfo
                    else utc.localize(comp_start)
                )
                e_utc = (
                    comp_end.astimezone(utc)
                    if comp_end.tzinfo
                    else utc.localize(comp_end)
                )
                if e_utc <= s_utc:
                    logger.error(
                        f"End ({end_dt_obj}) must be after start ({start_dt_obj}). UTC compare: {e_utc} <= {s_utc}"
                    )
                    return None
            elif isinstance(comp_start, date) and isinstance(comp_end, date):
                if comp_end <= comp_start:
                    logger.error(
                        f"End date ({end_dt_obj}) must be after start date ({start_dt_obj})."
                    )
                    return None

        event = iCalEvent()
        event.add("uid", uid or self._generate_event_uid())
        event.add("summary", summary)

        # CHANGE 1 Applied here for DTSTART
        self._add_datetime_property_with_tzid(
            event, "dtstart", start_dt_obj, effective_input_tz_str
        )

        if end_dt_obj:
            # CHANGE 1 Applied here for DTEND
            self._add_datetime_property_with_tzid(
                event, "dtend", end_dt_obj, effective_input_tz_str
            )
        elif duration_str:
            try:
                parsed_duration_val_unit = utils._parse_duration_string(duration_str)
                if not parsed_duration_val_unit:
                    raise ValueError(f"Could not parse duration string: {duration_str}")
                value, unit = parsed_duration_val_unit
                td: Optional[timedelta] = None
                if unit == "minutes":
                    td = timedelta(minutes=value)
                elif unit == "hours":
                    td = timedelta(hours=value)
                elif unit == "days":
                    td = timedelta(days=value)
                elif unit == "weeks":
                    td = timedelta(weeks=value)
                elif unit == "months":
                    if is_start_pure_date:
                        calculated_end_dt = start_dt_obj + relativedelta(months=value)
                        # CHANGE 1 Applied here for DTEND (calculated from month duration for all-day)
                        self._add_datetime_property_with_tzid(
                            event, "dtend", calculated_end_dt, effective_input_tz_str
                        )
                    else:
                        logger.error(
                            "Month-based duration for timed events not directly supported via DURATION. Calculate DTEND."
                        )
                        return None
                else:
                    raise ValueError(f"Unsupported duration unit: {unit}")
                if td:
                    event.add(
                        "duration", td
                    )  # If timedelta was formed (not month-based for date)
            except ValueError as e:
                logger.error(f"Invalid duration string: {duration_str}. Error: {e}")
                return None

        now_utc = utils.get_current_datetime_utc()
        event.add("dtstamp", now_utc)
        event.add("created", now_utc)
        event.add("last-modified", now_utc)

        if description:
            event.add("description", description)
        if location:
            event.add("location", location)

        if rrule_dict:
            if "UNTIL" in rrule_dict and isinstance(
                rrule_dict["UNTIL"], (datetime, date)
            ):
                until_parsed = self._ensure_timezone_aware(
                    rrule_dict["UNTIL"], effective_input_tz_str
                )
                # For vRecur, UNTIL should be aware if it's a datetime.
                # If it's a date, it's fine. If it's a datetime, icalendar handles its TZID.
                # We ensure it's aware here for consistency with how icalendar expects it.
                # vRecur itself will handle if TZID needs to be part of the RRULE string for UNTIL.
                rrule_dict["UNTIL"] = until_parsed
            event.add("rrule", vRecur(rrule_dict))

        self.calendar.add_component(event)
        if self._save_calendar():
            logger.info(f"Created event '{summary}' (UID: {event.get('uid')})")
            return self._event_to_dict(event)
        return None

    def get_event_details(self, uid: str) -> Optional[Dict[str, Any]]:
        """Retrieves the details of an event by its UID."""
        for component in self.calendar.walk("VEVENT"):
            if str(component.get("uid")) == uid:  # UID property might be string or vUID
                return self._event_to_dict(component)
        logger.debug(f"Event with UID '{uid}' not found for get_event_details.")
        return None

    def update_event(
        self, uid: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        original_event: Optional[iCalEvent] = None
        component_index = -1

        for i, component in enumerate(self.calendar.subcomponents):
            if isinstance(component, iCalEvent) and str(component.get("uid")) == uid:
                original_event = component
                component_index = i
                break
        if not original_event:
            logger.warning(f"Cannot update. Event with UID '{uid}' not found.")
            return None
        try:
            temp_event = iCalEvent.from_ical(original_event.to_ical())
        except Exception as e:
            logger.error(
                f"Failed to create temporary copy of event {uid} for update: {e}"
            )
            return None

        effective_input_tz_str = updates.get("timezone_str", config.MIRZA_TIMEZONE)
        updated_fields_log = []

        current_dtstart_val = (
            temp_event.get("dtstart").dt if temp_event.get("dtstart") else None
        )
        current_dtend_val = (
            temp_event.get("dtend").dt if temp_event.get("dtend") else None
        )
        current_duration_val = (
            temp_event.get("duration").dt if temp_event.get("duration") else None
        )

        def get_decoded_prop_for_compare(prop_obj):
            if prop_obj is None:
                return ""
            if isinstance(prop_obj, vRecur):
                return prop_obj.to_ical().decode()
            ical_form = (
                prop_obj.to_ical() if hasattr(prop_obj, "to_ical") else str(prop_obj)
            )
            return (
                ical_form.decode() if isinstance(ical_form, bytes) else str(ical_form)
            )

        if (
            "summary" in updates
            and get_decoded_prop_for_compare(temp_event.get("summary"))
            != updates["summary"]
        ):
            temp_event["summary"] = vText(updates["summary"])
            updated_fields_log.append("summary")

        if "start_dt_str" in updates:
            parsed_start = utils.parse_datetime_flexible(
                updates["start_dt_str"],
                tz_aware=True,
                default_localize_tz_str=effective_input_tz_str,
            )
            if parsed_start:
                new_start_val = self._ensure_timezone_aware(
                    parsed_start, effective_input_tz_str
                )
                if new_start_val != current_dtstart_val:
                    current_dtstart_val = new_start_val
                    # CHANGE 1 Applied in update for dtstart
                    self._add_datetime_property_with_tzid(
                        temp_event,
                        "dtstart",
                        current_dtstart_val,
                        effective_input_tz_str,
                    )
                    updated_fields_log.append("dtstart")
            else:
                logger.warning(
                    f"Invalid start_dt_str in update for UID {uid}: {updates['start_dt_str']}."
                )

        dtend_updated_in_this_call = False
        if "end_dt_str" in updates:
            parsed_end = utils.parse_datetime_flexible(
                updates["end_dt_str"],
                tz_aware=True,
                default_localize_tz_str=effective_input_tz_str,
            )
            if parsed_end:
                new_end_val = self._ensure_timezone_aware(
                    parsed_end, effective_input_tz_str
                )
                if new_end_val != current_dtend_val or ("duration" in temp_event):
                    current_dtend_val = new_end_val
                    # CHANGE 1 Applied in update for dtend
                    self._add_datetime_property_with_tzid(
                        temp_event, "dtend", current_dtend_val, effective_input_tz_str
                    )
                    if "duration" in temp_event:
                        del temp_event["duration"]
                    current_duration_val = None
                    updated_fields_log.append("dtend")
                dtend_updated_in_this_call = True
            else:
                logger.warning(
                    f"Invalid end_dt_str in update for UID {uid}: {updates['end_dt_str']}."
                )

        if "duration_str" in updates and not dtend_updated_in_this_call:
            duration_val_str = updates["duration_str"]
            new_duration_val = None
            new_dtend_for_month_duration = None
            if duration_val_str is None:
                if current_duration_val is not None:
                    if "duration" in temp_event:
                        del temp_event["duration"]
                    current_duration_val = None
                    updated_fields_log.append("duration (cleared)")
            else:
                try:
                    parsed_duration_val_unit = utils._parse_duration_string(
                        duration_val_str
                    )
                    if not parsed_duration_val_unit:
                        raise ValueError("Parse failed")
                    value, unit = parsed_duration_val_unit
                    td = None
                    is_current_start_pure_date = isinstance(
                        current_dtstart_val, date
                    ) and not isinstance(current_dtstart_val, datetime)

                    if unit == "minutes":
                        td = timedelta(minutes=value)
                    elif unit == "hours":
                        td = timedelta(hours=value)
                    elif unit == "days":
                        td = timedelta(days=value)
                    elif unit == "weeks":
                        td = timedelta(weeks=value)
                    elif unit == "months":
                        if is_current_start_pure_date and current_dtstart_val:
                            new_dtend_for_month_duration = (
                                current_dtstart_val + relativedelta(months=value)
                            )
                        else:
                            raise ValueError("Month duration for timed event")
                    else:
                        raise ValueError(f"Unsupported duration unit: {unit}")
                    if td:
                        new_duration_val = td

                    if (
                        new_duration_val != current_duration_val
                        or (
                            new_dtend_for_month_duration
                            and new_dtend_for_month_duration != current_dtend_val
                        )
                        or ("dtend" in temp_event and new_duration_val is not None)
                    ):
                        if new_dtend_for_month_duration:
                            current_dtend_val = new_dtend_for_month_duration
                            # CHANGE 1 Applied in update for dtend (from duration)
                            self._add_datetime_property_with_tzid(
                                temp_event,
                                "dtend",
                                current_dtend_val,
                                effective_input_tz_str,
                            )
                            if "duration" in temp_event:
                                del temp_event["duration"]
                                current_duration_val = None
                        elif new_duration_val:
                            temp_event["duration"] = new_duration_val
                            current_duration_val = new_duration_val
                            if "dtend" in temp_event:
                                del temp_event["dtend"]
                                current_dtend_val = None
                        updated_fields_log.append("duration/dtend from duration")
                except ValueError as e:
                    logger.warning(
                        f"Invalid duration string for update: {duration_val_str}. Error: {e}."
                    )

        # --- Validation Block (applied to potentially modified current_dtstart_val, current_dtend_val, current_duration_val) ---
        # This block is crucial and should use the state of current_dtstart_val, current_dtend_val, current_duration_val
        # ... (validation block from your original code, ensure it uses these current_* variables) ...
        final_start_on_temp = current_dtstart_val
        final_end_on_temp = None
        if current_dtend_val:
            final_end_on_temp = current_dtend_val
        elif final_start_on_temp and current_duration_val:
            if isinstance(final_start_on_temp, (datetime, date)):
                final_end_on_temp = final_start_on_temp + current_duration_val
            else:
                logger.error(
                    f"Internal error: final_start_on_temp not date/datetime for UID {uid} during update validation."
                )
                return None
        if not final_start_on_temp:
            logger.error(f"Update for UID {uid} results in event without a start time.")
            return None
        is_final_start_pure_date = isinstance(
            final_start_on_temp, date
        ) and not isinstance(final_start_on_temp, datetime)
        if final_end_on_temp:
            is_final_end_pure_date = isinstance(
                final_end_on_temp, date
            ) and not isinstance(final_end_on_temp, datetime)
            if is_final_start_pure_date != is_final_end_pure_date:
                logger.error(
                    f"Update for UID {uid} creates invalid state: Start/End type mismatch. Start is date: {is_final_start_pure_date}, End is date: {is_final_end_pure_date}."
                )
                return None
            comp_start, comp_end = final_start_on_temp, final_end_on_temp
            if isinstance(comp_start, datetime) and isinstance(comp_end, datetime):
                utc = pytz.utc
                s_utc = (
                    comp_start.astimezone(utc)
                    if comp_start.tzinfo
                    else utc.localize(comp_start)
                )
                e_utc = (
                    comp_end.astimezone(utc)
                    if comp_end.tzinfo
                    else utc.localize(comp_end)
                )
                if e_utc <= s_utc:
                    logger.error(
                        f"Update for UID {uid} creates invalid state: End ({final_end_on_temp}) not after start ({final_start_on_temp}). UTC Compare: {e_utc} <= {s_utc}"
                    )
                    return None
            elif isinstance(comp_start, date) and isinstance(comp_end, date):
                if comp_end <= comp_start:
                    logger.error(
                        f"Update for UID {uid} creates invalid state: End date ({final_end_on_temp}) not after start date ({final_start_on_temp})."
                    )
                    return None
        elif not is_final_start_pure_date:
            logger.error(
                f"Update for UID {uid} results in timed event without determined end."
            )
            return None
        # --- End of Validation Block ---

        for key in ["description", "location"]:
            if (
                key in updates
                and get_decoded_prop_for_compare(temp_event.get(key)) != updates[key]
            ):
                temp_event[key] = vText(updates[key])
                updated_fields_log.append(key)

        if "rrule_dict" in updates:
            rrule_val = updates["rrule_dict"]
            current_rrule_obj = temp_event.get("rrule")
            current_rrule_str = (
                current_rrule_obj.to_ical().decode()
                if current_rrule_obj and hasattr(current_rrule_obj, "to_ical")
                else ""
            )
            new_rrule_str = ""
            if rrule_val is None:
                if "rrule" in temp_event:
                    del temp_event["rrule"]
            elif isinstance(rrule_val, dict):
                if "UNTIL" in rrule_val and isinstance(
                    rrule_val["UNTIL"], (datetime, date)
                ):
                    rrule_val["UNTIL"] = self._ensure_timezone_aware(
                        rrule_val["UNTIL"], effective_input_tz_str
                    )
                new_rrule_obj = vRecur(rrule_val)
                temp_event["rrule"] = new_rrule_obj
                new_rrule_str = new_rrule_obj.to_ical().decode()
            if new_rrule_str != current_rrule_str:
                updated_fields_log.append("rrule")

        if not updated_fields_log:
            logger.info(f"No effective changes applied for update of event UID {uid}.")
            return self._event_to_dict(original_event)

        if "last-modified" in temp_event:
            del temp_event["last-modified"]
        if "dtstamp" in temp_event:
            del temp_event["dtstamp"]

        now_utc = utils.get_current_datetime_utc()
        temp_event.add(
            "last-modified", now_utc
        )  # RFC 5545 basic format (YYYYMMDDTHHMMSSZ)
        temp_event.add("dtstamp", now_utc)
        if "last-modified/dtstamp" not in updated_fields_log:
            updated_fields_log.append("last-modified/dtstamp")

        self.calendar.subcomponents[component_index] = temp_event
        if self._save_calendar():
            logger.info(
                f"Updated event '{uid}'. Changed: {', '.join(updated_fields_log)}"
            )
            return self._event_to_dict(temp_event)

        logger.error(
            f"Failed to save calendar after updating event {uid}. Reverting in-memory changes."
        )
        self.calendar.subcomponents[component_index] = original_event
        return None

    def delete_event(self, uid: str) -> bool:
        """Deletes an event by its UID."""
        initial_len = len(self.calendar.subcomponents)
        # Filter out VEVENTs, keep other components like VTIMEZONE
        self.calendar.subcomponents = [
            comp
            for comp in self.calendar.subcomponents
            if not (isinstance(comp, iCalEvent) and str(comp.get("uid")) == uid)
        ]
        if len(self.calendar.subcomponents) < initial_len:
            if self._save_calendar():
                logger.info(f"Deleted event with UID '{uid}'.")
                return True
            else:  # Failed to save, theoretically should revert but that's complex
                logger.error(
                    f"Found and removed event '{uid}' from memory, but failed to save calendar."
                )
                # To ensure consistency, reload the calendar from its last saved state
                self.calendar = self._load_calendar()
                return False
        logger.warning(f"Event with UID '{uid}' not found for deletion.")
        return False

    def list_events_in_range(
        self,
        start_range_str: str,
        end_range_str: str,
        target_timezone_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        effective_target_tz_str = target_timezone_str or config.MIRZA_TIMEZONE
        try:
            target_pytz_obj = pytz.timezone(effective_target_tz_str)
        except pytz.UnknownTimeZoneError:
            logger.error(
                f"Unknown target timezone: {effective_target_tz_str}. Defaulting to UTC."
            )
            target_pytz_obj = pytz.utc
            effective_target_tz_str = "UTC"  # Update for _ensure_timezone_aware

        start_dt_parsed = utils.parse_datetime_flexible(
            start_range_str,
            tz_aware=True,
            default_localize_tz_str=effective_target_tz_str,
        )
        end_dt_parsed = utils.parse_datetime_flexible(
            end_range_str,
            tz_aware=True,
            default_localize_tz_str=effective_target_tz_str,
        )

        if not start_dt_parsed or not end_dt_parsed:
            logger.error("Invalid start or end date range for listing events.")
            return []

        # Ensure query start/end are aware using the target_pytz_obj's string name for consistency
        query_start_dt = self._ensure_timezone_aware(
            start_dt_parsed, str(target_pytz_obj)
        )
        query_end_dt = self._ensure_timezone_aware(end_dt_parsed, str(target_pytz_obj))

        # Normalize for comparison if both are datetimes
        comp_start_query, comp_end_query = query_start_dt, query_end_dt
        if isinstance(comp_start_query, datetime) and isinstance(
            comp_end_query, datetime
        ):
            utc = pytz.utc
            s_utc = (
                comp_start_query.astimezone(utc)
                if comp_start_query.tzinfo
                else utc.localize(comp_start_query)
            )
            e_utc = (
                comp_end_query.astimezone(utc)
                if comp_end_query.tzinfo
                else utc.localize(comp_end_query)
            )
            if e_utc <= s_utc:
                logger.error(
                    f"Query range error: end ({query_end_dt}) must be after start ({query_start_dt})."
                )
                return []
        elif isinstance(comp_start_query, date) and isinstance(comp_end_query, date):
            if (
                comp_end_query < comp_start_query
            ):  # For dates, end can be same as start for single day query
                logger.error(
                    f"Query range error: end date ({query_end_dt}) must be same or after start date ({query_start_dt})."
                )
                return []
        # Mixed date/datetime for query range start/end should ideally be caught by parsing or be consistent

        if not self.calendar_file.exists() or self.calendar_file.stat().st_size == 0:
            logger.info("Calendar file is empty or does not exist. No events to list.")
            return []

        # It's good practice to ensure the latest in-memory state is on disk before an external tool reads it.
        if not self._save_calendar():
            logger.warning(
                "Failed to save calendar before listing events. Results might be from a previous state."
            )
            # Decide if to proceed or return empty/raise error

        try:
            # Pass target_pytz_obj to icalevents.events()
            # This tells icalevents to return event times in this timezone.
            fetched_ical_events = fetch_events(
                file=str(self.calendar_file),
                start=query_start_dt,
                end=query_end_dt,
                tzinfo=target_pytz_obj,
            )

            dict_events = [
                self._icalevents_event_to_dict(ev) for ev in fetched_ical_events
            ]
            # Sort by start time, ensuring dates are comparable with datetimes
            dict_events.sort(
                key=lambda x: (
                    x["start"]
                    if isinstance(x["start"], datetime)
                    else datetime.combine(x["start"], datetime.min.time()).replace(
                        tzinfo=(
                            target_pytz_obj
                            if isinstance(x["start"], date) and target_pytz_obj.zone
                            else None
                        )
                    )  # Make date objects comparable by converting to datetime at midnight in target_tz
                )
            )
            return dict_events
        except Exception as e:
            logger.error(
                f"Error listing events with icalevents: {type(e).__name__} - {e}"
            )
            # import traceback
            # logger.error(traceback.format_exc()) # For more detailed stack trace if needed
            return []

    def get_schedule_for_date(
        self, date_str: str, target_timezone_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        effective_target_tz_str = target_timezone_str or config.MIRZA_TIMEZONE
        dt_parsed = utils.parse_datetime_flexible(
            date_str, tz_aware=True, default_localize_tz_str=effective_target_tz_str
        )
        if not dt_parsed:
            logger.warning(f"Could not parse date_str for schedule: {date_str}")
            return []

        if isinstance(dt_parsed, datetime):
            day_date_part = dt_parsed.date()
        elif isinstance(dt_parsed, date):
            day_date_part = dt_parsed
        else:
            return []

        # CHANGE 2: Use explicit datetime objects for the day's range
        try:
            target_pytz_obj_for_day_range = pytz.timezone(effective_target_tz_str)
        except pytz.UnknownTimeZoneError:
            logger.warning(
                f"Timezone '{effective_target_tz_str}' for day range construction unknown. Using UTC."
            )
            target_pytz_obj_for_day_range = pytz.utc
            # Update effective_target_tz_str if it fell back, for consistency in list_events_in_range call
            effective_target_tz_str = "UTC"

        start_of_day_naive = datetime.combine(day_date_part, datetime.min.time())
        start_of_day_aware = target_pytz_obj_for_day_range.localize(start_of_day_naive)

        # End of day is exclusive start of next day
        end_of_day_naive = datetime.combine(
            day_date_part + timedelta(days=1), datetime.min.time()
        )
        end_of_day_aware = target_pytz_obj_for_day_range.localize(end_of_day_naive)

        return self.list_events_in_range(
            utils.format_datetime_iso(start_of_day_aware),
            utils.format_datetime_iso(end_of_day_aware),
            effective_target_tz_str,  # Pass the (potentially updated) effective string
        )

    def get_todays_schedule(
        self, target_timezone_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        today_local_date = utils.get_current_datetime_local().date()
        return self.get_schedule_for_date(
            utils.format_datetime_iso(today_local_date), target_timezone_str
        )

    def get_tomorrows_schedule(
        self, target_timezone_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        tomorrow_local_date = (
            utils.get_current_datetime_local() + timedelta(days=1)
        ).date()
        return self.get_schedule_for_date(
            utils.format_datetime_iso(tomorrow_local_date), target_timezone_str
        )


if __name__ == "__main__":
    # Test suite remains largely the same, but should now pass with the fixes.
    # The logging configuration is moved to the top of this class for when it's run standalone.

    test_cal_file = (
        config.DATA_DIR / "test_calendar_manager_ultimate.ics"
    )  # Changed name
    if test_cal_file.exists():
        test_cal_file.unlink()

    manager = CalendarManager(calendar_file_path=test_cal_file)
    print(f"\n--- CalendarManager Test Initialized ---")
    print(f"Using test calendar file: {manager.calendar_file}")

    # --- Test Create Event ---
    print("\n--- Testing Create Event ---")
    now_local = utils.get_current_datetime_local()

    event1_start_str = utils.format_datetime_for_llm(now_local + timedelta(hours=1))
    event1_end_str = utils.format_datetime_for_llm(now_local + timedelta(hours=2))
    ev1 = manager.create_event(
        "Meeting with Team",
        event1_start_str,
        end_dt_str=event1_end_str,
        description="Discuss project X",
        location="Room 101",
    )
    assert (
        ev1 and ev1["summary"] == "Meeting with Team"
    ), f"ev1 creation failed or summary mismatch. ev1: {ev1}"
    print(
        f"Created Event 1 (UID: {ev1['uid']}): {ev1['summary']} from {utils.format_datetime_for_llm(ev1['dtstart'])} to {utils.format_datetime_for_llm(ev1['dtend'])}"
    )

    # All-day event
    event2_start_date_str = (now_local + timedelta(days=2)).strftime("%Y-%m-%d")
    ev2 = manager.create_event(
        "Doctor's Appointment (All-day)",
        event2_start_date_str,
        duration_str="1d",
        location="Clinic",
    )
    assert ev2, f"ev2 (all-day) creation failed. ev2: {ev2}"
    assert isinstance(
        ev2["dtstart"], date
    ), f"ev2 dtstart should be date, but got {type(ev2['dtstart'])}"
    print(
        f"Created All-day Event 2 (UID: {ev2['uid']}): {ev2['summary']} on {utils.format_datetime_for_llm(ev2['dtstart'])}"
    )

    # Recurring event
    today_weekday = now_local.weekday()
    days_until_monday = (0 - today_weekday + 7) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # Schedule for next Monday if today is Monday

    next_monday_start_dt = (now_local + timedelta(days=days_until_monday)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    rrule_weekly_mondays = {"FREQ": "WEEKLY", "BYDAY": "MO", "COUNT": 3}
    ev3_start_str = utils.format_datetime_for_llm(next_monday_start_dt)
    ev3 = manager.create_event(
        "Weekly Sync",
        ev3_start_str,
        duration_str="PT1H",  # ISO 8601 duration
        rrule_dict=rrule_weekly_mondays,
        location="Online",
    )
    assert (
        ev3 and ev3["rrule"] is not None
    ), f"ev3 (recurring) creation failed. ev3: {ev3}"
    print(
        f"Created Recurring Event 3 (UID: {ev3['uid']}): {ev3['summary']}, starts {utils.format_datetime_for_llm(ev3['dtstart'])}, RRULE: {ev3['rrule']}"
    )

    # Test (10): Create event with explicit timezone_str for naive input
    print("\n--- Testing Create Event with explicit timezone_str (Change 10) ---")
    naive_start_str = (now_local + timedelta(hours=5)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # Naive string
    naive_end_str = (now_local + timedelta(hours=6)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # Naive string
    test_tz = (
        "America/New_York"  # Try a different timezone to see if it handles correctly
    )

    ev_tz_test = manager.create_event(
        "Timezone Test Event",
        naive_start_str,
        end_dt_str=naive_end_str,
        timezone_str=test_tz,  # This is the key for Change (10)
    )
    assert ev_tz_test, "Timezone test event creation failed."
    assert (
        ev_tz_test["dtstart"].tzinfo is not None
    ), "Timezone test event start time is naive."
    assert (
        str(ev_tz_test["dtstart"].tzinfo) == test_tz
    ), f"Timezone mismatch. Expected {test_tz}, got {ev_tz_test['dtstart'].tzinfo}"
    print(
        f"Created Event with timezone '{test_tz}' (UID: {ev_tz_test['uid']}): {ev_tz_test['summary']} at {utils.format_datetime_for_llm(ev_tz_test['dtstart'])}"
    )

    # --- Test Get Event Details ---
    print("\n--- Testing Get Event Details ---")
    details_ev1 = manager.get_event_details(ev1["uid"])
    assert (
        details_ev1 and details_ev1["summary"] == "Meeting with Team"
    ), f"Failed to get details for ev1. Got: {details_ev1}"
    print(f"Details for Event 1 (UID: {ev1['uid']}): {details_ev1['summary']}")

    # --- Test Update Event (Change 9: Input Validation) ---
    print("\n--- Testing Update Event ---")
    updated_ev1 = manager.update_event(
        ev1["uid"],
        {"summary": "Extended Meeting with Team", "location": "Main Conference Hall"},
    )
    assert (
        updated_ev1 and updated_ev1["summary"] == "Extended Meeting with Team"
    ), f"Valid update for ev1 failed. Got: {updated_ev1}"
    assert updated_ev1["location"] == "Main Conference Hall"
    print(
        f"Updated Event 1 summary to: '{updated_ev1['summary']}' and location to: '{updated_ev1['location']}'"
    )

    # Test invalid update: end before start
    invalid_end_dt_val = updated_ev1["dtstart"] - timedelta(
        hours=1
    )  # Ensure this is a datetime object
    invalid_end_str = utils.format_datetime_for_llm(invalid_end_dt_val)  # Format it

    invalid_update_result = manager.update_event(
        ev1["uid"], {"end_dt_str": invalid_end_str}
    )
    assert (
        invalid_update_result is None
    ), "Update with end before start should have failed."
    print(f"Correctly failed to update ev1 with end_dt_str before start_dt_str.")

    # Verify original event details are unchanged by failed update
    ev1_after_failed_update = manager.get_event_details(ev1["uid"])
    assert (
        ev1_after_failed_update is not None
    ), "Event ev1 should still exist after failed update."
    # Compare relevant fields that should not have changed
    assert (
        ev1_after_failed_update["dtend"] == updated_ev1["dtend"]
    ), f"Event dtend changed after a failed update. Expected {updated_ev1['dtend']}, got {ev1_after_failed_update['dtend']}"
    assert (
        ev1_after_failed_update["summary"] == updated_ev1["summary"]
    ), f"Event summary changed after a failed update. Expected {updated_ev1['summary']}, got {ev1_after_failed_update['summary']}"

    new_rrule_ev3 = {
        "FREQ": "WEEKLY",
        "BYDAY": "MO",
        "COUNT": 5,
    }  # This is the crucial update
    updated_ev3 = manager.update_event(ev3["uid"], {"rrule_dict": new_rrule_ev3})
    assert updated_ev3, f"Update of ev3 RRULE failed. updated_ev3 is None."
    assert (
        "COUNT=5" in updated_ev3["rrule"].upper()
    ), f"Failed to update rrule for ev3. Got: {updated_ev3['rrule']}, expected COUNT=5"
    print(f"Updated Event 3 RRULE to: {updated_ev3['rrule']}")

    # --- Test List Events in Range (Change 4: icalevents tzinfo) ---
    print("\n--- Testing List Events in Range ---")
    # Extend range to ensure all 5 occurrences of the updated recurring event are captured
    range_start_dt_obj = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    range_end_dt_obj = range_start_dt_obj + timedelta(
        days=35
    )  # 5 weeks to catch 5 weekly events
    range_start_str = utils.format_datetime_iso(range_start_dt_obj)
    range_end_str = utils.format_datetime_iso(range_end_dt_obj)

    list_target_tz = "America/Los_Angeles"  # A different TZ for testing conversion
    if list_target_tz == config.MIRZA_TIMEZONE:
        list_target_tz = "Europe/Paris"  # Ensure it's different

    print(
        f"Listing events from {range_start_str} to {range_end_str} in timezone {list_target_tz}"
    )
    events_in_range = manager.list_events_in_range(
        range_start_str, range_end_str, target_timezone_str=list_target_tz
    )
    print(f"Found {len(events_in_range)} event occurrences in range:")

    found_ev1_in_range = False
    found_ev2_in_range = False
    recurring_instances_found = 0
    for event_occurrence in events_in_range:
        if isinstance(event_occurrence["start"], datetime):  # Dates won't have tzinfo
            assert (
                str(event_occurrence["start"].tzinfo) == list_target_tz
            ), f"Event start time TZ mismatch. Expected {list_target_tz}, got {event_occurrence['start'].tzinfo} for UID {event_occurrence['uid']}"
        print(
            f"  - UID: {event_occurrence['uid']}, Summary: {event_occurrence['summary']}, Start: {utils.format_datetime_for_llm(event_occurrence['start'])}, All-day: {event_occurrence['all_day']}"
        )
        if event_occurrence["uid"] == ev1["uid"]:
            found_ev1_in_range = True
        if event_occurrence["uid"] == ev2["uid"]:
            found_ev2_in_range = True
        if (
            event_occurrence["uid"] == ev3["uid"]
        ):  # Use original ev3 UID to count instances
            recurring_instances_found += 1

    assert found_ev1_in_range, "Event 1 not found in range query."
    assert found_ev2_in_range, "Event 2 (all-day) not found in range query."
    assert (
        recurring_instances_found == 5
    ), f"Expected 5 occurrences for recurring event 3 (UID: {ev3['uid']}), found {recurring_instances_found}."
    print(
        "Recurring event instances correctly expanded by icalevents with target timezone."
    )

    # --- Test Get Today's Schedule ---
    print("\n--- Testing Get Today's Schedule ---")
    today_event_start_str = utils.format_datetime_for_llm(
        now_local + timedelta(minutes=30)
    )
    ev_today = manager.create_event(
        "Quick Task Today", today_event_start_str, duration_str="15m"
    )
    assert ev_today, "Failed to create today's event."

    todays_events = manager.get_todays_schedule()
    print(f"Today's schedule ({len(todays_events)} events):")
    is_ev_today_found = any(te["uid"] == ev_today["uid"] for te in todays_events)
    for te in todays_events:
        print(f"  - {te['summary']} @ {utils.format_datetime_for_llm(te['start'])}")
    assert (
        is_ev_today_found
    ), "Newly created today's event not found in today's schedule."

    # --- Test Delete Event ---
    print("\n--- Testing Delete Event ---")
    delete_success_ev1 = manager.delete_event(ev1["uid"])
    assert delete_success_ev1, "Failed to delete Event 1."
    assert (
        manager.get_event_details(ev1["uid"]) is None
    ), "Event 1 still found after deletion."
    print(f"Deleted Event 1 (UID: {ev1['uid']}) successfully.")

    events_after_delete = manager.list_events_in_range(range_start_str, range_end_str)
    uids_after_delete = [e["uid"] for e in events_after_delete]
    assert (
        ev1["uid"] not in uids_after_delete
    ), "Deleted Event 1 still appears in range query."
    print("Event 1 correctly removed from range query after deletion.")

    print(
        f"\n--- Final count of unique event UIDs in calendar: {len(list(manager.calendar.walk('VEVENT')))} ---"
    )

    if test_cal_file.exists():
        print(f"\nTest file is at: {test_cal_file}")
    print("\n--- CalendarManager Final Testing Complete ---")
