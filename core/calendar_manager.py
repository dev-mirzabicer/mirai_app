# mirai_app/core/calendar_manager.py

import logging
import uuid
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from icalendar import (
    Calendar as iCalCalendar,
    Event as iCalEvent,
    vRecur,
    Timezone as iCalTimezone,
)
from icalevents.icalevents import events as fetch_events
from icalevents import icalparser
import pytz  # For robust timezone handling

from mirai_app import config
from mirai_app.core import utils

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
                with self.calendar_file.open("rb") as f:  # Open in binary mode
                    cal = iCalCalendar.from_ical(f.read())
                logger.info(
                    f"Loaded {len(list(cal.walk('VEVENT')))} events from {self.calendar_file}"
                )
                return cal
            except Exception as e:  # Catch broad exceptions from icalendar parsing
                logger.error(
                    f"Error parsing existing calendar file {self.calendar_file}: {e}. Creating a new calendar."
                )

        # Create a new calendar if file doesn't exist, is empty, or parsing failed
        cal = iCalCalendar()
        cal.add("prodid", "-//MIRAI Personal Assistant//mirzabicer.dev//")
        cal.add("version", "2.0")
        # Attempt to save immediately to ensure the file is created
        self._save_calendar(cal)
        logger.info(f"Created a new calendar at {self.calendar_file}")
        return cal

    def _save_calendar(self, calendar_obj: Optional[iCalCalendar] = None) -> bool:
        """Saves the provided calendar object (or self.calendar) to the .ics file."""
        cal_to_save = calendar_obj if calendar_obj is not None else self.calendar
        try:
            with self.calendar_file.open("wb") as f:  # Open in binary write mode
                f.write(cal_to_save.to_ical())
            logger.debug(f"Successfully saved calendar to {self.calendar_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calendar to {self.calendar_file}: {e}")
            return False

    def _generate_event_uid(self) -> str:
        """Generates a unique ID for an event."""
        return str(uuid.uuid4())

    def _ensure_timezone_aware(
        self,
        dt_input: Union[datetime, date],
        default_tz_str: str = config.MIRZA_TIMEZONE,
    ) -> Union[datetime, date]:
        """Ensures datetime is timezone-aware. Dates are returned as is."""
        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None or dt_input.tzinfo.utcoffset(dt_input) is None:
                # Naive datetime, localize it
                local_tz = pytz.timezone(default_tz_str)
                return local_tz.localize(dt_input)
            # Already timezone-aware
            return dt_input
        # It's a date object, return as is (icalendar handles date objects for all-day events)
        return dt_input

    def _event_to_dict(self, event: iCalEvent) -> Dict[str, Any]:
        """Converts an icalendar.Event object to a dictionary."""
        data = {}
        for prop_name, prop_value in event.items():
            key = prop_name.lower()
            # For simple properties that are often vText or similar
            if hasattr(prop_value, "to_ical"):
                # For complex types, try to get a simpler representation
                if isinstance(
                    prop_value, (icalparser.TYPES_TIME, icalparser.TYPES_DATE)
                ):  # from icalevents
                    data[key] = prop_value
                elif hasattr(prop_value, "dt"):  # For vDDDTypes like DTSTART, DTEND
                    data[key] = prop_value.dt
                else:
                    try:  # Attempt to decode if it's a text-like property
                        data[key] = (
                            prop_value.to_ical().decode("utf-8")
                            if isinstance(prop_value.to_ical(), bytes)
                            else str(prop_value)
                        )
                    except:  # Fallback for other types
                        data[key] = str(prop_value)

            else:  # For simple Python types that might be directly set
                data[key] = prop_value

        # Ensure standard keys are present if possible
        data["uid"] = str(event.get("uid", ""))
        data["summary"] = str(event.get("summary", ""))
        data["dtstart"] = event.get("dtstart").dt if event.get("dtstart") else None
        data["dtend"] = event.get("dtend").dt if event.get("dtend") else None
        data["duration"] = event.get("duration").dt if event.get("duration") else None
        data["description"] = str(event.get("description", ""))
        data["location"] = str(event.get("location", ""))
        data["last-modified"] = (
            event.get("last-modified").dt if event.get("last-modified") else None
        )
        data["dtstamp"] = event.get("dtstamp").dt if event.get("dtstamp") else None

        # Handle recurrence rules
        rrule = event.get("rrule")
        if rrule:
            if isinstance(
                rrule, list
            ):  # RRULE can be a list if multiple are defined (though unusual)
                data["rrule"] = [r.to_ical().decode("utf-8") for r in rrule]
            else:
                data["rrule"] = rrule.to_ical().decode("utf-8")
        else:
            data["rrule"] = None

        return data

    def _icalevents_event_to_dict(
        self, icalevents_event: icalparser.Event, target_tz_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Converts an icalevents.icalparser.Event to our standard dictionary format."""
        target_tz = pytz.timezone(target_tz_str or config.MIRZA_TIMEZONE)

        start_dt = icalevents_event.start
        end_dt = icalevents_event.end

        if isinstance(start_dt, datetime):
            start_dt = (
                start_dt.astimezone(target_tz)
                if start_dt.tzinfo
                else target_tz.localize(start_dt)
            )
        if isinstance(end_dt, datetime):
            end_dt = (
                end_dt.astimezone(target_tz)
                if end_dt.tzinfo
                else target_tz.localize(end_dt)
            )

        return {
            "uid": str(icalevents_event.uid),
            "summary": str(icalevents_event.summary),
            "start": start_dt,
            "end": end_dt,
            "duration": icalevents_event.duration,
            "description": str(icalevents_event.description or ""),
            "location": str(icalevents_event.location or ""),
            "all_day": icalevents_event.all_day,
            # icalevents.Event doesn't directly expose rrule as string, it's used for expansion
        }

    def create_event(
        self,
        summary: str,
        start_dt_str: str,
        end_dt_str: Optional[str] = None,
        duration_str: Optional[str] = None,  # e.g., "1h30m", "2d"
        description: Optional[str] = None,
        location: Optional[str] = None,
        uid: Optional[str] = None,
        rrule_dict: Optional[
            Dict[str, Any]
        ] = None,  # e.g., {'FREQ': 'WEEKLY', 'BYDAY': 'MO', 'UNTIL': datetime}
        timezone_str: Optional[
            str
        ] = None,  # Timezone for parsing naive start/end_dt_str
    ) -> Optional[Dict[str, Any]]:

        if not summary or not start_dt_str:
            logger.error("Summary and start datetime are required to create an event.")
            return None
        if not end_dt_str and not duration_str:
            logger.error("Either end datetime or duration is required.")
            return None

        tz_for_inputs = timezone_str or config.MIRZA_TIMEZONE

        start_dt_obj = self._ensure_timezone_aware(
            utils.parse_datetime_flexible(start_dt_str), tz_for_inputs
        )
        if not start_dt_obj:
            logger.error(f"Invalid start datetime string: {start_dt_str}")
            return None

        event = iCalEvent()
        event.add("uid", uid or self._generate_event_uid())
        event.add("summary", summary)
        event.add("dtstart", start_dt_obj)

        if end_dt_str:
            end_dt_obj = self._ensure_timezone_aware(
                utils.parse_datetime_flexible(end_dt_str), tz_for_inputs
            )
            if not end_dt_obj:
                logger.error(f"Invalid end datetime string: {end_dt_str}")
                return None
            if isinstance(start_dt_obj, date) and not isinstance(
                end_dt_obj, date
            ):  # All-day start, timed end
                logger.error(
                    "Cannot mix all-day start with timed end. Make both dates or both datetimes."
                )
                return None
            if isinstance(start_dt_obj, datetime) and isinstance(
                end_dt_obj, date
            ):  # Timed start, all-day end
                logger.error(
                    "Cannot mix timed start with all-day end. Make both dates or both datetimes."
                )
                return None
            event.add("dtend", end_dt_obj)
        elif duration_str:
            # Parse duration_str like "PT1H30M", "P2D"
            try:
                # utils.calculate_expiry_date is for adding duration, we need to parse ISO 8601 duration
                # For simplicity, let's expect timedelta-like strings for now or handle basic ones
                # A more robust parser for ISO 8601 durations would be needed for full spec.
                # Example: "1h30m" -> timedelta(hours=1, minutes=30)
                # "2d" -> timedelta(days=2)
                parsed_duration = utils._parse_duration_string(
                    duration_str
                )  # Using the helper from utils
                if parsed_duration:
                    value, unit = parsed_duration
                    if unit == "minutes":
                        td = timedelta(minutes=value)
                    elif unit == "hours":
                        td = timedelta(hours=value)
                    elif unit == "days":
                        td = timedelta(days=value)
                    elif unit == "weeks":
                        td = timedelta(weeks=value)
                    else:
                        raise ValueError("Unsupported duration unit for event creation")
                    event.add("duration", td)
                else:
                    raise ValueError(f"Could not parse duration string: {duration_str}")
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
            # Ensure UNTIL is timezone aware if it's a datetime
            if "UNTIL" in rrule_dict and isinstance(rrule_dict["UNTIL"], datetime):
                rrule_dict["UNTIL"] = self._ensure_timezone_aware(
                    rrule_dict["UNTIL"], tz_for_inputs
                )
            event.add("rrule", vRecur(rrule_dict))

        self.calendar.add_component(event)
        # Automatically add VTIMEZONE components if needed based on tz-aware datetimes
        try:
            self.calendar.add_missing_timezones(pytz)  # Pass pytz as the provider
        except Exception as e:
            logger.warning(
                f"Could not automatically add VTIMEZONE components: {e}. Manual VTIMEZONE might be needed or ensure UTC datetimes."
            )

        if self._save_calendar():
            logger.info(f"Created event '{summary}' (UID: {event.get('uid')})")
            return self._event_to_dict(event)
        return None

    def get_event_details(self, uid: str) -> Optional[Dict[str, Any]]:
        """Retrieves the details of an event by its UID."""
        for component in self.calendar.walk("VEVENT"):
            if str(component.get("uid")) == uid:
                return self._event_to_dict(component)
        logger.warning(f"Event with UID '{uid}' not found.")
        return None

    def update_event(
        self, uid: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        event_to_update: Optional[iCalEvent] = None
        component_index = -1

        for i, component in enumerate(self.calendar.subcomponents):
            if isinstance(component, iCalEvent) and str(component.get("uid")) == uid:
                event_to_update = component
                component_index = i
                break

        if not event_to_update:
            logger.warning(f"Cannot update. Event with UID '{uid}' not found.")
            return None

        tz_for_inputs = updates.get("timezone_str", config.MIRZA_TIMEZONE)
        updated_fields_log = []

        if "summary" in updates:
            event_to_update["summary"] = updates["summary"]
            updated_fields_log.append("summary")
        if "start_dt_str" in updates:
            start_dt = self._ensure_timezone_aware(
                utils.parse_datetime_flexible(updates["start_dt_str"]), tz_for_inputs
            )
            if start_dt:
                event_to_update["dtstart"] = start_dt
            updated_fields_log.append("dtstart")

        # Handle end/duration (clear one if other is set)
        if "end_dt_str" in updates:
            end_dt = self._ensure_timezone_aware(
                utils.parse_datetime_flexible(updates["end_dt_str"]), tz_for_inputs
            )
            if end_dt:
                event_to_update["dtend"] = end_dt
                if "duration" in event_to_update:
                    del event_to_update["duration"]
            updated_fields_log.append("dtend")
        elif "duration_str" in updates:
            try:
                parsed_duration = utils._parse_duration_string(updates["duration_str"])
                if parsed_duration:
                    value, unit = parsed_duration
                    if unit == "minutes":
                        td = timedelta(minutes=value)
                    elif unit == "hours":
                        td = timedelta(hours=value)
                    elif unit == "days":
                        td = timedelta(days=value)
                    elif unit == "weeks":
                        td = timedelta(weeks=value)
                    else:
                        raise ValueError("Unsupported duration unit")
                    event_to_update["duration"] = td
                    if "dtend" in event_to_update:
                        del event_to_update["dtend"]
                else:
                    raise ValueError("Could not parse duration")
                updated_fields_log.append("duration")
            except ValueError as e:
                logger.warning(
                    f"Invalid duration string for update: {updates['duration_str']}. Error: {e}"
                )

        if "description" in updates:
            event_to_update["description"] = updates["description"]
            updated_fields_log.append("description")
        if "location" in updates:
            event_to_update["location"] = updates["location"]
            updated_fields_log.append("location")

        if "rrule_dict" in updates:
            rrule_val = updates["rrule_dict"]
            if rrule_val is None and "rrule" in event_to_update:  # Clear existing rrule
                del event_to_update["rrule"]
            elif isinstance(rrule_val, dict):
                if "UNTIL" in rrule_val and isinstance(rrule_val["UNTIL"], datetime):
                    rrule_val["UNTIL"] = self._ensure_timezone_aware(
                        rrule_val["UNTIL"], tz_for_inputs
                    )
                event_to_update["rrule"] = vRecur(rrule_val)
            updated_fields_log.append("rrule")

        event_to_update["last-modified"] = utils.get_current_datetime_utc()
        event_to_update["dtstamp"] = utils.get_current_datetime_utc()
        updated_fields_log.append("last-modified/dtstamp")

        # Replace the component in the calendar's subcomponents list
        self.calendar.subcomponents[component_index] = event_to_update

        try:
            self.calendar.add_missing_timezones(pytz)
        except Exception as e:
            logger.warning(
                f"Could not automatically add VTIMEZONE components during update: {e}."
            )

        if self._save_calendar():
            logger.info(
                f"Updated event '{uid}'. Changed: {', '.join(updated_fields_log)}"
            )
            return self._event_to_dict(event_to_update)
        return None

    def delete_event(self, uid: str) -> bool:
        """Deletes an event by its UID."""
        initial_len = len(self.calendar.subcomponents)
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
                return False  # Indicates save failure
        logger.warning(f"Event with UID '{uid}' not found for deletion.")
        return False

    def list_events_in_range(
        self,
        start_range_str: str,
        end_range_str: str,
        target_timezone_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lists all event occurrences (including expanded recurring events) within a given date range.
        Uses 'icalevents' for robust parsing and recurrence handling.
        Input date strings are parsed relative to MIRZA_TIMEZONE if naive.
        Output event times are in target_timezone_str (defaults to MIRZA_TIMEZONE).
        """
        query_tz_str = target_timezone_str or config.MIRZA_TIMEZONE
        query_tz = pytz.timezone(query_tz_str)

        start_dt = utils.parse_datetime_flexible(start_range_str)
        end_dt = utils.parse_datetime_flexible(end_range_str)

        if not start_dt or not end_dt:
            logger.error("Invalid start or end date range for listing events.")
            return []

        # Ensure start_dt and end_dt are timezone-aware for icalevents
        # If they are date objects, icalevents handles them for all-day ranges.
        # If datetime, make them aware.
        if isinstance(start_dt, datetime):
            start_dt = self._ensure_timezone_aware(start_dt, query_tz_str)
        if isinstance(end_dt, datetime):
            end_dt = self._ensure_timezone_aware(end_dt, query_tz_str)

        # Ensure the calendar file is saved before icalevents reads it
        if not self._save_calendar():
            logger.error(
                "Could not save calendar file before querying with icalevents. Results may be stale."
            )
            # Potentially return [] or raise error, but let's try to proceed.

        if not self.calendar_file.exists() or self.calendar_file.stat().st_size == 0:
            logger.info("Calendar file is empty or does not exist. No events to list.")
            return []

        try:
            # icalevents.events expects start and end.
            # It uses the tzinfo of start/end if present, or assumes system local if naive.
            # The 'tzinfo' param to events() is for the *output* event times.
            fetched_ical_events = fetch_events(
                file=str(self.calendar_file),
                start=start_dt,
                end=end_dt,
                # tzinfo=query_tz # This sets the timezone of the returned event's start/end
            )

            dict_events = [
                self._icalevents_event_to_dict(ev, query_tz_str)
                for ev in fetched_ical_events
            ]
            # Sort by start time
            dict_events.sort(key=lambda x: x["start"])
            return dict_events
        except Exception as e:
            logger.error(f"Error listing events with icalevents: {e}")
            return []

    def get_schedule_for_date(
        self, date_str: str, target_timezone_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Gets all event occurrences for a specific date."""
        dt = utils.parse_datetime_flexible(date_str)
        if not dt:
            return []

        # Ensure we use date objects if the input was just a date string
        if isinstance(dt, datetime):  # If it parsed as datetime, use it
            start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:  # It's a date object
            start_of_day = dt
            end_of_day = (
                dt  # icalevents handles single date for start/end as the whole day
            )

        return self.list_events_in_range(
            utils.format_datetime_iso(start_of_day),
            utils.format_datetime_iso(end_of_day),
            target_timezone_str,
        )

    def get_todays_schedule(
        self, target_timezone_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        today_local_dt = utils.get_current_datetime_local()
        return self.get_schedule_for_date(
            utils.format_datetime_iso(today_local_dt.date()), target_timezone_str
        )

    def get_tomorrows_schedule(
        self, target_timezone_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        tomorrow_local_dt = utils.get_current_datetime_local() + timedelta(days=1)
        return self.get_schedule_for_date(
            utils.format_datetime_iso(tomorrow_local_dt.date()), target_timezone_str
        )


if __name__ == "__main__":
    test_cal_file = config.DATA_DIR / "test_calendar_manager.ics"
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
    assert ev1 and ev1["summary"] == "Meeting with Team"
    print(
        f"Created Event 1 (UID: {ev1['uid']}): {ev1['summary']} from {ev1['dtstart']} to {ev1['dtend']}"
    )

    # All-day event
    event2_start_date_str = (now_local + timedelta(days=2)).strftime("%Y-%m-%d")
    ev2 = manager.create_event(
        "Doctor's Appointment (All-day)",
        event2_start_date_str,
        duration_str="1d",
        location="Clinic",
    )  # Duration "1d" for all-day
    assert ev2 and isinstance(ev2["dtstart"], date)
    print(
        f"Created All-day Event 2 (UID: {ev2['uid']}): {ev2['summary']} on {ev2['dtstart']}"
    )

    # Recurring event: Weekly on Mondays for 3 occurrences
    # Find next Monday
    today_weekday = now_local.weekday()  # Monday is 0, Sunday is 6
    days_until_monday = (0 - today_weekday + 7) % 7
    days_until_monday = (
        7 if days_until_monday == 0 and now_local.hour > 10 else days_until_monday
    )  # if today is Mon past 10am, take next Mon
    days_until_monday = (
        1
        if days_until_monday == 0 and (now_local + timedelta(days=7)).weekday() != 0
        else days_until_monday
    )  # ensure it's not today if it's already monday
    if days_until_monday == 0:
        days_until_monday = 7  # if it's monday today, schedule for next monday

    next_monday_start_dt = now_local.replace(
        hour=10, minute=0, second=0, microsecond=0
    ) + timedelta(days=days_until_monday)

    rrule_weekly_mondays = {"FREQ": "WEEKLY", "BYDAY": "MO", "COUNT": 3}
    ev3_start_str = utils.format_datetime_for_llm(next_monday_start_dt)
    ev3 = manager.create_event(
        "Weekly Sync",
        ev3_start_str,
        duration_str="1h",
        rrule_dict=rrule_weekly_mondays,
        location="Online",
    )
    assert ev3 and ev3["rrule"] is not None
    print(
        f"Created Recurring Event 3 (UID: {ev3['uid']}): {ev3['summary']}, starts {ev3['dtstart']}, RRULE: {ev3['rrule']}"
    )

    # --- Test Get Event Details ---
    print("\n--- Testing Get Event Details ---")
    details_ev1 = manager.get_event_details(ev1["uid"])
    assert details_ev1 and details_ev1["summary"] == "Meeting with Team"
    print(f"Details for Event 1 (UID: {ev1['uid']}): {details_ev1['summary']}")

    # --- Test Update Event ---
    print("\n--- Testing Update Event ---")
    updated_ev1 = manager.update_event(
        ev1["uid"],
        {"summary": "Extended Meeting with Team", "location": "Main Conference Hall"},
    )
    assert updated_ev1 and updated_ev1["summary"] == "Extended Meeting with Team"
    assert updated_ev1["location"] == "Main Conference Hall"
    print(
        f"Updated Event 1 summary to: '{updated_ev1['summary']}' and location to: '{updated_ev1['location']}'"
    )

    # Update recurrence of ev3
    new_rrule_ev3 = {"FREQ": "WEEKLY", "BYDAY": "MO", "COUNT": 5}  # Change count to 5
    updated_ev3 = manager.update_event(ev3["uid"], {"rrule_dict": new_rrule_ev3})
    assert updated_ev3 and "COUNT=5" in updated_ev3["rrule"].upper()
    print(f"Updated Event 3 RRULE to: {updated_ev3['rrule']}")

    # --- Test List Events in Range (using icalevents) ---
    print("\n--- Testing List Events in Range ---")
    # Range covering the next 3 weeks to catch recurring events
    range_start_dt = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    range_end_dt = range_start_dt + timedelta(days=21)
    range_start_str = utils.format_datetime_iso(range_start_dt)
    range_end_str = utils.format_datetime_iso(range_end_dt)

    print(f"Listing events from {range_start_str} to {range_end_str}")
    events_in_range = manager.list_events_in_range(range_start_str, range_end_str)
    print(f"Found {len(events_in_range)} event occurrences in range:")
    found_ev1_in_range = False
    found_ev2_in_range = False
    recurring_instances_found = 0
    for event_occurrence in events_in_range:
        print(
            f"  - UID: {event_occurrence['uid']}, Summary: {event_occurrence['summary']}, Start: {utils.format_datetime_for_llm(event_occurrence['start'])}, All-day: {event_occurrence['all_day']}"
        )
        if event_occurrence["uid"] == ev1["uid"]:
            found_ev1_in_range = True
        if event_occurrence["uid"] == ev2["uid"]:
            found_ev2_in_range = True
        if event_occurrence["uid"] == ev3["uid"]:
            recurring_instances_found += 1

    assert found_ev1_in_range, "Event 1 not found in range query."
    assert found_ev2_in_range, "Event 2 (all-day) not found in range query."
    assert (
        recurring_instances_found == 5
    ), f"Expected 5 occurrences for recurring event 3, found {recurring_instances_found}."
    print("Recurring event instances correctly expanded by icalevents.")

    # --- Test Get Today's Schedule ---
    print("\n--- Testing Get Today's Schedule ---")
    # Create an event for today to test this specifically
    today_event_start = utils.format_datetime_for_llm(now_local + timedelta(minutes=30))
    ev_today = manager.create_event(
        "Quick Task Today", today_event_start, duration_str="15m"
    )

    todays_events = manager.get_todays_schedule()
    print(f"Today's schedule ({len(todays_events)} events):")
    is_ev_today_found = False
    for te in todays_events:
        print(f"  - {te['summary']} @ {utils.format_datetime_for_llm(te['start'])}")
        if te["uid"] == ev_today["uid"]:
            is_ev_today_found = True
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

    # Verify it's gone from range query
    events_after_delete = manager.list_events_in_range(range_start_str, range_end_str)
    uids_after_delete = [e["uid"] for e in events_after_delete]
    assert (
        ev1["uid"] not in uids_after_delete
    ), "Deleted Event 1 still appears in range query."
    print("Event 1 correctly removed from range query after deletion.")

    print(
        f"\n--- Final count of unique event UIDs in calendar: {len(list(manager.calendar.walk('VEVENT')))} ---"
    )
    # Expected: ev2, ev3, ev_today

    # --- Clean up test file ---
    if test_cal_file.exists():
        print(f"\nCleaning up test file: {test_cal_file}")
        # test_cal_file.unlink() # Uncomment to auto-delete
    else:
        print(f"\nTest file {test_cal_file} not found for cleanup.")

    print("\n--- CalendarManager Testing Complete ---")
