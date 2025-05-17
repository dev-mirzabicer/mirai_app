# mirai_app/core/calendar_manager.py

import logging
import uuid
from datetime import (
    datetime,
    date,
    timedelta,
)  # Removed timezone from here, use pytz for tz objects
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from icalendar import (
    Calendar as iCalCalendar,
    Event as iCalEvent,
    vRecur,
    # Timezone as iCalTimezone, # Not directly used for creation, add_missing_timezones handles it
    vText,  # For explicit string properties
    vDDDTypes,  # Parent for date/time/duration types
)
from icalevents.icalevents import events as fetch_events
from icalevents import icalparser  # Keep for type hints if needed
import pytz  # For robust timezone handling

from mirai_app import config
from mirai_app.core import utils

# Configure logging
logger = logging.getLogger(__name__)
# logging.basicConfig( # Removed to avoid reconfiguring if already set by main app
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )


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
            except Exception as e:
                logger.error(
                    f"Error parsing existing calendar file {self.calendar_file}: {type(e).__name__} - {e}. Creating a new calendar."
                )

        cal = iCalCalendar()
        cal.add("prodid", "-//MIRAI Personal Assistant//mirzabicer.dev//")
        cal.add("version", "2.0")
        self._save_calendar(cal)  # Attempt to save immediately
        logger.info(f"Created a new calendar at {self.calendar_file}")
        return cal

    def _save_calendar(self, calendar_obj: Optional[iCalCalendar] = None) -> bool:
        """Saves the provided calendar object (or self.calendar) to the .ics file."""
        cal_to_save = calendar_obj if calendar_obj is not None else self.calendar
        try:
            # Ensure all necessary VTIMEZONE components are present before saving
            # This is crucial if any events use timezones other than UTC.
            # It's better to do this here than in every create/update.
            # It will iterate through all events and add VTIMEZONEs for any TZIDs found.
            cal_to_save.add_missing_timezones(pytz_provider=pytz)

            with self.calendar_file.open("wb") as f:
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
        default_tz_str: str = config.MIRZA_TIMEZONE,
    ) -> Union[datetime, date]:
        """Ensures datetime is timezone-aware. Dates are returned as is."""
        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None or dt_input.tzinfo.utcoffset(dt_input) is None:
                local_tz = pytz.timezone(default_tz_str)
                return local_tz.localize(dt_input)
            return dt_input
        return dt_input  # It's a date object, return as is

    def _event_to_dict(
        self, event: iCalEvent
    ) -> Dict[str, Any]:  # Implementation of Change (1)
        """Converts an icalendar.Event object to a dictionary."""
        data = {}
        # Directly access known properties and convert them appropriately
        data["uid"] = str(
            event.get("uid", vText("")).to_ical().decode()
        )  # Ensure UID is string
        data["summary"] = str(event.get("summary", vText("")).to_ical().decode())
        data["description"] = str(
            event.get("description", vText("")).to_ical().decode()
        )
        data["location"] = str(event.get("location", vText("")).to_ical().decode())

        # Handle date/datetime/duration properties using .dt
        for prop_name_upper in [
            "DTSTART",
            "DTEND",
            "CREATED",
            "LAST-MODIFIED",
            "DTSTAMP",
        ]:
            prop_val = event.get(prop_name_upper)
            if prop_val and hasattr(prop_val, "dt"):
                data[prop_name_upper.lower().replace("-", "_")] = prop_val.dt
            else:
                data[prop_name_upper.lower().replace("-", "_")] = None

        duration_prop = event.get("DURATION")
        if duration_prop and hasattr(duration_prop, "dt"):
            data["duration"] = duration_prop.dt  # This will be a timedelta
        else:
            data["duration"] = None

        rrule_prop = event.get("rrule")
        if rrule_prop:
            if isinstance(rrule_prop, list):
                data["rrule"] = [r.to_ical().decode("utf-8") for r in rrule_prop]
            elif hasattr(rrule_prop, "to_ical"):  # Should be vRecur
                data["rrule"] = rrule_prop.to_ical().decode("utf-8")
            else:  # Fallback, should not happen for valid RRULE
                data["rrule"] = str(rrule_prop)
        else:
            data["rrule"] = None

        # Example for other properties if needed:
        # data["status"] = str(event.get("status", vText("")).to_ical().decode())
        # attendees_prop = event.get("attendee")
        # if attendees_prop:
        #     if isinstance(attendees_prop, list):
        #         data["attendees"] = [str(a.to_ical().decode()) for a in attendees_prop]
        #     else:
        #         data["attendees"] = [str(attendees_prop.to_ical().decode())]
        # else:
        #     data["attendees"] = []

        return data

    def _icalevents_event_to_dict(
        self,
        icalevents_event: icalparser.Event,  # Removed target_tz_str due to Change (4)
    ) -> Dict[str, Any]:
        """Converts an icalevents.icalparser.Event to our standard dictionary format."""
        # With Change (4), icalevents_event.start/end should already be in the target timezone
        # or be naive date objects if all-day.

        return {
            "uid": str(icalevents_event.uid),
            "summary": str(icalevents_event.summary),
            "start": icalevents_event.start,  # Already in target_tz or a date object
            "end": icalevents_event.end,  # Already in target_tz or a date object
            "duration": icalevents_event.duration,
            "description": str(icalevents_event.description or ""),
            "location": str(icalevents_event.location or ""),
            "all_day": icalevents_event.all_day,
        }

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
        timezone_str: Optional[
            str
        ] = None,  # Timezone for localizing naive input strings
    ) -> Optional[Dict[str, Any]]:

        if not summary or not start_dt_str:
            logger.error("Summary and start datetime are required to create an event.")
            return None
        if not end_dt_str and not duration_str:
            # For all-day events, if only start_dt_str (as a date) is given,
            # we can default to a 1-day duration.
            temp_start_obj = utils.parse_datetime_flexible(start_dt_str)
            if not (
                isinstance(temp_start_obj, date)
                and not isinstance(temp_start_obj, datetime)
            ):  # if not a pure date
                logger.error(
                    "Either end datetime or duration is required for timed events."
                )
                return None
            # If it's a pure date and no end/duration, assume 1-day all-day event
            duration_str = "1d"

        tz_for_inputs = timezone_str or config.MIRZA_TIMEZONE

        # Parse start and end, utils.parse_datetime_flexible now returns date or datetime
        start_dt_obj_parsed = utils.parse_datetime_flexible(start_dt_str)
        if not start_dt_obj_parsed:
            logger.error(f"Invalid start datetime string: {start_dt_str}")
            return None

        # Ensure timezone if it's a datetime object
        start_dt_obj = self._ensure_timezone_aware(start_dt_obj_parsed, tz_for_inputs)

        end_dt_obj = None
        if end_dt_str:
            end_dt_obj_parsed = utils.parse_datetime_flexible(end_dt_str)
            if not end_dt_obj_parsed:
                logger.error(f"Invalid end datetime string: {end_dt_str}")
                return None
            end_dt_obj = self._ensure_timezone_aware(end_dt_obj_parsed, tz_for_inputs)

        # Refined type checking for pure date vs datetime
        is_start_pure_date = isinstance(start_dt_obj, date) and not isinstance(
            start_dt_obj, datetime
        )
        is_end_pure_date = False
        if end_dt_obj:
            is_end_pure_date = isinstance(end_dt_obj, date) and not isinstance(
                end_dt_obj, datetime
            )

        if end_dt_obj:  # Only check if end_dt_obj is provided
            if is_start_pure_date and not is_end_pure_date:
                logger.error(
                    "Cannot mix all-day start (date) with timed end (datetime). Make both dates or both datetimes."
                )
                return None
            if not is_start_pure_date and is_end_pure_date:
                logger.error(
                    "Cannot mix timed start (datetime) with all-day end (date). Make both dates or both datetimes."
                )
                return None
            # Additional validation: end must be after start
            if end_dt_obj <= start_dt_obj:
                logger.error(
                    f"End datetime/date ({end_dt_obj}) must be after start datetime/date ({start_dt_obj})."
                )
                return None

        event = iCalEvent()
        event.add("uid", uid or self._generate_event_uid())
        event.add("summary", summary)
        event.add("dtstart", start_dt_obj)  # This will be date or datetime

        if end_dt_obj:
            event.add("dtend", end_dt_obj)
        elif duration_str:
            try:
                parsed_duration_val_unit = utils._parse_duration_string(duration_str)
                if not parsed_duration_val_unit:
                    raise ValueError(f"Could not parse duration string: {duration_str}")

                value, unit = parsed_duration_val_unit
                td = None
                if unit == "minutes":
                    td = timedelta(minutes=value)
                elif unit == "hours":
                    td = timedelta(hours=value)
                elif unit == "days":
                    td = timedelta(days=value)
                elif unit == "weeks":
                    td = timedelta(weeks=value)
                # Note: icalendar's DURATION doesn't directly support "months" via timedelta.
                # For "months", it's better to calculate DTEND if start is a date.
                # Or, if start is datetime, it's complex due to varying month lengths.
                # For simplicity, if duration is months and start is date, calculate DTEND.
                elif unit == "months" and is_start_pure_date:
                    # This requires dateutil.relativedelta for accurate month addition
                    from dateutil.relativedelta import relativedelta

                    calculated_end_dt = start_dt_obj + relativedelta(months=value)
                    event.add(
                        "dtend", calculated_end_dt
                    )  # Add as DTEND instead of DURATION
                    logger.info(
                        f"Calculated DTEND for month-based duration: {calculated_end_dt}"
                    )
                elif unit == "months":
                    logger.warning(
                        "Month-based duration for timed events is complex and not directly set as DURATION. Consider calculating DTEND."
                    )
                    return None  # Or handle differently
                else:
                    raise ValueError(
                        f"Unsupported duration unit for event creation: {unit}"
                    )

                if td:  # If timedelta was created (not month-based for date)
                    event.add("duration", td)

            except ValueError as e:
                logger.error(f"Invalid duration string: {duration_str}. Error: {e}")
                return None

        now_utc = utils.get_current_datetime_utc()
        event.add("dtstamp", now_utc)
        event.add("created", now_utc)  # Optional, but good practice
        event.add("last-modified", now_utc)

        if description:
            event.add("description", description)
        if location:
            event.add("location", location)

        if rrule_dict:
            if "UNTIL" in rrule_dict and isinstance(
                rrule_dict["UNTIL"], (datetime, date)
            ):
                rrule_dict["UNTIL"] = self._ensure_timezone_aware(
                    rrule_dict["UNTIL"], tz_for_inputs
                )
            event.add("rrule", vRecur(rrule_dict))

        self.calendar.add_component(event)
        if self._save_calendar():
            logger.info(f"Created event '{summary}' (UID: {event.get('uid')})")
            return self._event_to_dict(event)
        return None

    def get_event_details(self, uid: str) -> Optional[Dict[str, Any]]:
        """Retrieves the details of an event by its UID."""
        for component in self.calendar.walk("VEVENT"):
            if str(component.get("uid")) == uid:
                return self._event_to_dict(component)
        logger.debug(f"Event with UID '{uid}' not found for get_event_details.")
        return None

    def update_event(
        self, uid: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:  # Implementation of Change (9)
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

        # Preserve original start/end for validation if they are being changed
        original_start = (
            event_to_update.get("dtstart").dt
            if event_to_update.get("dtstart")
            else None
        )
        original_end = (
            event_to_update.get("dtend").dt if event_to_update.get("dtend") else None
        )
        # If DURATION is used, original_end might be None. Calculate it if needed for validation.
        if not original_end and original_start and event_to_update.get("duration"):
            original_duration = event_to_update.get("duration").dt
            if isinstance(original_start, (datetime, date)) and isinstance(
                original_duration, timedelta
            ):
                original_end = original_start + original_duration

        tz_for_inputs = updates.get("timezone_str", config.MIRZA_TIMEZONE)
        updated_fields_log = []

        new_start_dt = None
        new_end_dt = None

        if "summary" in updates:
            event_to_update["summary"] = vText(updates["summary"])
            updated_fields_log.append("summary")

        if "start_dt_str" in updates:
            parsed_start = utils.parse_datetime_flexible(updates["start_dt_str"])
            if parsed_start:
                new_start_dt = self._ensure_timezone_aware(parsed_start, tz_for_inputs)
                event_to_update["dtstart"] = new_start_dt
                updated_fields_log.append("dtstart")
            else:
                logger.warning(
                    f"Invalid start_dt_str in update for UID {uid}: {updates['start_dt_str']}"
                )
        else:
            new_start_dt = original_start  # Use original if not updated

        # Handle end/duration (clear one if other is set)
        if "end_dt_str" in updates:
            parsed_end = utils.parse_datetime_flexible(updates["end_dt_str"])
            if parsed_end:
                new_end_dt = self._ensure_timezone_aware(parsed_end, tz_for_inputs)
                event_to_update["dtend"] = new_end_dt
                if "duration" in event_to_update:
                    del event_to_update["duration"]
                updated_fields_log.append("dtend")
            else:
                logger.warning(
                    f"Invalid end_dt_str in update for UID {uid}: {updates['end_dt_str']}"
                )
        elif "duration_str" in updates:
            duration_val = updates["duration_str"]
            if duration_val is None and "duration" in event_to_update:  # Clear duration
                del event_to_update["duration"]
                # If clearing duration, DTEND might need to be set or event becomes invalid
                # This case needs careful handling or disallowing clearing duration without setting DTEND
                logger.warning(
                    "Cleared duration. Ensure DTEND is set or event might be invalid."
                )
                updated_fields_log.append("duration (cleared)")

            elif isinstance(duration_val, str):
                try:
                    parsed_duration_val_unit = utils._parse_duration_string(
                        duration_val
                    )
                    if not parsed_duration_val_unit:
                        raise ValueError(
                            f"Could not parse duration string: {duration_val}"
                        )

                    value, unit = parsed_duration_val_unit
                    td = None
                    # Similar logic as in create_event for duration
                    if unit == "minutes":
                        td = timedelta(minutes=value)
                    elif unit == "hours":
                        td = timedelta(hours=value)
                    elif unit == "days":
                        td = timedelta(days=value)
                    elif unit == "weeks":
                        td = timedelta(weeks=value)
                    elif unit == "months":  # Requires careful handling
                        is_current_start_pure_date = isinstance(
                            new_start_dt or original_start, date
                        ) and not isinstance(new_start_dt or original_start, datetime)
                        if is_current_start_pure_date:
                            from dateutil.relativedelta import relativedelta

                            calculated_end_dt = (
                                new_start_dt or original_start
                            ) + relativedelta(months=value)
                            event_to_update["dtend"] = calculated_end_dt
                            if "duration" in event_to_update:
                                del event_to_update["duration"]
                            new_end_dt = calculated_end_dt  # For validation
                        else:
                            logger.warning(
                                "Month-based duration for timed events is complex. Update DTEND instead."
                            )
                            # Potentially skip this update or return error
                    else:
                        raise ValueError(f"Unsupported duration unit: {unit}")

                    if td:
                        event_to_update["duration"] = td
                        if "dtend" in event_to_update:
                            del event_to_update["dtend"]
                        # Calculate new_end_dt for validation if start is known
                        if new_start_dt or original_start:
                            current_start_for_calc = new_start_dt or original_start
                            if isinstance(
                                current_start_for_calc, (datetime, date)
                            ):  # Ensure it's a date/datetime
                                new_end_dt = current_start_for_calc + td

                    updated_fields_log.append("duration")
                except ValueError as e:
                    logger.warning(
                        f"Invalid duration string for update: {duration_val}. Error: {e}"
                    )
        else:  # Neither end_dt_str nor duration_str in updates
            if "dtend" in event_to_update:  # Use existing DTEND if present
                new_end_dt = event_to_update.get("dtend").dt
            elif "duration" in event_to_update and (
                new_start_dt or original_start
            ):  # Use existing DURATION
                current_start_for_calc = new_start_dt or original_start
                if isinstance(current_start_for_calc, (datetime, date)):
                    new_end_dt = (
                        current_start_for_calc + event_to_update.get("duration").dt
                    )

        # Validation after potential start/end updates (Change 9)
        current_start_to_validate = (
            new_start_dt if "start_dt_str" in updates else original_start
        )
        current_end_to_validate = (
            new_end_dt  # This is calculated/set based on updates or original
        )

        if current_start_to_validate and current_end_to_validate:
            is_start_pure_date = isinstance(
                current_start_to_validate, date
            ) and not isinstance(current_start_to_validate, datetime)
            is_end_pure_date = isinstance(
                current_end_to_validate, date
            ) and not isinstance(current_end_to_validate, datetime)

            if is_start_pure_date and not is_end_pure_date:
                logger.error(
                    f"Update for UID {uid} creates invalid state: All-day start with timed end."
                )
                return None
            if not is_start_pure_date and is_end_pure_date:
                logger.error(
                    f"Update for UID {uid} creates invalid state: Timed start with all-day end."
                )
                return None
            if current_end_to_validate <= current_start_to_validate:
                logger.error(
                    f"Update for UID {uid} creates invalid state: End ({current_end_to_validate}) is not after start ({current_start_to_validate})."
                )
                return None

        if "description" in updates:
            event_to_update["description"] = vText(updates["description"])
            updated_fields_log.append("description")
        if "location" in updates:
            event_to_update["location"] = vText(updates["location"])
            updated_fields_log.append("location")

        if "rrule_dict" in updates:
            rrule_val = updates["rrule_dict"]
            if rrule_val is None and "rrule" in event_to_update:
                del event_to_update["rrule"]
            elif isinstance(rrule_val, dict):
                if "UNTIL" in rrule_val and isinstance(
                    rrule_val["UNTIL"], (datetime, date)
                ):
                    rrule_val["UNTIL"] = self._ensure_timezone_aware(
                        rrule_val["UNTIL"], tz_for_inputs
                    )
                event_to_update["rrule"] = vRecur(rrule_val)
            updated_fields_log.append("rrule")

        event_to_update["last-modified"] = utils.get_current_datetime_utc()
        # DTSTAMP should also be updated to reflect this modification time
        event_to_update["dtstamp"] = utils.get_current_datetime_utc()
        updated_fields_log.append("last-modified/dtstamp")

        self.calendar.subcomponents[component_index] = event_to_update

        if self._save_calendar():
            logger.info(
                f"Updated event '{uid}'. Changed: {', '.join(updated_fields_log) if updated_fields_log else 'No effective changes.'}"
            )
            return self._event_to_dict(event_to_update)
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
            else:
                logger.error(
                    f"Found and removed event '{uid}' from memory, but failed to save calendar."
                )
                # Potentially, should reload calendar from disk to revert in-memory change
                self.calendar = self._load_calendar()
                return False
        logger.warning(f"Event with UID '{uid}' not found for deletion.")
        return False

    def list_events_in_range(
        self,
        start_range_str: str,
        end_range_str: str,
        target_timezone_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:  # Implementation of Change (4)
        """
        Lists all event occurrences (including expanded recurring events) within a given date range.
        Uses 'icalevents' for robust parsing and recurrence handling.
        Input date strings are parsed. If naive, they are considered in MIRZA_TIMEZONE.
        Output event times are in target_timezone_str (defaults to MIRZA_TIMEZONE).
        """
        effective_target_tz_str = target_timezone_str or config.MIRZA_TIMEZONE
        try:
            target_pytz_obj = pytz.timezone(effective_target_tz_str)
        except pytz.UnknownTimeZoneError:
            logger.error(
                f"Unknown target timezone: {effective_target_tz_str}. Defaulting to UTC."
            )
            target_pytz_obj = pytz.utc

        # Parse input strings. utils.parse_datetime_flexible can return date or datetime.
        start_dt_parsed = utils.parse_datetime_flexible(start_range_str)
        end_dt_parsed = utils.parse_datetime_flexible(end_range_str)

        if not start_dt_parsed or not end_dt_parsed:
            logger.error("Invalid start or end date range for listing events.")
            return []

        # Ensure start/end are timezone-aware if they are datetimes for icalevents query
        # If they are date objects, icalevents handles them correctly for all-day ranges.
        query_start_dt = self._ensure_timezone_aware(
            start_dt_parsed, effective_target_tz_str
        )
        query_end_dt = self._ensure_timezone_aware(
            end_dt_parsed, effective_target_tz_str
        )

        if not self.calendar_file.exists() or self.calendar_file.stat().st_size == 0:
            logger.info("Calendar file is empty or does not exist. No events to list.")
            return []

        # Ensure calendar is saved before icalevents reads it.
        # This is important if there are unsaved changes in self.calendar
        self._save_calendar()

        try:
            # Pass target_pytz_obj to icalevents.events()
            # This tells icalevents to convert event times to this timezone.
            fetched_ical_events = fetch_events(
                file=str(self.calendar_file),
                start=query_start_dt,
                end=query_end_dt,
                tzinfo=target_pytz_obj,  # Change (4)
            )

            dict_events = [
                self._icalevents_event_to_dict(ev) for ev in fetched_ical_events
            ]
            dict_events.sort(key=lambda x: x["start"])
            return dict_events
        except Exception as e:
            logger.error(
                f"Error listing events with icalevents: {type(e).__name__} - {e}"
            )
            return []

    def get_schedule_for_date(
        self, date_str: str, target_timezone_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Gets all event occurrences for a specific date."""
        # utils.parse_datetime_flexible will return date if date_str is "YYYY-MM-DD"
        dt_parsed = utils.parse_datetime_flexible(date_str)
        if not dt_parsed:
            logger.warning(f"Could not parse date_str for schedule: {date_str}")
            return []

        # Determine start and end of the day for the query
        if isinstance(dt_parsed, datetime):
            # If a datetime was given, use its date part for the full day range
            day_date_part = dt_parsed.date()
            start_of_day_query = day_date_part
            # For icalevents, providing the same date for start and end covers the whole day
            end_of_day_query = day_date_part
        elif isinstance(dt_parsed, date):
            start_of_day_query = dt_parsed
            end_of_day_query = dt_parsed
        else:  # Should not happen
            return []

        return self.list_events_in_range(
            utils.format_datetime_iso(
                start_of_day_query
            ),  # format_datetime_iso handles date
            utils.format_datetime_iso(end_of_day_query),
            target_timezone_str,
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
    # Setup basic logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    test_cal_file = config.DATA_DIR / "test_calendar_manager_revised.ics"
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
        event2_start_date_str,  # Input is "YYYY-MM-DD"
        duration_str="1d",  # utils._parse_duration_string handles "1d"
        location="Clinic",
    )
    assert ev2, f"ev2 (all-day) creation failed. ev2: {ev2}"
    assert isinstance(
        ev2["dtstart"], date
    ), f"ev2 dtstart should be date, but got {type(ev2['dtstart'])}"
    print(
        f"Created All-day Event 2 (UID: {ev2['uid']}): {ev2['summary']} on {utils.format_datetime_for_llm(ev2['dtstart'])}"
    )

    # Recurring event: Weekly on Mondays for 3 occurrences
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
    # Use a different timezone than default MIRZA_TIMEZONE to verify
    test_tz = "America/New_York"
    # Ensure test_tz is different from config.MIRZA_TIMEZONE for a meaningful test
    if test_tz == config.MIRZA_TIMEZONE:
        test_tz = "Europe/London"  # Fallback if default is NY

    ev_tz_test = manager.create_event(
        "Timezone Test Event",
        naive_start_str,
        end_dt_str=naive_end_str,
        timezone_str=test_tz,
    )
    assert ev_tz_test, "Timezone test event creation failed."
    assert (
        ev_tz_test["dtstart"].tzinfo is not None
    ), "Timezone test event start time is naive."
    # Check if the timezone name matches (pytz objects might have different string representations)
    # A robust check is to convert to UTC and compare, or check offset.
    # For simplicity, let's check if the tzname from the object matches the input.
    # pytz tzinfo objects stringify to the IANA name.
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
    # Valid update
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

    # Test invalid update: end before start (Change 9)
    invalid_end_str = utils.format_datetime_for_llm(
        updated_ev1["dtstart"] - timedelta(hours=1)
    )
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
        ev1_after_failed_update["dtend"] == updated_ev1["dtend"]
    ), "Event changed after a failed update."

    # Update recurrence of ev3
    new_rrule_ev3 = {"FREQ": "WEEKLY", "BYDAY": "MO", "COUNT": 5}
    updated_ev3 = manager.update_event(ev3["uid"], {"rrule_dict": new_rrule_ev3})
    assert (
        updated_ev3 and "COUNT=5" in updated_ev3["rrule"].upper()
    ), f"Failed to update rrule for ev3. Got: {updated_ev3}"
    print(f"Updated Event 3 RRULE to: {updated_ev3['rrule']}")

    # --- Test List Events in Range (Change 4: icalevents tzinfo) ---
    print("\n--- Testing List Events in Range ---")
    range_start_dt = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    range_end_dt = range_start_dt + timedelta(days=21)  # 3 weeks
    range_start_str = utils.format_datetime_iso(range_start_dt)
    range_end_str = utils.format_datetime_iso(range_end_dt)

    # Test with a specific target timezone for list_events_in_range
    list_target_tz = "America/Los_Angeles"
    if list_target_tz == config.MIRZA_TIMEZONE:
        list_target_tz = "Europe/Paris"  # Ensure different for test

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
        # Verify timezone of returned events
        if isinstance(event_occurrence["start"], datetime):
            assert (
                str(event_occurrence["start"].tzinfo) == list_target_tz
            ), f"Event start time TZ mismatch. Expected {list_target_tz}, got {event_occurrence['start'].tzinfo}"
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
    # Expected: ev2, ev3, ev_tz_test, ev_today

    if test_cal_file.exists():
        print(f"\nTest file is at: {test_cal_file}")
        # test_cal_file.unlink() # Keep for inspection
    print("\n--- CalendarManager Revised Testing Complete ---")
