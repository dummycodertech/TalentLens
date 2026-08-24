"""
modules/calendar_scheduler.py — Google Calendar API + Meet link generation.

OAuth2 flow (headless-safe for Streamlit Cloud):
  1. Generate token locally via google-auth-oauthlib InstalledAppFlow → creds.to_json()
  2. Base64-encode → store as GOOGLE_OAUTH_CLIENT_JSON in Streamlit secrets
  3. On load: base64-decode → Credentials.from_authorized_user_info(json.loads(token_json))
  4. Before each call: creds.refresh(Request()) if creds.expired

C2: JSON serialization only — no pickle.
C10: Event title includes candidate name + s_no.
C9: schedule_all() iterates over actual DB rows, slots assigned sequentially.
"""

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone


# ─── Credential helpers (C2) ───────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def load_google_credentials(oauth_json_b64: str):
    """
    Decode base64 → JSON → Credentials object.
    Refreshes the token if expired before returning.
    C2: Uses from_authorized_user_info(), never pickle.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_json = base64.b64decode(oauth_json_b64).decode("utf-8")
    token_dict = json.loads(token_json)

    creds = Credentials.from_authorized_user_info(token_dict, scopes=SCOPES)

    # Refresh if expired (C2: token will expire, must handle refresh)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return creds


def build_calendar_service(creds):
    """Build the Google Calendar API service client."""
    from googleapiclient.discovery import build
    return build("calendar", "v3", credentials=creds)


# ─── Event creation ────────────────────────────────────────────────────────────

def create_interview_event(
    service,
    candidate: dict,
    slot_datetime: datetime,
    recruiter_email: str,
    duration_mins: int = 45,
) -> dict:
    """
    Create a Google Calendar event with a Meet link and candidate attendee.
    Returns the created event dict.

    C10: Event title includes candidate name + s_no for inbox disambiguation.
    """
    sno = candidate.get("s_no")
    name = candidate.get("name") or f"Candidate {sno}"
    candidate_email = candidate.get("email")

    # C10: title includes name + s_no
    title = f"Interview — {name} (s_no {sno})"

    end_datetime = slot_datetime + timedelta(minutes=duration_mins)

    # Format for Google Calendar API (RFC3339)
    def fmt(dt: datetime) -> str:
        return dt.isoformat()

    attendees = [{"email": recruiter_email}]
    if candidate_email and candidate_email != recruiter_email:
        attendees.append({"email": candidate_email, "displayName": name})

    event_body = {
        "summary": title,
        "description": (
            f"Interview for myNachiketa GTM Engineering Internship\n"
            f"Candidate: {name} (s_no {sno})\n"
            f"Duration: {duration_mins} minutes"
        ),
        "start": {"dateTime": fmt(slot_datetime), "timeZone": "Asia/Kolkata"},
        "end": {"dateTime": fmt(end_datetime), "timeZone": "Asia/Kolkata"},
        "attendees": attendees,
        "conferenceData": {
            "createRequest": {
                "requestId": str(uuid.uuid4()),
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 60},
                {"method": "popup", "minutes": 15},
            ],
        },
    }

    created = service.events().insert(
        calendarId="primary",
        body=event_body,
        conferenceDataVersion=1,
        sendUpdates="all",  # Google sends its own invite emails to attendees
    ).execute()

    return created


def extract_meet_link(event: dict) -> str | None:
    """Extract the Google Meet link from a created event."""
    conf = event.get("conferenceData", {})
    for ep in conf.get("entryPoints", []):
        if ep.get("entryPointType") == "video":
            return ep.get("uri")
    return event.get("hangoutLink")


# ─── Batch scheduling ──────────────────────────────────────────────────────────

def schedule_all(
    db,
    oauth_json_b64: str,
    recruiter_email: str,
    slot_start: datetime,
    spacing_mins: int = 45,
    status_callback=None,
) -> dict[int, dict]:
    """
    Schedule interviews for all shortlisted candidates.
    Assigns sequential time slots starting from slot_start.

    C9: Iterates over actual DB query rows.
         Slots assigned from slot_start + i*spacing — no s_no arithmetic.
    C10: Event titles include name + s_no (handled in create_interview_event).

    Args:
        db: the db module
        oauth_json_b64: base64-encoded token JSON from Streamlit secrets
        recruiter_email: recruiter's Google account email
        slot_start: first interview slot (datetime, timezone-aware)
        spacing_mins: minutes between consecutive interviews
        status_callback: optional callable(s_no, status_str)

    Returns: dict of s_no -> scheduling result
    """
    try:
        creds = load_google_credentials(oauth_json_b64)
        service = build_calendar_service(creds)
    except Exception as e:
        return {"error": f"Failed to initialize Google Calendar: {e}"}

    shortlisted = db.get_by_status("shortlisted")
    results = {}

    # Ensure slot_start is timezone-aware
    if slot_start.tzinfo is None:
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        slot_start = ist.localize(slot_start)

    for i, row in enumerate(shortlisted):
        sno = row["s_no"]
        candidate = dict(row)
        slot = slot_start + timedelta(minutes=i * spacing_mins)

        if status_callback:
            status_callback(sno, f"scheduling interview at {slot.strftime('%H:%M')}…")

        try:
            event = create_interview_event(
                service=service,
                candidate=candidate,
                slot_datetime=slot,
                recruiter_email=recruiter_email,
                duration_mins=spacing_mins,
            )
            meet_link = extract_meet_link(event)
            event_id = event.get("id")

            db.upsert_interview_event(
                sno,
                calendar_event_id=event_id,
                meet_link=meet_link,
                scheduled_time=slot.isoformat(),
                invite_sent=1,
            )
            db.update_status(sno, "interview_scheduled")

            results[sno] = {
                "success": True,
                "event_id": event_id,
                "meet_link": meet_link,
                "scheduled_time": slot.isoformat(),
                "error": None,
            }

            if status_callback:
                status_callback(sno, "interview_scheduled")

        except Exception as e:
            error_msg = str(e)
            db.update_status(sno, "scheduling_failed", error=error_msg)
            results[sno] = {"success": False, "error": error_msg}
            if status_callback:
                status_callback(sno, "scheduling_failed")

    return results


# ─── Token generation helper (for README / local setup) ───────────────────────

def generate_token_locally(client_secrets_path: str, output_path: str = "token.json") -> None:
    """
    Run this ONCE locally to generate the OAuth token.
    Then base64-encode token.json and store in Streamlit secrets.

    Usage:
        python -c "from modules.calendar_scheduler import generate_token_locally; generate_token_locally('client_secret.json')"
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, scopes=SCOPES)
    creds = flow.run_local_server(port=0)

    token_json = creds.to_json()
    with open(output_path, "w") as f:
        f.write(token_json)

    encoded = base64.b64encode(token_json.encode()).decode()
    print(f"\nToken saved to {output_path}")
    print(f"\nAdd this to Streamlit secrets as GOOGLE_OAUTH_CLIENT_JSON:\n")
    print(encoded)
