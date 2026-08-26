"""
modules/emailer.py — Gmail SMTP email sender for test links and interview invites.

C10: Subject includes candidate name + s_no for disambiguation in shared recruiter inbox.
C9: send_test_links() iterates over actual DB rows, never range().
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# ─── Email construction ────────────────────────────────────────────────────────

def build_test_email_html(candidate_name: str, s_no: int, test_url: str) -> str:
    """Build a clean HTML email body for the test link."""
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1a1a2e; background: #f8f9fa; margin: 0; padding: 0; }}
    .container {{ max-width: 560px; margin: 40px auto; background: #fff;
                  border-radius: 12px; overflow: hidden;
                  box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               padding: 32px 40px; color: white; }}
    .header h1 {{ margin: 0; font-size: 22px; font-weight: 700; }}
    .header p {{ margin: 6px 0 0; opacity: 0.85; font-size: 14px; }}
    .body {{ padding: 32px 40px; }}
    .body p {{ line-height: 1.7; color: #444; margin: 0 0 16px; }}
    .cta {{ display: block; background: linear-gradient(135deg, #667eea, #764ba2);
            color: white !important; text-decoration: none; text-align: center;
            padding: 14px 28px; border-radius: 8px; font-weight: 600;
            font-size: 16px; margin: 24px 0; }}
    .footer {{ padding: 20px 40px; border-top: 1px solid #eee;
               color: #999; font-size: 12px; }}
    .info-box {{ background: #f0f4ff; border-left: 4px solid #667eea;
                 border-radius: 6px; padding: 14px 18px; margin: 16px 0; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🎯 You've been shortlisted!</h1>
      <p>Technical Assessment Invitation</p>
    </div>
    <div class="body">
      <p>Hi <strong>{candidate_name}</strong>,</p>
      <p>
        Congratulations! Based on your application, resume, and GitHub profile,
        you have been shortlisted for the next stage of our evaluation process.
      </p>
      <p>Please complete the technical assessment at the link below. 
         This assesses your logical and coding skills.</p>
      <a href="{test_url}" class="cta">Start Assessment →</a>
      <div class="info-box">
        <strong>⏱ Important:</strong> Complete the test in one sitting.
        Your progress is saved automatically.
      </div>
      <p>If you have any issues accessing the test, please reply to this email.</p>
      <p>Best of luck!<br><strong>Recruiting Team</strong></p>
    </div>
    <div class="footer">
      Ref: Candidate #{s_no} | Technical Assessment
    </div>
  </div>
</body>
</html>
"""


def build_interview_email_html(
    candidate_name: str,
    s_no: int,
    meet_link: str,
    scheduled_time: str,
) -> str:
    """Build an HTML email with the Google Meet link for the interview."""
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1a1a2e; background: #f8f9fa; margin: 0; padding: 0; }}
    .container {{ max-width: 560px; margin: 40px auto; background: #fff;
                  border-radius: 12px; overflow: hidden;
                  box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
    .header {{ background: linear-gradient(135deg, #2dd4a8 0%, #1a9c7a 100%);
               padding: 32px 40px; color: white; }}
    .header h1 {{ margin: 0; font-size: 22px; font-weight: 700; }}
    .header p {{ margin: 6px 0 0; opacity: 0.9; font-size: 14px; }}
    .body {{ padding: 32px 40px; }}
    .body p {{ line-height: 1.7; color: #444; margin: 0 0 16px; }}
    .cta {{ display: block; background: linear-gradient(135deg, #2dd4a8, #1a9c7a);
            color: white !important; text-decoration: none; text-align: center;
            padding: 16px 28px; border-radius: 8px; font-weight: 700;
            font-size: 17px; margin: 24px 0; letter-spacing: 0.01em; }}
    .info-box {{ background: #f0fff9; border-left: 4px solid #2dd4a8;
                 border-radius: 6px; padding: 14px 18px; margin: 16px 0; }}
    .info-box strong {{ color: #1a9c7a; }}
    .footer {{ padding: 20px 40px; border-top: 1px solid #eee;
               color: #999; font-size: 12px; }}
    .time-block {{ font-size: 18px; font-weight: 700; color: #1a1a2e;
                   background: #f0fff9; border-radius: 8px; padding: 12px 20px;
                   margin: 16px 0; border: 1px solid #2dd4a8; display: inline-block; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>📅 Interview Scheduled!</h1>
      <p>Your interview has been confirmed</p>
    </div>
    <div class="body">
      <p>Hi <strong>{candidate_name}</strong>,</p>
      <p>
        Great news — your interview has been scheduled. Here are your details:
      </p>
      <div class="time-block">🕐 {scheduled_time}</div>
      <p>Click the button below to join your Google Meet interview at the scheduled time:</p>
      <a href="{meet_link}" class="cta">🎥 Join Google Meet Interview</a>
      <div class="info-box">
        <strong>⏱ Tips:</strong><br>
        • Join 2–3 minutes early to test your audio and video<br>
        • Ensure a stable internet connection<br>
        • Have your work/projects ready to discuss
      </div>
      <p>If you have any questions or need to reschedule, please reply to this email.</p>
      <p>Looking forward to speaking with you!<br><strong>Recruiting Team</strong></p>
    </div>
    <div class="footer">
      Ref: Candidate #{s_no} | Interview Confirmation
    </div>
  </div>
</body>
</html>
"""


def build_test_subject(candidate_name: str, s_no: int) -> str:
    """
    C10: Include name + s_no in subject so recruiter's shared inbox is navigable.
    """
    return f"Assessment Link — {candidate_name} (s_no {s_no})"


def build_interview_subject(candidate_name: str, s_no: int) -> str:
    """Subject for the interview meet-link confirmation email."""
    return f"Interview Scheduled — {candidate_name} (s_no {s_no})"


# ─── SMTP send ─────────────────────────────────────────────────────────────────

def send_email(
    to_address: str,
    subject: str,
    html_body: str,
    gmail_address: str,
    app_password: str,
) -> dict:
    """
    Send a single HTML email via Gmail SMTP + app password.
    Returns {"success": bool, "error": str|None}.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"AI Screener <{gmail_address}>"
    msg["To"] = to_address

    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(gmail_address, app_password)
            server.sendmail(gmail_address, to_address, msg.as_string())
        return {"success": True, "error": None}
    except smtplib.SMTPAuthenticationError:
        return {"success": False, "error": "SMTP auth failed — check Gmail address and app password"}
    except smtplib.SMTPRecipientsRefused:
        return {"success": False, "error": f"Recipient refused: {to_address}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Batch send ────────────────────────────────────────────────────────────────

def send_test_links(
    candidates: list,  # list of sqlite3.Row or dict-like
    test_url: str,
    db,
    gmail_address: str,
    app_password: str,
    status_callback=None,
) -> dict[int, dict]:
    """
    Send test-link emails to a list of candidates.
    C9: Iterates over actual candidate rows — no sequential s_no assumptions.
    C10: Subject includes name + s_no via build_test_subject().

    Returns: dict of s_no -> send result
    """
    results = {}

    for row in candidates:
        sno = row["s_no"]
        name = row["name"] or f"Candidate {sno}"
        email = row["email"]

        if status_callback:
            status_callback(sno, "sending email…")

        subject = build_test_subject(name, sno)
        html_body = build_test_email_html(name, sno, test_url)

        result = send_email(email, subject, html_body, gmail_address, app_password)
        results[sno] = result

        if result["success"]:
            db.update_status(sno, "test_sent")
        else:
            db.update_status(sno, "email_failed", error=result["error"])

        if status_callback:
            status_callback(sno, "test_sent" if result["success"] else "email_failed")

    return results


def send_interview_links(
    candidates: list,  # list of dicts with keys: s_no, name, email, meet_link, scheduled_time
    gmail_address: str,
    app_password: str,
    status_callback=None,
) -> dict[int, dict]:
    """
    Send interview meet-link emails to scheduled candidates.
    Each email contains a clickable Google Meet button.

    Returns: dict of s_no -> send result
    """
    results = {}

    for row in candidates:
        sno = row.get("s_no")
        name = row.get("name") or f"Candidate {sno}"
        email = row.get("email", "")
        meet_link = row.get("meet_link", "")
        scheduled_time = row.get("scheduled_time", "")

        if not email or not meet_link:
            results[sno] = {"success": False, "error": "Missing email or meet link"}
            continue

        # Format time nicely if it's an ISO string
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(scheduled_time)
            time_str = dt.strftime("%A, %d %b %Y at %H:%M IST")
        except Exception:
            time_str = scheduled_time

        if status_callback:
            status_callback(sno, "sending interview invite…")

        subject = build_interview_subject(name, sno)
        html_body = build_interview_email_html(name, sno, meet_link, time_str)

        result = send_email(email, subject, html_body, gmail_address, app_password)
        results[sno] = result

        if status_callback:
            status_callback(sno, "invite_sent" if result["success"] else "invite_failed")

    return results
