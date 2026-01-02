"""Email notifications for signals."""

from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from typing import Optional

from src.crm.config import CRMConfig


def send_email(cfg: CRMConfig, subject: str, body: str) -> tuple[bool, Optional[str]]:
    """Send a plain-text email. Returns (ok, error_message)."""

    if not cfg.allow_email:
        return False, "email not configured"

    sender = cfg.email_from or cfg.email_user
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = cfg.email_to

    try:
        with smtplib.SMTP(cfg.email_smtp_host, cfg.email_smtp_port, timeout=10) as server:
            server.starttls()
            server.login(cfg.email_user, cfg.email_password)
            server.sendmail(sender, [cfg.email_to], msg.as_string())
        return True, None
    except Exception as exc:  # pragma: no cover
        return False, str(exc)

