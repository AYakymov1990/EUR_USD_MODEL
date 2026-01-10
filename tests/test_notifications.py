import unittest
from unittest import mock

from src.crm.config import CRMConfig
from src.crm.notifications import send_email


class NotificationsTest(unittest.TestCase):
    def test_send_email_not_configured(self):
        cfg = CRMConfig()  # allow_email -> False
        with mock.patch("src.crm.notifications.smtplib.SMTP") as smtp_mock:
            ok, err = send_email(cfg, "subj", "body")
        self.assertFalse(ok)
        self.assertEqual(err, "email not configured")
        smtp_mock.assert_not_called()

    def test_send_email_success(self):
        cfg = CRMConfig(
            email_smtp_host="smtp.example.com",
            email_smtp_port=587,
            email_user="user@example.com",
            email_password="pass",
            email_to="dest@example.com",
            email_from="from@example.com",
        )

        smtp_mock = mock.Mock()
        smtp_manager = mock.Mock()
        smtp_manager.__enter__ = mock.Mock(return_value=smtp_mock)
        smtp_manager.__exit__ = mock.Mock(return_value=None)

        with mock.patch("src.crm.notifications.smtplib.SMTP", return_value=smtp_manager) as smtp_ctor:
            ok, err = send_email(cfg, "subject", "body")

        self.assertTrue(ok)
        self.assertIsNone(err)
        smtp_ctor.assert_called_once_with(cfg.email_smtp_host, cfg.email_smtp_port, timeout=10)
        smtp_mock.starttls.assert_called_once()
        smtp_mock.login.assert_called_once_with(cfg.email_user, cfg.email_password)
        smtp_mock.sendmail.assert_called_once()
