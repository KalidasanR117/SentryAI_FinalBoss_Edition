import firebase_admin
from firebase_admin import credentials, messaging
from alerts.config import FIREBASE_KEY_PATH

# Initialize Firebase ONCE
if not firebase_admin._apps:
    cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
    firebase_admin.initialize_app(cred)


def send_fcm_data_message(
    *,
    title: str,
    body: str,
    token: str,
    report_path: str | None = None,
    screenshots: list[str] | None = None,
    extra: dict | None = None
):
    data = {
        "title": title,
        "body": body,
    }

    if report_path:
        data["report"] = report_path

    if screenshots:
        data["screenshots"] = ",".join(screenshots)

    if extra:
        for k, v in extra.items():
            data[k] = str(v)

    message = messaging.Message(
        data=data,   # ✅ DATA-ONLY MESSAGE
        token=token
    )

    response = messaging.send(message)
    return response
