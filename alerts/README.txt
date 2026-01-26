SENTRY ALERT MODULE (FCM)

FILES:
- firebase_key.json  -> Firebase service account key (YOU must add)
- fcm_client.py      -> Firebase push sender
- notifier.py        -> High-level alert trigger

STEPS:
1. Create Firebase project
2. Download service account key
3. Rename it to firebase_key.json
4. Place it inside alerts/
5. Paste your Android FCM token in notifier.py

DO NOT upload firebase_key.json to GitHub.