import os
import requests
import threading
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        
        if not self.token or not self.chat_id:
            print("[WARNING] Telegram credentials missing in .env")
            self.enabled = False
        else:
            self.enabled = True
            print("[INFO] Telegram Notifier Enabled")

    def _send_photo_thread(self, frame, caption):
        """Internal method to run in a separate thread"""
        try:
            # Encode image to memory buffer
            _, buffer = cv2.imencode(".jpg", frame)
            files = {'photo': ('alert.jpg', buffer.tobytes(), 'image/jpeg')}
            data = {'chat_id': self.chat_id, 'caption': caption}
            
            # Use standard requests (synchronous is fine inside a thread)
            requests.post(f"{self.base_url}/sendPhoto", data=data, files=files)
            print(f"[TELEGRAM] Alert sent: {caption.splitlines()[0]}")
        except Exception as e:
            print(f"[TELEGRAM ERROR] Failed to send photo: {e}")

    def send_alert(self, frame, severity, message):
        """
        Sends an instant alert with an image.
        """
        if not self.enabled or frame is None:
            return

        caption = f"🚨 SentryAI Alert: {severity}\n{message}"
        
        # Run in thread to prevent blocking the main video loop
        t = threading.Thread(target=self._send_photo_thread, args=(frame, caption))
        t.daemon = True
        t.start()

    def send_report(self, pdf_path, summary_text=None):
        """
        Sends the final PDF report.
        """
        if not self.enabled:
            return

        def _send_task():
            try:
                # 1. Send Summary Text
                if summary_text:
                    # Telegram limit is ~4096 chars, truncate if needed
                    safe_text = summary_text[:4000] + ("..." if len(summary_text) > 4000 else "")
                    requests.post(f"{self.base_url}/sendMessage", data={
                        'chat_id': self.chat_id,
                        'text': f"📝 **Session Summary**:\n{safe_text}",
                        'parse_mode': 'Markdown'
                    })

                # 2. Send PDF File
                with open(pdf_path, "rb") as f:
                    files = {'document': (os.path.basename(pdf_path), f, 'application/pdf')}
                    data = {'chat_id': self.chat_id, 'caption': "📊 Sentry Final Report"}
                    requests.post(f"{self.base_url}/sendDocument", data=data, files=files)
                    print(f"[TELEGRAM] Report sent: {os.path.basename(pdf_path)}")
            
            except Exception as e:
                print(f"[TELEGRAM ERROR] Report failed: {e}")

        # Thread the report sending too, just in case file upload is slow
        t = threading.Thread(target=_send_task)
        t.daemon = True
        t.start()

# Global Instance
telegram_bot = TelegramNotifier()