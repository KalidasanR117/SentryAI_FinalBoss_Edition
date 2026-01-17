import os
from fpdf import FPDF
from datetime import datetime
from PIL import Image

# ============================================================
# Professional PDF Report Generator – Sentry AI
# ============================================================

class PDFReport(FPDF):

    COLOR_GOLD = (212, 175, 55)
    COLOR_CHARCOAL = (34, 34, 34)
    COLOR_COVER_BG = (24, 24, 24)
    COLOR_WHITE = (255, 255, 255)
    COLOR_LIGHT_GREY = (200, 200, 200)
    COLOR_BORDER = (90, 90, 90)

    # ---------------- Header / Footer ----------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 9)
        self.set_text_color(150)
        self.cell(0, 8, "Sentry AI Surveillance Report", 0, 0, "L")
        self.cell(0, 8, f"Page {self.page_no()}", 0, 1, "R")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(
            0, 10,
            f"Sentry AI | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            0, 0, "C"
        )

    # ---------------- Page Background ----------------
    def add_page(self, *args, **kwargs):
        super().add_page(*args, **kwargs)
        if self.page_no() > 1:
            self.set_fill_color(*self.COLOR_CHARCOAL)
            self.rect(0, 0, self.w, self.h, "F")

    # ---------------- Cover Page ----------------
    def add_cover_page(self, cover_path):
        super().add_page()
        self.set_fill_color(*self.COLOR_COVER_BG)
        self.rect(0, 0, self.w, self.h, "F")

        if cover_path and os.path.exists(cover_path):
            img = Image.open(cover_path)
            iw, ih = img.size
            pw, ph = self.w, self.h
            scale = min(pw / iw, ph / ih)
            w, h = iw * scale, ih * scale
            x = (pw - w) / 2
            y = (ph - h) / 2
            self.image(cover_path, x=x, y=y, w=w, h=h)
        else:
            self.set_text_color(*self.COLOR_WHITE)
            self.set_font("Helvetica", "B", 24)
            self.cell(0, 20, "Sentry AI Report", ln=True, align="C")

    # ---------------- Summary ----------------
    def add_summary_page(self, summary, events):
        self.add_page()
        self.set_text_color(*self.COLOR_WHITE)
        self.set_font("Helvetica", "B", 22)
        self.cell(0, 14, "Executive Summary", ln=True)

        self.set_draw_color(*self.COLOR_GOLD)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(10)

        self.set_font("Helvetica", "", 12)
        self.set_text_color(*self.COLOR_LIGHT_GREY)
        self.multi_cell(0, 8, summary)
        self.ln(8)

        danger = len([e for e in events if e.get("final") == "danger"])
        suspicious = len([e for e in events if e.get("final") == "suspicious"])

        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(0, 10, "Key Findings", ln=True)

        self.set_font("Helvetica", "", 11)
        self.set_text_color(*self.COLOR_LIGHT_GREY)
        self.cell(0, 8, f"- Total Events: {len(events)}", ln=True)
        self.cell(0, 8, f"- Danger Events: {danger}", ln=True)
        self.cell(0, 8, f"- Suspicious Events: {suspicious}", ln=True)

    # ---------------- Timeline ----------------
    def add_timeline_page(self, events):
        self.add_page()
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(0, 14, "Event Timeline", ln=True)
        self.ln(6)

        self.set_font("Helvetica", "", 12)
        self.set_text_color(*self.COLOR_LIGHT_GREY)

        for e in events:
            start = e.get("start_time", "?")
            etype = e.get("type", "Unknown")
            sev = e.get("final", "normal").upper()
            text = f"[{start}s] {etype} ({sev})"
            self.multi_cell(0, 8, text)
            self.ln(1)

    # ---------------- Event Page ----------------
    def add_event_details(self, event):
        self.add_page()

        # Title
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(
            0, 10,
            f"Event Details - Frame {event.get('frame', 'N/A')}",
            ln=True
        )
        self.ln(4)

        # ---- Table ----
        col1 = 50
        col2 = self.w - self.l_margin - self.r_margin - col1

        self.set_draw_color(*self.COLOR_BORDER)

        self.set_font("Helvetica", "B", 11)
        self.cell(col1, 8, "Detection Type", 1)
        self.set_font("Helvetica", "", 11)
        self.cell(col2, 8, "Details", 1, ln=True)

        self.set_font("Helvetica", "B", 11)
        self.cell(col1, 8, "Violence Detected", 1)
        self.set_font("Helvetica", "", 11)
        self.cell(col2, 8, event.get("type", "None"), 1, ln=True)

        self.set_font("Helvetica", "B", 11)
        self.cell(col1, 8, "Final Severity", 1)

        sev = event.get("final", "normal")
        if sev == "danger":
            self.set_text_color(220, 60, 60)
        elif sev == "suspicious":
            self.set_text_color(*self.COLOR_GOLD)
        else:
            self.set_text_color(160, 255, 160)

        self.set_font("Helvetica", "B", 11)
        self.cell(col2, 8, sev.upper(), 1, ln=True)
        self.set_text_color(*self.COLOR_LIGHT_GREY)
        self.ln(6)

        # ---- Visual Evidence ----
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(0, 10, "Visual Evidence", ln=True)

        img = event.get("screenshot")
        x = self.l_margin
        y = self.get_y()

        if img and os.path.exists(img):
            self.image(img, x=x, y=y, w=90)
            self.set_xy(x + 95, y)
            self.set_font("Helvetica", "I", 11)
            self.multi_cell(90, 10, "Skeletal Structure\n(Pose / 3D View)")
            self.ln(65)
        else:
            self.set_font("Helvetica", "I", 10)
            self.cell(0, 8, "Screenshot not available", ln=True)

        # ---- Explainability ----
        cause = event.get("cause")
        if cause:
            self.ln(5)
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "Why this event was detected", ln=True)

            self.set_font("Helvetica", "", 11)
            text = (
                f"Trigger Source : {cause.get('trigger')}\n"
                f"Rule Name      : {cause.get('rule_name')}\n"
                f"Description    : {cause.get('description')}\n"
                f"Joints         : {', '.join(cause.get('joints_involved', []))}\n"
                f"Metrics        : {cause.get('metrics')}"
            )
            self.multi_cell(0, 7, text)


# ============================================================
# PDF Generator Entry
# ============================================================
def generate_pdf_report(event_buffer, summary_text, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf = PDFReport()
    pdf.set_auto_page_break(True, margin=15)

    project_root = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    cover = os.path.join(project_root, "cover.png")

    pdf.add_cover_page(cover)
    pdf.add_summary_page(summary_text, event_buffer)
    # pdf.add_timeline_page(event_buffer)

    for e in event_buffer:
        pdf.add_event_details(e)

    pdf.output(output_path)
    print(f"[PDF] Report generated -> {output_path}")
