"""Paleta y hoja de estilo (tema claro)."""

BG = "#F3F5F9"
CARD = "#FFFFFF"
BORDER = "#E3E8F0"
TEXT = "#18212F"
MUTED = "#6B778C"
ACCENT = "#2563EB"
OK = "#15803D"
OK_BG = "#DCFCE7"
WARN = "#B45309"
WARN_BG = "#FEF3C7"
DANGER = "#B91C1C"
DANGER_BG = "#FEE2E2"
NEUTRAL_BG = "#E8ECF3"
ECG = "#0E7490"
GRID = "#E6EBF2"

# Clases de latido (AAMI): N normal, S supraventricular, V ventricular, F fusión
CLASS_COLORS = {"N": "#15803D", "S": "#B45309", "V": "#B91C1C", "F": "#7C3AED"}
CLASS_NAMES = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular", "F": "Fusión"}

STYLESHEET = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
QLabel#h1 {{ font-size: 20px; font-weight: 700; }}
QLabel#sub {{ color: {MUTED}; font-size: 12px; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#bpm {{ font-size: 54px; font-weight: 700; }}
QLabel#unit {{ color: {MUTED}; font-size: 14px; font-weight: 600; }}
QLabel#kv {{ font-size: 12px; }}
QLabel#kvKey {{ color: {MUTED}; font-size: 12px; }}
QLabel#note {{ color: {MUTED}; font-size: 11px; }}
QPushButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 9px; padding: 8px 14px; font-weight: 600; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #A3ADBD; background: #F7F8FB; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: none; }}
QPushButton#primary:hover {{ background: #1D4ED8; }}
QPushButton#primary:disabled {{ background: #9DB7F3; color: white; }}
QPushButton#danger {{ background: {DANGER}; color: white; border: none; }}
QComboBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 9px; padding: 7px 10px; }}
QListWidget {{ background: {CARD}; border: none; outline: 0; }}
QListWidget::item {{ border-bottom: 1px solid {BORDER}; padding: 7px 4px; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #CBD3E1; border-radius: 4px; }}
"""


def pill_style(fg: str, bg: str) -> str:
    return (f"background:{bg}; color:{fg}; border-radius:12px; padding:5px 12px;"
            " font-weight:700; font-size:12px;")
