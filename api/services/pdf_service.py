import io
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── Color Palette ─────────────────────────────────────────────────
DARK_BG    = colors.HexColor("#0f172a")
CARD_BG    = colors.HexColor("#1e293b")
BLUE       = colors.HexColor("#38bdf8")
TEXT       = colors.HexColor("#e2e8f0")
MUTED      = colors.HexColor("#94a3b8")
RED        = colors.HexColor("#fca5a5")
YELLOW     = colors.HexColor("#fde68a")
GREEN      = colors.HexColor("#86efac")
LIGHT_BLUE = colors.HexColor("#93c5fd")
WHITE      = colors.white

RISK_COLORS = {
    "HIGH"  : colors.HexColor("#ef4444"),
    "MEDIUM": colors.HexColor("#f59e0b"),
    "LOW"   : colors.HexColor("#22c55e")
}

WARD_COLORS = {
    "ICU"    : colors.HexColor("#ef4444"),
    "MICU"   : colors.HexColor("#f59e0b"),
    "Private": colors.HexColor("#22c55e"),
    "General": colors.HexColor("#3b82f6")
}

def generate_patient_report(
    patient: dict,
    shap: dict,
    explanation: str,
    discharge_timeline: str
) -> bytes:
    """Generate a PDF report and return as bytes"""

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── Header ────────────────────────────────────────────────────
    header_style = ParagraphStyle(
        "Header",
        fontSize=22,
        textColor=BLUE,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=4
    )
    sub_style = ParagraphStyle(
        "Sub",
        fontSize=10,
        textColor=MUTED,
        fontName="Helvetica",
        alignment=TA_CENTER,
        spaceAfter=2
    )
    elements.append(Paragraph("🏥 Healthcare Risk Platform", header_style))
    elements.append(Paragraph("Patient Risk Assessment Report", sub_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        sub_style
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=BLUE, spaceAfter=12
    ))

    # ── Section Title Style ───────────────────────────────────────
    section_style = ParagraphStyle(
        "Section",
        fontSize=13,
        textColor=BLUE,
        fontName="Helvetica-Bold",
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        fontSize=10,
        textColor=colors.HexColor("#334155"),
        fontName="Helvetica",
        spaceAfter=4,
        leading=16
    )

    # ── Patient Summary Table ─────────────────────────────────────
    elements.append(Paragraph("Patient Summary", section_style))

    risk_color = RISK_COLORS.get(patient.get("risk_tier", "LOW"), GREEN)
    ward_color = WARD_COLORS.get(patient.get("predicted_ward", "General"), LIGHT_BLUE)

    summary_data = [
        ["Field", "Value"],
        ["Patient ID",    str(patient.get("subject_id", "N/A"))],
        ["Risk Score",    f"{patient.get('risk_score', 0):.1%}"],
        ["Risk Tier",     patient.get("risk_tier", "N/A")],
        ["Assigned Ward", patient.get("predicted_ward", "N/A")],
        ["Est. Stay",     f"{patient.get('estimated_los_days', 0):.1f} days"],
        ["Admissions",    str(int(patient.get("admission_count", 0)))],
        ["Emergency Rate",f"{patient.get('emergency_ratio', 0):.1%}"],
        ["Avg Stay",      f"{patient.get('avg_los_days', 0):.1f} days"],
    ]

    summary_table = Table(summary_data, colWidths=[6*cm, 10*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  11),
        ("BACKGROUND",  (0, 1), (-1, 1),  colors.HexColor("#f8fafc")),
        ("BACKGROUND",  (0, 3), (-1, 3),  risk_color),
        ("TEXTCOLOR",   (0, 3), (-1, 3),  WHITE),
        ("FONTNAME",    (0, 3), (-1, 3),  "Helvetica-Bold"),
        ("BACKGROUND",  (0, 4), (-1, 4),  ward_color),
        ("TEXTCOLOR",   (0, 4), (-1, 4),  WHITE),
        ("FONTNAME",    (0, 4), (-1, 4),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("FONTNAME",    (0, 1), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 1), (-1, -1), 10),
        ("PADDING",     (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 3), (-1, 3), [risk_color]),
        ("ROWBACKGROUNDS", (0, 4), (-1, 4), [ward_color]),
    ]))
    elements.append(summary_table)

    # ── SHAP Feature Contributions ────────────────────────────────
    elements.append(Paragraph("SHAP Feature Contributions", section_style))
    elements.append(Paragraph(
        "The following features contributed most to this patient's risk prediction:",
        body_style
    ))

    if shap:
        shap_sorted = sorted(
            shap.items(), key=lambda x: abs(x[1]), reverse=True
        )
        shap_data = [["Feature", "Impact", "Direction"]]
        for k, v in shap_sorted:
            feat_name = k.replace("shap_", "").replace("_", " ").title()
            direction = "↑ Increases Risk" if v > 0 else "↓ Decreases Risk"
            dir_color = colors.HexColor("#ef4444") if v > 0 else colors.HexColor("#22c55e")
            shap_data.append([feat_name, f"{abs(v):.4f}", direction])

        shap_table = Table(shap_data, colWidths=[8*cm, 4*cm, 6*cm])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("FONTSIZE",    (0, 1), (-1, -1), 9),
            ("PADDING",     (0, 0), (-1, -1), 7),
        ]))
        elements.append(shap_table)

    # ── AI Risk Explanation ───────────────────────────────────────
    elements.append(Paragraph("AI Risk Explanation", section_style))
    if explanation:
        clean_explanation = explanation.replace("\n", "<br/>")
        elements.append(Paragraph(clean_explanation, body_style))

    # ── Discharge Plan ────────────────────────────────────────────
    elements.append(Paragraph("Discharge Plan", section_style))
    if discharge_timeline:
        clean_timeline = discharge_timeline\
            .replace("**", "")\
            .replace("\n", "<br/>")
        elements.append(Paragraph(clean_timeline, body_style))

    # ── Footer ────────────────────────────────────────────────────
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=MUTED, spaceAfter=6
    ))
    footer_style = ParagraphStyle(
        "Footer",
        fontSize=8,
        textColor=MUTED,
        fontName="Helvetica",
        alignment=TA_CENTER
    )
    elements.append(Paragraph(
        "⚠️ This report is generated by an AI system for educational purposes only. "
        "Not for clinical use. Always consult a qualified healthcare professional.",
        footer_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()