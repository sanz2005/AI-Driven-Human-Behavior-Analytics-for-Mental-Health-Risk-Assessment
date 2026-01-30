from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def generate_pdf(path, summary, trends):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=A4)
    story = []

    story.append(Paragraph("Human Behavior Intelligence — Weekly Health Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Latest Snapshot", styles["Heading2"]))
    for k, v in summary.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recent Trends (last records)", styles["Heading2"]))
    for k, v in trends.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    doc.build(story)
