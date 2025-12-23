from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def generate_pdf_report(insights_text, file_path):
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("<b>AI-Generated Data Insights Report</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))

    for line in insights_text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    doc.build(story)

