"""
pdf_report.py — Feature 3: Clinical PDF Report Generator.

Builds a hospital-style PDF summarizing the full workflow output: patient
info, risk assessment, explainable AI summary, disease risks, clinical
evidence + guidelines used, medication plan, drug safety alerts, lifestyle
plan, follow-up plan, clinical road map, and confidence scores.

Pure Python (reportlab) — no external services. Called from app.py after the
workflow finishes, and offered to the user as a Streamlit download button.
"""
from __future__ import annotations

import datetime
import io
from typing import Any, Dict, Optional

from ml_disease_models import risk_level

CLINICAL_DISCLAIMER = (
    "This recommendation is intended to assist licensed healthcare professionals. "
    "It does not replace clinical judgment. Final diagnosis and treatment decisions "
    "remain the responsibility of the treating clinician."
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

_HOSPITAL_BLUE = colors.HexColor("#0B3D66")
_ALERT_RED = colors.HexColor("#B3261E")
_LIGHT_GREY = colors.HexColor("#F2F4F7")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="HospitalTitle", fontSize=18, leading=22, textColor=_HOSPITAL_BLUE,
        spaceAfter=4, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="HospitalSubtitle", fontSize=10, leading=13, textColor=colors.grey,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader", fontSize=13, leading=16, textColor=_HOSPITAL_BLUE,
        spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="Body", fontSize=9.5, leading=13.5, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="AlertBody", fontSize=9.5, leading=13.5, spaceAfter=4, textColor=_ALERT_RED,
    ))
    return styles


def _section_table(rows, col_widths=None, header=False):
    style_cmds = [
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D5DD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), _HOSPITAL_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle(style_cmds))
    return t


def _build_story(result: Dict[str, Any]) -> list:
    """
    Assembles the flowable "story" (all report content) from a LangGraph
    final state dict (same shape rendered by app.py's render_results),
    optionally augmented with:
      - result["explainability_report"] (explainability.ExplainabilityReport)
      - result["guideline_evidence"] (guideline_retrieval.GuidelineEvidence)
      - result["evidence_ranking"] (evidence_ranking.EvidenceRankingSummary)
      - result["followup_plan"] (followup_planner.FollowUpPlan)
    All are optional; sections are skipped gracefully if absent. Shared by
    both the file-path and in-memory (bytes) entry points below so the report
    layout logic lives in exactly one place.
    """
    styles = _styles()
    story = []

    patient = result.get("patient_profile")
    risk_assessment = result.get("risk_assessment")
    prescription_plan = result.get("prescription_plan")
    lifestyle_plan = result.get("lifestyle_plan")
    clinical_alert = result.get("clinical_alert")
    explainability_report = result.get("explainability_report")
    guideline_evidence = result.get("guideline_evidence")
    evidence_ranking = result.get("evidence_ranking")
    followup_plan = result.get("followup_plan")
    road_map = result.get("treatment_road_map", "")

    # --- Header ---
    story.append(Paragraph("Clinical Decision Support System", styles["HospitalTitle"]))
    story.append(Paragraph(
        f"AI-Assisted Clinical Report &nbsp;|&nbsp; Generated {datetime.date.today().isoformat()}",
        styles["HospitalSubtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=_HOSPITAL_BLUE))
    story.append(Spacer(1, 8))

    # --- Patient Information ---
    if patient is not None:
        story.append(Paragraph("Patient Information", styles["SectionHeader"]))
        rows = [
            ["Age", str(patient.age), "Gender", patient.gender],
            ["Height", f"{patient.height} m", "Weight", f"{patient.weight} kg"],
            ["BMI", f"{patient.bmi:.2f}", "Blood Pressure", f"{patient.systolic_bp}/{patient.diastolic_bp} mmHg"],
            ["Sugar Level", f"{patient.sugar_level} mmol/L", "Heart Rate", f"{patient.heartbeat_rate} bpm"],
            ["Symptoms", ", ".join(patient.symptoms) or "None", "Medical History", ", ".join(patient.medical_history) or "None"],
            ["Allergies", ", ".join(patient.allergies) or "None", "Immunizations", ", ".join(patient.immunizations) or "None"],
        ]
        story.append(_section_table(rows, col_widths=[1.1 * inch, 2.1 * inch, 1.3 * inch, 2.1 * inch]))

    # --- Risk Assessment (ML Disease Risk Predictions ONLY — risk_assessment.disease_risks
    # is populated exclusively by the four trained models; see workflow.py) ---
    if risk_assessment is not None:
        story.append(Paragraph("ML Disease Risk Predictions", styles["SectionHeader"]))
        story.append(Paragraph(risk_assessment.risk_summary, styles["Body"]))
        if risk_assessment.disease_risks:
            rows = [["Disease", "Probability", "Risk Level"]] + [
                [r.disease_name, f"{r.risk_score*100:.0f}%", risk_level(r.risk_score)]
                for r in risk_assessment.disease_risks
            ]
            story.append(Spacer(1, 4))
            story.append(_section_table(rows, col_widths=[3.5 * inch, 1.2 * inch, 1.3 * inch], header=True))
            # Surface known model limitations (e.g. hypertension leakage) right
            # next to the score they qualify, sourced from the real trained
            # model metadata (ml_disease_models.MODEL_RELIABILITY_NOTES) rather
            # than a generic disclaimer.
            ml_predictions = result.get("ml_disease_predictions") or []
            for pred in ml_predictions:
                if getattr(pred, "reliability_note", None):
                    story.append(Paragraph(f"<b>{pred.disease_name}:</b> {pred.reliability_note}", styles["AlertBody"]))
        if risk_assessment.risk_flags:
            story.append(Spacer(1, 4))
            story.append(Paragraph("Risk Flags: " + ", ".join(risk_assessment.risk_flags), styles["AlertBody"]))

    # --- Clinical Considerations (Feature 6, revised): LLM differential
    # reasoning for diseases with NO trained model. No score column — by
    # construction (ClinicalConsideration has no risk_score field) there is
    # nothing numeric to put in one. ---
    if risk_assessment is not None and risk_assessment.clinical_considerations:
        story.append(Paragraph("Clinical Considerations", styles["SectionHeader"]))
        story.append(Paragraph(
            "Raised by clinical reasoning, not a trained model. No ML probability is available for these.",
            styles["Body"]
        ))
        for c in risk_assessment.clinical_considerations:
            story.append(Paragraph(f"<b>Possible {c.disease_name}</b>", styles["Body"]))
            if c.contributing_factors:
                story.append(Paragraph("Reason: " + ", ".join(c.contributing_factors), styles["Body"]))
            elif c.reasoning:
                story.append(Paragraph(c.reasoning, styles["Body"]))
            story.append(Paragraph("No ML probability available.", styles["HospitalSubtitle"]))
            story.append(Spacer(1, 3))

    # --- Differential Diagnosis detail table (ML-scored diseases, ranked) ---
    if risk_assessment is not None and risk_assessment.disease_risks:
        story.append(Paragraph("Differential Diagnosis Detail (ML-Scored)", styles["SectionHeader"]))
        ranked = sorted(risk_assessment.disease_risks, key=lambda r: r.risk_score, reverse=True)
        rows = [["#", "Condition (Risk)", "Score", "Reasoning"]]
        for i, r in enumerate(ranked[:8], start=1):
            severity = risk_level(r.risk_score)
            rows.append([
                str(i), f"{r.disease_name} ({severity})", f"{r.risk_score*100:.0f}%",
                r.reasoning or "Based on aggregate biometric and history pattern.",
            ])
        story.append(_section_table(rows, col_widths=[0.3 * inch, 1.9 * inch, 0.6 * inch, 3.2 * inch], header=True))

    # --- Explainable AI Summary: automatically ML-only, since
    # explainability.generate_explanations() no longer accepts a non-ML
    # disease list at all (see explainability.py). A Clinical Consideration
    # can never appear in this section by construction. ---
    if explainability_report is not None and explainability_report.disease_explanations:
        story.append(Paragraph("Explainable AI Summary", styles["SectionHeader"]))
        story.append(Paragraph(f"Method: {explainability_report.method}", styles["Body"]))
        story.append(Paragraph(explainability_report.summary, styles["Body"]))
        for exp in explainability_report.disease_explanations:
            story.append(Paragraph(f"<b>{exp.disease_name} — {exp.risk_score*100:.0f}%</b>", styles["Body"]))
            rows = [["Feature", "Direction", "Contribution", "Value", "Reference Range"]]
            for c in exp.top_contributors:
                arrow = "UP" if c.direction == "increases_risk" else ("DOWN" if c.direction == "decreases_risk" else "-")
                rows.append([c.feature_name, arrow, f"{c.contribution_pct:+.1f}%", str(c.raw_value), c.reference_range])
            story.append(_section_table(rows, col_widths=[1.5 * inch, 0.7 * inch, 1.0 * inch, 0.9 * inch, 1.9 * inch], header=True))
            story.append(Spacer(1, 4))

    # --- Clinical Knowledge Graph (Feature 7) ---
    knowledge_graph = result.get("knowledge_graph")
    if knowledge_graph is not None and knowledge_graph.edges:
        story.append(Paragraph("Clinical Knowledge Graph", styles["SectionHeader"]))
        if knowledge_graph.narrative:
            story.append(Paragraph(knowledge_graph.narrative, styles["Body"]))
        for edge in knowledge_graph.edges:
            emphasis = "Body" if edge.patient_relevant else "HospitalSubtitle"
            story.append(Paragraph(f"{edge.source} &rarr; {edge.target}", styles[emphasis]))
        story.append(Spacer(1, 4))

    # --- Clinical Evidence & Guidelines ---
    if guideline_evidence is not None:
        story.append(Paragraph("Clinical Evidence & Guidelines", styles["SectionHeader"]))
        story.append(Paragraph(guideline_evidence.merged_summary or "No merged evidence summary available.", styles["Body"]))
        if guideline_evidence.guidelines:
            # Build a lookup of evidence_score from the ranked evidence summary
            # (Feature 5), keyed by (organization, page), so this table can show
            # Source / Page / Similarity / Evidence Score per guideline chunk.
            score_lookup = {}
            if evidence_ranking is not None:
                for item in evidence_ranking.ranked_items:
                    if item.source_type == "Clinical Guideline":
                        score_lookup[(item.source_name, item.page)] = item.evidence_score

            rows = [["Source", "Page", "Similarity", "Evidence Score"]]
            for g in guideline_evidence.guidelines:
                evidence_score = score_lookup.get((g.organization, g.page))
                rows.append([
                    g.organization,
                    str(g.page) if g.page is not None else "N/A",
                    f"{g.similarity_score*100:.0f}%",
                    f"{evidence_score*100:.0f}%" if evidence_score is not None else "N/A",
                ])
            story.append(Spacer(1, 4))
            story.append(_section_table(rows, col_widths=[1.6 * inch, 1.0 * inch, 1.4 * inch, 1.6 * inch], header=True))
        elif guideline_evidence.supported_by:
            badges = "  ".join(f"[{b}]" for b in guideline_evidence.supported_by)
            story.append(Paragraph(f"<b>Supported By:</b> {badges}", styles["Body"]))

    # --- Evidence Ranking / Confidence ---
    if evidence_ranking is not None and evidence_ranking.ranked_items:
        story.append(Paragraph("Evidence Ranking & Confidence", styles["SectionHeader"]))
        rows = [["Source", "Type", "Page", "Similarity", "Evidence Score", "Recency", "Confidence"]]
        for item in evidence_ranking.ranked_items[:10]:
            rows.append([
                item.source_name, item.source_type,
                str(item.page) if item.page is not None else "-",
                f"{item.similarity_score*100:.1f}%" if item.similarity_score is not None else "-",
                f"{item.evidence_score*100:.1f}%", f"{item.recency_score*100:.1f}%", f"{item.confidence*100:.1f}%"
            ])
        story.append(_section_table(
            rows,
            col_widths=[1.1 * inch, 1.1 * inch, 0.6 * inch, 0.8 * inch, 0.95 * inch, 0.8 * inch, 0.85 * inch],
            header=True,
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"<b>Overall Recommendation Confidence: {evidence_ranking.overall_recommendation_confidence*100:.1f}%</b>",
            styles["Body"]
        ))

    # --- Suggested Evidence-Based Treatment Options (Feature 1) ---
    if prescription_plan is not None:
        story.append(Paragraph("Suggested Evidence-Based Treatment Options", styles["SectionHeader"]))
        story.append(Paragraph(
            "The following are options for clinician review, not a final prescription.",
            styles["HospitalSubtitle"],
        ))
        if prescription_plan.medications:
            rows = [["Option", "Dose Range", "Reason", "Guideline", "Evidence", "Confidence"]]
            for m in prescription_plan.medications:
                rows.append([
                    m.name, m.dose, getattr(m, "reason", "") or "-",
                    getattr(m, "supporting_guideline", "") or "-",
                    getattr(m, "evidence_source", "") or "-",
                    getattr(m, "confidence", "") or "-",
                ])
            story.append(_section_table(
                rows, col_widths=[1.1 * inch, 0.8 * inch, 1.5 * inch, 1.0 * inch, 0.9 * inch, 0.7 * inch], header=True
            ))
        if prescription_plan.recommendations:
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Recommendations:</b> " + "; ".join(prescription_plan.recommendations), styles["Body"]))
        if prescription_plan.instructions:
            story.append(Paragraph("<b>Instructions:</b> " + "; ".join(prescription_plan.instructions), styles["Body"]))
        story.append(Spacer(1, 4))

    # --- Drug Safety Alerts ---
    if clinical_alert is not None and clinical_alert.interaction_flags:
        story.append(Paragraph("Drug Safety Alerts", styles["SectionHeader"]))
        for flag in clinical_alert.interaction_flags:
            story.append(Paragraph(f"- {flag}", styles["AlertBody"]))

    # --- Lifestyle Plan ---
    if lifestyle_plan is not None:
        story.append(Paragraph("Lifestyle Plan", styles["SectionHeader"]))
        for label, items in [
            ("Exercise", lifestyle_plan.exercises), ("Diet", lifestyle_plan.diet),
            ("Sleep", lifestyle_plan.sleep), ("Metabolic Advice", lifestyle_plan.metabolic_advice),
        ]:
            if items:
                story.append(Paragraph(f"<b>{label}:</b> " + "; ".join(items), styles["Body"]))

    # --- Follow-up Plan ---
    if followup_plan is not None:
        story.append(Paragraph("Follow-up Plan", styles["SectionHeader"]))
        rows = [
            ["Next Review", f"{followup_plan.next_review_date} ({followup_plan.review_interval_days} days)"],
            ["Recommended Tests", ", ".join(followup_plan.recommended_tests) or "None"],
            ["Imaging", ", ".join(followup_plan.imaging_studies) or "None required"],
            ["Monitoring Interval", followup_plan.monitoring_interval],
            ["Doctor Visit Schedule", followup_plan.doctor_visit_schedule],
            ["Warning Symptoms", ", ".join(followup_plan.warning_symptoms) or "None"],
        ]
        story.append(_section_table(rows, col_widths=[1.6 * inch, 4.4 * inch]))

    # --- Clinical Alert / Escalation (Feature 8) ---
    if clinical_alert is not None:
        story.append(Paragraph("Clinical Alert", styles["SectionHeader"]))
        rows = [
            ["Urgency", clinical_alert.urgency],
            ["Findings", clinical_alert.message],
            ["Conditions Flagged", ", ".join(clinical_alert.conditions_flagged) or "None"],
            ["Recommended Action", clinical_alert.recommended_action],
        ]
        story.append(_section_table(rows, col_widths=[1.4 * inch, 4.6 * inch]))
        story.append(Spacer(1, 4))

    # --- Structured Clinical Road Map (Priorities) ---
    roadmap_priorities = result.get("roadmap_priorities")
    if roadmap_priorities:
        story.append(Paragraph("Clinical Road Map — Priorities", styles["SectionHeader"]))
        rows = [["#", "Priority", "Goal", "Rationale"]]
        for p in sorted(roadmap_priorities, key=lambda x: x.priority_rank):
            rows.append([f"P{p.priority_rank}", p.title, p.goal, p.rationale or "-"])
        story.append(_section_table(rows, col_widths=[0.4 * inch, 1.5 * inch, 1.8 * inch, 2.3 * inch], header=True))
        story.append(Spacer(1, 4))

    # --- Evaluation & Quality Panel (Feature 4) ---
    quality_panel = result.get("quality_panel")
    if quality_panel is not None:
        story.append(Paragraph("Evaluation & Quality Panel", styles["SectionHeader"]))
        rows = [
            ["Groundedness", f"{quality_panel.groundedness*100:.0f}%" if quality_panel.groundedness is not None else "N/A"],
            ["Faithfulness", f"{quality_panel.faithfulness*100:.0f}%" if quality_panel.faithfulness is not None else "N/A"],
            ["Evidence Coverage", ", ".join(quality_panel.evidence_coverage) or "None"],
            ["Guideline Coverage", ", ".join(quality_panel.guideline_coverage) or "None"],
            ["Retrieved Sources", str(quality_panel.retrieved_sources_count)],
            ["Hallucination Risk", quality_panel.hallucination_risk],
            ["Recommendation Confidence", f"{quality_panel.recommendation_confidence*100:.0f}%" if quality_panel.recommendation_confidence is not None else "N/A"],
            ["Drug Safety Status", quality_panel.drug_safety_status],
        ]
        story.append(_section_table(rows, col_widths=[2.0 * inch, 4.0 * inch]))
        story.append(Spacer(1, 6))

    # --- Clinical Road Map ---
    story.append(Paragraph("Clinical Road Map", styles["SectionHeader"]))
    story.append(Paragraph(road_map or "Synthesis pending.", styles["Body"]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "This report is generated by an AI-assisted clinical decision support system and is intended "
        "to support, not replace, clinician judgment.",
        styles["HospitalSubtitle"]
    ))

    return story


def _new_doc(target):
    return SimpleDocTemplate(
        target, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.65 * inch, rightMargin=0.65 * inch,
    )


def generate_clinical_pdf(result: Dict[str, Any], output_path: str) -> str:
    """Writes the clinical PDF report to disk and returns the output path."""
    doc = _new_doc(output_path)
    doc.build(_build_story(result))
    return output_path


def generate_clinical_pdf_bytes(result: Dict[str, Any]) -> bytes:
    """Convenience wrapper for Streamlit's download_button, which needs bytes."""
    buffer = io.BytesIO()
    doc = _new_doc(buffer)
    doc.build(_build_story(result))
    return buffer.getvalue()