import uuid
import streamlit as st
from langgraph.types import Command
from workflow import (
    workflow, AgentState, PatientProfile, RiskAssessment, PrescriptionPlan,
    LifestylePlan, MedicalSearchQuery, MedicalEvidence, ClinicalAlert,
    CLINICAL_DISCLAIMER,
)
from pdf_report import generate_clinical_pdf_bytes
from ml_disease_models import risk_level, RISK_LEVEL_BADGE

_URGENCY_BADGE = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}


st.set_page_config(page_title="Healthcare AI Clinical Support", layout="wide")

st.title("🩺 Healthcare AI Clinical Decision Support System")
st.caption("Interactive patient monitoring, risk assessment, and lifestyle guidance using AI. "
           "All output is for clinician review — not a substitute for clinical judgment.")

# --- Session state: tracks a paused (interrupted) run across Streamlit reruns ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "final_result" not in st.session_state:
    st.session_state.final_result = None


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_triage_strip(result: dict, halted: bool) -> None:
    """Everything a clinician needs at a single glance, above the fold:
    urgency, top risk, next review, and recommendation confidence."""
    ca = result.get("clinical_alert")
    ra = result.get("risk_assessment")
    followup_plan = result.get("followup_plan")
    quality_panel = result.get("quality_panel")

    urgency_badge = _URGENCY_BADGE.get(ca.urgency, "⚪")
    top_risk = max(ra.disease_risks, key=lambda r: r.risk_score) if ra.disease_risks else None
    rec_confidence = (
        f"{quality_panel.recommendation_confidence * 100:.0f}%"
        if quality_panel is not None and quality_panel.recommendation_confidence is not None
        else "N/A"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Urgency", f"{urgency_badge} {ca.urgency}")
    c2.metric("Top Risk", top_risk.disease_name if top_risk else "N/A",
              f"{top_risk.risk_score * 100:.0f}%" if top_risk else None)
    c3.metric("Next Review", followup_plan.next_review_date if followup_plan is not None else "N/A")
    c4.metric("Recommendation Confidence", rec_confidence)

    alert_container = st.container(border=True)
    with alert_container:
        st.markdown(f"#### {urgency_badge} Clinical Alert — {ca.urgency}")
        st.markdown(f"**Findings:** {ca.message}")
        flag_cols = st.columns(2)
        with flag_cols[0]:
            st.markdown(f"**Conditions Flagged:** {', '.join(ca.conditions_flagged) or 'None'}")
        with flag_cols[1]:
            st.markdown(f"**Interaction Risks:** {', '.join(ca.interaction_flags) or 'None'}")
        st.markdown(f"**Recommended Action:** {ca.recommended_action}")
        st.caption(CLINICAL_DISCLAIMER)

    if halted:
        st.error("⛔ Workflow halted: clinician did not approve automated continuation. "
                  "Treatment and lifestyle recommendations were not generated.")
    elif ca.urgency in ("HIGH", "CRITICAL"):
        # Reaching here without halted=True is only possible after the clinician
        # approved at human_review_gate (see hitl_router/clinical_triage_router) —
        # so this branch is specifically the post-approval case.
        st.success("✅ Clinician review completed — downstream clinical decision-support workflow continued.")
        st.caption("Clinical decision-support roadmap generated successfully.")
    else:
        # Low-risk path: never routed through the clinician review gate at all
        # (see clinical_triage_router) — do not claim a review happened.
        st.success("✅ Clinical decision-support roadmap generated successfully.")


def _render_overview_tab(result: dict) -> None:
    st.subheader("🗺️ Clinical Road Map")
    roadmap_priorities = result.get("roadmap_priorities") or []
    if roadmap_priorities:
        cols = st.columns(len(roadmap_priorities))
        for col, p in zip(cols, sorted(roadmap_priorities, key=lambda x: x.priority_rank)):
            with col:
                with st.container(border=True):
                    st.markdown(f"**Priority {p.priority_rank}**")
                    st.markdown(f"**{p.title}**")
                    st.caption(f"Goal: {p.goal}")
                    if p.rationale:
                        st.caption(p.rationale)
    with st.expander("Full narrative road map"):
        st.info(result.get("treatment_road_map", "Synthesis pending..."))

    knowledge_graph = result.get("knowledge_graph")
    if knowledge_graph is not None and knowledge_graph.edges:
        st.subheader("🕸️ Clinical Knowledge Graph")
        if knowledge_graph.narrative:
            st.caption(knowledge_graph.narrative)
        try:
            st.graphviz_chart(
                "digraph { " + " ".join(
                    f'"{e.source}" -> "{e.target}";' for e in knowledge_graph.edges
                ) + " }"
            )
        except Exception:
            for edge in knowledge_graph.edges:
                st.markdown(f"- {edge.source} → {edge.target}")


def _render_diagnosis_tab(result: dict) -> None:
    ra = result.get("risk_assessment")
    ml_predictions = result.get("ml_disease_predictions") or []
    reliability_by_name = {p.disease_name: p for p in ml_predictions}

    st.subheader("🩻 Differential Diagnosis")
    st.markdown(f"**Risk Summary:** {ra.risk_summary}")

    # --- ML Disease Risk Predictions: the ONLY section with probabilities. ---
    # ra.disease_risks is populated exclusively by the four trained XGBoost
    # models (see workflow.py:early_disease_detection / ml_disease_models.py)
    # — the LLM has no path into this list anymore.
    st.markdown("#### ML Disease Risk Predictions")
    if not ra.disease_risks:
        st.caption("No ML-scored disease risks for this patient.")
    for i, risk in enumerate(sorted(ra.disease_risks, key=lambda r: r.risk_score, reverse=True), start=1):
        severity = risk_level(risk.risk_score)
        badge = RISK_LEVEL_BADGE[severity]
        pred = reliability_by_name.get(risk.disease_name)
        with st.container(border=True):
            header = f"{badge} **{severity}** — **{i}. {risk.disease_name}** — {risk.risk_score * 100:.0f}%"
            if pred and pred.reliability_severity == "critical":
                header += "  ⚠️ **known model limitation**"
            elif pred and pred.reliability_severity == "warning":
                header += "  ⚠ known model limitation"
            st.markdown(header)
            st.progress(min(risk.risk_score, 1.0))
            if risk.reasoning:
                st.caption(f"Reason: {risk.reasoning}")
            if pred and pred.reliability_note:
                st.caption(f"⚠ {pred.reliability_note}")

    if ra.risk_flags:
        st.markdown("**Risk Flags:**")
        for flag in ra.risk_flags:
            st.markdown(f"- {flag}")

    # --- Clinical Considerations: LLM differential reasoning for diseases
    # with NO trained model. Deliberately no percentage anywhere in this
    # block — ClinicalConsideration (workflow.py) has no risk_score field
    # to begin with, so there is nothing numeric to display even by mistake. ---
    considerations = getattr(ra, "clinical_considerations", [])
    if considerations:
        st.divider()
        st.markdown("#### Clinical Considerations")
        st.caption("Raised by clinical reasoning, not a trained model. No probability is available for these.")
        for c in considerations:
            with st.container(border=True):
                st.markdown(f"**Possible {c.disease_name}**")
                if c.contributing_factors:
                    st.markdown("**Reason**")
                    for factor in c.contributing_factors:
                        st.markdown(f"- {factor}")
                elif c.reasoning:
                    st.caption(c.reasoning)
                st.caption("No ML probability available.")

    # --- SHAP explainability: only ever built from ml_predictions (see
    # explainability.py:generate_explanations, which no longer accepts a
    # non-ML disease list at all) — so this section is structurally
    # guaranteed to never show a Clinical Consideration.
    explainability_report = result.get("explainability_report")
    if explainability_report and explainability_report.disease_explanations:
        st.divider()
        st.subheader("🔎 Why did the AI predict this risk?")
        st.caption(f"Method: {explainability_report.method}")
        st.markdown(explainability_report.summary)
        for exp in explainability_report.disease_explanations:
            with st.expander(f"{exp.disease_name} — {exp.risk_score * 100:.0f}%"):
                for c in exp.top_contributors:
                    arrow = "🔺" if c.direction == "increases_risk" else ("🔻" if c.direction == "decreases_risk" else "▪️")
                    st.markdown(
                        f"{arrow} **{c.feature_name}** ({c.contribution_pct:+.1f}%) "
                        f"— value: {c.raw_value} (normal: {c.reference_range})"
                    )
                st.caption(exp.confidence_note)


def _render_treatment_tab(result: dict, halted: bool) -> None:
    if halted:
        st.info("Treatment and lifestyle recommendations were not generated because the "
                 "workflow was halted at the clinician review gate.")
        return

    st.subheader("💊 Suggested Evidence-Based Treatment Options")
    st.caption("For clinician review — not a final prescription.")
    pp = result.get("prescription_plan")
    for med in pp.medications:
        with st.container(border=True):
            st.markdown(f"**{med.name}** — {med.dose}")
            if getattr(med, "reason", ""):
                st.markdown(f"*Reason:* {med.reason}")
            st.markdown(f"*Mechanism:* {med.mechanism}  \n*Notes:* {med.notes}")
            badge_cols = st.columns(3)
            with badge_cols[0]:
                st.caption(f"📘 Guideline: {getattr(med, 'supporting_guideline', '') or 'N/A'}")
            with badge_cols[1]:
                st.caption(f"📚 Evidence: {getattr(med, 'evidence_source', '') or 'N/A'}")
            with badge_cols[2]:
                st.caption(f"🎯 Confidence: {getattr(med, 'confidence', '') or 'N/A'}")

    rec_col, instr_col = st.columns(2)
    with rec_col:
        st.markdown("**Recommendations:**")
        for rec in pp.recommendations:
            st.markdown(f"- {rec}")
    with instr_col:
        st.markdown("**Instructions:**")
        for instr in pp.instructions:
            st.markdown(f"- {instr}")
    st.caption(CLINICAL_DISCLAIMER)

    st.divider()
    st.subheader("🥗 Personalized Lifestyle Plan")
    targets = result.get("lifestyle_targets")
    if targets is not None:
        t_cols = st.columns(3)
        with t_cols[0]:
            st.metric("Current BMI", f"{targets.current_bmi}")
            st.caption(f"Weight Goal: {targets.weight_goal}")
        with t_cols[1]:
            st.metric("Calorie Target", f"{targets.daily_calories_kcal} kcal/day")
            st.caption(f"Sodium Limit: {targets.sodium_limit_mg} mg/day")
        with t_cols[2]:
            st.metric("Walking", f"{targets.daily_walking_minutes} min/day")
            st.caption(f"Resistance Training: {targets.resistance_sessions_per_week}x/week")

    lp = result.get("lifestyle_plan")
    lp_cols = st.columns(4)
    with lp_cols[0]:
        st.markdown("**Exercises:**")
        for ex in lp.exercises:
            st.markdown(f"- {ex}")
    with lp_cols[1]:
        st.markdown("**Diet:**")
        for d in lp.diet:
            st.markdown(f"- {d}")
    with lp_cols[2]:
        st.markdown("**Sleep:**")
        for s in lp.sleep:
            st.markdown(f"- {s}")
    with lp_cols[3]:
        st.markdown("**Metabolic Advice:**")
        for m in lp.metabolic_advice:
            st.markdown(f"- {m}")


def _render_evidence_tab(result: dict, halted: bool) -> None:
    if halted:
        st.info("Guideline evidence and quality scoring were not generated because the "
                 "workflow was halted at the clinician review gate.")
        return

    guideline_evidence = result.get("guideline_evidence")
    if guideline_evidence is not None:
        st.subheader("📚 Clinical Evidence & Guidelines")
        st.markdown(guideline_evidence.merged_summary or "No merged evidence summary available.")
        if guideline_evidence.guidelines:
            score_lookup = {}
            evidence_ranking_summary = result.get("evidence_ranking")
            if evidence_ranking_summary is not None:
                for item in evidence_ranking_summary.ranked_items:
                    if item.source_type == "Clinical Guideline":
                        score_lookup[(item.source_name, item.page)] = item
            st.markdown("**Supported By:**")
            for g in guideline_evidence.guidelines:
                page_text = f"Page {g.page}" if g.page is not None else "Page N/A"
                ranked = score_lookup.get((g.organization, g.page))
                score_bits = ""
                if ranked is not None:
                    score_bits = f" — Evidence Score {ranked.evidence_score * 100:.0f}% — Confidence {ranked.confidence * 100:.0f}%"
                st.markdown(
                    f"- `✓ {g.organization}` — {g.title} — {page_text} — "
                    f"Similarity {g.similarity_score * 100:.0f}%{score_bits}"
                )
        elif guideline_evidence.supported_by:
            badges = "  ".join(f"`✓ {b}`" for b in guideline_evidence.supported_by)
            st.markdown(f"**Supported By:** {badges}")

    evidence_ranking_summary = result.get("evidence_ranking")
    if evidence_ranking_summary is not None and evidence_ranking_summary.ranked_items:
        st.divider()
        st.subheader("📈 Evidence Ranking & Confidence")
        st.metric(
            "Overall Recommendation Confidence",
            f"{evidence_ranking_summary.overall_recommendation_confidence * 100:.1f}%"
        )
        for item in evidence_ranking_summary.ranked_items:
            page_bit = f", Page {item.page}" if item.page is not None else ""
            similarity_bit = f", Similarity: {item.similarity_score * 100:.1f}%" if item.similarity_score is not None else ""
            st.markdown(
                f"- **{item.source_name}** ({item.source_type}{page_bit}) — "
                f"Evidence Score: {item.evidence_score * 100:.1f}%, "
                f"Recency: {item.recency_score * 100:.1f}%, "
                f"Confidence: {item.confidence * 100:.1f}%"
                f"{similarity_bit}"
            )

    quality_panel = result.get("quality_panel")
    if quality_panel is not None:
        st.divider()
        st.subheader("📋 Evaluation & Quality Panel")
        q_cols = st.columns(4)
        with q_cols[0]:
            st.metric("Groundedness", f"{quality_panel.groundedness * 100:.0f}%" if quality_panel.groundedness is not None else "N/A")
        with q_cols[1]:
            st.metric("Faithfulness", f"{quality_panel.faithfulness * 100:.0f}%" if quality_panel.faithfulness is not None else "N/A")
        with q_cols[2]:
            st.metric("Recommendation Confidence", f"{quality_panel.recommendation_confidence * 100:.0f}%" if quality_panel.recommendation_confidence is not None else "N/A")
        with q_cols[3]:
            risk_badge = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴", "UNKNOWN": "⚪"}.get(quality_panel.hallucination_risk, "⚪")
            st.metric("Hallucination Risk", f"{risk_badge} {quality_panel.hallucination_risk}")
        st.markdown(f"**Evidence Coverage:** {', '.join(quality_panel.evidence_coverage) or 'None'}")
        st.markdown(f"**Guideline Coverage:** {', '.join(quality_panel.guideline_coverage) or 'None'}")
        st.markdown(f"**Retrieved Sources:** {quality_panel.retrieved_sources_count}")
        safety_badge = "✅" if quality_panel.drug_safety_status == "PASS" else ("⚠️" if quality_panel.drug_safety_status == "FLAGGED" else "➖")
        st.markdown(f"**Drug Safety:** {safety_badge} {quality_panel.drug_safety_status}")


def _render_followup_tab(result: dict) -> None:
    followup_plan = result.get("followup_plan")
    if followup_plan is None:
        st.info("No follow-up plan available for this run.")
        return
    st.subheader("🗓️ Follow-up & Monitoring Plan")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Next Review:** {followup_plan.next_review_date} ({followup_plan.review_interval_days} days)")
        st.markdown(f"**Monitoring Interval:** {followup_plan.monitoring_interval}")
        st.markdown(f"**Doctor Visit Schedule:** {followup_plan.doctor_visit_schedule}")
    with col_b:
        st.markdown("**Recommended Tests:**")
        for t in followup_plan.recommended_tests:
            st.markdown(f"- {t}")
        if followup_plan.imaging_studies:
            st.markdown("**Imaging:**")
            for img in followup_plan.imaging_studies:
                st.markdown(f"- {img}")
    st.warning("**⚠️ Warning Signs (seek urgent care):**\n" +
               "\n".join(f"- {w}" for w in followup_plan.warning_symptoms))


def _render_report_tab(result: dict) -> None:
    st.subheader("📄 Clinical PDF Report")
    st.caption("A print-ready summary of this assessment for the patient chart or referral.")
    try:
        pdf_bytes = generate_clinical_pdf_bytes(result)
        st.download_button(
            label="⬇️ Download Clinical Report (PDF)",
            data=pdf_bytes,
            file_name=f"clinical_report_{result.get('patient_profile').age}yo.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"PDF report could not be generated: {e}")


def render_results(result: dict) -> None:
    """Renders the final workflow output, whether it completed normally
    or was halted at the human-review gate. Urgency and alert status are
    surfaced first so a clinician can triage before reading further."""
    halted = result.get("treatment_road_map", "").startswith("Pipeline halted")

    _render_triage_strip(result, halted)
    st.divider()

    tabs = st.tabs([
        "🗺️ Overview", "🩻 Diagnosis", "💊 Treatment & Lifestyle",
        "📚 Evidence & Quality", "🗓️ Follow-up", "📄 Report",
    ])
    with tabs[0]:
        _render_overview_tab(result)
    with tabs[1]:
        _render_diagnosis_tab(result)
    with tabs[2]:
        _render_treatment_tab(result, halted)
    with tabs[3]:
        _render_evidence_tab(result, halted)
    with tabs[4]:
        _render_followup_tab(result)
    with tabs[5]:
        _render_report_tab(result)


# ---------------------------------------------------------------------------
# Patient input form — grouped into clearly labeled clinical sections so a
# clinician can scan and fill it the way a chart is organized.
# ---------------------------------------------------------------------------
with st.form("patient_form"):
    st.markdown("### 🧍 Demographics")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=55)
    with d_col2:
        gender = st.selectbox("Gender", ["male", "female"])
    with d_col3:
        height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.75)
    with d_col4:
        weight = st.number_input("Weight (kg)", min_value=1, max_value=300, value=90)

    st.markdown("### ❤️ Vital Signs")
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        sbp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=145)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=92)
        hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=72)
    with v_col2:
        temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        resp = st.number_input("Respiratory Rate (/min)", min_value=0, max_value=60, value=16)
    with v_col3:
        spo2 = st.number_input("Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)

    st.markdown("### 🧪 Labs")
    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0, max_value=500, value=250)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=0, max_value=500, value=170)
    with l_col2:
        hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=0, max_value=500, value=35)
        tg = st.number_input("Triglycerides (mg/dL)", min_value=0, max_value=500, value=220)
    with l_col3:
        sugar = st.number_input("Fasting Sugar (mmol/L)", min_value=0.0, max_value=20.0, value=8.0)
        wbc = st.number_input("WBC Count (x10^9/L)", min_value=0.0, max_value=100.0, value=7.0, step=0.1)
    plt = st.number_input("Platelets (x10^9/L)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)

    st.markdown("### 🛌 Lifestyle")
    li_col1, li_col2 = st.columns(2)
    with li_col1:
        sleep_hours = st.number_input("Average Sleep (hrs/night)", min_value=0.0, max_value=24.0, value=5.5)
    with li_col2:
        steps = st.number_input("Average Daily Steps", min_value=0, max_value=20000, value=3000)

    st.markdown("### 📋 Symptoms & History")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        symptoms = st.text_input("Symptoms (comma-separated)", "chest_pain,fatigue")
        medical_history = st.text_input("Medical History (comma-separated)", "diabetes")
    with s_col2:
        allergies = st.text_input("Allergies (comma-separated)", "Penicillin")
        immunizations = st.text_input("Immunizations (comma-separated)", "Influenza")
        family_history = st.text_input("Family History (comma-separated)", "heart_disease")

    # --- New: inputs for the trained ML risk models (hypertension / stroke /
    # diabetes / cardiovascular — see ml_disease_models.py). Every field has a
    # clinically-reasonable default already baked into PatientProfile, so
    # these are worth collecting for real accuracy but the app still runs
    # fine if a clinician skips this section. ---
    st.markdown("### 🧬 Risk Model Inputs (Hypertension / Stroke / Diabetes / Heart)")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        salt_intake = st.number_input("Salt Intake (g/day)", min_value=0.0, max_value=30.0, value=9.0, step=0.5)
        stress_score = st.slider("Stress Score", min_value=0, max_value=10, value=6)
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol_intake = st.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])
    with m_col2:
        ever_married = st.selectbox("Ever Married", ["Yes", "No"]) == "Yes"
        occupation_type = st.selectbox("Occupation Type", ["Private", "Self-employed", "Govt_job", "Never_worked", "Children"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        chest_pain_type = st.selectbox("Chest Pain Type", ["Asymptomatic", "Typical Angina", "Atypical Angina", "Non-anginal Pain"])
    with m_col3:
        diet_fruits_daily = st.checkbox("Eats Fruit Daily", value=True)
        diet_veggies_daily = st.checkbox("Eats Vegetables Daily", value=True)
        difficulty_walking = st.checkbox("Difficulty Walking/Stairs", value=False)
        exercise_induced_angina = st.checkbox("Exercise-Induced Chest Pain", value=False)

    st.markdown("*Optional — leave as defaults if unknown:*")
    o_col1, o_col2, o_col3, o_col4 = st.columns(4)
    with o_col1:
        general_health_rating = st.selectbox("Self-Rated Health (1=excellent, 5=poor)", [None, 1, 2, 3, 4, 5], index=0)
    with o_col2:
        mental_health_poor_days = st.number_input("Poor Mental Health Days (last 30)", min_value=0, max_value=30, value=0)
    with o_col3:
        physical_health_poor_days = st.number_input("Poor Physical Health Days (last 30)", min_value=0, max_value=30, value=0)
    with o_col4:
        exercise_hours_per_week = st.number_input("Exercise Hours/Week", min_value=0.0, max_value=40.0, value=0.0, step=0.5)

    submitted = st.form_submit_button("🚀 Run Clinical Workflow", use_container_width=True)

if submitted:
    with st.spinner(text="Running clinical workflow...", show_time=True):
        # Prepare state
        raw_patient_data = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "cholesterol": chol,
            "ldl_cholesterol": ldl,
            "hdl_cholesterol": hdl,
            "triglycerides": tg,
            "heartbeat_rate": hr,
            "sugar_level": sugar,
            "avg_sleep_hours": sleep_hours,
            "avg_daily_steps": steps,
            "temperature": temp,
            "respiratory_rate": resp,
            "wbc_count": wbc,
            "platelets": plt,
            "oxygen_saturation": spo2,
            "symptoms": [s.strip() for s in symptoms.split(",")],
            "medical_history": [s.strip() for s in medical_history.split(",")],
            "allergies": [s.strip() for s in allergies.split(",")],
            "immunizations": [s.strip() for s in immunizations.split(",")],
            "family_history": [s.strip() for s in family_history.split(",")] if family_history.strip() else [],
            "salt_intake_g_per_day": salt_intake,
            "stress_score": stress_score,
            "smoking_status": smoking_status,
            "ever_married": ever_married,
            "occupation_type": occupation_type,
            "residence_type": residence_type,
            "diet_fruits_daily": diet_fruits_daily,
            "diet_veggies_daily": diet_veggies_daily,
            "alcohol_intake": alcohol_intake,
            "general_health_rating": general_health_rating,
            "mental_health_poor_days": mental_health_poor_days,
            "physical_health_poor_days": physical_health_poor_days,
            "difficulty_walking": difficulty_walking,
            "exercise_hours_per_week": exercise_hours_per_week if exercise_hours_per_week > 0 else None,
            "exercise_induced_angina": exercise_induced_angina,
            "chest_pain_type": chest_pain_type,
        }

        state = AgentState(
            patient_profile=PatientProfile(
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                systolic_bp=sbp,
                diastolic_bp=dbp,
                cholesterol=chol,
                ldl_cholesterol=ldl,
                hdl_cholesterol=hdl,
                triglycerides=tg,
                heartbeat_rate=hr,
                sugar_level=sugar,
                avg_sleep_hours=sleep_hours,
                avg_daily_steps=steps,
                temperature=temp,
                respiratory_rate=resp,
                wbc_count=wbc,
                platelets=plt,
                oxygen_saturation=spo2,
                symptoms=raw_patient_data["symptoms"],
                medical_history=raw_patient_data["medical_history"],
                allergies=raw_patient_data["allergies"],
                immunizations=raw_patient_data["immunizations"],
                family_history=raw_patient_data["family_history"],
                salt_intake_g_per_day=raw_patient_data["salt_intake_g_per_day"],
                stress_score=raw_patient_data["stress_score"],
                smoking_status=raw_patient_data["smoking_status"],
                ever_married=raw_patient_data["ever_married"],
                occupation_type=raw_patient_data["occupation_type"],
                residence_type=raw_patient_data["residence_type"],
                diet_fruits_daily=raw_patient_data["diet_fruits_daily"],
                diet_veggies_daily=raw_patient_data["diet_veggies_daily"],
                alcohol_intake=raw_patient_data["alcohol_intake"],
                general_health_rating=raw_patient_data["general_health_rating"],
                mental_health_poor_days=raw_patient_data["mental_health_poor_days"],
                physical_health_poor_days=raw_patient_data["physical_health_poor_days"],
                difficulty_walking=raw_patient_data["difficulty_walking"],
                exercise_hours_per_week=raw_patient_data["exercise_hours_per_week"],
                exercise_induced_angina=raw_patient_data["exercise_induced_angina"],
                chest_pain_type=raw_patient_data["chest_pain_type"],
            ),
            risk_assessment=RiskAssessment(disease_risks=[], risk_flags=[], risk_summary=""),
            prescription_plan=PrescriptionPlan(medications=[], recommendations=[], instructions=[]),
            lifestyle_plan=LifestylePlan(exercises=[], diet=[], sleep=[], metabolic_advice=[]),
            raw_patient_data=raw_patient_data,
            medical_search_query=MedicalSearchQuery(query=""),
            medical_evidence=MedicalEvidence(query="", retrieved_chunks_count=0, refined_context="", clinical_summary="", sources_used=[]),
            clinical_alert=ClinicalAlert(
                urgency="LOW",
                message="No alert",
                conditions_flagged=[],
                interaction_flags=[],
                recommended_action="Continue monitoring"
            ),
            treatment_road_map=""
        )

        # Each run gets its own thread_id so the checkpointer can track it,
        # and so a resumed Command(resume=...) call knows which run to continue.
        thread_id = str(uuid.uuid4())
        config = {
    "configurable": {
        "thread_id": thread_id,
    },
    "run_name": "clinical-decision-support",
    "tags": [
        "clinical",
        "langgraph",
    ],
    "metadata": {
        "application": "healthcare-ai",
    },
}

        # Execute workflow (this pauses automatically if the human-review gate fires)
        result = workflow.invoke(state, config=config)

        st.session_state.thread_id = thread_id
        st.session_state.final_result = None
        st.session_state.pending_interrupt = None

        if result.get("__interrupt__"):
            # Graph paused at human_review_gate — stash the payload for the approval UI below
            st.session_state.pending_interrupt = result["__interrupt__"][0].value
        else:
            st.session_state.final_result = result

# --- Human-in-the-loop review gate ---
if st.session_state.pending_interrupt:
    info = st.session_state.pending_interrupt
    urgency_badge = _URGENCY_BADGE.get(info.get("urgency"), "⚪")

    st.warning(
        "🧑‍⚕️ **Clinician review is required before the clinical-support workflow continues.** "
        "You are not approving a diagnosis and this is not the final AI output — you are deciding "
        "whether the workflow should proceed to the downstream clinical-support stage "
        "(evidence retrieval, treatment options, and follow-up planning)."
    )

    # --- A. Critical Alert ---
    with st.container(border=True):
        st.markdown(f"### {urgency_badge} A. Critical Alert — {info.get('urgency')}")
        st.markdown(f"**Findings & Potential Complications:** {info.get('message')}")
        st.caption(
            "This system provides clinical decision support. It does not replace clinical judgment. "
            "Final diagnosis and treatment decisions remain with the treating clinician."
        )

    # --- B. ML Risk Assessment (real model probabilities ONLY) ---
    with st.container(border=True):
        st.markdown("### B. ML Risk Assessment")
        ml_risks = info.get("ml_disease_risks") or []
        if ml_risks:
            for r in sorted(ml_risks, key=lambda x: x["risk_score"], reverse=True):
                level = risk_level(r["risk_score"])
                st.markdown(f"{RISK_LEVEL_BADGE[level]} **{r['disease_name']}** — {r['risk_score']*100:.0f}% ({level})")
                if r.get("reasoning"):
                    st.caption(r["reasoning"])
        else:
            st.caption("No ML-scored disease risks for this patient.")

    # --- C. Model Explanation (existing SHAP output, not recomputed) ---
    with st.container(border=True):
        st.markdown("### C. Model Explanation")
        shap_summary = info.get("shap_summary") or []
        if shap_summary:
            for exp in shap_summary:
                st.markdown(f"**{exp['disease_name']}** ({exp['risk_score']*100:.0f}%)")
                for c in exp["top_contributors"]:
                    arrow = "↑ risk" if c["direction"] == "increases_risk" else ("↓ risk" if c["direction"] == "decreases_risk" else "~")
                    st.caption(f"- {c['feature_name']} {arrow} ({c['contribution_pct']:+.1f}%)")
        else:
            st.caption("No SHAP explanation available for this run.")

    # --- D. Clinical Considerations (LLM differential reasoning, NO trained model, NO probability) ---
    with st.container(border=True):
        st.markdown("### D. Clinical Considerations")
        considerations = info.get("clinical_considerations") or []
        if considerations:
            st.caption("Raised by clinical reasoning, not a trained model. No probability is available for these.")
            for c in considerations:
                st.markdown(f"**{c['disease_name']}** — clinical consideration")
                if c.get("contributing_factors"):
                    st.caption("Reason: " + ", ".join(c["contributing_factors"]))
                elif c.get("reasoning"):
                    st.caption(c["reasoning"])
        else:
            st.caption("No additional clinical considerations were raised.")

    # --- G. Reliability / Warnings (surfaced before F/H/I so known model
    # limitations are visible ahead of the decision, not buried after it) ---
    reliability_warnings = info.get("reliability_warnings") or []
    if reliability_warnings:
        with st.container(border=True):
            st.markdown("### G. Reliability / Warnings")
            for w in reliability_warnings:
                icon = "⚠️" if w.get("severity") == "critical" else "⚠"
                st.markdown(f"{icon} **{w['disease_name']}:** {w['note']}")

    # --- F. Recommended Immediate Action ---
    with st.container(border=True):
        st.markdown("### F. Recommended Immediate Action")
        st.markdown(info.get("recommended_action"))
        if info.get("conditions_flagged"):
            st.caption(f"Conditions Flagged: {', '.join(info['conditions_flagged'])}")
        if info.get("interaction_flags"):
            st.caption(f"Interaction Risks: {', '.join(info['interaction_flags'])}")

    # --- H. Clinician Notes + I. Decision ---
    with st.container(border=True):
        st.markdown("### H. Clinician Notes")
        notes = st.text_area("Notes (optional)", key="hitl_notes", label_visibility="collapsed")

        st.markdown("### I. Decision")
        st.caption(
            "Approving continues the workflow to evidence retrieval, treatment options, and follow-up "
            "planning. Rejecting halts the workflow here — no treatment or lifestyle plan will be generated."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            approve = st.button("✅ Approve & Continue", use_container_width=True, type="primary")
        with col_b:
            reject = st.button("⛔ Reject & Halt", use_container_width=True)

    if approve or reject:
        with st.spinner(text="Resuming clinical workflow...", show_time=True):
            resume_config = {"configurable": {"thread_id": st.session_state.thread_id}}
            resumed_result = workflow.invoke(
                Command(resume={"approved": approve, "notes": notes}),
                config=resume_config
            )
            st.session_state.pending_interrupt = None
            st.session_state.final_result = resumed_result
        st.rerun()

# --- Final results ---
if st.session_state.final_result:
    render_results(st.session_state.final_result)