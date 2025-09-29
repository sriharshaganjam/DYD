# streamlit_riasec_app.py
# RIASEC Survey Streamlit app (radio buttons with no default)
# Usage:
#   pip install streamlit pandas
#   streamlit run streamlit_riasec_app.py

import streamlit as st
import pandas as pd
import os
import datetime
import json

# --------- CONFIG ----------
CLASS_CSV = "RIASEC QUESTIONS Flag.csv"  # adjust if your CSV is in a different location
LOG_PATH = "riasec_submissions.jsonl"    # local JSONL log of submissions
st.set_page_config(page_title="RIASEC Survey", layout="wide")

# --------- HELPERS ----------
def load_classification(path=CLASS_CSV):
    """Load question list and RIASEC flag from CSV or fallback list."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, header=0)
            cols = df.columns.tolist()
            if 'Question' in cols and ('RIASEC Flag' in cols or 'Flag' in cols):
                if 'RIASEC Flag' not in df.columns and 'Flag' in df.columns:
                    df.rename(columns={'Flag': 'RIASEC Flag'}, inplace=True)
                df = df[['Question', 'RIASEC Flag']].dropna(subset=['Question']).reset_index(drop=True)
            else:
                df = df.iloc[:, 0:2]
                df.columns = ['Question','RIASEC Flag']
            df['RIASEC Flag'] = df['RIASEC Flag'].astype(str).str.strip().str.upper().str[0]
            return df
        except Exception as e:
            st.warning(f"Could not parse classification CSV: {e}")
    # fallback embedded list
    fallback = [
        ("1. I like to work on cars","R"),
        ("7. I like to build things","R"),
        ("22. I like putting things together or assembling things.","R"),
        ("14. I like to take care of animals","R"),
        ("30. I like to cook","R"),
        ("32. I am a practical person","R"),
        ("37. I like working outdoors","R"),
        ("33. I like working with numbers or charts","I"),
        ("39. I’m good at math","I"),
        ("21. I enjoy trying to figure out how things work","I"),
        ("26. I like to analyze things (problems/ situations)","I"),
        ("2. I like to do puzzles","I"),
        ("18. I enjoy science","I"),
        ("11. I like to do experiments","I"),
        ("12. I like to teach or train people","S"),
        ("3. I am good at working independently","R"),
        ("23. I am a creative person","A"),
        ("31. I like acting in plays","A"),
        ("27. I like to play instruments or sing","A"),
        ("8. I like to read about art and music","A"),
        ("41. I like to draw","A"),
        ("17. I enjoy creative writing","A"),
        ("40. I like helping people","S"),
        ("34. I like to get into discussions about issues","S"),
        ("28. I enjoy learning about other cultures","S"),
        ("20. I am interested in healing people","S"),
        ("13. I like trying to help people solve their problems","S"),
        ("4. I like to work in teams","S"),
        ("5. I am an ambitious person, I set goals for myself","E"),
        ("29. I would like to start my own business","E"),
        ("19. I am quick to take on new responsibilities","E"),
        ("16. I like selling things","E"),
        ("36. I like to lead","E"),
        ("10. I like to try to influence or persuade people","E"),
        ("42. I like to give speeches","E"),
        ("6. I like to organize things, (files, desks/offices)","C"),
        ("9. I like to have clear instructions to follow","C"),
        ("15. I wouldn’t mind working 8 hours per day in an office","C"),
        ("24. I pay attention to details","C"),
        ("25. I like to do filing or typing","C"),
        ("35. I am good at keeping records of my work","C"),
        ("38. I would like to work in an office","C"),
    ]
    return pd.DataFrame(fallback, columns=['Question','RIASEC Flag'])

def save_log(record, path=LOG_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# --------- UI ----------
st.title("JAIN Design You Degree RIASEC Survey")
st.markdown(
    """
Answer each statement with **Yes** or **No**.  
⚠️ All questions are **mandatory** — you must choose for each one before submitting.
"""
)

questions_df = load_classification()

# Sidebar: student details
st.sidebar.header("Your details")
student_name = st.sidebar.text_input("Student name (required)", value="")
person_id = st.sidebar.text_input("Student ID or email (optional)", value="")
consent = st.sidebar.checkbox("I consent to store my responses for analysis", value=True)

# Render form
with st.form("riasec_form"):
    answers = {}
    cols = st.columns(2)
    for i, row in questions_df.iterrows():
        label = row['Question']
        col = cols[i % 2]
        # Radio with blank option at top to force explicit choice
        answers[label] = col.radio(
            label,
            options=["--", "Yes", "No"],
            index=0,
            key=f"q_{i}"
        )
    submitted = st.form_submit_button("Submit")

# --------- Submit handling ----------
if submitted:
    if not student_name:
        st.error("Please enter the Student name (required).")
    elif not consent:
        st.error("You must consent to saving the responses to proceed.")
    else:
        # Validate that all answered
        unanswered = [q for q,v in answers.items() if v == "--"]
        if unanswered:
            st.error(f"Please answer all {len(unanswered)} unanswered question(s) before submitting.")
        else:
            # Map answers to binary 1/0
            q_to_flag = dict(zip(questions_df['Question'], questions_df['RIASEC Flag']))
            vals = {q: (1 if answers[q]=="Yes" else 0) for q in answers}
            resp_df = pd.DataFrame([{'flag': q_to_flag[q], 'response': vals[q]} for q in vals])

            # Compute rates per trait
            trait_stats = resp_df.groupby('flag')['response'].agg(['sum','count']).reset_index()
            all_flags = ['R','I','A','S','E','C']
            for f in all_flags:
                if f not in trait_stats['flag'].values:
                    trait_stats = trait_stats.append({'flag': f, 'sum': 0, 'count': int((questions_df['RIASEC Flag']==f).sum())}, ignore_index=True)
            trait_stats['rate'] = trait_stats['sum'] / trait_stats['count']
            denom = trait_stats['rate'].sum()
            if denom == 0:
                st.warning("All responses are NO — cannot compute profile.")
            else:
                trait_stats['score'] = trait_stats['rate'] / denom
                scores = {r: float(trait_stats.loc[trait_stats['flag']==r,'score'].values[0]) for r in all_flags}

                st.subheader("Your RIASEC profile (normalized)")
                st.bar_chart(pd.DataFrame([scores], index=["score"]).T)

                st.write(pd.DataFrame([{'Trait': k, 'Score': round(v, 3)} for k,v in scores.items()]))

                top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                st.success("Top 3 traits: " + ", ".join([f"{t} ({v:.3f})" for t,v in top3]))
                st.markdown(f"**Interpretation:** Your strongest trait is **{top3[0][0]}**.")

                # Save log
                record = {
                    "submission_id": f"{student_name}_{datetime.datetime.utcnow().isoformat()}",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "student_name": student_name,
                    "person_id": person_id,
                    "answers": vals,
                    "riasec_vector": [scores['R'], scores['I'], scores['A'], scores['S'], scores['E'], scores['C']],
                    "top3": [t for t,_ in top3],
                    "method": "normalized_rate_sum_v1"
                }
                try:
                    save_log(record)
                    st.info(f"Saved submission to {LOG_PATH}")
                except Exception as e:
                    st.error(f"Failed to save log: {e}")

st.markdown("---")
st.markdown("App version 1.2 — RIASEC quick survey with radio buttons (blank default).")
