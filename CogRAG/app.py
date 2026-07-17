import streamlit as st

from CogRAG.adaptive_engine import (
    get_adaptive_question,
    process_answer,
    get_mastered_concepts,
    get_weak_concepts,
    get_overall_progress
)

# ----------------------------
# ✅ Page Config
# ----------------------------
st.set_page_config(
    page_title="Adaptive DBMS Tutor",
    layout="centered"
)

st.title("📚 Adaptive DBMS Tutor (BKT + Graph-Aware)")

# ----------------------------
# ✅ Session State Init
# ----------------------------
if "question_data" not in st.session_state:
    st.session_state.question_data = None

if "concept" not in st.session_state:
    st.session_state.concept = None

if "difficulty" not in st.session_state:
    st.session_state.difficulty = None

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "result" not in st.session_state:
    st.session_state.result = None

# ----------------------------
# ✅ Sidebar Dashboard
# ----------------------------
st.sidebar.header("📊 Dashboard")

progress = get_overall_progress()
st.sidebar.metric("Overall Knowledge (%)", progress)

st.sidebar.subheader("✅ Mastered Concepts")
st.sidebar.write(get_mastered_concepts())

st.sidebar.subheader("⚠️ Weak Concepts")
st.sidebar.write(get_weak_concepts())

# ----------------------------
# ✅ Generate Question
# ----------------------------
if st.button("🎯 Generate Question"):

    data = get_adaptive_question()

    if data and data["question"]:
        st.session_state.question_data = data["question"]
        st.session_state.concept = data["concept"]
        st.session_state.difficulty = data["difficulty"]
        st.session_state.submitted = False
        st.session_state.result = None

# ----------------------------
# ✅ Display Question
# ----------------------------
if st.session_state.question_data:

    q = st.session_state.question_data

    st.markdown(f"### 📚 Concept: `{st.session_state.concept}`")
    st.markdown(f"### 🧠 Difficulty: `{st.session_state.difficulty.upper()}`")

    st.write(q["question"])

    selected_option = st.radio(
        "Select your answer:",
        q["options"],
        index=None,
        key="options"
    )

    # ----------------------------
    # ✅ Submit Answer
    # ----------------------------
    if st.button("✅ Submit Answer"):

        if not selected_option:
            st.warning("Please select an option")
        else:
            user_answer = selected_option.split(".")[0]

            result = process_answer(
                st.session_state.concept,
                st.session_state.question_data,
                user_answer
            )

            st.session_state.result = result
            st.session_state.submitted = True

# ----------------------------
# ✅ Show Result
# ----------------------------
if st.session_state.submitted and st.session_state.result:

    q = st.session_state.question_data
    result = st.session_state.result
    mastery_data = result["mastery"]

    st.subheader("📌 Result")

    if result["correct"]:
        st.success("✅ Correct!")
    else:
        st.error("❌ Incorrect")

    st.write(f"**Correct Answer:** {q['answer']}")

    # ✅ BKT Knowledge Update
    st.subheader("📊 Knowledge Update (BKT)")

    st.write(f"📈 Probability of Knowing (p_know): **{round(mastery_data['p_know'], 3)}**")
    st.write(f"📚 Questions Attempted: {mastery_data['questions_attempted']}")
    st.write(f"✅ Correct Answers: {mastery_data['correct_answers']}")

    # ✅ Recommendation
    st.subheader("🚀 Learning Guidance")

    st.write(f"**Suggestion:** {result['recommendation']['recommendation']}")
    st.write(f"**Reason:** {result['recommendation']['reason']}")
    st.write(f"**Confidence (p_know):** {round(result['recommendation']['p_know'], 3)}")

    # ✅ Source
    st.subheader("📄 Source Chunk")
    st.code(q["source_chunk"])

    # ----------------------------
    # ✅ Next Question
    # ----------------------------
    if st.button("➡️ Next Question"):
        st.session_state.question_data = None
        st.session_state.submitted = False
        st.session_state.result = None
