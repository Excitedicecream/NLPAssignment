import streamlit as st
import pandas as pd
import re
from collections import Counter

# -----------------------------
# 0. TEXT CLEANING FUNCTION
# -----------------------------
def clean_transcription(text):
    text = str(text)

    # Remove section headers (ALL CAPS + colon)
    text = re.sub(r"\b[A-Z][A-Z\s/]+\:", "", text)

    # Remove numbered list markers (1., 2., 10.)
    text = re.sub(r"\b\d+\.\s*", "", text)

    # Remove vitals / measurements / percentages
    text = re.sub(r"\b\d+(/\d+)?\b", "", text)        # 124/78, 49
    text = re.sub(r"\b\d+%|\b\d+'\d+\"?", "", text)  # 70%, 5'9"

    # Remove extra commas
    text = re.sub(r",+", ",", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# 1. LOAD DATASET FROM GITHUB
# -----------------------------
URL = "https://raw.githubusercontent.com/Excitedicecream/NLPAssignment/refs/heads/main/mtsamples.csv"

@st.cache_data
def load_corpus():
    df = pd.read_csv(URL)
    df = df[["transcription"]].dropna()

    # Apply cleaning
    df["cleaned_transcription"] = df["transcription"].apply(clean_transcription)

    # Build corpus ONLY from cleaned text
    full_text = " ".join(df["cleaned_transcription"].astype(str).tolist())
    tokens = re.findall(r"[a-zA-Z']+", full_text.lower())

    return tokens, df


tokens, df = load_corpus()
total_words = len(tokens)

# Build frequency dictionary and vocabulary (CLEANED)
word_freq = Counter(tokens)
vocab = set(tokens)


# -----------------------------
# 2. EDIT DISTANCE FUNCTION
# -----------------------------
def edit_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


# -----------------------------
# 3. CANDIDATE GENERATION
# -----------------------------
def generate_candidates(word):
    candidates = []
    for vocab_word in vocab:
        if abs(len(vocab_word) - len(word)) <= 2:
            dist = edit_distance(word, vocab_word)
            if dist <= 2:
                candidates.append((vocab_word, dist))
    return candidates


# -----------------------------
# 4. RANKING MODEL
# -----------------------------
def rank_candidates(word):
    candidates = generate_candidates(word)
    if not candidates:
        return []

    scored = []
    for cand, dist in candidates:
        score = -dist + (word_freq[cand] / 10000)
        scored.append((cand, score))

    return [w for w, _ in sorted(scored, key=lambda x: -x[1])[:5]]


# -----------------------------
# 5. STREAMLIT UI
# -----------------------------
st.title("NLP Assignment – Spelling Correction Demo (Cleaned Clinical Notes)")
st.markdown(f"### 📊 Total Words in Cleaned Corpus: **{total_words:,}**")

tab1, tab2 = st.tabs(["✍️ Spelling Correction", "📄 Cleaned Dataset Examples"])

# =============================
# TAB 1: SPELLING CORRECTION
# =============================
with tab1:
    st.subheader("📝 Enter Text")

    if "editor_text" not in st.session_state:
        st.session_state.editor_text = ""

    input_text = st.text_area(
        "Write text here:",
        value=st.session_state.editor_text,
        height=200
    )

    input_tokens = re.findall(r"[a-zA-Z']+", input_text.lower())
    misspelled = [w for w in input_tokens if w not in vocab]

    st.sidebar.title("🔧 Corrections")

    if not misspelled:
        st.sidebar.success("No spelling errors detected!")
    else:
        st.sidebar.write("### ✏️ Suggestions")
        for word in misspelled:
            st.sidebar.markdown(f"**❌ {word}**")

            suggestions = rank_candidates(word)

            if suggestions:
                choice = st.sidebar.radio(
                    f"Replace '{word}' with:",
                    options=suggestions + ["(keep original)"],
                    key=word
                )

                if st.sidebar.button(f"Apply '{word}'", key=word + "_apply"):
                    if choice != "(keep original)":
                        st.session_state.editor_text = re.sub(
                            r"\b" + re.escape(word) + r"\b",
                            choice,
                            st.session_state.editor_text,
                            count=1
                        )
                    else:
                        st.session_state.editor_text = input_text

                    st.rerun()
            else:
                st.sidebar.warning("No suggestions found.")

    st.session_state.editor_text = input_text


# =============================
# TAB 2: CLEANED DATASET EXAMPLES
# =============================
with tab2:
    st.subheader("📄 Cleaned Sample Transcriptions")

    num_examples = st.slider(
        "Number of examples to show:",
        min_value=1,
        max_value=10,
        value=5
    )

    st.info(
        "These examples are preprocessed by removing section headers, "
        "numbers, vitals, and formatting noise before tokenization."
    )

    for i, text in enumerate(
        df["cleaned_transcription"].head(num_examples),
        start=1
    ):
        st.markdown(f"**Example {i}:**")
        st.write(text)
        st.markdown("---")
