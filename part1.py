import streamlit as st
import pandas as pd
import requests
import re
from collections import Counter, defaultdict

# -----------------------------
# 1. LOAD DATASET FROM GITHUB
# -----------------------------
URL = "https://raw.githubusercontent.com/Excitedicecream/NLPAssignment/refs/heads/main/mtsamples.csv"

@st.cache_data
def load_corpus():
    df = pd.read_csv(URL)
    df = df[["transcription"]].dropna()
    text = " ".join(df["transcription"].astype(str).tolist())
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return tokens, df

tokens, df = load_corpus()
total_words = len(tokens)

# Build frequency dictionary
word_freq = Counter(tokens)
vocab = set(tokens)

# -----------------------------
# 2. EDIT DISTANCE FUNCTION
# -----------------------------
def edit_distance(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,     # deletion
                dp[i][j-1] + 1,     # insertion
                dp[i-1][j-1] + cost # substitution
            )

    return dp[-1][-1]

# -----------------------------
# 3. CANDIDATE GENERATION
# -----------------------------
def generate_candidates(word):
    candidates = []
    for vocab_word in vocab:
        if abs(len(vocab_word) - len(word)) <= 2:  # small speed filter
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

    return [w for w, s in sorted(scored, key=lambda x: -x[1])[:5]]

# -----------------------------
# 5. STREAMLIT UI
# -----------------------------
st.title("NLP Assignment – Spelling Correction Demo")
st.markdown(f"### 📊 Total Words in Corpus: **{total_words:,}**")

st.markdown("---")
st.subheader("📝 Enter Text")

if "editor_text" not in st.session_state:
    st.session_state.editor_text = ""

input_text = st.text_area("Write text here:", value=st.session_state.editor_text, height=200)

# Tokenize input
input_tokens = re.findall(r"[a-zA-Z']+", input_text.lower())

# Detect misspelled words
misspelled = [w for w in input_tokens if w not in vocab]

# RIGHT SIDEBAR
st.sidebar.title("🔧 Corrections")

if not misspelled:
    st.sidebar.success("No spelling errors detected!")
else:
    st.sidebar.write("### ✏️ Suggestions")
    for word in misspelled:
        st.sidebar.markdown(f"**❌ {word}**")

        # generate top suggestions
        suggestions = rank_candidates(word)

        if suggestions:
            choice = st.sidebar.radio(
                f"Replace '{word}' with:",
                options=suggestions + ["(keep original)"],
                key=word
            )

            # Apply button
            if st.sidebar.button(f"Apply '{word}'", key=word+"_apply"):
                if choice != "(keep original)":
                    # replace only one instance
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

# Update editor
st.session_state.editor_text = input_text
