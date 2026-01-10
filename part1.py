import streamlit as st
import pandas as pd
import re
from collections import Counter, defaultdict

# -----------------------------
# 0. TEXT CLEANING
# -----------------------------
def clean_punctuation_simple(text: str) -> str:
    result = []
    n = len(text)

    for i, ch in enumerate(text):
        if ch in {",", "."}:
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i < n - 1 else ""
            if prev.isalpha() and nxt.isalpha():
                result.append(ch)
        else:
            result.append(ch)

    cleaned = "".join(result)
    cleaned = " ".join(cleaned.split())
    return cleaned.lower().strip()


# -----------------------------
# 1. LOAD & PREPARE CORPUS
# -----------------------------
URL = "https://raw.githubusercontent.com/Excitedicecream/NLPAssignment/refs/heads/main/mtsamples.csv"

@st.cache_data
def load_corpus():
    df = pd.read_csv(URL)
    df = df[["transcription"]].dropna()

    df["cleaned_transcription"] = df["transcription"].apply(clean_punctuation_simple)

    full_text = " ".join(df["cleaned_transcription"].astype(str))
    tokens = re.findall(r"[a-zA-Z']+", full_text)

    return tokens, df

tokens, df = load_corpus()
total_words = len(tokens)

word_freq = Counter(tokens)
vocab = set(tokens)

# -----------------------------
# 2. BIGRAM MODEL (REAL-WORD ERRORS)
# -----------------------------
bigram_freq = defaultdict(int)
for i in range(len(tokens) - 1):
    bigram_freq[(tokens[i], tokens[i + 1])] += 1

def bigram_score(prev_word, candidate):
    if not prev_word:
        return 0
    return bigram_freq.get((prev_word, candidate), 0)


# -----------------------------
# 3. EDIT DISTANCE
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
# 4. CANDIDATE GENERATION
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
# 5. RANKING (EDIT DISTANCE + FREQUENCY + BIGRAM)
# -----------------------------
def rank_candidates(word, prev_word=None):
    candidates = generate_candidates(word)
    if not candidates:
        return []

    scored = []
    for cand, dist in candidates:
        score = (
            -dist +
            (word_freq[cand] / total_words) +
            (bigram_score(prev_word, cand) * 0.001)
        )
        scored.append((cand, score))

    return [w for w, _ in sorted(scored, key=lambda x: -x[1])[:5]]


# -----------------------------
# 6. STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="NLP Spelling Correction", layout="wide")

st.title("🧠 NLP Assignment – Spelling Correction System")
st.markdown(f"### 📊 Total Words in Corpus: **{total_words:,}**")

tab1, tab2, tab3 = st.tabs([
    "✍️ Spelling Correction",
    "📄 Dataset Examples",
    "📖 System Workflow & Corpus Design"
])

# =============================
# TAB 1: SPELLING CORRECTION
# =============================
with tab1:

    st.subheader("✍️ Spelling Correction Demo")

    # -----------------------------
    # INITIALISE SESSION STATE
    # -----------------------------
    if "editor_text" not in st.session_state:
        st.session_state.editor_text = ""

    # -----------------------------
    # SAMPLE TEXTS
    # -----------------------------
    sample_texts = {
        "": "",
        "Clinical Report Sample": (
            "The patient was admitted to the medical ward for further evaluation of chronic chest pain and shortness of breath. According to the clinical history, the patient has been experiecing intermittent chest discomfort for the past six months, which has gradually increased in frequncy and severity. The pain is described as a dull pressure sensation that radiates form the center of the chest to the left shoulder and upper arm. The patient reported that the pain usually occurs during physical exertion but sometimes appears at rest, especialy during periods of emotional stress. Past medical history reveals a long-standing history of hypertension and type two diabetes mellitus. The patient admited to poor medication compliance due to financial constraints."
        ),
        "Business Report Sample": (
            "This report analyses the recent performance of the organisation and evaluates the effectiveness of its current strategic initiatives. Over the past fiscal year, the company has experienced a steady increase in revenue. However, several operational challanges were identified that may impact future growth if they are not addressed in a timely manner.Delays in raw material deliveries resulted in production bottlenecks, and some orders were delivered later then promised. Their is evidence that communication between departments needs improvement."
        ),
        "NLP Research Sample": (
            "Natural language processing has emerged as a critical area of research within artificial intelligence. Despite recent advancements, challanges remain in handling ambiguity and context. Real word spelling errors are often overlooked by traditional spell checkers because the incorrect word still exists in the dictionary. This highlights the importance of incorporating contextual information such as bigram language models."
        ),
    }

    # -----------------------------
    # DROPDOWN (THIS WILL SHOW)
    # -----------------------------
    selected_sample = st.selectbox(
        "📌 Choose a sample text to load:",
        options=list(sample_texts.keys()),
        index=0
    )

    if selected_sample != "":
        if st.button("Use selected sample"):
            st.session_state.editor_text = sample_texts[selected_sample]
            st.rerun()

    # -----------------------------
    # TEXT INPUT
    # -----------------------------
    input_text = st.text_area(
        "Enter text (max 500 characters):",
        value=st.session_state.editor_text,
        height=250,
        max_chars=500
    )

    tokens_input = re.findall(r"[a-zA-Z']+", input_text.lower())

    misspelled = []
    for i, word in enumerate(tokens_input):
        if word not in vocab:
            misspelled.append((i, word))

    # -----------------------------
    # HIGHLIGHT ERRORS
    # -----------------------------
    highlighted = input_text
    for _, w in misspelled:
        highlighted = re.sub(
            rf"\b{w}\b",
            f"**:red[{w}]**",
            highlighted,
            flags=re.IGNORECASE
        )

    st.markdown("### 🔍 Highlighted Spelling Errors")
    st.markdown(highlighted)

    # -----------------------------
    # SIDEBAR CORRECTIONS
    # -----------------------------
    st.sidebar.title("🔧 Corrections")

    if not misspelled:
        st.sidebar.success("No spelling errors detected!")
    else:
        for idx, word in misspelled:
            prev_word = tokens_input[idx - 1] if idx > 0 else None

            st.sidebar.markdown(f"**❌ {word}**")
            suggestions = rank_candidates(word, prev_word)

            if suggestions:
                choice = st.sidebar.radio(
                    f"Replace '{word}' with:",
                    options=suggestions + ["(keep original)"],
                    key=f"{word}_{idx}"
                )

                if st.sidebar.button(f"Apply", key=f"apply_{word}_{idx}"):
                    if choice != "(keep original)":
                        st.session_state.editor_text = re.sub(
                            rf"\b{word}\b",
                            choice,
                            st.session_state.editor_text,
                            count=1
                        )
                    st.rerun()

# =============================
# TAB 2: DATASET EXAMPLES
# =============================
with tab2:
    st.subheader("📄 Cleaned Corpus Samples")

    n = st.slider("Number of samples:", 1, 10, 5)
    for i, text in enumerate(df["cleaned_transcription"].head(n), start=1):
        st.markdown(f"**Example {i}:**")
        st.write(text)
        st.markdown("---")


# =============================
# TAB 3: SYSTEM WORKFLOW & CORPUS DESIGN
# =============================
with tab3:
    st.subheader("🧩 System Workflow and Corpus Design")

    st.markdown("""
    This section explains how the spelling correction system works internally
    and how the corpus influences its behaviour.
    """)

    # -----------------------------
    # WORKFLOW STEPS
    # -----------------------------
    with st.expander("Step 1 – Corpus Preparation"):
        st.markdown("""
        - Raw corpus text is cleaned and normalised.
        - Text is converted to lowercase to ensure consistency.
        - The corpus is tokenised into individual words.
        - A vocabulary of valid words is created.
        - Word frequency statistics are computed.
        - Bigram frequencies are generated to capture contextual patterns.
        """)

    with st.expander("Step 2 – User Input Processing"):
        st.markdown("""
        - User input text is tokenised using the same method as the corpus.
        - Consistent tokenisation ensures accurate comparison.
        - Each input word is checked against the corpus vocabulary.
        """)

    with st.expander("Step 3 – Spelling Error Detection"):
        st.markdown("""
        - Words not found in the vocabulary are flagged as non-word errors.
        - This approach efficiently detects misspellings without requiring
          external dictionaries.
        """)

    with st.expander("Step 4 – Candidate Generation (Edit Distance)"):
        st.markdown("""
        - Correction candidates are generated using Minimum Edit Distance.
        - Only words with small edit distances are considered.
        - This ensures corrections are linguistically plausible.
        """)

    with st.expander("Step 5 – Context-Aware Candidate Ranking"):
        st.markdown("""
        - Candidates are ranked using a hybrid scoring strategy:
            - Edit distance (string similarity)
            - Word frequency (corpus likelihood)
            - Bigram context (previous word relationship)
        - This enables correction of both non-word and real-word errors.
        """)

    with st.expander("Step 6 – Interactive Correction"):
        st.markdown("""
        - Top-ranked suggestions are presented to the user.
        - The user selects a replacement or keeps the original word.
        - Corrections are applied incrementally and the text is re-evaluated.
        """)

    # -----------------------------
    # BENEFITS
    # -----------------------------
    with st.expander("🏥 Benefits of Using Larger and Domain-Specific Corpora"):
        st.markdown("""
        **1. Improved Vocabulary Coverage**  
        Domain-specific terms (e.g. medical terminology) are recognised
        as valid words, reducing false error detection.

        **2. More Accurate Corrections**  
        Edit distance candidates are biased toward domain-relevant words,
        improving correction quality.

        **3. Better Contextual Accuracy**  
        Bigram models learn domain-specific word usage patterns, improving
        real-word error correction.

        **4. Domain Adaptability**  
        The system architecture remains unchanged. Replacing the corpus
        automatically adapts the system to new domains.

        **5. Scalability**  
        Larger corpora improve frequency estimation and reduce sparsity,
        leading to more reliable probabilistic ranking.
        """)

    # -----------------------------
    # LIMITATIONS & FUTURE WORK
    # -----------------------------
    with st.expander("⚠️ Limitations and Future Improvements"):
        st.markdown("""
        **Current Limitations**
        - Uses bigram context only and cannot model long-range dependencies.
        - Relies on vocabulary presence to detect errors.
        - Computational cost increases with very large vocabularies.

        **Future Improvements**
        - Extend to trigram or neural language models.
        - Incorporate Part-of-Speech (POS) tagging to enforce grammatical rules.
        - Use semantic embeddings to improve meaning-based correction.
        - Apply smoothing techniques to improve probability estimation.
        """)

