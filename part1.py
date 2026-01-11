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
    df = df.sample(frac=0.1, random_state=42)  # ~200k tokens

    df["cleaned_transcription"] = df["transcription"].apply(clean_punctuation_simple)

    full_text = " ".join(df["cleaned_transcription"].astype(str))
    tokens = re.findall(r"[a-zA-Z']+", full_text)

    return tokens, df

tokens, df = load_corpus()
total_words = len(tokens)

word_freq = Counter(tokens)
vocab = set(tokens)

# -----------------------------
# 2. BIGRAM MODEL
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
    candidates = [(word, 0)]

    for vocab_word in vocab:
        if vocab_word == word:
            continue
        if abs(len(vocab_word) - len(word)) <= 2:
            dist = edit_distance(word, vocab_word)
            if dist <= 2:
                candidates.append((vocab_word, dist))

    return candidates


# -----------------------------
# 5. RANKING WITH SCORES
# -----------------------------
def rank_candidates_with_scores(word, prev_word=None):
    scored = []

    for cand, dist in generate_candidates(word):
        score = (
            -dist +
            (word_freq[cand] / total_words) +
            (bigram_score(prev_word, cand) * 0.001)
        )
        scored.append((cand, score))

    return sorted(scored, key=lambda x: -x[1])


# -----------------------------
# 6. STREAMLIT UI
# -----------------------------
REAL_WORD_THRESHOLD = 0.3

st.set_page_config(page_title="NLP Spelling Correction", layout="wide")
st.title("🧠 NLP Assignment Part 1 – Spelling Correction System")
st.markdown(f"### 📊 Total Words in Corpus: **{total_words:,}**")

tab1, tab2, tab3 = st.tabs(["✍️ Spelling Correction", "📄 Dataset Examples", "📖 System Workflow"])

# =============================
# TAB 1
# =============================
with tab1:

    if "editor_text" not in st.session_state:
        st.session_state.editor_text = ""

    if "replacements" not in st.session_state:
        st.session_state.replacements = {}
    # -----------------------------
    # SAMPLE TEXTS
    # -----------------------------
    sample_texts = {
        "": "",
        "Clinical Report Sample": (
            "The patient was admitd to medical ward for chest pain and shortness of brath. "
            "He has had intermittet painn form september 2025, which worsens with exertion or stress. "
            "Pain dull and radiates to left arm. On exam, nasal mucos was erythematus. "
            "PMH: hypertension and type 2 diabetes. Patient admits poor med compliance due 2 financial issues."
        ),
        "Business Report Sample": (
            "This report analyses the recent performance of the organisation and evaluates the effectiveness "
            "of its current strategic initiatives. Over the past fiscal year, the company has experienced a "
            "steady increase in revenue. However, several operational challanges were identified that may "
            "impact future growth if they are not addressed in a timely manner. Delays in raw material "
            "deliveries resulted in production bottlenecks, and some orders were delivered later then promised. "
            "Their is evidence that communication between departments needs improvement."
        ),
        "NLP Research Sample": (
            "Natural language processing has emerged as a critical area of research within artificial "
            "intelligence. Despite recent advancements, challanges remain in handling ambiguity and context. "
            "Real word spelling errors are often overlooked by traditional spell checkers because the "
            "incorrect word still exists in the dictionary. This highlights the importance of incorporating "
            "contextual information such as bigram language models."
        ),
    }
    
    # -----------------------------
    # SAMPLE DROPDOWN
    # -----------------------------
    selected_sample = st.selectbox(
        "📌 Choose a sample text to load:",
        options=list(sample_texts.keys()),
        index=0
    )
    
    if selected_sample and st.button("Use selected sample"):
        st.session_state.editor_text = sample_texts[selected_sample]
        st.session_state.replacements = {}  # reset pending edits
        st.rerun()
    
    input_text = st.text_area(
        "Enter text (max 500 characters):",
        value=st.session_state.editor_text,
        height=250,
        max_chars=500
    )

    tokens_input = re.findall(r"[a-zA-Z']+", input_text.lower())
    misspelled = []

    for i, word in enumerate(tokens_input):
        prev_word = tokens_input[i - 1] if i > 0 else None
        ranked = rank_candidates_with_scores(word, prev_word)

        if not ranked:
            continue

        best_word, best_score = ranked[0]
        orig_score = next(s for w, s in ranked if w == word)

        if (
            word not in vocab or
            (best_word != word and (best_score - orig_score) > REAL_WORD_THRESHOLD)
        ):
            misspelled.append((i, word, ranked))

    # -----------------------------
    # Highlight errors
    # -----------------------------
    highlighted = input_text
    for _, w, _ in misspelled:
        highlighted = re.sub(
            rf"\b{w}\b",
            f"**:red[{w}]**",
            highlighted,
            flags=re.IGNORECASE
        )

    st.markdown("### 🔍 Highlighted Spelling Errors")
    st.markdown(highlighted)

    # -----------------------------
    # Sidebar selections
    # -----------------------------
    st.sidebar.title("🔧 Corrections")

    for idx, word, ranked in misspelled:

        options = [w for w, _ in ranked[:5] if w != word]

        if not options:
            options = ["(keep original)"]
        else:
            options = options + ["(keep original)"]

        choice = st.sidebar.radio(
            f"Replace '{word}' with:",
            options=options,
            key=f"{word}_{idx}"
        )

        st.session_state.replacements[idx] = choice

    # -----------------------------
    # APPLY ALL CHANGES (BOTTOM)
    # -----------------------------
    if st.button("✅ Apply All Changes"):

        updated_text = input_text

        # Apply from last index to first to avoid offset issues
        for idx, word, _ in sorted(misspelled, reverse=True):
            replacement = st.session_state.replacements.get(idx)

            if replacement and replacement != "(keep original)":
                updated_text = re.sub(
                    rf"\b{word}\b",
                    replacement,
                    updated_text,
                    count=1
                )

        st.session_state.editor_text = updated_text
        st.session_state.replacements = {}
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
# TAB 3: SYSTEM WORKFLOW
# =============================
with tab3:
    st.subheader("🧩 System Workflow")

    st.markdown("""
    This section describes the internal workflow of the spelling correction
    system and explains how the corpus is processed and used during correction.
    The system is designed around a **medical-domain corpus** and applies
    probabilistic techniques to correct spelling errors.
    """)

    with st.expander("Part 1 – Corpus Preparation"):
        st.markdown("""
        - A medical-domain corpus is loaded from an external dataset.
        - Raw text is cleaned by removing unnecessary punctuation.
        - All text is converted to lowercase.
        - The corpus is tokenised into individual words.
        """)
        
    with st.expander("Part 2 – Corpus Processing"):
        st.markdown("""
        - A vocabulary set is created from all unique tokens.
        - Word frequency counts are calculated.
        - Bigram frequency counts are generated from consecutive word pairs.
        - These processed components are stored for later use during correction.
        """)

    with st.expander("Part 3 – User Input Processing"):
        st.markdown("""
        - User input is limited to 500 characters.
        - Input text is tokenised using the same rules as the corpus.
        - Each word is analysed together with its previous word.
        """)

    with st.expander("Part 4 – Spelling Error Detection"):
        st.markdown("""
        - Every input word is evaluated using the same scoring mechanism.
        - Candidate corrections are generated for all words, regardless of
          whether the word exists in the vocabulary.
        - A confidence threshold of 0.3 is applied to compare the best candidate score
          against the original word score.
        - A word is flagged only if a candidate exceeds the original word score
          by the defined threshold.
        """)

    with st.expander("Part 5 – Candidate Generation"):
        st.markdown("""
        - Correction candidates are generated using Minimum Edit Distance.
        - Only candidates with small edit distances are considered.
        - All candidates must exist in the corpus vocabulary.
        """)

    with st.expander("Part 6 – Candidate Ranking"):
        st.markdown("""
        - Candidates are scored using:
            - Edit distance
            - Word frequency probability
            - Bigram context score
        - The highest scoring candidate is suggested as the correction.
        """)

    with st.expander("Part 7 – Interactive Correction"):
        st.markdown("""
        - Misspelled words are highlighted in the editor.
        - Up to five ranked correction options are shown.
        - Users may select a correction or keep the original word.
        - All selected changes are applied in a single action.
        """)

    # -----------------------------
    # BENEFITS
    # -----------------------------
    with st.expander("🏥 Benefits of the Current System"):
        st.markdown("""
        - Medical terminology is recognised as valid words.
        - Contextual bigram information improves correction accuracy.
        - The system supports both non-word and real-word errors.
        - The architecture can be adapted to other domains by changing the corpus.
        """)

    # -----------------------------
    # LIMITATIONS
    # -----------------------------
    with st.expander("⚠️ System Limitations"):
        st.markdown("""
        - Performance is optimised for medical text only.
        - Business and financial terms may not be recognised.
        - Acronyms and abbreviations are not explicitly handled.
        - Grammar and sentence structure are not modelled.
        - Only bigram context is used; long-range dependencies are ignored.
        """)

    # -----------------------------
    # FUTURE IMPROVEMENTS
    # -----------------------------
    with st.expander("🚀 Future Improvements"):
        st.markdown("""
        - Integrate Part-of-Speech (POS) tagging for grammatical awareness.
        - Extend context modelling beyond bigrams.
        - Add acronym and abbreviation handling.
        - Incorporate semantic representations for improved accuracy.
        """)
