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
    "📖 Dictionary Explorer"
])

# =============================
# TAB 1: SPELLING CORRECTION
# =============================
with tab1:

    # --------------------------------
    # PRELOADED EXAMPLE TEXTS (≥500 WORDS)
    # --------------------------------
    EXAMPLES = {
        "Clinical Report": """
The patient was admitted to the medical ward for further evaluation of chronic chest pain and shortness of breath. According to the clinical history, the patient has been experiecing intermittent chest discomfort for the past six months, which has gradually increased in frequncy and severity. The pain is described as a dull pressure sensation that radiates form the center of the chest to the left shoulder and upper arm. The patient reported that the pain usually occurs during physical exertion but sometimes appears at rest, especialy during periods of emotional stress.

Past medical history reveals a long-standing history of hypertension and type two diabetes mellitus. The patient admited to poor medication compliance due to financial constraints and lack of understanding regarding the importance of regular treatment. Blood presure recordings during admission were consistently elevated, with systolic readings ranging between 160 and 180 mmHg. Laboratory investigations showed elevated fasting blood glucose levels and mildly increased cholesterol. Their was no previous history of myocardial infarction or stroke documented in the medical records.

Physical examination revealed an overweight male patient in mild respiratory distress. Cardiovascular examination showed normal heart sounds with no audible murmurs, rubs, or gallops. Respiratory examination demonstrated reduced air entry at the lung bases bilateraly, with occasional crackles. The abdomen was soft and non tender, with no palpable organomegaly. Peripheral pulses were palpable and symmetrical, and there were no signs of peripheral edema.

An electrocardiogram was performed and showed non specific ST segment changes but no acute ischemic findings. Chest radiograph revealed mild cardiomegaly with increased pulmonary vascular markings. Based on the clinical presentation and investigation results, the working diagnosis was stable angina secondary to coronary artery disease. Further tests, including echocardiography and stress testing, were planned to confirm the diagnozed condition.

The patient was started on antiplatelet therapy, statins, and antihypertensive medications. Lifestyle modifications, including dietary changes, smoking cessation, and regular physical activity, were strongly advised. Education was provided to ensure the patient understands the importance of medication adherence and follow up appointments. With appropriate management, the prognosis is favorable, although long term outcomes depend largely on compliance and risk factor control.
""",

        "Business Report": """
This report analyses the recent performance of the organisation and evaluates the effectiveness of its current strategic initiatives. Over the past fiscal year, the company has experienced a steady increase in revenue, primarily driven by expansion into new markets and improved operational efficiency. However, several operational challanges were identified that may impact future growth if they are not addressed in a timely manner.

One of the major concerns highlighted by management is the inconsistency in supply chain coordination. Delays in raw material deliveries have resulted in production bottlenecks, which negatively affected customer satisfaction. In some cases, orders were delivered later then the promised deadlines, leading to complaints and loss of trust. These issues were further compounded by limited communication between procurement and production departments.

Financial analysis indicates that while gross profit margins have improved, operating expenses remain higher than industry benchmarks. This is largely attributed to rising administrative costs and inefficient resource allocation. The finance team noted that certain departments exceeded their allocated budgets without proper justification or approval. Their is also evidence to suggest that redundant processes exist, which could be streamlined to reduce costs.

Human resource management plays a critical role in organisational performance. Employee surveys revealed moderate job satisfaction levels, with concerns raised regarding workload distribution and career development opportunities. Many employees expressed that training programs were either insufficient or not aligned with their actual job requirements. As a result, staff turnover rates have increased slightly compared to the previous year.

To address these issues, several recommendations are proposed. First, the company should invest in an integrated supply chain management system to enhance coordination and transparency. Second, stricter budget controls should be implemented to ensure financial discipline. Third, human resource policies should be revised to emphasise continuous professional development and employee engagement. If these strategies are implemented effectively, the organisation is likely to achieve sustainable growth and maintain its competitive advantage in the market.
""",

        "NLP Research Article": """
Natural language processing has emerged as a critical area of research within artificial intelligence, with applications spanning healthcare, finance, education, and social media analysis. Recent advancements in machine learning and deep learning have significantly improved the accuracy of text based systems. Despite these advancements, several challanges remain, particularly in handling ambiguity, context, and noisy real world data.

One of the fundamental tasks in natural language processing is text preprocessing, which involves tokenization, normalization, and noise removal. Errors introduced during data collection, such as misspellings and grammatical inconsistencies, can adversely affect model performance. For example, real word spelling errors are often overlooked by traditional spell checkers because the incorrect word still exists in the dictionary. This highlights the importance of incorporating contextual information into language models.

Statistical language models, such as bigram and trigram models, estimate the probability of word sequences based on observed frequencies in a corpus. While these models are relatively simple, they provide valuable insights into local word dependencies. However, they often fail to capture long range semantic relationships. Neural based approaches, including word embeddings and transformer architectures, address some of these limitations by learning distributed representations of words.

In experimental evaluations, models that integrate both rule based and probabilistic techniques tend to perform better than those relying on a single approach. For instance, combining edit distance algorithms with language models allows systems to correct both non word and real word errors more effectively. Nevertheless, these hybrid systems require careful tuning to balance accuracy and computational efficiency.

Future research directions include the integration of syntactic and semantic knowledge, such as part of speech tagging and named entity recognition, to further enhance contextual understanding. Additionally, ethical considerations related to bias, privacy, and transparency must be addressed as natural language processing systems become increasingly pervasive. Continued research and interdisciplinary collaboration will be essential to overcome these challenges and fully realise the potential of intelligent language technologies.
"""
    }

    # --------------------------------
    # LOAD EXAMPLE BUTTON
    # --------------------------------
    selected_example = st.selectbox(
        "📌 Load an example text (optional):",
        ["(Choose an example)"] + list(EXAMPLES.keys())
    )

    if selected_example != "(Choose an example)":
        st.session_state.editor_text = EXAMPLES[selected_example]

    # --------------------------------
    # TEXT INPUT
    # --------------------------------
    if "editor_text" not in st.session_state:
        st.session_state.editor_text = ""

    input_text = st.text_area(
        "✍️ Enter text (max 500 characters):",
        value=st.session_state.editor_text,
        height=220,
        max_chars=500
    )

    tokens_input = re.findall(r"[a-zA-Z']+", input_text.lower())

    misspelled = []
    for i, word in enumerate(tokens_input):
        if word not in vocab:
            misspelled.append((i, word))

    # --------------------------------
    # HIGHLIGHT ERRORS
    # --------------------------------
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

    # --------------------------------
    # SIDEBAR CORRECTIONS
    # --------------------------------
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

                if st.sidebar.button(f"Apply '{word}'", key=f"apply_{word}_{idx}"):
                    if choice != "(keep original)":
                        st.session_state.editor_text = re.sub(
                            rf"\b{word}\b",
                            choice,
                            st.session_state.editor_text,
                            count=1
                        )
                    st.rerun()
            else:
                st.sidebar.warning("No suggestions found.")


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
# TAB 3: DICTIONARY EXPLORER
# =============================
with tab3:
    st.subheader("📖 Corpus Dictionary")

    search = st.text_input("Search for a word:")
    vocab_list = sorted(vocab)

    if search:
        vocab_list = [w for w in vocab_list if search.lower() in w]

    st.write(f"Showing {len(vocab_list)} words")
    st.write(vocab_list[:500])
