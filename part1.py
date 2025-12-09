import streamlit as st
import pandas as pd
from collections import Counter
import re

# --------------------------
#   SIMPLE NGRAM FUNCTIONS
# --------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_unigram_model(tokens):
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = {w: counts[w] / total for w in counts}
    return probs, counts

def generate_candidates(word, vocab):
    """Simple edit-distance-1 candidate generation."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    inserts = [L + c + R for L, R in splits for c in letters]
    substitutes = [L + c + R[1:] for L, R in splits if R for c in letters]

    all_edits = set(deletes + inserts + substitutes)
    return [w for w in all_edits if w in vocab]

def suggest(word, vocab, probs):
    if word in vocab:
        return []  # no suggestions needed

    cands = generate_candidates(word, vocab)
    if not cands:
        return []

    # Rank candidates by unigram probability
    ranked = sorted(cands, key=lambda w: probs.get(w, 0), reverse=True)
    return ranked[:5]


# --------------------------
#        STREAMLIT UI
# --------------------------

st.set_page_config(page_title="Custom Corpus Spell Checker", layout="wide")

st.title("📝 Custom Corpus Spell Checker (Streamlit Version)")

# Upload CSVs
st.sidebar.header("📁 Upload Your Corpus CSV Files")
csv1 = st.sidebar.file_uploader("Upload Dataset 1 (one column)", type=["csv"])
csv2 = st.sidebar.file_uploader("Upload Dataset 2 (one column)", type=["csv"])

if csv1 and csv2:
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # merge
    merged = pd.concat([df1, df2], ignore_index=True)

    # assume first column is text
    text_data = merged.iloc[:,0].astype(str)

    full_text = " ".join(text_data)
    tokens = tokenize(full_text)

    # Build vocab + unigram model
    probs, counts = build_unigram_model(tokens)
    vocab = set(counts.keys())

    # Display dataset stats
    st.sidebar.subheader("📊 Corpus Statistics")
    st.sidebar.write(f"Total rows merged: **{len(merged)}**")
    st.sidebar.write(f"Total words in dataset: **{len(tokens):,}**")
    st.sidebar.write(f"Unique words: **{len(vocab):,}**")

    st.divider()

    # ---------------------
    # TEXT EDITOR + CORRECTIONS
    # ---------------------
    st.subheader("✏️ Text Editor")
    user_text = st.text_area("Enter text to check:", height=250)

    if user_text.strip():
        user_tokens = user_text.split()

        # word selection
        st.sidebar.subheader("🔍 Correction Suggestions")

        selected_word = st.sidebar.selectbox(
            "Select a word to check:",
            user_tokens
        )

        suggestions = suggest(selected_word.lower(), vocab, probs)

        if suggestions:
            st.sidebar.write("Suggestions:")
            chosen_fix = st.sidebar.radio("Pick correction:", suggestions)
        else:
            chosen_fix = None
            st.sidebar.write("No suggestions found.")

        # APPLY button
        if st.sidebar.button("✅ APPLY CORRECTION"):
            if chosen_fix:
                updated_tokens = [
                    chosen_fix if w == selected_word else w for w in user_tokens
                ]
                corrected_text = " ".join(updated_tokens)

                st.success("Word updated!")
                st.text_area("Corrected Text:", corrected_text, height=250)
            else:
                st.warning("No correction selected.")
else:
    st.info("Please upload **both CSV files** to begin.")
