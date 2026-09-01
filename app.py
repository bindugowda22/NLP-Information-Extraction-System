import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import modules (we will create these next)
import streamlit as st
import pandas as pd
import spacy
import nltk
import os
from modules.relation_extractor import relation_extraction
from modules.event_extractor import event_extraction
from modules.temporal_order import temporal_ordering
from modules.graph_builder import build_graph  

with open(Path(__file__).resolve().parent / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from nltk import word_tokenize, pos_tag

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------- POS TAGGING ---------------- #

def pos_tagging(text):

    doc = nlp(text)

    spacy_data = []

    for token in doc:
        spacy_data.append({
            "Token": token.text,
            "POS": token.pos_,
            "Tag": token.tag_,
            "Dependency": token.dep_
        })

    spacy_df = pd.DataFrame(spacy_data)

    nltk_tokens = word_tokenize(text)

    nltk_tags = pos_tag(nltk_tokens)

    nltk_df = pd.DataFrame(
        nltk_tags,
        columns=["Token", "POS"]
    )

    return spacy_df, nltk_df


# ---------------- NER ---------------- #

def named_entity_recognition(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append({

            "Entity": ent.text,
            "Label": ent.label_,
            "Start": ent.start_char,
            "End": ent.end_char

        })

    entity_df = pd.DataFrame(entities)

    return entity_df

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Information Extraction System",
    page_icon="🧠",
    layout="wide"
)


def load_css():

    with open(Path(__file__).resolve().parent / "style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
# ---------------- TITLE ---------------- #

st.markdown("""
<div style="
background:linear-gradient(90deg,#4facfe,#00f2fe);
padding:25px;
border-radius:18px;
text-align:center;
color:white;
">

<h1>🧠 Information Extraction System</h1>

<h4>
POS Tagging • Named Entity Recognition • Relation Extraction • Event Extraction • Timeline • Knowledge Graph
</h4>

</div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ---------------- #
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
    width=80
)

st.sidebar.markdown("## NLP Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📄 Dataset",

        "📝 POS Tagging",

        "🏷️ Named Entity Recognition",

        "🔗 Relation Extraction",

        "📅 Event Extraction",

        "⏳ Timeline",

        "🌐 Knowledge Graph"

    ]

)
#---------------load dataset-------------#
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "train.csv"


@st.cache_data
def load_dataset():

    if not DATASET_PATH.exists():

        st.error(
            f"Dataset not found: {DATASET_PATH}"
        )

        return None

    try:

        df = pd.read_csv(
            DATASET_PATH
        )

        df.columns = df.columns.str.strip()

        df["Title"] = (
            df["Title"]
            .fillna("")
            .astype(str)
        )

        df["Description"] = (
            df["Description"]
            .fillna("")
            .astype(str)
        )

        df["Text"] = (
            df["Title"]
            + ". "
            + df["Description"]
        )

        return df

    except Exception as e:

        st.error(
            f"Dataset error: {e}"
        )

        return None


dataset = load_dataset()
# ============================================================
# PAGE CONTENT
# ============================================================

# Make sure there is always some text available
if "text" not in st.session_state:

    if dataset is not None and len(dataset) > 0:
        st.session_state["text"] = dataset.iloc[0]["Text"]
    else:
        st.session_state["text"] = ""


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="home-card">
        <h2> Welcome to the NLP Dashboard</h2>
        <p>
        This system performs information extraction from text using
        Natural Language Processing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📝 POS Tagging\n\nIdentify grammatical categories.")

    with col2:
        st.info("🏷️ Named Entity Recognition\n\nExtract people, places and organizations.")

    with col3:
        st.info("🔗 Relation Extraction\n\nFind relationships between entities.")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.info("📅 Event Extraction\n\nIdentify important events.")

    with col5:
        st.info("⏳ Timeline\n\nArrange events chronologically.")

    with col6:
        st.info("🌐 Knowledge Graph\n\nVisualize entity relationships.")


# ============================================================
# DATASET
# ============================================================

elif page == "📄 Dataset":

    st.header("📊 Dataset")

    if dataset is None:

        st.error("❌ Dataset could not be loaded.")

    else:

        st.success("✅ Dataset Loaded Successfully")

        # Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Rows",
                len(dataset)
            )

        with col2:
            st.metric(
                "Total Columns",
                len(dataset.columns)
            )

        with col3:
            st.metric(
                "Text Samples",
                len(dataset["Text"])
            )

        st.divider()

        # Dataset table
        st.subheader("📋 Dataset Preview")

        st.dataframe(
            dataset.head(10),
            use_container_width=True
        )

        st.divider()

        # Columns
        st.subheader("📌 Dataset Columns")

        st.write(
            list(dataset.columns)
        )

        st.divider()

        # Select row
        st.subheader("🔎 Select a Text")

        row_number = st.number_input(
            "Select row number",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1
        )

        selected_text = dataset.iloc[row_number]["Text"]

        st.text_area(
            "Selected Text",
            selected_text,
            height=200
        )

        if st.button(
            "🚀 Use This Text",
            key="dataset_use_text"
        ):

            st.session_state["text"] = selected_text

            st.success(
                "✅ Text selected! You can now use the NLP modules."
            )

        st.divider()

        # Download
        csv_data = dataset.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Dataset",
            csv_data,
            "dataset.csv",
            "text/csv"
        )


# ============================================================
# POS TAGGING
# ============================================================

elif page == "📝 POS Tagging":

    st.header("📝 Part of Speech Tagging")

    text = st.session_state.get(
        "text",
        ""
    )

    if not text:

        st.warning(
            "⚠️ No text available. Select a dataset row or enter custom text."
        )

    else:

        st.subheader("📄 Input Text")

        st.info(text)

        spacy_df, nltk_df = pos_tagging(text)

        st.subheader("🔹 spaCy POS Tags")

        st.dataframe(
            spacy_df,
            use_container_width=True
        )

        st.subheader("🔹 NLTK POS Tags")

        st.dataframe(
            nltk_df,
            use_container_width=True
        )


# ============================================================
# NAMED ENTITY RECOGNITION
# ============================================================

elif page == "🏷️ Named Entity Recognition":

    st.header("🏷️ Named Entity Recognition")

    text = st.session_state.get(
        "text",
        ""
    )

    if not text:

        st.warning(
            "⚠️ No text available. Select a dataset row or enter custom text."
        )

    else:

        st.subheader("📄 Input Text")

        st.info(text)

        entity_df = named_entity_recognition(text)

        st.subheader("🏷️ Extracted Entities")

        if entity_df.empty:

            st.warning(
                "No named entities found."
            )

        else:

            st.dataframe(
                entity_df,
                use_container_width=True
            )

            st.subheader("📊 Entity Frequency")

            st.bar_chart(
                entity_df["Label"].value_counts()
            )


# ============================================================
# RELATION EXTRACTION
# ============================================================

elif page == "🔗 Relation Extraction":

    st.header("🔗 Relation Extraction")

    text = st.session_state.get(
        "text",
        ""
    )

    if not text:

        st.warning(
            "⚠️ No text available."
        )

    else:

        st.subheader("📄 Input Text")

        st.info(text)

        relation_df = relation_extraction(text)

        st.subheader("🔗 Extracted Relations")

        if relation_df is None or relation_df.empty:

            st.warning(
                "No relations found."
            )

        else:

            st.dataframe(
                relation_df,
                use_container_width=True
            )


# ============================================================
# EVENT EXTRACTION
# ============================================================

elif page == "📅 Event Extraction":

    st.header("📅 Event Extraction")

    text = st.session_state.get(
        "text",
        ""
    )

    if not text:

        st.warning(
            "⚠️ No text available."
        )

    else:

        st.subheader("📄 Input Text")

        st.info(text)

        event_df = event_extraction(text)

        st.subheader("📅 Extracted Events")

        if event_df is None or event_df.empty:

            st.warning(
                "No events found."
            )

        else:

            st.dataframe(
                event_df,
                use_container_width=True
            )


# ============================================================
# TIMELINE
# ============================================================

elif page == "⏳ Timeline":

    st.header("⏳ Event Timeline")

    text = st.session_state.get(
        "text",
        ""
    )

    if not text:

        st.warning(
            "⚠️ No text available."
        )

    else:

        st.subheader("📄 Input Text")

        st.info(text)

        timeline_df = temporal_ordering(text)

        st.subheader("⏳ Temporal Order")

        if timeline_df is None or timeline_df.empty:

            st.warning(
                "No temporal information found."
            )

        else:

            st.dataframe(
                timeline_df,
                use_container_width=True
            )


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

elif page == "🌐 Knowledge Graph":

    st.header("🌐 Knowledge Graph")

    text = st.session_state.get(
        "text",
        ""
    )

    if not text:

        st.warning(
            "⚠️ No text available."
        )

    else:

        st.subheader("📄 Input Text")

        st.info(text)

        try:

            graph = build_graph(text)

            st.subheader("🌐 Knowledge Graph")

            if graph is not None:

                st.write(graph)

            else:

                st.warning(
                    "Knowledge graph could not be generated."
                )

        except Exception as e:

            st.error(
                f"Knowledge graph error: {e}"
            )

# ============================================================
# CUSTOM TEXT
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown("### ✏️ Custom Input")

user_text = st.sidebar.text_area(
    "Enter your text:",
    height=150,
    placeholder="Example: Apple was founded by Steve Jobs in California."
)

if st.sidebar.button("🚀 Use Custom Text"):

    if user_text.strip():

        st.session_state["text"] = user_text.strip()

        st.sidebar.success("✅ Text loaded!")

    else:

        st.sidebar.warning(
            "Please enter some text."
        )

# ---------------- FOOTER ---------------- #

st.sidebar.markdown("---")
st.sidebar.info("Developed using Streamlit + spaCy + NLTK")

