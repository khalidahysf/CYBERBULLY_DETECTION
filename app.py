# ============================================================
# STREAMLIT APP: PENGESANAN BULI SIBER
# Final Model: RoBERTa Bahasa
# ============================================================

import os
import re
import json
import html
import textwrap

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Pengesanan Buli Siber",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }

    .title-container {
        text-align: center;
        padding-bottom: 10px;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 15px;
        color: #6b7280;
        margin-top: 0px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #1f2937;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .info-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .small-label {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #1f2937;
        text-align: center;
    }

    .metric-label {
        font-size: 13px;
        color: #6b7280;
        text-align: center;
    }

    .result-safe {
        background-color: #d1fae5;
        border: 2px solid #10b981;
        color: #047857;
        padding: 18px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .result-danger {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
        color: #b91c1c;
        padding: 18px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .result-desc {
        font-size: 14px;
        font-weight: 400;
        margin-top: 8px;
    }

    .prob-box {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .highlight-word {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #ef4444;
        border-radius: 6px;
        padding: 3px 6px;
        margin: 2px;
        display: inline-block;
        font-weight: 600;
    }

    .normal-word {
        margin: 2px;
        display: inline-block;
    }

    .note-box {
        background-color: #f9fafb;
        border-left: 5px solid #3b82f6;
        padding: 14px;
        border-radius: 8px;
        color: #374151;
        font-size: 14px;
    }

    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #2563eb;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SETTINGS: FINAL RoBERTa Bahasa MODEL
# ============================================================
MODEL_PATH = "roberta_bahasa_final_model"
MODEL_NAME = "RoBERTa Bahasa"

# Final test result after error analysis + calibration threshold tuning
ACCURACY = "82.64%"
BALANCED_ACCURACY = "82.75%"
PRECISION_BULI = "86.03%"
RECALL_BULI = "79.12%"
F1_BULI = "82.43%"
MACRO_F1 = "82.64%"
DEFAULT_THRESHOLD = 0.11
MAX_LENGTH = 128


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_and_threshold():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    threshold_path = os.path.join(MODEL_PATH, "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
        threshold = float(threshold_data.get("threshold", DEFAULT_THRESHOLD))
    else:
        threshold = DEFAULT_THRESHOLD

    return tokenizer, model, threshold


try:
    tokenizer, model, threshold = load_model_and_threshold()
except Exception as e:
    st.error(
        "Model gagal dimuatkan. Pastikan folder 'roberta_bahasa_final_model' "
        "berada dalam folder yang sama dengan app.py."
    )
    st.exception(e)
    st.stop()


# ============================================================
# SESSION STATE FOR SIMPLE STATISTICS
# ============================================================
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0

if "total_cyberbullying" not in st.session_state:
    st.session_state.total_cyberbullying = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def predict_text(text):
    """
    Predict text using final RoBERTa Bahasa model.
    Important: final label is decided using calibrated threshold, not highest probability.
    Label 0 = Bukan Buli
    Label 1 = Buli Siber
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    non_cyber_prob = float(probabilities[0])
    cyber_prob = float(probabilities[1])

    if cyber_prob >= threshold:
        prediction = "Buli Siber"
    else:
        prediction = "Bukan Buli Siber"

    # This is displayed as model score, not absolute certainty.
    decision_score = cyber_prob if prediction == "Buli Siber" else non_cyber_prob

    return prediction, decision_score, cyber_prob, non_cyber_prob


def count_words(text):
    return len(text.split())


# ============================================================
# SHAP EXPLAINABILITY FUNCTIONS
# ============================================================
def model_predict_for_shap(texts):
    """
    Used by SHAP.
    Returns probabilities in fixed order:
    [Bukan Buli probability, Buli Siber probability]
    """
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    return probabilities


@st.cache_resource
def load_shap_explainer():
    import shap

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        model_predict_for_shap,
        masker,
        output_names=["Bukan Buli Siber", "Buli Siber"]
    )
    return explainer


def clean_shap_token(token):
    token_text = str(token)
    token_text = token_text.replace("Ġ", " ").replace("▁", " ")
    token_text = token_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    return token_text.strip()


def get_shap_words(text, target_class_index=1, max_words=10):
    """
    target_class_index:
    0 = Bukan Buli Siber
    1 = Buli Siber
    """
    explainer = load_shap_explainer()

    shap_values = explainer(
        [text],
        max_evals=100,
        batch_size=8
    )

    tokens = shap_values.data[0]
    values = shap_values.values[0]

    if len(values.shape) == 2:
        class_values = values[:, target_class_index]
    else:
        class_values = values

    word_scores = []

    for token, score in zip(tokens, class_values):
        token_text = clean_shap_token(token)

        if token_text == "" or token_text in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        word_scores.append((token_text, float(score)))

    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)
    return word_scores[:max_words]


def display_shap_explanation(text):
    """
    Display SHAP explanation for Buli Siber class.
    Positive SHAP value = pushes prediction towards Buli Siber.
    Negative SHAP value = pushes prediction away from Buli Siber.
    """
    target_class_index = 1
    explanation_label = "Perkataan yang mempengaruhi kecenderungan model terhadap kelas Buli Siber"

    with st.spinner("SHAP sedang menjana penjelasan. Ini mungkin mengambil sedikit masa..."):
        shap_words = get_shap_words(text, target_class_index=target_class_index)

    token_html = ""

    if len(shap_words) == 0:
        token_html = "<span style='color:#6b7280;font-size:13px;'>Tiada token penting dapat dipaparkan oleh SHAP untuk teks ini.</span>"
    else:
        for word, score in shap_words:
            if score >= 0:
                color = "#fee2e2"
                text_color = "#991b1b"
                border = "#ef4444"
            else:
                color = "#d1fae5"
                text_color = "#047857"
                border = "#10b981"

            token_html += (
                f"<span style='"
                f"background-color:{color};"
                f"color:{text_color};"
                f"border:1px solid {border};"
                f"border-radius:6px;"
                f"padding:6px 9px;"
                f"margin:4px;"
                f"display:inline-block;"
                f"font-weight:600;"
                f"white-space:nowrap;"
                f"'>"
                f"{html.escape(word)} ({score:.4f})"
                f"</span>"
            )

    with st.container(border=True):
        st.markdown(f"**{explanation_label}**")

        st.markdown(
            token_html,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <span style="font-size:13px;color:#6b7280;line-height:1.7;">
            Nilai SHAP positif menunjukkan perkataan tersebut meningkatkan kecenderungan model kepada kelas Buli Siber.
            Nilai SHAP negatif menunjukkan perkataan tersebut mengurangkan kecenderungan model kepada kelas Buli Siber.
            SHAP digunakan untuk menerangkan ramalan model, bukan untuk membuat ramalan baharu.
            </span>
            """,
            unsafe_allow_html=True
        )


def highlight_important_words(text):
    """
    Simple keyword-based fallback highlighting.
    SHAP is more reliable because it uses the model explanation.
    """
    bullying_keywords = [
        "bodoh", "bangang", "babi", "sial", "anjing", "hodoh",
        "gemuk", "lembab", "loser", "mati", "mampus", "mampos",
        "hina", "buruk", "kerek", "poyo", "kedut", "londeh", "lonteh",
        "busuk", "bongok", "gila", "kepala", "puqi", "palat", "hawau",
        "kepam", "sakai", "barua" , "hauk" , "gampang"
    ]

    words = text.split()
    highlighted = []
    found_words = []

    for word in words:
        clean_word = re.sub(r"[^a-zA-ZÀ-ÿ0-9]", "", word).lower()
        escaped_word = html.escape(word)

        if clean_word in bullying_keywords:
            highlighted.append(f"<span class='highlight-word'>{escaped_word}</span>")
            found_words.append(clean_word)
        else:
            highlighted.append(f"<span class='normal-word'>{escaped_word}</span>")

    return " ".join(highlighted), list(set(found_words))

def detect_risky_keywords(text):
    text_lower = text.lower()

    target_words = ["kau", "ko", "awak", "engkau", "hang", "dia"]
    insult_words = [
        "gemuk", "hodoh", "buruk", "busuk", "bodoh", "bangang",
        "mampus", "mati", "benak", "sakai", "kepam", "obes",
        "tak jaga diri", "babi", "sial"
    ]

    has_target = any(word in text_lower for word in target_words)
    has_insult = any(word in text_lower for word in insult_words)

    return has_target and has_insult


def plot_probability_chart(cyber_prob, non_cyber_prob):
    labels = ["Buli Siber", "Bukan Buli Siber"]
    values = [cyber_prob * 100, non_cyber_prob * 100]
    colors = ["#dc2626", "#10b981"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.45)

    ax.axhline(threshold * 100, linestyle="--", linewidth=1.5, color="#374151")
    ax.text(
        0.5,
        threshold * 100 + 2,
        f"Threshold Buli Siber = {threshold * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#374151"
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Peratus (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, bar in enumerate(bars):
        value = values[i]
        height = bar.get_height()

        if height > 15:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold"
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 3,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                color=colors[i],
                fontsize=12,
                fontweight="bold"
            )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ Maklumat Model")
    st.markdown(f"""
    <div class="info-card">
        <b>Model:</b> {MODEL_NAME}<br><br>
        <span class="small-label">Accuracy:</span> {ACCURACY}<br>
        <span class="small-label">Balanced Accuracy:</span> {BALANCED_ACCURACY}<br>
        <span class="small-label">Precision Buli:</span> {PRECISION_BULI}<br>
        <span class="small-label">Recall Buli:</span> {RECALL_BULI}<br>
        <span class="small-label">F1 Buli:</span> {F1_BULI}<br>
        <span class="small-label">Macro F1:</span> {MACRO_F1}<br>
        <span class="small-label">Threshold:</span> {threshold:.2f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Statistik")
    st.markdown(f"""
    <div class="info-card">
        <span class="small-label">Teks Dianalisis Hari Ini</span><br>
        <b style="font-size:24px;">{st.session_state.total_analyzed}</b><br><br>
        <span class="small-label">Buli Siber Dikesan</span><br>
        <b style="font-size:24px;">{st.session_state.total_cyberbullying}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Tentang")
    st.markdown("""
    <div class="info-card">
        Sistem ini menggunakan model RoBERTa Bahasa yang dilatih untuk mengesan kandungan buli siber dalam teks Bahasa Melayu di media sosial.
        Keputusan dibuat menggunakan threshold kalibrasi model.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN TITLE
# ============================================================
st.markdown("""
<div class="title-container">
    <div class="main-title">🛡️ Pengesanan Buli Siber</div>
    <div class="subtitle">Berasaskan Pembelajaran Mendalam (RoBERTa Bahasa)</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔍 Analisis Teks", "📈 Metrik Model", "ℹ️ Panduan"])


# ============================================================
# TAB 1: TEXT ANALYSIS
# ============================================================
with tab1:
    st.markdown('<div class="section-title">Masukkan Teks untuk Analisis</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        label="",
        placeholder="Contoh: kau ni memang bodoh dan menyusahkan",
        height=170
    )

    st.caption("💡 Tip: Masukkan teks komen atau post media sosial dalam Bahasa Melayu untuk dianalisis.")

    use_shap = st.checkbox("Paparkan Penjelasan SHAP", value=False)
    analyze_button = st.button("🔍 Analisis Teks")

    if analyze_button:
        if user_text.strip() == "":
            st.warning("Sila masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Model sedang menganalisis teks..."):
               prediction, decision_score, cyber_prob, non_cyber_prob = predict_text(user_text)
               risky_warning = detect_risky_keywords(user_text)

            st.session_state.total_analyzed += 1

            if prediction == "Buli Siber":
                st.session_state.total_cyberbullying += 1

            st.markdown("---")
            st.markdown('<div class="section-title">📋 Hasil Analisis</div>', unsafe_allow_html=True)

            if prediction == "Buli Siber":
                st.markdown("""
                <div class="result-danger">
                    ⚠️ BULI SIBER DIKESAN
                    <div class="result-desc">Teks mempunyai skor Buli Siber yang melepasi threshold model.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-safe">
                    ✅ BUKAN BULI SIBER
                    <div class="result-desc">Teks tidak melepasi threshold Buli Siber model.</div>
                </div>
                """, unsafe_allow_html=True)

            if prediction == "Bukan Buli Siber" and risky_warning:
                st.warning(
                  "⚠️ Model mengklasifikasikan teks ini sebagai Bukan Buli Siber, "
                  "tetapi teks mengandungi perkataan berisiko yang mungkin menunjukkan "
                  "serangan personal, body shaming atau penghinaan. Semakan manusia disarankan."
            )

            # Explanation note for calibrated threshold decision
            if prediction == "Buli Siber" and non_cyber_prob > cyber_prob:
               st.info(
                 f"ℹ️ Nota: Skor Bukan Buli Siber lebih tinggi ({non_cyber_prob * 100:.1f}%), "
                 f"tetapi skor Buli Siber ({cyber_prob * 100:.1f}%) telah melepasi threshold model "
                 f"({threshold * 100:.1f}%). Oleh itu, sistem mengklasifikasikan teks ini sebagai "
                 f"Buli Siber berisiko rendah/sederhana dan semakan manusia disarankan."
            )

            elif prediction == "Bukan Buli Siber":
               st.info(
                 f"ℹ️ Nota: Skor Buli Siber ialah {cyber_prob * 100:.1f}%, iaitu lebih rendah daripada "
                 f"threshold model ({threshold * 100:.1f}%). Oleh itu, sistem mengklasifikasikan teks ini "
                 f"sebagai Bukan Buli Siber dan semakan manusia disarankan."
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <div class="metric-label">Skor Buli Siber</div>
                    <div class="metric-number">{cyber_prob * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <div class="metric-label">Threshold Model</div>
                    <div class="metric-number">{threshold * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="info-card">
                    <div class="metric-label">Klasifikasi</div>
                    <div class="metric-number" style="font-size:24px;">{prediction}</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="info-card">
                    <div class="metric-label">Bilangan Perkataan</div>
                    <div class="metric-number">{count_words(user_text)}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">📊 Skor Model dan Threshold</div>', unsafe_allow_html=True)

            with st.container(border=True):
                fig = plot_probability_chart(cyber_prob, non_cyber_prob)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown('<div class="section-title">🔍 Perkataan Penting yang Dikenalpasti</div>', unsafe_allow_html=True)

            if use_shap:
                try:
                    display_shap_explanation(user_text)
                except Exception as e:
                    st.error("SHAP gagal dijana untuk teks ini.")
                    st.info("Sistem masih boleh digunakan untuk ramalan model. Cuba pendekkan teks atau jalankan semula SHAP.")
                    st.exception(e)
            else:
                highlighted_text, found_words = highlight_important_words(user_text)

                if len(found_words) > 0:
                    st.markdown(f"""
                    <div class="info-card">
                        {highlighted_text}
                        <br><br>
                        <span class="small-label">
                            Paparan ini berdasarkan kata kunci ringkas. Tandakan pilihan SHAP untuk penjelasan berdasarkan model RoBERTa Bahasa.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-card">
                        ✅ Tiada perkataan yang menunjukkan unsur buli siber yang kuat berdasarkan senarai kata kunci ringkas.
                        <br><br>
                        <span class="small-label">
                            Tandakan pilihan SHAP untuk melihat penjelasan berdasarkan model RoBERTa Bahasa.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================
# TAB 2: MODEL METRICS
# ============================================================
with tab2:
    st.markdown('<div class="section-title">📈 Metrik Prestasi Model</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-number">{ACCURACY}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">Balanced Accuracy</div>
            <div class="metric-number">{BALANCED_ACCURACY}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">F1 Buli</div>
            <div class="metric-number">{F1_BULI}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">Macro F1</div>
            <div class="metric-number">{MACRO_F1}</div>
        </div>
        """, unsafe_allow_html=True)

    col5, col6, col7 = st.columns(3)

    with col5:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">Precision Buli</div>
            <div class="metric-number">{PRECISION_BULI}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">Recall Buli</div>
            <div class="metric-number">{RECALL_BULI}</div>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">Threshold</div>
            <div class="metric-number">{threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note-box">
        <b>Penerangan:</b><br>
        Accuracy menunjukkan peratus ramalan yang betul secara keseluruhan.
        Balanced Accuracy digunakan untuk menilai prestasi kedua-dua kelas secara lebih seimbang.
        Precision Buli menunjukkan ketepatan model apabila mengklasifikasikan sesuatu teks sebagai Buli Siber.
        Recall Buli menunjukkan keupayaan model mengesan teks Buli Siber sebenar.
        Macro F1 digunakan kerana ia mengambil kira prestasi kedua-dua kelas secara adil.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 3: GUIDE
# ============================================================
with tab3:
    st.markdown('<div class="section-title">ℹ️ Panduan Penggunaan Sistem</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <b>Langkah 1:</b> Masukkan teks komen atau hantaran media sosial dalam Bahasa Melayu.<br><br>
        <b>Langkah 2:</b> Klik butang <b>Analisis Teks</b>.<br><br>
        <b>Langkah 3:</b> Sistem akan memaparkan keputusan sama ada teks tersebut diklasifikasikan sebagai 
        <b>Buli Siber</b> atau <b>Bukan Buli Siber</b>.<br><br>
        <b>Langkah 4:</b> Lihat skor Buli Siber, threshold model, skor kebarangkalian dan perkataan penting.
        Tandakan pilihan SHAP jika ingin melihat penjelasan model.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note-box">
        <b>Nota:</b> Sistem ini bertujuan sebagai alat sokongan analisis.
        Keputusan model perlu ditafsir dengan berhati-hati kerana konteks bahasa, sindiran, slang dan maksud tersirat boleh mempengaruhi keputusan.
        SHAP digunakan untuk membantu menerangkan ramalan model, bukan untuk menggantikan penilaian manusia.
    </div>
    """, unsafe_allow_html=True)
