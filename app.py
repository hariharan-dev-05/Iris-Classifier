"""
Iris Species Classifier - Streamlit UI
Loads `iris.pkl` (a dict with 'model' and 'features', saved via joblib)
and predicts the species from four flower measurements.

Run:  streamlit run app.py
"""

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "iris.pkl"
CSV_PATH = Path(__file__).parent / "IRIS.csv"  # optional, used for the chart
TEST_ACCURACY = 0.9667  # from the notebook (QDA, 80/20 split, random_state=42)

st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABELS = {
    "sepal_length": "Sepal length",
    "sepal_width": "Sepal width",
    "petal_length": "Petal length",
    "petal_width": "Petal width",
}
# (min, max, default) for each slider - based on the dataset's describe() output
RANGES = {
    "sepal_length": (4.0, 8.0, 5.1),
    "sepal_width": (2.0, 4.5, 3.5),
    "petal_length": (1.0, 7.0, 1.4),
    "petal_width": (0.1, 2.6, 0.2),
}
# Min/max actually seen in the training data (used for the out-of-range warning)
TRAIN_MIN_MAX = {
    "sepal_length": (4.3, 7.9),
    "sepal_width": (2.0, 4.4),
    "petal_length": (1.0, 6.9),
    "petal_width": (0.1, 2.5),
}

SPECIES = {
    "Iris-setosa": {
        "emoji": "🌸",
        "color": "#14b8a6",
        "blurb": "Small, compact flower with short, broad petals. "
        "Easily the most distinct of the three species.",
    },
    "Iris-versicolor": {
        "emoji": "🌼",
        "color": "#f59e0b",
        "blurb": "Medium-sized flower with mid-length petals. "
        "Sits between setosa and virginica in most measurements.",
    },
    "Iris-virginica": {
        "emoji": "🌺",
        "color": "#8b5cf6",
        "blurb": "The largest of the three, with long, wide petals "
        "and longer sepals.",
    },
}
PRESETS = {
    "🌸 Setosa": dict(sepal_length=5.0, sepal_width=3.4, petal_length=1.5, petal_width=0.2),
    "🌼 Versicolor": dict(sepal_length=5.9, sepal_width=2.8, petal_length=4.3, petal_width=1.3),
    "🌺 Virginica": dict(sepal_length=6.7, sepal_width=3.0, petal_length=5.6, petal_width=2.1),
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
  header[data-testid="stHeader"] {background: transparent;}

  .hero {
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.4rem;
    background: linear-gradient(120deg, #0f766e 0%, #6d28d9 100%);
    color: #fff;
  }
  .hero h1 {margin: 0; font-size: 2rem; font-weight: 750; color: #fff; padding: 0;}
  .hero p  {margin: .35rem 0 0 0; opacity: .88; font-size: 1.02rem;}

  .section-title {font-size: 1.1rem; font-weight: 650; margin: 0 0 .2rem 0;}
  .section-sub   {font-size: .88rem; opacity: .65; margin-bottom: .8rem;}

  .result-card {
    border-radius: 18px; padding: 1.5rem 1.6rem; color: #fff;
    box-shadow: 0 8px 24px rgba(0,0,0,.15);
  }
  .result-card .tag   {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; opacity: .85;}
  .result-card .name  {font-size: 2rem; font-weight: 800; line-height: 1.15; margin: .25rem 0 .1rem 0;}
  .result-card .conf  {font-size: 1.05rem; font-weight: 600; opacity: .95;}
  .result-card .blurb {font-size: .92rem; opacity: .9; margin-top: .7rem;}
  .result-card .emoji {font-size: 2.6rem; float: right; line-height: 1;}

  .prob-row   {margin: .55rem 0;}
  .prob-label {display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: 4px;}
  .prob-track {height: 10px; border-radius: 99px; background: rgba(128,128,128,.2); overflow: hidden;}
  .prob-fill  {height: 100%; border-radius: 99px;}

  .stat-pill {
    display: inline-block; padding: .25rem .7rem; border-radius: 99px; margin: 0 .35rem .35rem 0;
    font-size: .8rem; background: rgba(128,128,128,.15);
  }
  div[data-testid="stButton"] button {border-radius: 10px; font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    """Return (model, feature_names). Works with a bundle dict or a bare model."""
    obj = joblib.load(path)
    model = obj["model"] if isinstance(obj, dict) else obj

    # Prefer the names the model was actually fitted with. The notebook's saved
    # 'features' list also contains the target column ('species'), so don't
    # trust it blindly.
    if hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
    elif isinstance(obj, dict) and "features" in obj:
        feats = [f for f in obj["features"] if f != "species"]
    else:
        feats = FEATURES
    return model, feats


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Dataset used only for the context chart. Falls back to sklearn's copy."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = FEATURES + ["target"]
    names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    df["species"] = df.pop("target").map(dict(enumerate(names)))
    return df


if not MODEL_PATH.exists():
    st.error(f"Could not find **{MODEL_PATH.name}** next to app.py. "
             "Place your saved model in the same folder and reload.")
    st.stop()

model, feature_names = load_model(MODEL_PATH)
ref_df = load_reference_data()
classes = list(model.classes_)

# ----------------------------------------------------------------------------
# Session state / callbacks
# ----------------------------------------------------------------------------
for f in FEATURES:
    st.session_state.setdefault(f, RANGES[f][2])


def apply_preset(values: dict):
    for k, v in values.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌸 Iris Classifier")
    st.caption("Predict the species of an iris flower from its measurements.")
    st.divider()

    st.markdown("**Quick examples**")
    for label, vals in PRESETS.items():
        st.button(label, key=f"preset_{label}", width="stretch",
                  on_click=apply_preset, args=(vals,))
    st.button("↺ Reset to default", width="stretch",
              on_click=apply_preset,
              args=({f: RANGES[f][2] for f in FEATURES},))

    st.divider()
    st.markdown("**About the model**")
    st.markdown(
        f"""
<span class="stat-pill">Quadratic Discriminant Analysis</span>
<span class="stat-pill">Test accuracy {TEST_ACCURACY:.1%}</span>
<span class="stat-pill">{len(classes)} classes</span>
<span class="stat-pill">150 samples</span>
""",
        unsafe_allow_html=True,
    )
    st.caption("Trained on the classic Iris dataset (80/20 train-test split). "
               "All measurements are in centimetres.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🌸 Iris Species Classifier</h1>
  <p>Adjust the sliders to describe a flower — the model predicts its species instantly.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Inputs + Prediction
# ----------------------------------------------------------------------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">📏 Flower measurements</div>'
                '<div class="section-sub">Values in centimetres</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        values = {}
        for i, f in enumerate(FEATURES):
            lo, hi, _ = RANGES[f]
            with (c1 if i < 2 else c2):
                values[f] = st.slider(LABELS[f], lo, hi, step=0.1,
                                      format="%.1f cm", key=f)

# Build the input in the exact column order the model was trained on
X_input = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
proba = model.predict_proba(X_input)[0]
pred_idx = int(np.argmax(proba))
pred = classes[pred_idx]
conf = float(proba[pred_idx])
info = SPECIES.get(pred, {"emoji": "🌷", "color": "#6366f1", "blurb": ""})

with right:
    st.markdown('<div class="section-title">🔮 Prediction</div>'
                '<div class="section-sub">Updates live as you change the inputs</div>',
                unsafe_allow_html=True)
    display_name = pred.replace("Iris-", "Iris ").title()
    st.markdown(
        f"""
<div class="result-card" style="background: linear-gradient(135deg, {info['color']} 0%, {info['color']}cc 100%);">
<span class="emoji">{info['emoji']}</span>
<div class="tag">Predicted species</div>
<div class="name">{display_name}</div>
<div class="conf">{conf:.1%} confidence</div>
<div class="blurb">{info['blurb']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Warnings
    out_of_range = [LABELS[f] for f in FEATURES
                    if not (TRAIN_MIN_MAX[f][0] <= values[f] <= TRAIN_MIN_MAX[f][1])]
    if out_of_range:
        st.warning("**" + ", ".join(out_of_range) + "** is outside the range the model "
                   "was trained on, so the prediction may be unreliable.", icon="⚠️")
    elif conf < 0.75:
        st.info("The model is not very sure here — these measurements sit close to "
                "the boundary between two species.", icon="🤔")

# ----------------------------------------------------------------------------
# Probabilities
# ----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
p_col, c_col = st.columns([1, 1.4], gap="large")

with p_col:
    st.markdown('<div class="section-title">📊 Class probabilities</div>'
                '<div class="section-sub">How likely each species is, according to the model</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        order = np.argsort(proba)[::-1]
        rows = ""
        for i in order:
            c = SPECIES.get(classes[i], {"color": "#6366f1", "emoji": ""})
            name = classes[i].replace("Iris-", "Iris ").title()
            rows += (
                f'<div class="prob-row"><div class="prob-label">'
                f'<span>{c["emoji"]} {name}</span><b>{proba[i]:.1%}</b></div>'
                f'<div class="prob-track"><div class="prob-fill" '
                f'style="width:{max(proba[i]*100, 0.5):.1f}%; background:{c["color"]}"></div></div></div>'
            )
        st.markdown(rows, unsafe_allow_html=True)

    with st.expander("Your input"):
        st.dataframe(X_input.rename(columns=LABELS), hide_index=True,
                     width="stretch")

# ----------------------------------------------------------------------------
# Context chart
# ----------------------------------------------------------------------------
with c_col:
    st.markdown('<div class="section-title">🧭 Where does your flower sit?</div>'
                '<div class="section-sub">Your input (◆) plotted against the 150 flowers in the dataset</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        a, b = st.columns(2)
        x_axis = a.selectbox("X axis", FEATURES, index=2,
                             format_func=LABELS.get, key="x_axis")
        y_axis = b.selectbox("Y axis", FEATURES, index=3,
                             format_func=LABELS.get, key="y_axis")

        color_scale = alt.Scale(domain=list(SPECIES.keys()),
                                range=[v["color"] for v in SPECIES.values()])
        base = alt.Chart(ref_df)
        points = base.mark_circle(size=60, opacity=0.65).encode(
            x=alt.X(x_axis, title=f"{LABELS[x_axis]} (cm)", scale=alt.Scale(zero=False)),
            y=alt.Y(y_axis, title=f"{LABELS[y_axis]} (cm)", scale=alt.Scale(zero=False)),
            color=alt.Color("species:N", scale=color_scale,
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["species"] + FEATURES,
        )
        you = alt.Chart(pd.DataFrame({x_axis: [values[x_axis]], y_axis: [values[y_axis]]})
                        ).mark_point(shape="diamond", size=380, filled=True,
                                     color=info["color"], stroke="white",
                                     strokeWidth=2.5, opacity=1).encode(x=x_axis, y=y_axis)
        st.altair_chart((points + you).properties(height=300), width="stretch")

st.caption("Built with Streamlit · Model: Quadratic Discriminant Analysis (scikit-learn)")
