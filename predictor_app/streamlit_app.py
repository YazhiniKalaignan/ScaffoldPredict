import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from typing import List, Dict

# -----------------------------
# Helper: load resources
# -----------------------------
@st.cache_resource
def load_model(path: str = "model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------
# Load artifacts (adjust filenames if different)
# -----------------------------
MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_names.json"
ENCODING_PATH = "encoding_map.json"

model = load_model(MODEL_PATH)
feature_names: List[str] = load_json(FEATURES_PATH)
encoding_map: Dict[str, int] = load_json(ENCODING_PATH)

# Ensure feature names is a list
if not isinstance(feature_names, list):
    feature_names = list(feature_names)

# -----------------------------
# Try to identify biomaterial / cell / env columns by position with fallbacks
# -----------------------------
# You told: 60 biomaterials, 61st = cell type, 62-69 env (8 cols)
biomaterial_count = 60
expected_env_count = 8

if len(feature_names) >= biomaterial_count + 1 + expected_env_count:
    biomaterial_cols = feature_names[:biomaterial_count]
    cell_col = feature_names[biomaterial_count]
    env_cols = feature_names[biomaterial_count + 1 : biomaterial_count + 1 + expected_env_count]
else:
    # Fallback heuristics: try to find a column with 'Cell' in name for cell_col
    cell_candidates = [c for c in feature_names if "cell" in c.lower() or "cell type" in c.lower() or "cell_" in c.lower()]
    if cell_candidates:
        cell_col = cell_candidates[0]
    else:
        # last-expected position fallback
        cell_col = feature_names[min(biomaterial_count, max(0, len(feature_names)-1))]

    # Biomaterials: take first up to 60 or everything before cell_col
    if cell_col in feature_names:
        idx = feature_names.index(cell_col)
        biomaterial_cols = feature_names[:idx]
        env_start = idx + 1
    else:
        biomaterial_cols = feature_names[:biomaterial_count]
        env_start = biomaterial_count + 1

    env_cols = feature_names[env_start: env_start + expected_env_count]

# Final safety: ensure biomaterial_cols length <= existing
biomaterial_cols = [c for c in biomaterial_cols if c in feature_names]
env_cols = [c for c in env_cols if c in feature_names and c != cell_col]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Scaffold Property Predictor", layout="wide")
st.title("🧬 Scaffold Property Predictor")
st.write("Predict **Cell Response**, **Printability**, and **Scaffold Quality** from user inputs.")

# Sidebar with metadata
with st.sidebar:
    st.header("Model & Data")
    st.write(f"Loaded model: `{MODEL_PATH}`")
    st.write(f"# features: {len(feature_names)}")
    st.write("Tip: Select biomaterials then enter their values. Unselected materials are treated as 0.")
    st.caption("Scaffold Quality = Cell Response × Printability (computed)")

# Layout: inputs on left, results on right
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1) Biomaterials — select and set values")
    selected = st.multiselect("Select biomaterials used", options=biomaterial_cols, default=[]) 

    # container for dynamic inputs for selected biomaterials
    biomat_inputs = {}
    if selected:
        st.write("Enter values for selected biomaterials:")
        for name in selected:
            # default 0.0, step inferred
            val = st.number_input(f"{name}", value=0.0, format="%.6f", key=f"bm_{name}")
            biomat_inputs[name] = float(val)
    else:
        st.info("No biomaterials selected — all biomaterial columns will be set to 0.")

    st.markdown("---")

    st.subheader("2) Cell line (choose one)")
    cell_choice = st.selectbox("Cell Line", options=list(encoding_map.keys()))
    encoded_cell = encoding_map.get(cell_choice)
    st.write(f"Selected: **{cell_choice}** → encoded value **{encoded_cell}**")

    st.markdown("---")

    st.subheader("3) Environment / Process parameters")
    env_inputs = {}
    if env_cols:
        for c in env_cols:
            # sensible default 0.0
            v = st.number_input(f"{c}", value=0.0, format="%.6f", key=f"env_{c}")
            env_inputs[c] = float(v)
    else:
        st.info("No environment columns automatically detected — ensure your feature_names.json follows the expected layout.")

    st.markdown("---")

    # Extra: optional CSV upload to fill multiple biomaterials together
    st.subheader("Optional: Upload CSV of inputs")
    upload = st.file_uploader("CSV with columns matching feature names", type=["csv"] )
    uploaded_row = None
    if upload is not None:
        try:
            uploaded_df = pd.read_csv(upload)
            # take first row
            uploaded_row = uploaded_df.iloc[[0]]
            st.success("Loaded row from CSV (first row will be used)")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.markdown("---")

    run_pred = st.button("🔍 Predict Scaffold Properties")

with col2:
    st.subheader("Prediction Output")
    result_box = st.empty()
    details_box = st.empty()

# -----------------------------
# Prediction logic
# -----------------------------

def build_input_row(feature_names: List[str], biomat_inputs: Dict[str, float], cell_col: str, encoded_cell: int, env_inputs: Dict[str, float]):
    # initialize with floats to avoid dtype warnings
    row = pd.DataFrame([{col: 0.0 for col in feature_names}])

    # fill biomaterials
    for col, val in biomat_inputs.items():
        if col in row.columns:
            row.loc[0, col] = float(val)

    # fill cell
    if cell_col in row.columns:
        row.loc[0, cell_col] = float(encoded_cell)

    # fill env
    for col, val in env_inputs.items():
        if col in row.columns:
            row.loc[0, col] = float(val)

    return row


def try_predict_probas(model, X_row):
    """Try to get per-target probabilities if available. Return list of (pred, prob) pairs."""
    probs = []
    try:
        # MultiOutputClassifier: estimator list per target
        # Some wrapped estimators provide predict_proba
        if hasattr(model, "estimators_"):
            for i, est in enumerate(model.estimators_):
                if hasattr(est, "predict_proba"):
                    p = est.predict_proba(X_row)  # returns array (n_samples, n_classes)
                    # take max prob as confidence
                    max_prob = float(np.max(p, axis=1)[0])
                else:
                    max_prob = None
                probs.append(max_prob)
        else:
            probs = [None, None]
    except Exception:
        probs = [None, None]
    return probs


if run_pred:
    # if user uploaded CSV row, prefer that (but still map cell encoding if necessary)
    if uploaded_row is not None:
        # ensure column order and missing cols filled
        row = pd.DataFrame([{col: 0.0 for col in feature_names}])
        for c in uploaded_row.columns:
            if c in row.columns:
                row.loc[0, c] = uploaded_row.iloc[0][c]
        # if cell choice selected, override cell_col
        if cell_col in row.columns:
            row.loc[0, cell_col] = float(encoded_cell)
    else:
        row = build_input_row(feature_names, biomat_inputs, cell_col, encoded_cell, env_inputs)

    # final safety: ensure same column order as feature_names
    row = row[feature_names]

    # Predict
    try:
        pred = model.predict(row)[0]
        # convert to Python ints when appropriate
        try:
            pred_vals = [int(p) for p in pred]
        except Exception:
            # if regression or float outputs
            pred_vals = [float(p) for p in pred]

        # Extract
        cell_resp_pred = pred_vals[0]
        print_pred = pred_vals[1]
        try:
            scaffold_pred = int(cell_resp_pred * print_pred)
        except Exception:
            scaffold_pred = cell_resp_pred * print_pred

        # try probabilities/confidence
        confs = try_predict_probas(model, row)

        # Display
        result_text = f"**Cell Response:** {cell_resp_pred}"
        if confs[0] is not None:
            result_text += f"  — confidence: {confs[0]*100:.1f}%"
        result_text += "\n\n"
        result_text += f"**Printability:** {print_pred}"
        if confs[1] is not None:
            result_text += f"  — confidence: {confs[1]*100:.1f}%"
        result_text += "\n\n"
        result_text += f"**Scaffold Quality (derived):** {scaffold_pred}"

        result_box.markdown(result_text)

        # show input row and model meta if user wants details
        with details_box.expander("Show input row and raw model output"):
            st.write("Input row (first 20 cols):")
            st.dataframe(row.iloc[:, :20])
            st.write("Raw model output:")
            st.write(pred)
            st.write({"confidences": confs})

    except Exception as e:
        result_box.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ — Streamlit. Make sure `model.pkl`, `feature_names.json`, and `encoding_map.json` are present in the app folder.")
