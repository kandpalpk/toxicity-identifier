import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import numpy as np

# Load the TF-IDF vocabulary specific to the category (same as Flask app)
with open(r"toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open(r"toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)

# ... (same code as in your Flask app for loading models)

# Main Streamlit App
st.title("Toxicity Prediction")

# Replace Flask's form input with Streamlit's text input widget
user_input = st.text_input("Enter text to analyze for toxicity")

# Prediction logic (same as your Flask app)
if user_input:
    data = [user_input]

    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:,1]
    
    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:,1]

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:,1]

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:,1]

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:,1]

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:,1]
    # ... (prediction code)
    # ... (rounding and other operations)
    out_tox = round(pred_tox[0], 2)
    out_sev = round(pred_sev[0], 2)
    out_obs = round(pred_obs[0], 2)
    out_ins = round(pred_ins[0], 2)
    out_thr = round(pred_thr[0], 2)
    out_ide = round(pred_ide[0], 2)

    # Display results (replacing Flask's render_template)
    st.write(f'Prob (Toxic): {out_tox}')
    st.write(f'Prob (Severe Toxic): {out_sev}')
    st.write(f'Prob (Obscene): {out_obs}')
    st.write(f'Prob (Insult): {out_ins}')
    st.write(f'Prob (Threat): {out_thr}')
    st.write(f'Prob (Identity Hate): {out_ide}')

# #Create two columns
# col1, col2 = st.columns(2)

# # Display the first image in the first column
# col1.image("/Users/prakharkandpal/Projects/toxic_comments_classifier_copy/f1score.png", caption="Image 1", use_column_width=True)

# # Display the second image in the second column
# col2.image("/Users/prakharkandpal/Projects/toxic_comments_classifier_copy/percentagecomm.png", caption="Image 2", use_column_width=True)

st.image("/Users/prakharkandpal/Projects/toxic_comments_classifier_copy/f1score.png", caption="Fig 1. F1 Score Comparision", use_column_width=True)
st.image("/Users/prakharkandpal/Projects/toxic_comments_classifier_copy/percentagecomm.png", caption="Fig 2. Percentage of comments used as an insult/abuse", use_column_width=True)
