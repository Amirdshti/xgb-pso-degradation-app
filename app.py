# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:58:17 2025

@author: 24550372
"""

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("XGBPSOModel.pkl")

# Page config
st.set_page_config(page_title="Photocatalytic Degradation Predictor", layout="wide")

# Title and description
st.title("🧪 XGB-PSO Model for Photocatalytic Degradation (%) of Pollutants by ZnO")
st.markdown("""
This tool predicts the degradation efficiency of a pollutant by ZnO photocatlyst based on input parameters  
using an **XGBoost model optimized with Particle Swarm Optimization (PSO)**.
""")

# Input features with labels
inputs_with_labels = [
    ("PollutantMw", "Pollutant Mw (g/mol)"),
    ("TPSA", "Topological Polar Surface Area (Å²)"),
    ("HBDC", "H-bond Donor Count"),
    ("HBAC", "H-bond Acceptor Count"),
    ("C0", "Initial concentration (mg/L)"),
    ("pH", "Solution pH"),
    ("Light", "Light Source (1 = UV, 2 = VIS)"),
    ("Dosage", "Catalyst dosage (mg/L)"),
    ("DopantMw", "Dopant Molecular Weight (g/mol)"),
    ("DopZn", "Dopant/Zn Weight Ratio"),
    ("Time", "Time (min)")
]

# Sidebar input
st.sidebar.header("📥 Enter Input Parameters")
user_input = []
for var, label in inputs_with_labels:
    val = st.sidebar.number_input(label, step=0.1, format="%.4f")
    user_input.append(val)

# Predict button
if st.button("Predict Degradation"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"✅ Predicted Degradation: **{prediction:.2f} %**")

# Footer
st.markdown("---")
st.markdown("Made by [Amir Dashti](https://github.com/Amirdshti) • UTS CIvil and Environmental Engineering, 2025")
