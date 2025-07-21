import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your pipeline and reference df
pipe = pickle.load(open('pipe.pkl', 'rb'))
df    = pickle.load(open('df.pkl',   'rb'))

st.title("Laptop Price Predictor")

# --- 1) Streamlit inputs
company     = st.selectbox('Brand',            df['Company'].unique())
type_name   = st.selectbox('Type',             df['TypeName'].unique())
ram         = st.selectbox('RAM (in GB)',      [2,4,6,8,12,16,24,32,64])
weight      = st.number_input('Weight (kg)',    min_value=0.5, max_value=5.0, value=1.3, step=0.1)
touchscreen = st.selectbox('Touchscreen',      ['No','Yes'])
ips         = st.selectbox('IPS Panel',        ['No','Yes'])
screen_size = st.slider('Screen size (inches)',10.0,18.0,13.0,0.1)
resolution  = st.selectbox('Resolution',       [
                 '1920x1080','1366x768','1600x900','3840x2160',
                 '3200x1800','2880x1800','2560x1600',
                 '2560x1440','2304x1440'])
cpu_brand   = st.selectbox('CPU Brand',        df['Cpu brand'].unique())
hdd         = st.selectbox('HDD (in GB)',      [0,128,256,512,1024,2048])
ssd         = st.selectbox('SSD (in GB)',      [0,8,128,256,512,1024])
gpu_brand   = st.selectbox('GPU Brand',        df['Gpu brand'].unique())
os          = st.selectbox('Operating System', df['os'].unique())

# --- 2) On button click, prepare and predict
if st.button('Predict Price'):
    # encode Yes/No
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val         = 1 if ips         == 'Yes' else 0

    # compute PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # build input dict—keys match your training-time column names (case-insensitive)
    input_dict = {
        'Company':     company,
        'TypeName':    type_name,
        'Ram':         ram,
        'Weight':      weight,
        'Touchscreen': touchscreen_val,
        'Ips':         ips_val,
        'PPI':         ppi,
        'Cpu brand':   cpu_brand,
        'HDD':         hdd,
        'SSD':         ssd,
        'Gpu brand':   gpu_brand,
        'os':          os
    }

    # one‑row DataFrame
    input_df = pd.DataFrame([input_dict])

    # --- ALIGN COLUMNS AUTOMATICALLY ---
    expected = list(pipe.feature_names_in_)        # e.g. ['Company','TypeName',…,'PPI',…]
    col_map = {}
    for exp in expected:
        for actual in input_df.columns:
            if actual.lower() == exp.lower():
                col_map[actual] = exp
                break

    # rename and reorder
    input_df = input_df.rename(columns=col_map)[expected]

    #  predict (pipeline outputs log-price)
    log_pred   = pipe.predict(input_df)[0]
    price_pred = int(np.exp(log_pred))

    st.subheader(f"Predicted Laptop Price: ₹{price_pred}")
