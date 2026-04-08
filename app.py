import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import io

# Page Config
st.set_page_config(page_title="HAR Universal Engine", layout="wide")

# Load models
try:
    with open("hmm_models.pkl", "rb") as f:
        models = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 hmm_models.pkl not found! Please run the training code first.")

st.title("🏃 Human Activity Recognition Engine")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Accelerometer CSV (Any Format)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        
        # 1. Detect and Clean Format
        if ";" in content[:1000]: # Raw WISDM
            data = []
            for line in content.splitlines():
                line = line.strip().replace(";", "")
                parts = line.split(",")
                if len(parts) == 6: data.append(parts)
            df = pd.DataFrame(data, columns=["user", "activity", "timestamp", "x", "y", "z"])
        else: # Standard CSV or G-Force CSV
            df = pd.read_csv(io.StringIO(content), comment='#')
            # Column mapping dictionary
            rename_map = {
                'ax (m/s^2)': 'x', 'ay (m/s^2)': 'y', 'az (m/s^2)': 'z',
                'gFx': 'x', 'gFy': 'y', 'gFz': 'z',
                'Acceleration x (m/s^2)': 'x', 'Acceleration y (m/s^2)': 'y', 'Acceleration z (m/s^2)': 'z'
            }
            df = df.rename(columns=rename_map)

        # Convert to numeric
        for col in ["x", "y", "z"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["x", "y", "z"])

        if not df.empty:
            # 2. AUTO-CORRECT UNITS (The "Smart" Part)
            # Calculate raw magnitude to check the scale
            raw_mag_mean = np.mean(np.sqrt(df['x']**2 + df['y']**2 + df['z']**2))
            
            # If magnitude is ~1.0, it's G-Force -> Multiply by 9.8
            if 0.5 < raw_mag_mean < 1.5:
                df[['x', 'y', 'z']] *= 9.8
                st.info("💡 Detected G-Force units. Scaled to m/s².")
            
            # If magnitude is ~0, it's Linear Acceleration -> Shift by 9.8
            elif raw_mag_mean < 0.5:
                # We'll add 9.8 later during magnitude calculation for better stability
                st.info("💡 Detected Linear Acceleration. Applying Gravity Offset.")
            
            # 3. Visualization
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📡 Sensor Data Visualization")
                fig = go.Figure()
                for axis in ['x', 'y', 'z']:
                    fig.add_trace(go.Scatter(y=df[axis].values[:500], name=f"Axis {axis.upper()}"))
                fig.update_layout(height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            # 4. Prediction
            with col2:
                st.subheader("🤖 Model Prediction")
                window_size = 100
                if len(df) >= window_size:
                    seq_raw = df[["x", "y", "z"]].values[:window_size]
                    
                    # Final Feature Engineering
                    mag_calc = np.sqrt(np.sum(seq_raw**2, axis=1)).reshape(-1, 1)
                    
                    # Apply offset if it was linear (was near 0)
                    if raw_mag_mean < 0.5:
                        mag_calc += 9.8
                    
                    seq_final = np.hstack((seq_raw, mag_calc))

                    # Scoring
                    scores = {activity: model.score(seq_final) for activity, model in models.items()}
                    prediction = max(scores, key=scores.get)
                    
                    st.success(f"**Activity:** {prediction}")
                    st.write("Model Confidence Scores:")
                    st.bar_chart(pd.Series(scores))
                    st.metric("Mean Magnitude", f"{np.mean(mag_calc):.2f} m/s²")
                else:
                    st.error("Need 100 rows of data.")
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("Universal HAR Engine v2.0 | Supports WISDM, Phyphox, and G-Force CSVs")