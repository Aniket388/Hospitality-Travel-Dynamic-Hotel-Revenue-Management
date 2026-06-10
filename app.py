import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ==============================================================================
# 🎨 1. ENTERPRISE UI CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Dynamic Revenue Engine v2",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS Injection
st.markdown("""
    <style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #0F172A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #475569; margin-bottom: 2rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05); }
    .status-badge { font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🔒 2. SAFE PIPELINE ARTIFACT LOADING
# ==============================================================================
@st.cache_resource
def load_production_assets():
    try:
        dt_model = joblib.load("superior_dt_model.pkl")
        kmeans_model = joblib.load("superior_kmeans_model.pkl")
        scaler = joblib.load("superior_scaler.pkl")
        encoder = joblib.load("superior_encoder.pkl")
        with open("engine_meta.json", "r") as f:
            meta = json.load(f)
        return dt_model, kmeans_model, scaler, encoder, meta
    except Exception as e:
        st.error(f"❌ Initialization Error: Critical asset files could not be mapped. Details: {str(e)}")
        return None, None, None, None, None

dt_model, kmeans_model, scaler, encoder, meta = load_production_assets()

# Run the app if all system configurations load securely
if dt_model is not None:
    
    # Header Banner Area
    st.markdown('<div class="main-header">🏨 Real-Time Dynamic Hotel Revenue Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production Optimization Platform Driven by Tuned Cyclical Regression Vectors.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ==============================================================================
    # 📥 3. OPTIMIZED HIGH-IMPACT INPUT SIDEBAR
    # ==============================================================================
    st.sidebar.header("🎯 Core Pricing Drivers")
    
    hotel_choice = st.sidebar.selectbox("Hotel Operation Mode", ["City Hotel", "Resort Hotel"])
    is_resort = 1 if hotel_choice == "Resort Hotel" else 0

    arrival_month = st.sidebar.selectbox("Planned Arrival Month", 
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    
    lead_time = st.sidebar.slider("Lead Time (Days Prior to Check-in)", 0, 365, 45)
    arrival_week = st.sidebar.slider("Arrival Calendar Week Number", 1, 53, 26)

    st.sidebar.markdown("---")
    st.sidebar.header("👥 Guest & Stay Profiles")

    total_guests = st.sidebar.slider("Total Guests Count", 1, 10, 2)
    is_family = st.sidebar.checkbox("Is Family Booking? (More than 2 guests)", value=(total_guests > 2))
    is_family_val = 1 if is_family else 0
    
    total_stay_nights = st.sidebar.slider("Total Length of Stay (Nights)", 1, 30, 3)
    weekend_nights = st.sidebar.slider("How many of those are Weekend Nights?", 0, total_stay_nights, 1)
    weekend_ratio = weekend_nights / total_stay_nights if total_stay_nights > 0 else 0.0

    st.sidebar.markdown("---")
    st.sidebar.header("💼 Distribution & Segment Channels")

    # Only present high-impact categories confirmed by the pruning diagnostic report
    market_segment = st.sidebar.selectbox("Market Segment", ["Online TA", "Direct", "Groups", "Offline TA/TO"])
    distribution_channel = st.sidebar.selectbox("Distribution Channel", ["TA/TO", "Direct", "Corporate"])
    reserved_room_type = st.sidebar.selectbox("Reserved Inventory Room Class", ["A", "D", "B", "F", "E", "G", "C"])
    customer_type = st.sidebar.selectbox("Customer Contract Type", ["Transient", "Transient-Party", "Contract"])

    # Hardcoded system constants for low-variance features to save user UI friction
    special_requests = 1
    has_special_requests = 1
    booking_changes = 0
    is_repeat_guest = 0

    # ==============================================================================
    # ⚙️ 4. REAL-TIME DATA MATH TRANSFORMATIONS (MATCHES TRAINING PIPELINE)
    # ==============================================================================
    # A. Calculate Cyclical Time Signatures (Trigonometric Transformations)
    month_map = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
    month_num = month_map[arrival_month]
    month_sin = np.sin(2 * np.pi * month_num / 12.0)
    month_cos = np.cos(2 * np.pi * month_num / 12.0)

    # B. Resolve K-Means Demand Cluster Tier
    cluster_input = np.array([[lead_time, arrival_week, total_stay_nights, special_requests]])
    scaled_cluster_input = scaler.transform(cluster_input)
    raw_cluster_id = kmeans_model.predict(scaled_cluster_input)[0]
    demand_cluster = meta["sorting_cluster_map"][str(raw_cluster_id)]

    # C. Assemble Numerical Baseline DataFrame Matrix
    numerical_payload = {
        "demand_cluster": int(demand_cluster), "month_sin": float(month_sin), "month_cos": float(month_cos),
        "lead_time": int(lead_time), "total_stay_nights": int(total_stay_nights), "total_guests": int(total_guests),
        "is_family": int(is_family_val), "is_repeat_guest": int(is_repeat_guest), "has_special_requests": int(has_special_requests),
        "is_resort": int(is_resort), "booking_changes": int(booking_changes), "weekend_ratio": float(weekend_ratio)
    }
    df_numerical = pd.DataFrame([numerical_payload])

    # D. Reconstruct One-Hot Categorical Vector Arrays
    categorical_payload = [[market_segment, distribution_channel, reserved_room_type, customer_type]]
    df_categorical_raw = pd.DataFrame(categorical_payload, columns=meta["categorical_features_to_encode"])
    encoded_categorical_matrix = encoder.transform(df_categorical_raw)
    df_categorical_encoded = pd.DataFrame(encoded_categorical_matrix, columns=encoder.get_feature_names_out(), index=df_numerical.index)

    # E. Align Column Layout with Best-Fit Model Features Schema
    X_live_inference = pd.concat([df_numerical, df_categorical_encoded], axis=1)
    X_live_inference = X_live_inference[meta["expected_final_features"]]

    # F. Compute Optimized Room Rate via Decision Tree Logic Nodes
    recommended_adr = dt_model.predict(X_live_inference)[0]

    # ==============================================================================
    # 📊 5. GRAPHICAL DISPLAY INTERFACE
    # ==============================================================================
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Optimal Revenue Recommendation")
        st.markdown(f"<h1 style='color:#0F172A; margin:0;'>${recommended_adr:.2f} <span style='font-size:1.2rem; color:#64748B;'>/ Night</span></h1>", unsafe_allow_html=True)
        
        # Color-coded strategy signals matching calculated operational clusters
        demand_states = {
            0: ("🟢 Low-Demand Base Rate Rules Active", "Value-incentive promotion parameters triggered to lock in core baseline room occupancy."),
            1: ("🟡 Mid-Market Stable Rate Active", "Standard pricing equilibrium configured. Baseline seasonal contract rules applied."),
            2: ("🔴 Peak Optimization Yield Strategy Active", "High-occupancy parameters active. Premium dynamic revenue yield scaling enabled.")
        }
        state_badge, state_desc = demand_states[demand_cluster]
        
        st.markdown(f"<p style='margin-top:15px; font-weight:700; color:#2563EB; margin-bottom:5px;'>{state_badge}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#475569; font-size:0.95rem; margin:0;'>{state_desc}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.subheader("📊 Model Architecture Validation")
        st.info("💡 This clean engine strips out zero-variance variables (like GDS channels and group customer blocks) to reduce data noise and processing latency.")
        
        # Render clean system health scores
        st.markdown(f"🧬 Model Fit Confidence ($R^2$ Score): **`{meta['metrics']['R2']:.4f}`**")
        st.markdown(f"🎯 Prediction Precision Range (MAE): **`±${meta['metrics']['MAE']:.2f} / Night`**")
        
        # Real-time data trace tracking summary
        st.markdown("##### Runtime Input Vector Breakdowns:")
        breakdown_df = pd.DataFrame({
            "Operational Field": ["Property Configuration", "Horizon Window", "Calculated Stay Duration", "Inventory Group Selection"],
            "Parsed Metrics": [f"{hotel_choice}", f"{lead_time} Days Prior", f"{total_stay_nights} Total Nights", f"Room Class {reserved_room_type}"]
        })
        st.table(breakdown_df)

else:
    st.error("System Failure: The machine learning pipeline assets could not be located in this directory workspace environment.")
