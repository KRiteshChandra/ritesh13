# 1. Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 2. Load Trained ML Models
# - Linear Regression for watts prediction
# - Random Forest for usage class (Low, Medium, High)
# - Label Encoder for node/device encoding (historic data)
reg_model = joblib.load("linear_model.pkl")
clf_model = joblib.load("rf_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# Constants for emissions and tariffs
CO2_FACTOR = 0.475            # kg CO₂ per kWh (grid emission factor)
TARIFF_DAY, TARIFF_NIGHT = 0.15, 0.09   # Electricity tariff ($/kWh)

# 3. Device Specs (15 unique devices you requested)
device_specs = {
    # Cooling & Comfort
    "Air Conditioner (Office)": {"power_kw": 3.5, "eff": 0.85},
    "Chiller Unit": {"power_kw": 15.0, "eff": 0.88},
    "Cooling Tower Fan": {"power_kw": 5.0, "eff": 0.90},

    # Heating & Process
    "Boiler System": {"power_kw": 30.0, "eff": 0.75},
    "Industrial Oven": {"power_kw": 25.0, "eff": 0.85},
    "Furnace Unit": {"power_kw": 50.0, "eff": 0.70},

    # Motors, Pumps
    "Water Pump (General)": {"power_kw": 15.0, "eff": 0.90},
    "Air Compressor Machine": {"power_kw": 11.0, "eff": 0.85},
    "Industrial Blower Fan": {"power_kw": 10.0, "eff": 0.88},

    # Manufacturing
    "CNC Machine": {"power_kw": 10.0, "eff": 0.90},
    "Injection Moulding Machine": {"power_kw": 20.0, "eff": 0.85},
    "Welding System": {"power_kw": 12.0, "eff": 0.85},

    # Utilities / Logistics
    "Elevator Lift": {"power_kw": 7.0, "eff": 0.85},
    "Battery Forklift": {"power_kw": 12.0, "eff": 0.85},
    "Cold Storage Chamber": {"power_kw": 30.0, "eff": 0.85}
}

# 4. Streamlit UI Setup
st.set_page_config(layout="wide")  # Full width dashboard

st.title("Carbon Footprint Calculator Using Random Forest")
st.markdown("""
This dashboard combines **ML predictions (Linear Regression + Random Forest)**  
with your **custom inputs (hours/day, load %, units of appliances)**  
to forecast **Energy, Cost, and CO₂ emissions**.  
It also highlights optimization insights 👉 top cost drivers, worst days, savings opportunities.
""")

# Forecast horizon inputs
days = st.slider("Days to Forecast", 3, 30, 7)
start_date = st.date_input("Start Date", pd.to_datetime("today"))

# Appliance selection
all_devices = list(device_specs.keys())
selected_devices = st.multiselect("Select Appliances:", all_devices, default=[all_devices[0]])

# Appliance Input Map (hours/day, load%, units)
device_inputs = {}
st.subheader("⚙️ Configure Appliance Parameters")
for device in selected_devices:
    col1, col2, col3 = st.columns(3)
    with col1:
        hrs = st.slider(f"{device} - Hours per Day", 1, 24, 8, key=f"h{device}")
    with col2:
        load = st.slider(f"{device} Load %", 10, 100, 80, key=f"l{device}") / 100
    with col3:
        units = st.number_input(f"{device} Units", 1, 10, 1, key=f"u{device}")
    device_inputs[device] = {"hours": hrs, "load": load, "units": units}

# 5. Forecast Loop
if st.button("🚀 Run Forecast"):
    hours_ahead = 24 * days
    future = pd.date_range(start=start_date, periods=hours_ahead, freq="h")
    results = []

    for device in selected_devices:
        spec = device_specs[device]
        params = device_inputs[device]

        # Build feature DF for ML predictions
        df = pd.DataFrame({
            "hour": future.hour,
            "day": future.day,
            "weekday": future.weekday,
            "node_encoded": 0,   
            "timestamp": future
        })

        # Predict base watts with regression
        watts_pred = reg_model.predict(df.drop(columns=["timestamp"]))

        # Scale prediction with real-world base kW × efficiency
        base_kw = spec["power_kw"] * spec["eff"] * 1000   # → watts
        scaling_factor = base_kw / np.mean(watts_pred) if np.mean(watts_pred) > 0 else 1
        df["Watts"] = watts_pred * scaling_factor

        # Usage class → Random Forest
        try:
            df["UsageClass"] = clf_model.predict(
                df.drop(columns=["timestamp","Watts"], errors="ignore")
            )
        except:
            df["UsageClass"] = "Medium"

        # Apply custom user inputs (hours × units × load %)
        df["Energy_kWh"] = (df["Watts"] * params["hours"] * params["units"] * params["load"]) / 1000
        df["CO2_kg"] = df["Energy_kWh"] * CO2_FACTOR
        df["Cost_$"] = df.apply(
            lambda r: r["Energy_kWh"] * 
                     (TARIFF_NIGHT if (r["hour"] >= 22 or r["hour"] < 6) else TARIFF_DAY),
            axis=1
        )
        df["Device"] = device
        results.append(df)

    forecast_df = pd.concat(results)

    # Aggregate totals
    summary = forecast_df.groupby("Device")[["Energy_kWh","CO2_kg","Cost_$"]].sum()
    sys_daily = forecast_df.groupby("timestamp")[["Energy_kWh","CO2_kg","Cost_$"]].sum()
    device_daily = forecast_df.groupby(["timestamp","Device"])[["Cost_$"]].sum().reset_index()

    # 6. Insights
    st.subheader("📊 Appliance Totals")
    st.dataframe(summary.style.format({"Energy_kWh":"{:.1f}", "CO2_kg":"{:.1f}", "Cost_$":"${:.2f}"}))

    st.markdown(f"""
    ### 📦 System Totals for {days} Days
    - 🔋 Total Energy: **{summary['Energy_kWh'].sum():.1f} kWh**
    - 💵 Total Cost: **${summary['Cost_$'].sum():.2f}**
    - 🌍 Total CO₂: **{summary['CO2_kg'].sum():.1f} kg** (~{summary['CO2_kg'].sum()*4:.0f} km driving 🚗)
    """)

    st.subheader("💡 Optimization Insights")
    st.warning(f"💰 Highest Cost Driver: **{summary['Cost_$'].idxmax()}** (~${summary['Cost_$'].max():.2f})")
    st.error(f"🌍 Biggest Carbon Emitter: **{summary['CO2_kg'].idxmax()}** (~{summary['CO2_kg'].max():.1f} kg CO₂)")
    best_day, worst_day = sys_daily['Cost_$'].idxmin(), sys_daily['Cost_$'].idxmax()
    st.success(f"✅ Best Day: {best_day.date()} (~${sys_daily.loc[best_day,'Cost_$']:.2f})")
    st.error(f"⚠️ Worst Day: {worst_day.date()} (~${sys_daily.loc[worst_day,'Cost_$']:.2f})")

    # 7. Big dashboard (inside the forecast block to avoid NameError)
    fig, axes = plt.subplots(1, 3, figsize=(50,20))

    # PIE
    summary["Cost_$"].plot.pie(
        ax=axes[0],
        autopct='%1.1f%%',
        colormap="tab20",
        legend=False,
        textprops={'fontsize': 26}
    )
    axes[0].set_ylabel("")
    axes[0].set_title("Cost Share by Device", fontsize=34)

    # BAR
    summary["CO2_kg"].plot.bar(ax=axes[1], color="green")
    axes[1].set_title("CO₂ by Device", fontsize=34)
    axes[1].set_ylabel("kg CO₂", fontsize=28)
    axes[1].set_xlabel("Device", fontsize=28)
    axes[1].tick_params(axis='x', labelsize=22, rotation=30)
    axes[1].tick_params(axis='y', labelsize=22)

    # MULTI-LINE
    for device in selected_devices:
        device_series = device_daily[device_daily["Device"]==device]
        axes[2].plot(
            device_series["timestamp"],
            device_series["Cost_$"],
            marker="o",
            label=device,
            linewidth=4, markersize=12
        )

    axes[2].set_title("Daily Cost Trend per Device", fontsize=34)
    axes[2].set_ylabel("USD ($)", fontsize=28)
    axes[2].set_xlabel("Date", fontsize=28)
    axes[2].tick_params(axis='x', labelsize=22, rotation=30)
    axes[2].tick_params(axis='y', labelsize=22)
    axes[2].legend(fontsize=24)

    plt.tight_layout()
    st.pyplot(fig)
