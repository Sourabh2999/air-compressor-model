# compressor_model_app.py 
import streamlit as st
import numpy as np
import pandas as pd

# ----------------------------
# Step 1: Define System Boundaries and Components
# ----------------------------
st.set_page_config(page_title="Compressor Infrastructure Energy Model", layout="wide")
st.title("Compressed Air Infrastructure Optimization for Logistics centres")

st.sidebar.header("System Parameters")

# Basic Compressor Specs
st.sidebar.subheader("Installed Compressor Infrastructure Specifications")
flow_rates = []
powers = []
selected_models = []

# Load available models from CSV
import pandas as pd
import os
if os.path.exists("/mnt/data/Compressors Models_python.csv"):
    df_models = pd.read_csv("/mnt/data/Compressors Models_python.csv", encoding='ISO-8859-1')
    df_models.rename(columns={df_models.columns[0]: "Model"}, inplace=True)
    unique_models = sorted(df_models['Model'].dropna().unique().tolist())
else:
    unique_models = []

for i in range(1, 4):
    model = st.sidebar.selectbox(f"Compressor {i} Model", unique_models, index=0 if unique_models else None, key=f"model{i}")
    model_data = df_models[df_models['Model'] == model]
    available_flows = sorted(model_data['Flow Rate (mÂ³/min)'].astype(float).unique())
    flow = st.sidebar.selectbox(f"Rated Flow Compressor {i} (m3/min)", available_flows, key=f"flow{i}")
    pressure_match = model_data.iloc[(model_data['Flow Rate (mÂ³/min)'].astype(float) - flow).abs().argsort()[:1]]
    rated_power = pressure_match['Drive Motor Rated Power (kW)'].values[0] if not pressure_match.empty else 0.0
    st.sidebar.markdown(f"**Rated Power Compressor {i}:** {rated_power:.2f} kW")
    flow = st.sidebar.number_input(f"Rated Flow Compressor {i} (m3/min)", min_value=0.0, value=15.0, key=f"flow{i}")
    power = st.sidebar.number_input(f"Rated Power Compressor {i} (kW)", min_value=0.0, value=150.0, key=f"power{i}")
    selected_models.append(model)
    flow_rates.append(flow)
    powers.append(power)
st.sidebar.subheader("Operating Conditions")
ambient_temp_c = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-40.0, value=20.0)
ambient_temp = ambient_temp_c + 273.15
ambient_pressure_bar = st.sidebar.number_input("Ambient Pressure (bar)", min_value=0.5, value=1.013)
ambient_pressure = ambient_pressure_bar * 100000
set_pressure_bar = st.sidebar.number_input("Compressor Set Pressure (bar)", min_value=1.0, value=7.0)
set_pressure = set_pressure_bar * 100000

# Pressure Drops
st.sidebar.subheader("Optional Pressure Drops")
aftercooler_drop = st.sidebar.number_input("Aftercooler Pressure Drop (bar)", min_value=0.0, value=0.1)
dryer_drop = st.sidebar.number_input("Dryer Pressure Drop (bar)", min_value=0.0, value=0.2)
filter_drop = st.sidebar.number_input("Filter Pressure Drop (bar)", min_value=0.0, value=0.1)

total_pressure_drop = (aftercooler_drop + dryer_drop + filter_drop) * 100000  # bar to Pa
adjusted_set_pressure = set_pressure + total_pressure_drop

# Receiver Tank Volume
st.sidebar.subheader("Receiver Tank")
receiver_tank_liters = st.sidebar.number_input("Receiver Tank Volume (liters)", min_value=100.0, value=1000.0, step=50.0)
receiver_tank_m3 = receiver_tank_liters / 1000.0

# Constants
R = 287
k = 1.4
n = 1.3  # Polytropic exponent (assumed value between isothermal and adiabatic)
air_density = 1.225

def calculate_ideal_work(Pa, P2, Ta, Qm, flow_rate_m3_min=None, model=None):
    if model and not df_models.empty:
        model_data = df_models[df_models['Model'] == model]
        if not model_data.empty and flow_rate_m3_min is not None:
            try:
                # Filter for nearest operating pressure
                pressure_match = model_data.iloc[(model_data['Operating Pressure (bar)'] - P2/100000).abs().argsort()[:3]]
                flow_vals = pressure_match['Flow Rate (mÂ³/min)'].astype(float)
                power_vals = pressure_match['Drive Motor Rated Power (kW)'].astype(float)
                if len(flow_vals.unique()) > 1:
                    from scipy.interpolate import interp1d
                    interpolator = interp1d(flow_vals, power_vals, bounds_error=False, fill_value="extrapolate")
                    return interpolator(flow_rate_m3_min) * 1000  # kW to W
            except Exception as e:
                pass
    term = (P2 / Pa)**((n - 1) / n) - 1
    return (n / (n - 1)) * Qm * R * Ta * term

def calculate_tank_energy(Pa, P2, V):
    return (P2 * V / (k - 1)) * ((P2 / Pa)**((k - 1) / k) - 1)

# Step 1 Output
total_ideal_work = 0
for i in range(3):
    flow_rate = flow_rates[i] / 60
    Qm = flow_rate * air_density
    work = calculate_ideal_work(ambient_pressure, adjusted_set_pressure, ambient_temp, Qm, flow_rate_m3_min=flow_rates[i], model=selected_models[i])
    total_ideal_work += work

st.subheader("Ideal Compressor Work Calculation")
st.markdown(f"**Total Ideal Compressor Work (3 Compressors, with Pressure Losses):** {total_ideal_work/1000:.2f} kW")

# Utility function to apply ON/OFF masking to power readings
def apply_on_off_mask(df, i):
    power_col = f"Power{i}"
    on_col = f"C{i} On Time"
    if power_col in df.columns and on_col in df.columns:
        df[power_col] = df[power_col] * df[on_col]
    return df

# Apply ON/OFF mask after loading data
st.subheader("Upload Historical Compressor Data")
uploaded_file = st.file_uploader("Upload Compressor Data File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    rename_map = {
        "C1 - delivery volume flow rate": "Flow1",
        "C1 - intake temperature": "Temp1",
        "C1 - electrical power consumption": "Power1",
        "C2 - delivery volume flow rate": "Flow2",
        "C2 - intake temperature": "Temp2",
        "C2 - electrical power consumption": "Power2",
        "C3 - delivery volume flow rate": "Flow3",
        "C3 - intake temperature": "Temp3",
        "C3 - electrical power consumption": "Power3"
    }
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns=rename_map, inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    for i in range(1, 4):
        df = apply_on_off_mask(df, i)

    st.write("### File Preview")
    st.dataframe(df.head())

    # Step 3: Real Efficiency Calculation
    st.subheader("Real Compressor Efficiency Summary")
    summaries = []
    for i in range(1, 4):
        flow_col = f'Flow{i}'
        temp_col = f'Temp{i}'
        power_col = f'Power{i}'
        on_col = f'C{i} On Time'

        if flow_col in df.columns and temp_col in df.columns and power_col in df.columns and on_col in df.columns:
            flow_m3s = df[flow_col] / 60
            temp_K = df[temp_col] + 273.15
            Qm = flow_m3s * air_density
            df[f'Ideal_Power_{i}_kW'] = calculate_ideal_work(ambient_pressure, adjusted_set_pressure, temp_K, Qm, flow_rate_m3_min=df[flow_col], model=selected_models[i - 1]) / 1000
            actual_power = df[power_col]
            df[f'Efficiency_{i}'] = df[f'Ideal_Power_{i}_kW'] / actual_power.replace(0, np.nan)
            df[f'Efficiency_{i}'] = df[f'Efficiency_{i}'].clip(upper=1.5)

            duty_cycle = df[on_col].mean() * 100

            summaries.append({
                "Compressor": f"C{i}",
                "Avg Air Generated (m³/min)": f"{df[flow_col].mean():.2f}",
                "Avg Power Consumed (kW)": f"{actual_power.mean():.2f}",
                "Avg Temp (°C)": f"{df[temp_col].mean():.2f}",
                "Avg Ideal Power (kW)": f"{df[f'Ideal_Power_{i}_kW'].mean():.2f}",
                "Avg Efficiency (%)": f"{(df[f'Efficiency_{i}'].mean() * 100):.2f}",
                "Duty Cycle (%)": f"{duty_cycle:.2f}"
            })

    if summaries:
        st.write("### Compressor Efficiency Summary Table")
        st.dataframe(pd.DataFrame(summaries))

    # Step 4 header and logic
    st.subheader("Effectiveness and Carbon Emission Evaluation")
    with st.expander("🔁 Compare with Modified Configuration"):
        mod_receiver_tank_liters = st.number_input("Modified Receiver Tank Volume (liters)", min_value=100.0, value=receiver_tank_liters, step=50.0)
        mod_receiver_tank_m3 = mod_receiver_tank_liters / 1000.0
        mod_set_pressure_bar = st.number_input("Modified Set Pressure (bar)", value=set_pressure_bar)
        mod_aftercooler_drop = st.number_input("Modified Aftercooler Drop (bar)", value=aftercooler_drop)
        mod_dryer_drop = st.number_input("Modified Dryer Drop (bar)", value=dryer_drop)
        mod_filter_drop = st.number_input("Modified Filter Drop (bar)", value=filter_drop)

        mod_total_drop = (mod_aftercooler_drop + mod_dryer_drop + mod_filter_drop) * 100000
        mod_set_pressure = mod_set_pressure_bar * 100000 + mod_total_drop

        effectiveness_rows = []
        total_energy_base = 0
        total_energy_mod = 0
        total_cost_base = 0
        total_cost_mod = 0
        total_base_efficiency = 0
        total_mod_efficiency = 0
        count = 0

        base_tank_energy_kWh = calculate_tank_energy(ambient_pressure, adjusted_set_pressure, receiver_tank_m3) / 3600
        mod_tank_energy_kWh = calculate_tank_energy(ambient_pressure, mod_set_pressure, mod_receiver_tank_m3) / 3600

        for i in range(1, 4):
            flow_col = f"Flow{i}"
            temp_col = f"Temp{i}"
            power_col = f"Power{i}"

            if flow_col in df.columns and temp_col in df.columns and power_col in df.columns:
                flow_m3s = df[flow_col] / 60
                temp_K = df[temp_col] + 273.15
                Qm = flow_m3s * air_density

                base_ideal_power = df[f"Ideal_Power_{i}_kW"]
                mod_ideal_power = calculate_ideal_work(ambient_pressure, mod_set_pressure, temp_K, Qm) / 1000

                energy_base = (base_ideal_power * (5 / 60)).sum() + base_tank_energy_kWh
                energy_mod = (mod_ideal_power * (5 / 60)).sum() + mod_tank_energy_kWh
                cost_base = energy_base * 0.12
                cost_mod = energy_mod * 0.12

                actual_power = df[power_col]
                base_efficiency = base_ideal_power / actual_power.replace(0, np.nan)
                mod_efficiency = mod_ideal_power / actual_power.replace(0, np.nan)
                base_efficiency = base_efficiency.clip(upper=1.5)
                mod_efficiency = mod_efficiency.clip(upper=1.5)

                effectiveness = 1 - (energy_mod / energy_base)

                effectiveness_rows.append({
                    "Compressor": f"C{i}",
                    "Effectiveness (%)": f"{effectiveness * 100:.2f}",
                    "Original Design Energy (kWh)": f"{energy_base:.2f}",
                    "Modified Design Energy (kWh)": f"{energy_mod:.2f}",
                    "Original Design Tank Energy (kWh)": f"{base_tank_energy_kWh:.2f}",
                    "Modified Design Tank Energy (kWh)": f"{mod_tank_energy_kWh:.2f}",
                    "Original Design Cost (€/yr)": f"{cost_base:.2f}",
                    "Modified Design Cost (€/yr)": f"{cost_mod:.2f}",
                    "Original Design Efficiency (%)": f"{(base_efficiency.mean() * 100):.2f}",
                    "Modified Design Efficiency (%)": f"{(mod_efficiency.mean() * 100):.2f}"
                })

                total_energy_base += energy_base
                total_energy_mod += energy_mod
                total_cost_base += cost_base
                total_cost_mod += cost_mod
                total_base_efficiency += base_efficiency.mean()
                total_mod_efficiency += mod_efficiency.mean()
                count += 1

        if effectiveness_rows and count > 0:
            effectiveness_rows.append({
                "Compressor": "System Total",
                "Effectiveness (%)": f"{(1 - (total_energy_mod / total_energy_base)) * 100:.2f}",
                "Original Design Energy (kWh)": f"{total_energy_base:.2f}",
                "Modified Design Energy (kWh)": f"{total_energy_mod:.2f}",
                "Original Design Tank Energy (kWh)": f"{base_tank_energy_kWh:.2f}",
                "Modified Design Tank Energy (kWh)": f"{mod_tank_energy_kWh:.2f}",
                "Original Design Cost (€/yr)": f"{total_cost_base:.2f}",
                "Modified Design Cost (€/yr)": f"{total_cost_mod:.2f}",
                "Original Design Efficiency (%)": f"{(total_base_efficiency / count * 100):.2f}",
                "Modified Design Efficiency (%)": f"{(total_mod_efficiency / count * 100):.2f}"
            })

            st.write("### Effectiveness Comparison Table")
            df_summary = pd.DataFrame(effectiveness_rows)
            st.dataframe(df_summary)

            st.write("### 🌍 Carbon Emissions (TCO₂e)")
            co2_factor = 0.341 / 1000  # TCO2e per kWh
            tco2e_base = total_energy_base * co2_factor
            tco2e_mod = total_energy_mod * co2_factor

            st.markdown(f"**Original Design Emissions:** {tco2e_base:.2f} TCO₂e/year")
            st.markdown(f"**Modified Design Emissions:** {tco2e_mod:.2f} TCO₂e/year")
            st.markdown(f"**Reduction:** {tco2e_base - tco2e_mod:.2f} TCO₂e/year")
