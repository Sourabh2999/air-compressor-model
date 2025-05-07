# compressor_model_app.py 
import streamlit as st
import numpy as np
import pandas as pd

# ----------------------------
# Step 1: Define System Boundaries and Components
# ----------------------------
st.set_page_config(page_title="Air Compressor Energy Model", layout="wide")
st.title("Industrial Air Compressor System Energy Modeling")

st.sidebar.header("System Parameters")

# Basic Compressor Specs
st.sidebar.subheader("Compressor Specifications")
flow_rates = []
powers = []
for i in range(1, 4):
    flow = st.sidebar.number_input(f"Rated Flow Compressor {i} (m3/min)", min_value=0.0, value=15.0, key=f"flow{i}")
    power = st.sidebar.number_input(f"Rated Power Compressor {i} (kW)", min_value=0.0, value=150.0, key=f"power{i}")
    flow_rates.append(flow)
    powers.append(power)

motor_eff = st.sidebar.slider("Motor Efficiency (%)", 80, 100, 95)

# Operating Conditions
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
air_density = 1.225

def calculate_ideal_work(Pa, P2, Ta, Qm):
    term = (P2 / Pa)**((k - 1) / k) - 1
    return (k / (k - 1)) * Qm * R * Ta * term

def calculate_tank_energy(Pa, P2, V):
    return (P2 * V / (k - 1)) * ((P2 / Pa)**((k - 1) / k) - 1)

# Step 1 Output
total_ideal_work = 0
for i in range(3):
    flow_rate = flow_rates[i] / 60
    Qm = flow_rate * air_density
    work = calculate_ideal_work(ambient_pressure, adjusted_set_pressure, ambient_temp, Qm)
    total_ideal_work += work

st.subheader("Step 1: Ideal Compressor Work Calculation")
st.markdown(f"**Total Ideal Compressor Work (3 Compressors, with Pressure Losses):** {total_ideal_work/1000:.2f} kW")

# ----------------------------
# Step 2: Upload Historical Compressor Data
# ----------------------------
st.subheader("Step 2: Upload Historical Compressor Data")
uploaded_file = st.file_uploader("Upload Compressor Data File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df.rename(columns=lambda x: x.strip(), inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    st.write("### File Preview")
    st.dataframe(df.head())

    # ----------------------------
    # Step 3: Real Efficiency Calculation
    # ----------------------------
    st.subheader("Step 3: Real Compressor Efficiency Summary")
    summaries = []
    for i in range(1, 4):
        flow_col = f'Flow{i}'
        temp_col = f'Temp{i}'
        power_col = f'Power{i}'

        if flow_col in df.columns and temp_col in df.columns and power_col in df.columns:
            flow_m3s = df[flow_col] / 60
            temp_K = df[temp_col] + 273.15
            Qm = flow_m3s * air_density
            df[f'Ideal_Power_{i}_kW'] = calculate_ideal_work(ambient_pressure, adjusted_set_pressure, temp_K, Qm) / 1000
            df[f'Efficiency_{i}'] = df[f'Ideal_Power_{i}_kW'] / df[power_col]
            df[f'Efficiency_{i}'] = df[f'Efficiency_{i}'].clip(upper=1.5)

            summaries.append({
                "Compressor": f"C{i}",
                "Avg Flow (m³/min)": f"{df[flow_col].mean():.2f}",
                "Avg Power (kW)": f"{df[power_col].mean():.2f}",
                "Avg Temp (°C)": f"{df[temp_col].mean():.2f}",
                "Avg Ideal Power (kW)": f"{df[f'Ideal_Power_{i}_kW'].mean():.2f}",
                "Avg Efficiency (%)": f"{(df[f'Efficiency_{i}'].mean() * 100):.2f}"
            })

    if summaries:
        st.write("### Compressor Efficiency Summary Table")
        st.dataframe(pd.DataFrame(summaries))

    # ----------------------------
    # Step 4 and Step 5 Updated
    # ----------------------------
    st.subheader("Step 4 & 5: Effectiveness and Carbon Emission Evaluation")
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
                base_efficiency = base_ideal_power / actual_power
                mod_efficiency = mod_ideal_power / actual_power
                base_efficiency = base_efficiency.clip(upper=1.5)
                mod_efficiency = mod_efficiency.clip(upper=1.5)

                effectiveness = 1 - (energy_mod / energy_base)

                effectiveness_rows.append({
                    "Compressor": f"C{i}",
                    "Effectiveness (%)": f"{effectiveness * 100:.2f}",
                    "Energy Base (kWh)": f"{energy_base:.2f}",
                    "Energy Mod (kWh)": f"{energy_mod:.2f}",
                    "Tank Energy Base (kWh)": f"{base_tank_energy_kWh:.2f}",
                    "Tank Energy Mod (kWh)": f"{mod_tank_energy_kWh:.2f}",
                    "Cost Base (€/yr)": f"{cost_base:.2f}",
                    "Cost Mod (€/yr)": f"{cost_mod:.2f}",
                    "Base Efficiency (%)": f"{(base_efficiency.mean() * 100):.2f}",
                    "Mod Efficiency (%)": f"{(mod_efficiency.mean() * 100):.2f}"
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
                "Energy Base (kWh)": f"{total_energy_base:.2f}",
                "Energy Mod (kWh)": f"{total_energy_mod:.2f}",
                "Tank Energy Base (kWh)": f"{base_tank_energy_kWh:.2f}",
                "Tank Energy Mod (kWh)": f"{mod_tank_energy_kWh:.2f}",
                "Cost Base (€/yr)": f"{total_cost_base:.2f}",
                "Cost Mod (€/yr)": f"{total_cost_mod:.2f}",
                "Base Efficiency (%)": f"{(total_base_efficiency / count * 100):.2f}",
                "Mod Efficiency (%)": f"{(total_mod_efficiency / count * 100):.2f}"
            })

            st.write("### Effectiveness Comparison Table")
            df_summary = pd.DataFrame(effectiveness_rows)
            st.dataframe(df_summary)

            st.write("### 🌍 Carbon Emissions (TCO₂e)")
            co2_factor = 0.341 / 1000  # TCO2e per kWh
            tco2e_base = total_energy_base * co2_factor
            tco2e_mod = total_energy_mod * co2_factor

            st.markdown(f"**Base Emissions:** {tco2e_base:.2f} TCO₂e/year")
            st.markdown(f"**Modified Emissions:** {tco2e_mod:.2f} TCO₂e/year")
            st.markdown(f"**Reduction:** {tco2e_base - tco2e_mod:.2f} TCO₂e/year")
