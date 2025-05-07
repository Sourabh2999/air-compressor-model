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
    # Step 3: Efficiency and Effectiveness Analysis
    # ----------------------------
    st.subheader("Step 3: Effectiveness and Tank Impact Analysis")

    mod_set_pressure_bar = st.number_input("Modified Set Pressure (bar)", value=set_pressure_bar)
    mod_aftercooler_drop = st.number_input("Modified Aftercooler Drop (bar)", value=aftercooler_drop)
    mod_dryer_drop = st.number_input("Modified Dryer Drop (bar)", value=dryer_drop)
    mod_filter_drop = st.number_input("Modified Filter Drop (bar)", value=filter_drop)

    mod_total_drop = (mod_aftercooler_drop + mod_dryer_drop + mod_filter_drop) * 100000
    mod_set_pressure = mod_set_pressure_bar * 100000 + mod_total_drop

    base_tank_energy_kWh = calculate_tank_energy(ambient_pressure, adjusted_set_pressure, receiver_tank_m3) / 3600
    mod_tank_energy_kWh = calculate_tank_energy(ambient_pressure, mod_set_pressure, receiver_tank_m3) / 3600

    st.write(f"**Receiver Tank Energy - Base:** {base_tank_energy_kWh:.2f} kWh")
    st.write(f"**Receiver Tank Energy - Modified:** {mod_tank_energy_kWh:.2f} kWh")

    st.write("\n")
    st.markdown("**Effectiveness (%):**")
    if base_tank_energy_kWh > 0:
        effectiveness = (base_tank_energy_kWh - mod_tank_energy_kWh) / base_tank_energy_kWh * 100
    else:
        effectiveness = 0.0

    st.metric(label="Effectiveness from Tank Adjustment", value=f"{effectiveness:.2f}%")

    st.subheader("Step 4: Carbon Impact")
    co2_factor = 0.341 / 1000
    tco2e_base = base_tank_energy_kWh * co2_factor
    tco2e_mod = mod_tank_energy_kWh * co2_factor

    st.markdown(f"**Base Emissions:** {tco2e_base:.2f} TCO₂e/year")
    st.markdown(f"**Modified Emissions:** {tco2e_mod:.2f} TCO₂e/year")
    st.markdown(f"**Reduction:** {tco2e_base - tco2e_mod:.2f} TCO₂e/year")
