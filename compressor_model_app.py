import streamlit as st
st.set_page_config(page_title="Compressed Air Optimization", layout="wide")
st.title("Compressed Air Infrastructure Optimization for Logistics Centres")
import pandas as pd
import numpy as np

# Define constants
air_density = 1.2  # kg/m^3

def calculate_ideal_work(p1, p2, T, Q):
    k = 1.4
    R = 287
    return (k / (k - 1)) * R * T * Q * (((p2 / p1) ** ((k - 1) / k)) - 1)

# ----------------------------
# Step 1: System Parameters
# ----------------------------
st.subheader("Step 1: Compressor System Parameters")
with st.expander("⚙️ System Configuration"):
    flow_rates = []
    powers = []
    for i in range(1, 4):
        flow = st.number_input(f"Rated Flow Compressor {i} (m3/min)", min_value=0.0, value=15.0, key=f"flow{i}")
        power = st.number_input(f"Rated Power Compressor {i} (kW)", min_value=0.0, value=150.0, key=f"power{i}")
        flow_rates.append(flow)
        powers.append(power)

    motor_eff = st.slider("Motor Efficiency (%)", 80, 100, 95, key="motor_eff")

    set_pressure_bar = st.number_input("Set Pressure (bar)", value=7.0, key="set_pressure")
    aftercooler_drop = st.number_input("Aftercooler Pressure Drop (bar)", value=0.2, key="aftercooler")
    dryer_drop = st.number_input("Dryer Pressure Drop (bar)", value=0.2, key="dryer")
    filter_drop = st.number_input("Filter Pressure Drop (bar)", value=0.1, key="filter")
    receiver_volume = st.number_input("Receiver Tank Volume (liters)", value=500.0, key="receiver")

    ambient_temp_c = st.number_input("Ambient Temperature (°C)", min_value=-40.0, value=20.0, key="temp")
    ambient_temp = ambient_temp_c + 273.15
    ambient_pressure_bar = st.number_input("Ambient Pressure (bar)", min_value=0.5, value=1.013, key="amb_press")
    ambient_pressure = ambient_pressure_bar * 100000

    total_pressure_drop = (aftercooler_drop + dryer_drop + filter_drop) * 100000
    adjusted_set_pressure = set_pressure_bar * 100000 + total_pressure_drop

    total_ideal_work = 0
    for i in range(3):
        flow_rate = flow_rates[i] / 60
        Qm = flow_rate * air_density
        work = calculate_ideal_work(ambient_pressure, adjusted_set_pressure, ambient_temp, Qm)
        total_ideal_work += work

    st.markdown(f"**Total Ideal Compressor Work (3 Compressors, with Pressure Losses):** {total_ideal_work/1000:.2f} kW")

# ----------------------------
# Step 2: Upload and Preview Data
# ----------------------------
st.subheader("Step 2: Upload Compressor Data")
uploaded_file = st.file_uploader("Upload Excel or CSV file with compressor data", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.rename(columns={
        "C1 - delivery volume flow rate": "Flow1",
        "C1 - airend discharge temperature": "Temp1",
        "C1 - electrical power consumption": "Power1",
        "C1 On Time": "ON1",
        "C2 - delivery volume flow rate": "Flow2",
        "C2 - airend discharge temperature": "Temp2",
        "C2 - electrical power consumption": "Power2",
        "C2 On Time": "ON2",
        "C3 - delivery volume flow rate": "Flow3",
        "C3 - airend discharge temperature": "Temp3",
        "C3 - electrical power consumption": "Power3",
        "C3 On Time": "ON3"
    }, inplace=True)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M", errors='coerce')
    st.write("### Preview Data")
    st.dataframe(df.head())

    # ----------------------------
    # Step 3: Real Compressor Efficiency Summary
    # ----------------------------
    st.subheader("Step 3: Real Compressor Efficiency Summary")

    summaries = []
    for i in range(1, 4):
        flow_col = f'Flow{i}'
        temp_col = f'Temp{i}'
        power_col = f'Power{i}'
        ontime_col = f'ON{i}'

        if all(col in df.columns for col in [flow_col, temp_col, power_col, ontime_col]):
            flow_m3s = df[flow_col] / 60
            temp_K = df[temp_col] + 273.15
            Qm = flow_m3s * air_density
            df[f'Ideal_Power_{i}_kW'] = calculate_ideal_work(ambient_pressure, adjusted_set_pressure, temp_K, Qm) / 1000
            df[f'Efficiency_{i}'] = df[f'Ideal_Power_{i}_kW'] / df[power_col]
            df[f'Efficiency_{i}'] = df[f'Efficiency_{i}'].clip(upper=1.5)

            summaries.append({
                "Compressor": f"C{i}",
                "Flow (m³/min)": f"{df[flow_col].mean():.2f}",
                "Power (kW)": f"{df[power_col].mean():.2f}",
                "Temp (°C)": f"{df[temp_col].mean():.2f}",
                "Ideal Power (kW)": f"{df[f'Ideal_Power_{i}_kW'].mean():.2f}",
                "Efficiency (%)": f"{(df[f'Efficiency_{i}'].mean() * 100):.2f}",
                "Duty Cycle (%)": f"{(df[ontime_col].mean() / 300 * 100):.2f}"
            })

    if summaries:
        st.write("### Compressor Efficiency Summary Table")
        st.dataframe(pd.DataFrame(summaries))


    # ----------------------------
    # Step 4: Receiver Tank Storage Analysis
    # ----------------------------
    st.subheader("Step 4: Receiver Tank Storage Analysis")

    tank_volume_m3 = receiver_volume / 1000
    pressure_range_bar = st.number_input("Receiver Tank Pressure Range (bar)", min_value=0.1, value=1.0, key="tank_range")
    p1 = adjusted_set_pressure / 100000
    p2 = p1 + pressure_range_bar
    storage_capacity_nm3 = tank_volume_m3 * (p2 - p1) / 1.013
    avg_system_flow = df[[col for col in df.columns if "Flow" in col]].mean(axis=1).mean()

    if avg_system_flow > 0:
        buffer_minutes = storage_capacity_nm3 / avg_system_flow
    else:
        buffer_minutes = 0

    st.markdown(f"**Effective Storage Capacity:** {storage_capacity_nm3:.2f} Nm³")
    st.markdown(f"**Estimated Buffer Time:** {buffer_minutes:.1f} minutes at average system flow of {avg_system_flow:.2f} m³/min")

    # ----------------------------
    # Step 5: Effectiveness Simulation
    # ----------------------------
    st.subheader("Step 5: Effectiveness Simulation")
    with st.expander("🔁 Compare with Modified Configuration"):
        mod_set_pressure_bar = st.number_input("Modified Set Pressure (bar)", value=set_pressure_bar, key="mod_setp")
        mod_aftercooler_drop = st.number_input("Modified Aftercooler Drop (bar)", value=aftercooler_drop, key="mod_ac")
        mod_dryer_drop = st.number_input("Modified Dryer Drop (bar)", value=dryer_drop, key="mod_dryer")
        mod_filter_drop = st.number_input("Modified Filter Drop (bar)", value=filter_drop, key="mod_filter")
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

        for i in range(1, 4):
            flow_col = f"Flow{i}"
            temp_col = f"Temp{i}"
            power_col = f"Power{i}"
            ontime_col = f"ON{i}"

            if all(col in df.columns for col in [flow_col, temp_col, power_col, ontime_col]):
                flow_m3s = df[flow_col] / 60
                temp_K = df[temp_col] + 273.15
                Qm = flow_m3s * air_density

                base_ideal_power = df[f"Ideal_Power_{i}_kW"]
                mod_ideal_power = calculate_ideal_work(ambient_pressure, mod_set_pressure, temp_K, Qm) / 1000

                ε = (base_ideal_power - mod_ideal_power) / base_ideal_power
                ε = ε.clip(lower=0, upper=1)

                duty_cycle = df[ontime_col] / 300.0
                interval_hours = 5 / 60

                energy_base = (base_ideal_power * duty_cycle * interval_hours).sum()
                energy_mod = (mod_ideal_power * duty_cycle * interval_hours).sum()
                cost_base = energy_base * 0.12
                cost_mod = energy_mod * 0.12

                actual_power = df[power_col]
                base_efficiency = (base_ideal_power / actual_power).clip(upper=1.5)
                mod_efficiency = (mod_ideal_power / actual_power).clip(upper=1.5)

                effectiveness_rows.append({
                    "Compressor": f"C{i}",
                    "Energy Base (kWh)": f"{energy_base:.2f}",
                    "Energy Mod (kWh)": f"{energy_mod:.2f}",
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
                "Energy Base (kWh)": f"{total_energy_base:.2f}",
                "Energy Mod (kWh)": f"{total_energy_mod:.2f}",
                "Cost Base (€/yr)": f"{total_cost_base:.2f}",
                "Cost Mod (€/yr)": f"{total_cost_mod:.2f}",
                "Base Efficiency (%)": f"{(total_base_efficiency / count * 100):.2f}",
                "Mod Efficiency (%)": f"{(total_mod_efficiency / count * 100):.2f}"
            })

            st.write("### Effectiveness Comparison Table")
            st.dataframe(pd.DataFrame(effectiveness_rows))

            st.write("### 🌍 Carbon Emissions (TCO₂e)")
            co2_factor = 0.341 / 1000
            tco2e_base = total_energy_base * co2_factor
            tco2e_mod = total_energy_mod * co2_factor

            st.markdown(f"**Base Emissions:** {tco2e_base:.2f} TCO₂e/year")
            st.markdown(f"**Modified Emissions:** {tco2e_mod:.2f} TCO₂e/year")
            st.markdown(f"**Reduction:** {tco2e_base - tco2e_mod:.2f} TCO₂e/year")
