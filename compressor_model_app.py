import streamlit as st
import pandas as pd
import numpy as np

# Define constants
ambient_pressure = 101325  # Pa
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
    set_pressure_bar = st.number_input("Set Pressure (bar)", value=7.0)
    aftercooler_drop = st.number_input("Aftercooler Pressure Drop (bar)", value=0.2)
    dryer_drop = st.number_input("Dryer Pressure Drop (bar)", value=0.2)
    filter_drop = st.number_input("Filter Pressure Drop (bar)", value=0.1)
    receiver_volume = st.number_input("Receiver Tank Volume (liters)", value=500.0)

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

    st.write("### Preview Data")
    st.dataframe(df.head())

    # ----------------------------
    # Step 3: Calculate Ideal Power
    # ----------------------------
    st.subheader("Step 3: Calculate Ideal Power")
    total_drop = (aftercooler_drop + dryer_drop + filter_drop) * 100000
    set_pressure = set_pressure_bar * 100000 + total_drop

    for i in range(1, 4):
        flow_col = f"Flow{i}"
        temp_col = f"Temp{i}"

        if flow_col in df.columns and temp_col in df.columns:
            flow_m3s = df[flow_col] / 60
            temp_K = df[temp_col] + 273.15
            Qm = flow_m3s * air_density

            df[f"Ideal_Power_{i}_kW"] = calculate_ideal_work(ambient_pressure, set_pressure, temp_K, Qm) / 1000

    st.success("Ideal power calculations added to dataframe.")
    st.dataframe(df[[col for col in df.columns if "Ideal_Power" in col]].head())

    # ----------------------------
    # Step 4: Optional Losses Summary
    # ----------------------------
    st.subheader("Step 4: Optional Pressure Drop Losses Summary")
    st.markdown(f"**Aftercooler Loss:** {aftercooler_drop} bar")
    st.markdown(f"**Dryer Loss:** {dryer_drop} bar")
    st.markdown(f"**Filter Loss:** {filter_drop} bar")
    st.markdown(f"**Receiver Tank Volume:** {receiver_volume} liters")

    # ----------------------------
    # Step 5: Effectiveness Simulation using ON/OFF and Receiver Tank
    # ----------------------------
    st.subheader("Step 5: Effectiveness Simulation")
    with st.expander("🔁 Compare with Modified Configuration"):
        mod_set_pressure_bar = st.number_input("Modified Set Pressure (bar)", value=set_pressure_bar)
        mod_aftercooler_drop = st.number_input("Modified Aftercooler Drop (bar)", value=aftercooler_drop)
        mod_dryer_drop = st.number_input("Modified Dryer Drop (bar)", value=dryer_drop)
        mod_filter_drop = st.number_input("Modified Filter Drop (bar)", value=filter_drop)
        mod_receiver_volume = st.number_input("Modified Receiver Tank Volume (liters)", value=receiver_volume)

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
            ontime_col = f"C{i} ON Time"

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
