import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Compressor Infrastructure Energy Model", layout="wide")
st.title("Compressed Air Infrastructure Optimization for Logistics Centres")

st.sidebar.header("System Parameters")

flow_rates = []
powers = []
selected_models = []

try:
    df_models = pd.read_csv("Compressors Models_python.csv", encoding='ISO-8859-1')
    df_models.rename(columns={df_models.columns[0]: "Model"}, inplace=True)
    unique_models = sorted(df_models['Model'].dropna().unique().tolist())
except Exception:
    df_models = pd.DataFrame()
    unique_models = []

for i in range(1, 4):
    model = st.sidebar.selectbox(f"Compressor {i} Model", unique_models, index=0 if unique_models else None, key=f"model{i}")
    model_data = df_models[df_models['Model'] == model]
    available_flows = sorted(model_data['Flow Rate (mÂ³/min)'].astype(float).unique())
    flow = st.sidebar.selectbox(f"Rated Flow Compressor {i} (m3/min)", available_flows, key=f"flow{i}")
    pressure_match = model_data.iloc[(model_data['Flow Rate (mÂ³/min)'].astype(float) - flow).abs().argsort()[:1]]
    rated_power = pressure_match['Drive Motor Rated Power (kW)'].values[0] if not pressure_match.empty else 0.0
    st.sidebar.markdown(f"**Rated Power Compressor {i}:** {rated_power:.2f} kW")

    selected_models.append(model)
    flow_rates.append(flow)
    powers.append(rated_power)

st.sidebar.subheader("Operating Conditions")
ambient_temp_c = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-40.0, value=20.0)
ambient_temp = ambient_temp_c + 273.15
ambient_pressure_bar = st.sidebar.number_input("Ambient Pressure (bar)", min_value=0.5, value=1.013)
ambient_pressure = ambient_pressure_bar * 100000
set_pressure_bar = st.sidebar.number_input("Compressor Set Pressure (bar)", min_value=1.0, value=7.0)
set_pressure = set_pressure_bar * 100000

aftercooler_drop = st.sidebar.number_input("Aftercooler Pressure Drop (bar)", min_value=0.0, value=0.1)
dryer_drop = st.sidebar.number_input("Dryer Pressure Drop (bar)", min_value=0.0, value=0.2)
filter_drop = st.sidebar.number_input("Filter Pressure Drop (bar)", min_value=0.0, value=0.1)

total_pressure_drop = (aftercooler_drop + dryer_drop + filter_drop) * 100000
adjusted_set_pressure = set_pressure + total_pressure_drop

st.sidebar.subheader("Receiver Tank")
receiver_tank_liters = st.sidebar.number_input("Receiver Tank Volume (liters)", min_value=100.0, value=1000.0, step=50.0)
receiver_tank_m3 = receiver_tank_liters / 1000.0

R = 287
k = 1.4
n_default = 1.3
air_density = 1.225

def calculate_ideal_work(Pa, P2, Ta, Qm, flow_rate_m3_min=None, model=None, n=n_default):
    if model and not df_models.empty:
        model_data = df_models[df_models['Model'] == model]
        if not model_data.empty and flow_rate_m3_min is not None:
            try:
                pressure_match = model_data.iloc[(model_data['Operating Pressure (bar)'] - P2/100000).abs().argsort()[:3]]
                flow_vals = pressure_match['Flow Rate (mÂ³/min)'].astype(float)
                power_vals = pressure_match['Drive Motor Rated Power (kW)'].astype(float)
                if len(flow_vals.unique()) > 1:
                    from scipy.interpolate import interp1d
                    interpolator = interp1d(flow_vals, power_vals, bounds_error=False, fill_value="extrapolate")
                    return interpolator(flow_rate_m3_min) * 1000
            except Exception:
                pass
    term = (P2 / Pa)**((n - 1) / n) - 1
    return (n / (n - 1)) * Qm * R * Ta * term

def calculate_tank_energy(Pa, P2, V):
    return (P2 * V / (k - 1)) * ((P2 / Pa)**((k - 1) / k) - 1)
st.subheader("Ideal Compressor Work Calculation")
total_ideal_work = 0
for i in range(3):
    flow_rate = flow_rates[i] / 60  # convert to m3/s
    Qm = flow_rate * air_density  # mass flow rate (kg/s)
    work = calculate_ideal_work(
        ambient_pressure,
        adjusted_set_pressure,
        ambient_temp,
        Qm,
        flow_rate_m3_min=flow_rates[i],
        model=selected_models[i]
    )
    total_ideal_work += work

st.markdown(f"**Total Ideal Compressor Work (3 Compressors, with Pressure Losses):** {total_ideal_work / 1000:.2f} kW")

st.subheader("Upload Historical Compressor Data")
uploaded_file = st.file_uploader("Upload Compressor Data File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    rename_map = {
        "C1 - delivery volume flow rate": "C1 - delivery volume flow rate",
        "C2 - delivery volume flow rate": "C2 - delivery volume flow rate",
        "C3 - delivery volume flow rate": "C3 - delivery volume flow rate",
        "C1 - intake temperature": "Temp1",
        "C2 - intake temperature": "Temp2",
        "C3 - intake temperature": "Temp3",
        "C1 - airend discharge temperature": "T2_1",
        "C2 - airend discharge temperature": "T2_2",
        "C3 - airend discharge temperature": "T2_3",
        "C1 - local net pressure": "P2_1",
        "C2 - local net pressure": "P2_2",
        "C3 - local net pressure": "P2_3",
        "C1 - electrical power consumption": "Power1",
        "C2 - electrical power consumption": "Power2",
        "C3 - electrical power consumption": "Power3"
    }
    df.rename(columns=rename_map, inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    st.write("### File Preview")
    st.dataframe(df.head())

    st.subheader("Polytropic Exponent (n) Calculation from Data")
    for i in range(1, 4):
        on_col = f"C{i} On Time"
        power_col = f"Power{i}"
        intake_temp_col = f"Temp{i}"
        discharge_temp_col = f"T2_{i}"
        pressure_col = f"P2_{i}"

        if all(col in df.columns for col in [on_col, power_col, intake_temp_col, discharge_temp_col, pressure_col]):
            comp_df = df[(df[on_col] == 1) & (df[power_col] > 3)].copy()
            valid = (
                (comp_df[pressure_col] > ambient_pressure_bar * 1.1) &
                (comp_df[intake_temp_col] > 0) &
                (comp_df[discharge_temp_col] > 0) &
                ((comp_df[discharge_temp_col] - comp_df[intake_temp_col]).abs() > 5)
            )
            comp_df = comp_df[valid]

            P1 = ambient_pressure_bar
            P2 = comp_df[pressure_col]
            T1 = comp_df[intake_temp_col] + 273.15
            T2 = comp_df[discharge_temp_col] + 273.15

            V1_by_V2 = (T1 * P2) / (T2 * P1)
            n_vals = np.log(P2 / P1) / np.log(V1_by_V2)
            n_vals = n_vals.replace([np.inf, -np.inf], np.nan).dropna()

            df.loc[comp_df.index, f'n_{i}'] = n_vals

    st.subheader("Real Compressor Efficiency Summary")
    summaries = []
    for i in range(1, 4):
        flow_col = f"C{i} - delivery volume flow rate"
        temp_col = f'Temp{i}'
        power_col = f'Power{i}'
        on_col = f'C{i} On Time'
        n_col = f'n_{i}'

        if flow_col in df.columns and temp_col in df.columns and power_col in df.columns and on_col in df.columns:
            valid = df[(df[on_col] == 1) & (df[power_col] > 3) & (df[flow_col] > 0)]
            if not valid.empty:
                flow_m3s = valid[flow_col] / 60
                temp_K = valid[temp_col] + 273.15
                Qm = flow_m3s * air_density
                n_values = valid[n_col].fillna(n_default)

                df.loc[valid.index, f'Ideal_Power_{i}_kW'] = calculate_ideal_work(
                    ambient_pressure, adjusted_set_pressure, temp_K, Qm,
                    flow_rate_m3_min=valid[flow_col], model=selected_models[i - 1], n=n_values) / 1000

                actual_power = valid[power_col]
                df.loc[valid.index, f'Efficiency_{i}'] = df.loc[valid.index, f'Ideal_Power_{i}_kW'] / actual_power.replace(0, np.nan)
                df.loc[valid.index, f'Efficiency_{i}'] = df.loc[valid.index, f'Efficiency_{i}'].clip(upper=1.5)

                avg_power = actual_power.mean()
                avg_flow = valid[flow_col].mean()
                sec_kw_per_m3min = avg_power / avg_flow if avg_flow > 0 else np.nan

                duty_cycle = df[on_col].mean() * 100
                avg_n_filtered = n_values.mean()

                summaries.append({
                    "Compressor": f"C{i}",
                    "Avg Air Generated (m³/min)": f"{avg_flow:.2f}",
                    "Avg Power Consumed (kW)": f"{avg_power:.2f}",
                    "Avg Temp (°C)": f"{valid[temp_col].mean():.2f}",
                    "Avg Ideal Power (kW)": f"{df.loc[valid.index, f'Ideal_Power_{i}_kW'].mean():.2f}",
                    "Avg Efficiency (%)": f"{(df.loc[valid.index, f'Efficiency_{i}'].mean() * 100):.2f}",
                    "Duty Cycle (%)": f"{duty_cycle:.2f}",
                    "Avg n ": f"{avg_n_filtered:.3f}",
                    "SEC (kW/m³/min)": f"{sec_kw_per_m3min:.3f}"
                })

    if summaries:
        st.write("### Compressor Efficiency Summary Table")
        st.dataframe(pd.DataFrame(summaries))





        # Step 4: Effectiveness and Carbon Emission Evaluation
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
                flow_col = f"C{i} - delivery volume flow rate"
                temp_col = f"Temp{i}"
                power_col = f"Power{i}"
                on_col = f"C{i} On Time"
                n_col = f"n_{i}"

                if flow_col in df.columns and temp_col in df.columns and power_col in df.columns:
                    valid_mask = (df[on_col] == 1) & (df[power_col] > 3) & (df[flow_col] > 0)
                    flow_m3s = df[flow_col] / 60
                    temp_K = df[temp_col] + 273.15
                    Qm = flow_m3s * air_density
                    n_values = df[n_col].where(valid_mask, np.nan).fillna(n_default)

                    base_ideal_power = df[f"Ideal_Power_{i}_kW"]
                    mod_ideal_power = calculate_ideal_work(ambient_pressure, mod_set_pressure, temp_K, Qm, n=n_values) / 1000

                    energy_base = (base_ideal_power[valid_mask] * (5 / 60)).sum() + base_tank_energy_kWh
                    energy_mod = (mod_ideal_power[valid_mask] * (5 / 60)).sum() + mod_tank_energy_kWh
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
                        "Original Energy (kWh)": f"{energy_base:.2f}",
                        "Modified Energy (kWh)": f"{energy_mod:.2f}",
                        "Original Cost (€/yr)": f"{cost_base:.2f}",
                        "Modified Cost (€/yr)": f"{cost_mod:.2f}",
                        "Original Efficiency (%)": f"{(base_efficiency[valid_mask].mean() * 100):.2f}",
                        "Modified Efficiency (%)": f"{(mod_efficiency[valid_mask].mean() * 100):.2f}"
                    })

                    total_energy_base += energy_base
                    total_energy_mod += energy_mod
                    total_cost_base += cost_base
                    total_cost_mod += cost_mod
                    total_base_efficiency += base_efficiency[valid_mask].mean()
                    total_mod_efficiency += mod_efficiency[valid_mask].mean()
                    count += 1

            if effectiveness_rows and count > 0:
                effectiveness_rows.append({
                    "Compressor": "System Total",
                    "Effectiveness (%)": f"{(1 - (total_energy_mod / total_energy_base)) * 100:.2f}",
                    "Original Energy (kWh)": f"{total_energy_base:.2f}",
                    "Modified Energy (kWh)": f"{total_energy_mod:.2f}",
                    "Original Cost (€/yr)": f"{total_cost_base:.2f}",
                    "Modified Cost (€/yr)": f"{total_cost_mod:.2f}",
                    "Original Efficiency (%)": f"{(total_base_efficiency / count * 100):.2f}",
                    "Modified Efficiency (%)": f"{(total_mod_efficiency / count * 100):.2f}"
                })

                st.write("### Effectiveness Comparison Table")
                df_summary = pd.DataFrame(effectiveness_rows)
                st.dataframe(df_summary)

                st.write("### 🌍 Carbon Emissions (TCO₂e)")
                co2_factor = 0.341 / 1000  # TCO₂e per kWh
                tco2e_base = total_energy_base * co2_factor
                tco2e_mod = total_energy_mod * co2_factor

                st.markdown(f"**Original Design Emissions:** {tco2e_base:.2f} TCO₂e/year")
                st.markdown(f"**Modified Design Emissions:** {tco2e_mod:.2f} TCO₂e/year")
                st.markdown(f"**Reduction:** {tco2e_base - tco2e_mod:.2f} TCO₂e/year")
