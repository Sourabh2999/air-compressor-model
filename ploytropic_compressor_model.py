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

st.sidebar.subheader("Motor Efficiency")
motor_efficiency = st.sidebar.slider("Motor Efficiency (%)", min_value=80, max_value=100, value=95) / 100.0

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
    flow_rate = flow_rates[i] / 60  # m3/s
    Qm = flow_rate * air_density    # kg/s
    work = calculate_ideal_work(
        ambient_pressure,
        adjusted_set_pressure,
        ambient_temp,
        Qm,
        flow_rate_m3_min=flow_rates[i],
        model=selected_models[i]
    ) / motor_efficiency  # Adjusted by motor efficiency
    total_ideal_work += work

st.markdown(f"**Total Ideal Compressor Work (adjusted for motor efficiency):** {total_ideal_work / 1000:.2f} kW")

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
                    flow_rate_m3_min=valid[flow_col], model=selected_models[i - 1], n=n_values) / (1000 * motor_efficiency)

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
            # Constants for simulation
            R = 287
            T = 293.15
            rho = 1.225

            # Flow preprocessing
            for i in range(1, 4):
                df[f"C{i}_flow_m3s"] = df[f"C{i} - delivery volume flow rate"] / 60
                df[f"C{i}_on"] = df[f"C{i} On Time"]

            df["Q_in"] = sum(df[f"C{i}_flow_m3s"] * df[f"C{i}_on"] for i in range(1, 4))
            df["Q_out"] = df["CO1 - consumption volume flow rate"] / 60
            df["dt"] = df["Timestamp"].diff().dt.total_seconds().fillna(0)

            # Initial pressure
            initial_pressure_bar = df["CO1 - net pressure"].iloc[0]
            initial_pressure_Pa = initial_pressure_bar * 100000
            V_tank_mod = mod_receiver_tank_liters / 1000.0

            # Pressure simulation
            pressure_mod_Pa = [initial_pressure_Pa]
            for idx in range(1, len(df)):
                Q_in = df["Q_in"].iloc[idx]
                Q_out = df["Q_out"].iloc[idx]
                dt = df["dt"].iloc[idx]
                dP = (R * T * rho / V_tank_mod) * (Q_in - Q_out) * dt
                P_new = pressure_mod_Pa[-1] + dP
                P_new = max(P_new, 100000)
                P_new = min(P_new, 1200000)
                pressure_mod_Pa.append(P_new)

            df["Modified_Tank_Pressure_bar"] = np.array(pressure_mod_Pa) / 100000

            # Approximate energy impact due to tank buffer effect
            df["dE_kWh"] = 0.0
            for i in range(1, len(df)):
                P1 = pressure_mod_Pa[i-1]
                P2 = pressure_mod_Pa[i]
                if P2 > 0 and P1 > 0:
                    try:
                        dE = V_tank_mod * (P2 - P1) / (3600 * 1000)
                    except ZeroDivisionError:
                        dE = 0.0
                else:
                    dE = 0.0
                df.at[df.index[i], "dE_kWh"] = dE

            total_buffer_energy_kWh = df["dE_kWh"].sum()
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
                    energy_mod = (mod_ideal_power[valid_mask] * (5 / 60)).sum() + mod_tank_energy_kWh + total_buffer_energy_kWh
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
    available_flows = sorted(model_data['Flow Rate (m³/min)'].astype(float).unique())
    flow = st.sidebar.selectbox(f"Rated Flow Compressor {i} (m3/min)", available_flows, key=f"flow{i}")
    pressure_match = model_data.iloc[(model_data['Flow Rate (m³/min)'].astype(float) - flow).abs().argsort()[:1]]
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

st.sidebar.subheader("Motor Efficiency")
motor_efficiency = st.sidebar.slider("Motor Efficiency (%)", min_value=80, max_value=100, value=95) / 100.0

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
                flow_vals = pressure_match['Flow Rate (m³/min)'].astype(float)
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
    flow_rate = flow_rates[i] / 60  # m3/s
    Qm = flow_rate * air_density    # kg/s
    work = calculate_ideal_work(
        ambient_pressure,
        adjusted_set_pressure,
        ambient_temp,
        Qm,
        flow_rate_m3_min=flow_rates[i],
        model=selected_models[i]
    ) / motor_efficiency  # Adjusted by motor efficiency
    total_ideal_work += work

st.markdown(f"**Total Ideal Compressor Work (adjusted for motor efficiency):** {total_ideal_work / 1000:.2f} kW")

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
                    flow_rate_m3_min=valid[flow_col], model=selected_models[i - 1], n=n_values) / (1000 * motor_efficiency)

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
            # Constants for simulation
            R = 287
            T = 293.15
            rho = 1.225

            # Flow preprocessing
            for i in range(1, 4):
                df[f"C{i}_flow_m3s"] = df[f"C{i} - delivery volume flow rate"] / 60
                df[f"C{i}_on"] = df[f"C{i} On Time"]

            df["Q_in"] = sum(df[f"C{i}_flow_m3s"] * df[f"C{i}_on"] for i in range(1, 4))
            df["Q_out"] = df["CO1 - consumption volume flow rate"] / 60
            df["dt"] = df["Timestamp"].diff().dt.total_seconds().fillna(0)

            # Initial pressure
            initial_pressure_bar = df["CO1 - net pressure"].iloc[0]
            initial_pressure_Pa = initial_pressure_bar * 100000
            V_tank_mod = mod_receiver_tank_liters / 1000.0

            # Pressure simulation
            pressure_mod_Pa = [initial_pressure_Pa]
            for idx in range(1, len(df)):
                Q_in = df["Q_in"].iloc[idx]
                Q_out = df["Q_out"].iloc[idx]
                dt = df["dt"].iloc[idx]
                dP = (R * T * rho / V_tank_mod) * (Q_in - Q_out) * dt
                P_new = pressure_mod_Pa[-1] + dP
                P_new = max(P_new, 100000)
                P_new = min(P_new, 1200000)
                pressure_mod_Pa.append(P_new)

            df["Modified_Tank_Pressure_bar"] = np.array(pressure_mod_Pa) / 100000

            # Approximate energy impact due to tank buffer effect
            df["dE_kWh"] = 0.0
            for i in range(1, len(df)):
                P1 = pressure_mod_Pa[i-1]
                P2 = pressure_mod_Pa[i]
                if P2 > 0 and P1 > 0:
                    try:
                        dE = V_tank_mod * (P2 - P1) / (3600 * 1000)
                    except ZeroDivisionError:
                        dE = 0.0
                else:
                    dE = 0.0
                df.at[df.index[i], "dE_kWh"] = dE

            total_buffer_energy_kWh = df["dE_kWh"].sum()
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
                    energy_mod = (mod_ideal_power[valid_mask] * (5 / 60)).sum() + mod_tank_energy_kWh + total_buffer_energy_kWh
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
# ---------------------------------------------------------------
# 🔍 Compressor System Optimization: Best 2 or 3 model recommendation
# ---------------------------------------------------------------

import itertools

best_configurations = []

# Ensure model data is loaded
if 'df' in locals() and not df.empty and not df_models.empty and "CO1 - consumption volume flow rate" in df.columns:
    demand_series = df["CO1 - consumption volume flow rate"].fillna(0).values  # m3/min
    demand_series = np.clip(demand_series, a_min=0, a_max=None)
    time_interval_hr = 5 / 60  # each row = 5 minutes
    electricity_rate = 0.12  # €/kWh
    annual_hours = 24 * 365
    total_steps = len(demand_series)

    # Preprocess compressor models
    df_models_clean = df_models.copy()
    df_models_clean.rename(columns={df_models_clean.columns[0]: "Model"}, inplace=True)
    df_models_clean.dropna(subset=["Model", "Flow Rate (m³/min)", "Drive Motor Rated Power (kW)"], inplace=True)
    df_models_clean["Flow"] = df_models_clean["Flow Rate (m³/min)"].astype(float)
    df_models_clean["Power"] = df_models_clean["Drive Motor Rated Power (kW)"].astype(float)
    if "Speed Control" in df_models_clean.columns:
        df_models_clean["Type"] = df_models_clean["Speed Control"].fillna("Fixed")
    else:
        df_models_clean["Type"] = "Fixed"

    # Compute average and peak demand
    avg_demand = demand_series.mean()
    peak_demand = demand_series.max()
    min_required_flow = avg_demand * 1.1  # 10% margin

    # Filter models that can supply the minimum required demand
    def is_model_suitable(row):
        if row["Type"] == "VSD":
            return row["Flow"] >= 1  # VSD assumed min = 1 m3/min
        else:
            return row["Flow"] >= min_required_flow

    df_models_clean = df_models_clean[df_models_clean.apply(is_model_suitable, axis=1)]

    all_models = df_models_clean[["Model", "Flow", "Power", "Type"]].drop_duplicates()

    # Evaluate all 2-combo and 3-combo systems
    for r in [2, 3]:
        for combo in itertools.combinations(all_models.itertuples(index=False), r):
            models = list(combo)
            total_energy_kWh = 0.0
            sufficient = True
            on_time_counts = {m.Model: 0 for m in models}

            for demand in demand_series:
                assigned = 0.0
                step_energy = 0.0
                remaining = demand
                sorted_models = sorted(models, key=lambda x: (x.Type != "VSD", -x.Flow))

                for m in sorted_models:
                    if remaining <= 0:
                        break
                    flow = min(m.Flow, remaining)
                    if flow > 0:
                        if m.Type == "VSD":
                            power = (flow / m.Flow) * m.Power
                        else:
                            power = m.Power
                            on_time_counts[m.Model] += 1  # Only fixed-speed is counted for on-time

                        assigned += flow
                        step_energy += power * time_interval_hr
                        remaining -= flow

                if assigned < demand:
                    sufficient = False
                    break
                total_energy_kWh += step_energy

            if sufficient:
                cost = total_energy_kWh * electricity_rate
                total_demand_volume = demand_series.sum()
                duty_cycles = {model: round((count / total_steps) * 100, 2) for model, count in on_time_counts.items()}
                best_configurations.append({
                    "Models": [m.Model for m in models],
                    "Types": [m.Type for m in models],
                    "Total Energy (kWh)": round(total_energy_kWh, 2),
                    "Estimated Cost (€/yr)": round(cost, 2),
                    "SEC (kW/m³/min)": round(total_energy_kWh / total_demand_volume, 4),
                    "Duty Cycles (%)": duty_cycles
                })

    # Rank and display top configurations
    best_configurations.sort(key=lambda x: x["Total Energy (kWh)"])
    st.subheader("🔍 Recommended Compressor System Configuration")
    st.write("Based on your actual demand profile and available compressor models:")
    # Flatten and display cleanly
    display_df = []
    for row in best_configurations[:2]:
        flat_row = {
            "Compressor 1": row["Models"][0],
            "Type 1": row["Types"][0],
            "Compressor 2": row["Models"][1] if len(row["Models"]) > 1 else None,
            "Type 2": row["Types"][1] if len(row["Types"]) > 1 else None,
            "Compressor 3": row["Models"][2] if len(row["Models"]) > 2 else None,
            "Type 3": row["Types"][2] if len(row["Types"]) > 2 else None,
            "Total Energy (kWh)": row["Total Energy (kWh)"],
            "Estimated Cost (€/yr)": row["Estimated Cost (€/yr)"],
            "SEC (kW/m³/min)": row["SEC (kW/m³/min)"]
        }
        for model, duty in row["Duty Cycles (%)"].items():
            flat_row[f"Duty {model} (%)"] = duty
        display_df.append(flat_row)

    st.dataframe(pd.DataFrame(display_df))
