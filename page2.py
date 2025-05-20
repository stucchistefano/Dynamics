import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Data extraction

# Lista dei nomi dei soggetti
subject_names = ['Subject1', 'Subject2','Subject3','Subject4', 'Subject5','Subject6',
                 'Subject7', 'Subject8','Subject9','Subject10', 'Subject11','Subject12',
                 'Subject13', 'Subject14','Subject15']

# Dizionario di dizionari
Dict_dict ={}

# Itera su ogni nome soggetto e estrai dizionario di dataframe come prima
for n in subject_names:
    folder_path = f'GRFFolder/{n}'

    # Dizionario per contenere i dataframe
    dataframes_dict = {}

    if n == 'Subject12':
        file_names = ['SJ1', 'SJ3', 'DJ1', 'DJ2', 'DJ3']
    else:
      if n == 'Subject13':
        file_names = ['SJ2', 'SJ3', 'DJ1', 'DJ2', 'DJ3']
      else:
        file_names = ['SJ1', 'SJ2', 'SJ3', 'DJ1', 'DJ2', 'DJ3']

    # Itera su ogni nome file e carica l'excel in un DataFrame
    for name in file_names:
        file_path = os.path.join(folder_path, f'{name}_vector.xlsx')
        try:
            df = pd.read_excel(file_path)
            dataframes_dict[name] = df
            #print(f"Caricato: {name}")
        except Exception as e:
            print(f"Errore nel caricamento di {name}: {e}")
    Dict_dict[n] = dataframes_dict
    print(f"Caricato: {n}")


st.title("Dynamics GRF")

# Selectbox (into the sidebar)
# Object-like approach
subject_input= st.selectbox(
    "Select your player",
    ('Subject1','Subject2','Subject3','Subject4','Subject5','Subject6',
     'Subject7','Subject8','Subject9', 'Subject10','Subject11','Subject12',
     'Subject13','Subject14','Subject15')
    )

if subject_input == 'Subject12':
    type_input = st.selectbox(
        "Select the exercise",
        ('SJ1', 'SJ3', 'DJ1', 'DJ2', 'DJ3')
    )
else:
    if subject_input == 'Subject13':
        type_input = st.selectbox(
            "Select the exercise",
            ('SJ2', 'SJ3', 'DJ1', 'DJ2', 'DJ3')
        )
    else:
        type_input = st.selectbox(
            "Select the exercise",
            ('SJ1', 'SJ2', 'SJ3', 'DJ1', 'DJ2', 'DJ3')
        )


# Esempio per accedere a un dataframe:
df = Dict_dict[subject_input][type_input]
df


st.header("Interactive GRF plot")
# Make sure time and GRF columns are float
df["Time"] = df["Time"].astype(float)
df["Total_vert_normalized_GRF"] = df["Total_vert_normalized_GRF"].astype(float)
df["Left_vert_GRF_normalized"] = df["Left_vert_GRF_normalized"].astype(float)
df["Right_vert_GRF_normalized"] = df["Right_vert_GRF_normalized"].astype(float)

# --- Plot 1: Total GRF ---
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["Time"], y=df["Total_vert_normalized_GRF"],
                          mode="lines", name="Total GRF/BW", line=dict(color="blue")))
fig1.update_layout(title="Total GRF Normalized on BW",
                   xaxis_title="Time [s]",
                   yaxis_title="Total GRF [N/BW]",
                   template="plotly_white")
st.plotly_chart(fig1)

# --- Plot 2: Left and Right GRF Overplotted ---
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["Time"], y=df["Left_vert_GRF_normalized"],
                          mode="lines", name="Left GRF/BW", line=dict(color="green")))
fig2.add_trace(go.Scatter(x=df["Time"], y=df["Right_vert_GRF_normalized"],
                          mode="lines", name="Right GRF/BW", line=dict(color="red")))
fig2.update_layout(title="Left and Right GRF Normalized on BW",
                   xaxis_title="Time [s]",
                   yaxis_title="GRF [N/BW]",
                   template="plotly_white")
st.plotly_chart(fig2)


# Try of scrollitelling
st.header("Try of scrollytelling with mouse")
# Slider to simulate scrolling (1 to full length of data)
max_index = len(df)
step_size = 1  # Controls how many samples to reveal per scroll
scroll_position = st.slider("Scroll to explore jump", 1, max_index, 50, step=step_size)

# Subset of data to show
visible_df = df.iloc[:scroll_position]

# Plot 1: Total GRF
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Total_vert_normalized_GRF"],
    mode="lines", name="Total GRF/BW", line=dict(color="blue")
))
fig1.update_layout(
    title="Total GRF Normalized on BW",
    xaxis_title="Time [s]",
    yaxis_title="Total GRF [N/BW]",
    template="plotly_white"
)
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Left and Right GRF
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Left_vert_GRF_normalized"],
    mode="lines", name="Left GRF/BW", line=dict(color="green")
))
fig2.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Right_vert_GRF_normalized"],
    mode="lines", name="Right GRF/BW", line=dict(color="red")
))
fig2.update_layout(
    title="Left and Right GRF Normalized on BW",
    xaxis_title="Time [s]",
    yaxis_title="GRF [N/BW]",
    template="plotly_white"
)
st.plotly_chart(fig2, use_container_width=True)

# Cut of the signals
st.header("Cut of the erroneus peaks")
from scipy.signal import find_peaks
# --- PEAK DETECTION: Remove First Two Peaks ---
peaks, _ = find_peaks(df["Total_vert_normalized_GRF"], height=1.2, distance=1)
if len(peaks) >= 2:
    second_peak_idx = peaks[1]
    df_trimmed = df.iloc[second_peak_idx + 4:].reset_index(drop=True)
else:
    df_trimmed = df.copy()  # fallback if < 2 peaks

# Scroll slider
max_index = len(df_trimmed)
step_size = 10
scroll_position = st.slider("Scroll to explore jump", step_size, max_index, step_size, step=step_size)

# Get visible data
visible_df = df_trimmed.iloc[:scroll_position]

# --- Plot 1: Total GRF ---
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Total_vert_normalized_GRF"],
    mode="lines", name="Total GRF/BW", line=dict(color="blue")
))
fig1.update_layout(
    title="Total GRF Normalized on BW (After First Two Peaks)",
    xaxis_title="Time [s]",
    yaxis_title="Total GRF [N/BW]",
    template="plotly_white"
)
st.plotly_chart(fig1, use_container_width=True)

# --- Plot 2: Left and Right GRF ---
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Left_vert_GRF_normalized"],
    mode="lines", name="Left GRF/BW", line=dict(color="green")
))
fig2.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Right_vert_GRF_normalized"],
    mode="lines", name="Right GRF/BW", line=dict(color="red")
))
fig2.update_layout(
    title="Left and Right GRF Normalized on BW (After First Two Peaks)",
    xaxis_title="Time [s]",
    yaxis_title="GRF [N/BW]",
    template="plotly_white"
)
st.plotly_chart(fig2, use_container_width=True)


# Points annotations
st.header("Points Annotations")
# --- Remove first two GRF peaks ---
peaks, _ = find_peaks(df["Total_vert_normalized_GRF"], height=1.2, distance=10)
if len(peaks) >= 2:
    second_peak_idx = peaks[1]
    df_trimmed = df.iloc[second_peak_idx + 4:].reset_index(drop=True)
else:
    df_trimmed = df.copy()

# --- Detect Zero Crossings for Total GRF ---
grf = df_trimmed["Total_vert_normalized_GRF"].values
zero_crossings = np.where(np.diff(np.sign(grf - 0)) != 0)[0]

if len(zero_crossings) >= 2:
    takeoff_idx = zero_crossings[0]
    landing_idx = zero_crossings[-1]

    # Max force before take-off
    max_force_idx = df_trimmed.iloc[:takeoff_idx]["Total_vert_normalized_GRF"].idxmax()

else:
    takeoff_idx = landing_idx = max_force_idx = None

# --- Slider ---
max_index = len(df_trimmed)
step_size = 1
scroll_position = st.slider("Scroll to explore jump", step_size, max_index, 50, step=step_size)

visible_df = df_trimmed.iloc[:scroll_position]

# --- Plot 1: Total GRF ---
fig1 = go.Figure()

# GRF curve
fig1.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Total_vert_normalized_GRF"],
    mode="lines", name="Total GRF/BW", line=dict(color="blue")
))

# Mark take-off
if takeoff_idx is not None and takeoff_idx < scroll_position:
    fig1.add_trace(go.Scatter(
        x=[df_trimmed["Time"][takeoff_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][takeoff_idx]],
        mode="markers+text",
        name="Take-off",
        marker=dict(color="orange", size=10),
        text=["Take-off"],
        textposition="top right"
    ))

# Mark landing
if landing_idx is not None and landing_idx < scroll_position:
    fig1.add_trace(go.Scatter(
        x=[df_trimmed["Time"][landing_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][landing_idx]],
        mode="markers+text",
        name="Landing",
        marker=dict(color="green", size=10),
        text=["Landing"],
        textposition="top right"
    ))

# Mark max force
if max_force_idx is not None and max_force_idx < scroll_position:
    fig1.add_trace(go.Scatter(
        x=[df_trimmed["Time"][max_force_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][max_force_idx]],
        mode="markers+text",
        name="Max Force",
        marker=dict(color="purple", size=10),
        text=["Max Force"],
        textposition="top right"
    ))

fig1.update_layout(
    title="Total GRF Normalized on BW (After First Two Peaks)",
    xaxis_title="Time [s]",
    yaxis_title="Total GRF [N/BW]",
    template="plotly_white"
)
st.plotly_chart(fig1, use_container_width=True)



# Plot with area/impulse
st.header("Points Annotations + Impulse Area")

# --- Remove first two GRF peaks ---
peaks, _ = find_peaks(df["Total_vert_normalized_GRF"], height=1.2, distance=10)
if len(peaks) >= 2:
    second_peak_idx = peaks[1]
    df_trimmed = df.iloc[second_peak_idx + 4:].reset_index(drop=True)
else:
    df_trimmed = df.copy()

# --- Detect Zero Crossings for Total GRF ---
grf = df_trimmed["Total_vert_normalized_GRF"].values
time = df_trimmed["Time"].values
zero_crossings = np.where(np.diff(np.sign(grf)) != 0)[0]

# --- Identify points ---
takeoff_idx = landing_idx = max_force_idx = drift_start_idx = None
impulse_area = None

if len(zero_crossings) >= 2:
    takeoff_idx = zero_crossings[0]
    landing_idx = zero_crossings[-1]
    max_force_idx = df_trimmed.iloc[:takeoff_idx]["Total_vert_normalized_GRF"].idxmax()

    # Detect point before max_force where GRF starts drifting (increasing)
    pre_max = grf[:max_force_idx]

    # Find where GRF starts increasing
    slope = np.diff(pre_max) > 0
    slope_indices = np.where(slope)[0]
    # Find where GRF is crossing upward through 1.0
    above_one = pre_max > 1.0
    crossing_one_indices = np.where((~above_one[:-1]) & (above_one[1:]))[0]
    # Combine both: upward slope and crossing above 1.0
    combined_indices = np.intersect1d(slope_indices, crossing_one_indices)
    
    #drift_indices = np.where(np.diff(pre_max) > 0)[0]
    if len(combined_indices) > 0:
        #drift_start_idx = drift_indices[-12]
        drift_start_idx = combined_indices[-1]

        # Compute impulse (area under the curve from drift start to take-off)
        impulse_time = time[drift_start_idx:takeoff_idx+1]
        impulse_grf = grf[drift_start_idx:takeoff_idx+1]
        impulse_area = np.trapezoid(impulse_grf, impulse_time)

# --- Slider for visible portion ---
max_index = len(df_trimmed)
step_size = 1
scroll_position = st.slider("Scroll to explore jump", step_size, max_index, 60, step=step_size)
visible_df = df_trimmed.iloc[:scroll_position]

# --- Plot with annotations and impulse ---
# --- Get fixed x-axis range from full trimmed data ---
x_min = df_trimmed["Time"].min()
x_max = df_trimmed["Time"].max()
y_min = df_trimmed["Total_vert_normalized_GRF"].min()
y_max = df_trimmed["Total_vert_normalized_GRF"].max()

fig1 = go.Figure()

# GRF curve
fig1.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Total_vert_normalized_GRF"],
    mode="lines", name="Total GRF/BW", line=dict(color="blue")
))

# Impulse area fill
if drift_start_idx is not None and takeoff_idx is not None and takeoff_idx < scroll_position:
    fig1.add_trace(go.Scatter(
        x=time[drift_start_idx:takeoff_idx+1].tolist() + [time[takeoff_idx], time[drift_start_idx]],
        y=grf[drift_start_idx:takeoff_idx+1].tolist() + [0, 0],
        fill="toself", fillcolor="rgba(255, 165, 0, 0.3)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Impulse ≈ {impulse_area:.3f} (N·s/BW)"
    ))

# Annotate take-off
if takeoff_idx is not None and takeoff_idx < scroll_position:
    fig1.add_trace(go.Scatter(
        x=[time[takeoff_idx]],
        y=[grf[takeoff_idx]],
        mode="markers+text",
        name="Take-off",
        marker=dict(color="orange", size=10),
        text=["Take-off"],
        textposition="top right"
    ))

# Annotate landing
if landing_idx is not None and landing_idx < scroll_position:
    fig1.add_trace(go.Scatter(
        x=[time[landing_idx]],
        y=[grf[landing_idx]],
        mode="markers+text",
        name="Landing",
        marker=dict(color="green", size=10),
        text=["Landing"],
        textposition="top right"
    ))

# Annotate max force
if max_force_idx is not None and max_force_idx < scroll_position:
    fig1.add_trace(go.Scatter(
        x=[time[max_force_idx]],
        y=[grf[max_force_idx]],
        mode="markers+text",
        name="Max Force",
        marker=dict(color="purple", size=10),
        text=["Max Force"],
        textposition="top right"
    ))

fig1.update_layout(
    title="Total GRF Normalized on BW with Impulse Area",
    xaxis_title="Time [s]",
    yaxis_title="Total GRF [N/BW]",
    xaxis_range=[x_min, x_max],  # <--- FIXED X-AXIS RANGE
    yaxis_range=[y_min, y_max],  # <--- FIXED Y-AXIS RANGE
    template="plotly_white"
)

st.plotly_chart(fig1, use_container_width=True)



# Example of scrollytelling
st.header("Example of scrollytelling")
from scroll_component.scroll_component import scroll_position
# --- Preprocess: Remove first 2 peaks ---
peaks, _ = find_peaks(df["Total_vert_normalized_GRF"], height=1.2, distance=10)
if len(peaks) >= 2:
    df_trimmed = df.iloc[peaks[1] + 4:].reset_index(drop=True)
else:
    df_trimmed = df.copy()

# --- Find events ---
grf = df_trimmed["Total_vert_normalized_GRF"].values
zc = np.where(np.diff(np.sign(grf)) != 0)[0]

if len(zc) >= 2:
    takeoff_idx, landing_idx = zc[0], zc[-1]
    max_force_idx = df_trimmed.iloc[:takeoff_idx]["Total_vert_normalized_GRF"].idxmax()
else:
    takeoff_idx = landing_idx = max_force_idx = None

# --- Get scroll position in pixels ---
scroll_px = scroll_position()
scroll_fraction = min(scroll_px / 1000, 1.0)
scroll_index = int(len(df_trimmed) * scroll_fraction)
visible_df = df_trimmed.iloc[:scroll_index]

# --- Plot ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Total_vert_normalized_GRF"],
    mode="lines", name="Total GRF/BW", line=dict(color="blue")
))

if takeoff_idx and takeoff_idx < scroll_index:
    fig.add_trace(go.Scatter(
        x=[df_trimmed["Time"][takeoff_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][takeoff_idx]],
        mode="markers+text", marker=dict(color="orange", size=10),
        text=["Take-off"], textposition="top right"
    ))

if landing_idx and landing_idx < scroll_index:
    fig.add_trace(go.Scatter(
        x=[df_trimmed["Time"][landing_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][landing_idx]],
        mode="markers+text", marker=dict(color="blue", size=10),
        text=["Landing"], textposition="top right"
    ))

if max_force_idx and max_force_idx < scroll_index:
    fig.add_trace(go.Scatter(
        x=[df_trimmed["Time"][max_force_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][max_force_idx]],
        mode="markers+text", marker=dict(color="purple", size=10),
        text=["Max Force"], textposition="top right"
    ))

fig.update_layout(
    title="Scroll-based GRF Plot with Event Markers",
    xaxis_title="Time [s]",
    yaxis_title="GRF [N/BW]",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.title("In locale dovrebbe funzionare, qui NO!")

# fake scrollytelling with button and animation
st.header("Fake scrollytelling")
# --- Preprocess: Remove first 2 peaks ---
peaks, _ = find_peaks(df["Total_vert_normalized_GRF"], height=1.2, distance=10)
if len(peaks) >= 2:
    df_trimmed = df.iloc[peaks[1] + 4:].reset_index(drop=True)
else:
    df_trimmed = df.copy()

# --- Find events ---
grf = df_trimmed["Total_vert_normalized_GRF"].values
zc = np.where(np.diff(np.sign(grf)) != 0)[0]

if len(zc) >= 2:
    takeoff_idx, landing_idx = zc[0], zc[-1]
    max_force_idx = df_trimmed.iloc[:takeoff_idx]["Total_vert_normalized_GRF"].idxmax()
else:
    takeoff_idx = landing_idx = max_force_idx = None

# --- Scroll Simulation ---
step_size = 10
max_index = len(df_trimmed)

# Simulate scroll with a button (state-based counter)
if "scroll_pos" not in st.session_state:
    st.session_state.scroll_pos = step_size

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Scroll ➡️"):
        st.session_state.scroll_pos = min(st.session_state.scroll_pos + step_size, max_index)

scroll_index = st.session_state.scroll_pos
visible_df = df_trimmed.iloc[:scroll_index]

# --- Plot ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=visible_df["Time"], y=visible_df["Total_vert_normalized_GRF"],
    mode="lines", name="Total GRF/BW", line=dict(color="blue")
))

if takeoff_idx and takeoff_idx < scroll_index:
    fig.add_trace(go.Scatter(
        x=[df_trimmed["Time"][takeoff_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][takeoff_idx]],
        mode="markers+text", marker=dict(color="orange", size=10),
        text=["Take-off"], textposition="top right"
    ))

if landing_idx and landing_idx < scroll_index:
    fig.add_trace(go.Scatter(
        x=[df_trimmed["Time"][landing_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][landing_idx]],
        mode="markers+text", marker=dict(color="blue", size=10),
        text=["Landing"], textposition="top right"
    ))

if max_force_idx and max_force_idx < scroll_index:
    fig.add_trace(go.Scatter(
        x=[df_trimmed["Time"][max_force_idx]],
        y=[df_trimmed["Total_vert_normalized_GRF"][max_force_idx]],
        mode="markers+text", marker=dict(color="purple", size=10),
        text=["Max Force"], textposition="top right"
    ))

fig.update_layout(
    title="Simulated Scrolling GRF Plot with Event Markers",
    xaxis_title="Time [s]",
    yaxis_title="GRF [N/BW]",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)