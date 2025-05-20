import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

st.title("Dynamics KPIs")

# Selectbox (into the sidebar)
# Object-like approach
subject_input= st.selectbox(
    "Select your player",
    ('Subject1','Subject2','Subject3','Subject4','Subject5','Subject6',
     'Subject7','Subject8','Subject9', 'Subject10','Subject11','Subject12',
     'Subject13','Subject14','Subject15')
    )


# List of Excel file paths (you can also automate this using os.listdir if all files are in one folder)
file_paths = [
    "DataFolder/Subject1.xlsx",
    "DataFolder/Subject2.xlsx",
    "DataFolder/Subject3.xlsx",
    "DataFolder/Subject4.xlsx",
    "DataFolder/Subject5.xlsx",
    "DataFolder/Subject6.xlsx",
    "DataFolder/Subject7.xlsx",
    "DataFolder/Subject8.xlsx",
    "DataFolder/Subject9.xlsx",
    "DataFolder/Subject10.xlsx",
    "DataFolder/Subject11.xlsx",
    "DataFolder/Subject12.xlsx",
    "DataFolder/Subject13.xlsx",
    "DataFolder/Subject14.xlsx",
    "DataFolder/Subject15.xlsx"
]

# Initialize dictionary to store the resulting DataFrames
subject_data = {}

# Loop through each Excel file
for file_path in file_paths:
    # Extract subject name from filename (e.g., "Subject1")
    subject_name = os.path.splitext(os.path.basename(file_path))[0]

    # Read all sheets from the Excel file
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

    # Collect all unique column names from all sheets
    all_columns = set()
    for sheet_df in all_sheets.values():
        all_columns.update(sheet_df.columns)

    # Create a new DataFrame with rows as column names and columns as sheet names
    result_df = pd.DataFrame(index=sorted(all_columns), columns=all_sheets.keys())

    # Fill the result DataFrame with first-row values from each column of each sheet
    for sheet_name, sheet_df in all_sheets.items():
        for col in sheet_df.columns:
            if not sheet_df.empty:
                result_df.at[col, sheet_name] = sheet_df[col].iloc[0]

    # Store the DataFrame in the dictionary
    subject_data[subject_name] = result_df


# Set to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#subject_data['Subject1']


# Check if the subject exists in the dictionary
if subject_input in subject_data:
    # Extract the DataFrame
    extracted_df = subject_data[subject_input]

    # Display the extracted DataFrame
    print(f"Data for {subject_input} founded!")
else:
    print(f"Subject '{subject_input}' not found in the dataset.")

#extracted_df

# List of desired row labels to extract
selected_metrics = [
    "height_OC",
    "GRF_maxforce_OC",
    "timeRSI_OC",
    "max_power_OC",
    "I_OC",
    "Flight_time_OC",
    "Vel_takeoff_OC"
]

# Filter the extracted DataFrame
filtered_df = extracted_df.loc[extracted_df.index.isin(selected_metrics)]

# Display the result
print("Filtered DataFrame with selected metrics realised!")
#filtered_df

# New column names in desired order
new_column_names = ["SJ1", "SJ2", "SJ3", "DJ1", "DJ2", "DJ3"]

# Replace old column names with new ones
# Assumes filtered_df has exactly 6 columns in order: Sheet1, Sheet2, ..., Sheet6
filtered_df.columns = new_column_names

# Display the updated DataFrame
print("Filtered DataFrame with renamed columns realised!")
filtered_df

# Ask the user to choose "SJ" or "DJ"
focus_type = st.radio(
    "Select the type of jump",
    ('SJ','DJ')
    )

# Validate input and filter columns
if focus_type in ["SJ", "DJ"]:
    # Build list of relevant columns
    selected_columns = [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]

    # Filter the DataFrame
    #focused_df = filtered_df[selected_columns]
    focused_df = filtered_df[selected_columns].copy()

    # Display result
    print(f"\nData focusing on {focus_type} extracted!")
else:
    print("Invalid input. Please enter 'SJ' or 'DJ'.")

#focused_df

# Remove columns that are entirely NaN
focused_df = focused_df.dropna(axis=1, how='all')

# Display the cleaned DataFrame
print("DataFrame after removing columns filled with NaN realised!")
focused_df

# Calculate statistics
focused_df["Average"] = focused_df.mean(axis=1)
focused_df["Max"] = focused_df.max(axis=1)
focused_df["Min"] = focused_df.min(axis=1)
focused_df["Std"] = focused_df.std(axis=1)

# Calculate % change: (col2 - col1)/col1 * 100 and (col3 - col2)/col2 * 100
focused_df["% Δ 2 vs 1"] = ((focused_df[f"{focus_type}2"] - focused_df[f"{focus_type}1"]) / focused_df[f"{focus_type}1"]) * 100
focused_df["% Δ 3 vs 2"] = ((focused_df[f"{focus_type}3"] - focused_df[f"{focus_type}2"]) / focused_df[f"{focus_type}2"]) * 100

focused_df

st.title("Height plot")
# Extract height_OC values from the first three columns
height_values = focused_df.loc["height_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepare values for plotting
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['skyblue']
alphas = [1] * 3 + [0.5]  # Lower opacity for the average bar

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bar_containers = []

# Plot each bar individually to apply individual alpha
for i in range(len(bars)):
    bar = ax.bar(i, bars[i], color=colors[i], alpha=alphas[i], width=0.6)
    bar_containers.append(bar)

# Add standard deviation whiskers on top of the average bar only
for i, val in enumerate(bars):
    ax.text(i, val + 0.06, f"{val:.2f}", ha='center', va='bottom',
            fontweight='bold', color='skyblue')
    # Add whiskers for std on the average bar only
    if i < 3:  # Only first three bars get whiskers
        ax.vlines(i, val, val + std_dev, colors='black', linewidth=2)
        ax.hlines(val + std_dev, i - 0.1, i + 0.1, colors='black', linewidth=2)

# Customize axes
ax.set_ylabel("Jump Height [m]")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30)
ax.set_ylim(0, max(bars) + std_dev + 0.05)
ax.set_title("Jump Height per Trial and Average")

plt.tight_layout()
st.pyplot(fig)

st.subheader("Height plot interattivo")
import plotly.graph_objects as go

st.title("Height Plot")

# Estrai i valori
height_values = focused_df.loc["height_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['lightblue']
alphas = [1] * 3 + [0.5]

# Crea il grafico Plotly
fig = go.Figure()

# Aggiungi barre
for i, val in enumerate(bars):
    fig.add_trace(go.Bar(
        x=[labels[i]],
        y=[val],
        name=labels[i],
        marker=dict(color=colors[i]),
        opacity=alphas[i],
        text=f"{val:.2f}",
        textposition="outside"
    ))

# Aggiungi le barre di errore solo ai primi tre
for i in range(3):
    fig.add_trace(go.Scatter(
        x=[labels[i]],
        y=[bars[i] + std_dev],
        mode="markers+lines",
        marker=dict(symbol="line-ns-open", color="black", size=10),
        showlegend=False
    ))

# Layout
fig.update_layout(
    yaxis_title="Jump Height [m]",
    title="Jump Height per Trial and Average",
    xaxis=dict(tickangle=30),
    bargap=0.4
)

st.plotly_chart(fig)


st.header("Force plot")
# Extract height_OC values from the first three columns
height_values = focused_df.loc["GRF_maxforce_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepare values for plotting
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['skyblue']
alphas = [1] * 3 + [0.5]  # Lower opacity for the average bar

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bar_containers = []

# Plot each bar individually to apply individual alpha
for i in range(len(bars)):
    bar = ax.bar(i, bars[i], color=colors[i], alpha=alphas[i], width=0.6)
    bar_containers.append(bar)

# Add standard deviation whiskers on top of the average bar only
for i, val in enumerate(bars):
    ax.text(i, val + 0.06, f"{val:.2f}", ha='center', va='bottom',
            fontweight='bold', color='skyblue')
    # Add whiskers for std on the average bar only
    if i < 3:  # Only first three bars get whiskers
        ax.vlines(i, val, val + std_dev, colors='black', linewidth=2)
        ax.hlines(val + std_dev, i - 0.1, i + 0.1, colors='black', linewidth=2)

# Customize axes
ax.set_ylabel("Max Force [N/BW]")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30)
ax.set_ylim(0, max(bars) + std_dev + 0.5)
ax.set_title("BW-Normalized Max Force per Trial and Average")

plt.tight_layout()
st.pyplot(fig)


st.subheader("Interactive Force Plot")

# Estrai i valori
height_values = focused_df.loc["GRF_maxforce_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepara dati
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['lightblue']
alphas = [1] * 3 + [0.5]

# Crea la figura Plotly
fig = go.Figure()

# Aggiungi barre
for i, val in enumerate(bars):
    fig.add_trace(go.Bar(
        x=[labels[i]],
        y=[val],
        name=labels[i],
        marker=dict(color=colors[i]),
        opacity=alphas[i],
        text=f"{val:.2f}",
        textposition="outside"
    ))

# Aggiungi whiskers (deviazione standard) per i primi 3
for i in range(3):
    fig.add_trace(go.Scatter(
        x=[labels[i], labels[i]],
        y=[bars[i], bars[i] + std_dev],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))
    # Aggiungi la "lineetta" orizzontale sopra
    fig.add_trace(go.Scatter(
        x=[labels[i]],
        y=[bars[i] + std_dev],
        mode="markers",
        marker=dict(symbol="line-ns-open", color="black", size=12),
        showlegend=False
    ))

# Layout
fig.update_layout(
    yaxis_title="Max Force [N/BW]",
    title="BW-Normalized Max Force per Trial and Average",
    xaxis=dict(tickangle=30),
    bargap=0.4
)

# Mostra in Streamlit
st.plotly_chart(fig)



st.header("RSI plot")
# Extract height_OC values from the first three columns
height_values = focused_df.loc["timeRSI_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepare values for plotting
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['skyblue']
alphas = [1] * 3 + [0.5]  # Lower opacity for the average bar

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bar_containers = []

# Plot each bar individually to apply individual alpha
for i in range(len(bars)):
    bar = ax.bar(i, bars[i], color=colors[i], alpha=alphas[i], width=0.6)
    bar_containers.append(bar)

# Add standard deviation whiskers on top of the average bar only
for i, val in enumerate(bars):
    ax.text(i, val + 0.1, f"{val:.2f}", ha='center', va='bottom',
            fontweight='bold', color='skyblue')
    # Add whiskers for std on the average bar only
    if i < 3:  # Only first three bars get whiskers
        ax.vlines(i, val, val + std_dev, colors='black', linewidth=2)
        ax.hlines(val + std_dev, i - 0.1, i + 0.1, colors='black', linewidth=2)

# Customize axes
ax.set_ylabel("RSI [m/s]")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30)
ax.set_ylim(0, max(bars) + std_dev + 0.25)
ax.set_title("RSI per Trial and Average")

plt.tight_layout()
st.pyplot(fig)


st.subheader("Interaction RSI Plot")

# Estrai i valori
height_values = focused_df.loc["timeRSI_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepara dati
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['lightblue']
alphas = [1] * 3 + [0.5]

# Crea figura Plotly
fig = go.Figure()

# Aggiungi barre
for i, val in enumerate(bars):
    fig.add_trace(go.Bar(
        x=[labels[i]],
        y=[val],
        name=labels[i],
        marker=dict(color=colors[i]),
        opacity=alphas[i],
        text=f"{val:.2f}",
        textposition="outside"
    ))

# Aggiungi whiskers per std solo ai primi 3 trials
for i in range(3):
    fig.add_trace(go.Scatter(
        x=[labels[i], labels[i]],
        y=[bars[i], bars[i] + std_dev],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))
    # Aggiungi tacca orizzontale in cima al whisker
    fig.add_trace(go.Scatter(
        x=[labels[i]],
        y=[bars[i] + std_dev],
        mode="markers",
        marker=dict(symbol="line-ns-open", color="black", size=12),
        showlegend=False
    ))

# Layout grafico
fig.update_layout(
    yaxis_title="RSI [m/s]",
    title="RSI per Trial and Average",
    xaxis=dict(tickangle=30),
    bargap=0.4
)

# Mostra in Streamlit
st.plotly_chart(fig)


st.header("Power plot")
# Extract height_OC values from the first three columns
height_values = focused_df.loc["max_power_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepare values for plotting
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['skyblue']
alphas = [1] * 3 + [0.5]  # Lower opacity for the average bar

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bar_containers = []

# Plot each bar individually to apply individual alpha
for i in range(len(bars)):
    bar = ax.bar(i, bars[i], color=colors[i], alpha=alphas[i], width=0.6)
    bar_containers.append(bar)

# Add standard deviation whiskers on top of the average bar only
for i, val in enumerate(bars):
    ax.text(i, val + 0.1, f"{val:.2f}", ha='center', va='bottom',
            fontweight='bold', color='skyblue')
    # Add whiskers for std on the average bar only
    if i < 3:  # Only first three bars get whiskers
        ax.vlines(i, val, val + std_dev, colors='black', linewidth=2)
        ax.hlines(val + std_dev, i - 0.1, i + 0.1, colors='black', linewidth=2)

# Customize axes
ax.set_ylabel("Max Power [(N*m)/(s*BW)]")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30)
ax.set_ylim(0, max(bars) + std_dev + 0.5)
ax.set_title("BW-Normalized Max Power per Trial and Average")

plt.tight_layout()
st.pyplot(fig)


st.subheader("Interactive Power plot")
# Estrai i valori
height_values = focused_df.loc["max_power_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepara i dati
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['lightblue']
alphas = [1] * 3 + [0.5]

# Crea la figura Plotly
fig = go.Figure()

# Aggiungi barre
for i, val in enumerate(bars):
    fig.add_trace(go.Bar(
        x=[labels[i]],
        y=[val],
        name=labels[i],
        marker=dict(color=colors[i]),
        opacity=alphas[i],
        text=f"{val:.2f}",
        textposition="outside"
    ))

# Aggiungi whiskers solo ai primi 3 trials
for i in range(3):
    # Linea verticale (whisker)
    fig.add_trace(go.Scatter(
        x=[labels[i], labels[i]],
        y=[bars[i], bars[i] + std_dev],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))
    # Tacca orizzontale in cima al whisker
    fig.add_trace(go.Scatter(
        x=[labels[i]],
        y=[bars[i] + std_dev],
        mode="markers",
        marker=dict(symbol="line-ns-open", color="black", size=12),
        showlegend=False
    ))

# Layout del grafico
fig.update_layout(
    yaxis_title="Max Power [(N·m)/(s·BW)]",
    title="BW-Normalized Max Power per Trial and Average",
    xaxis=dict(tickangle=30),
    bargap=0.4
)

# Visualizza in Streamlit
st.plotly_chart(fig)


st.header("Impulse plot")
# Extract height_OC values from the first three columns
height_values = focused_df.loc["I_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepare values for plotting
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['skyblue']
alphas = [1] * 3 + [0.5]  # Lower opacity for the average bar

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bar_containers = []

# Plot each bar individually to apply individual alpha
for i in range(len(bars)):
    bar = ax.bar(i, bars[i], color=colors[i], alpha=alphas[i], width=0.6)
    bar_containers.append(bar)

# Add standard deviation whiskers on top of the average bar only
for i, val in enumerate(bars):
    ax.text(i, val + 7, f"{val:.2f}", ha='center', va='bottom',
            fontweight='bold', color='skyblue')
    # Add whiskers for std on the average bar only
    if i < 3:  # Only first three bars get whiskers
        ax.vlines(i, val, val + std_dev, colors='black', linewidth=2)
        ax.hlines(val + std_dev, i - 0.1, i + 0.1, colors='black', linewidth=2)

# Customize axes
ax.set_ylabel("Impulse [(kg*m)/s]")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30)
ax.set_ylim(0, max(bars) + std_dev + 25)
ax.set_title("Impulse per Trial and Average")

plt.tight_layout()
st.pyplot(fig)


st.subheader("Interactive Impulse plot")
# Estrai i valori
height_values = focused_df.loc["I_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepara i dati
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['lightblue']
alphas = [1] * 3 + [0.5]

# Crea la figura Plotly
fig = go.Figure()

# Aggiungi barre
for i, val in enumerate(bars):
    fig.add_trace(go.Bar(
        x=[labels[i]],
        y=[val],
        name=labels[i],
        marker=dict(color=colors[i]),
        opacity=alphas[i],
        text=f"{val:.2f}",
        textposition="outside"
    ))

# Aggiungi whiskers della deviazione standard (solo ai primi 3 trials)
for i in range(3):
    fig.add_trace(go.Scatter(
        x=[labels[i], labels[i]],
        y=[bars[i], bars[i] + std_dev],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[labels[i]],
        y=[bars[i] + std_dev],
        mode="markers",
        marker=dict(symbol="line-ns-open", color="black", size=12),
        showlegend=False
    ))

# Layout del grafico
fig.update_layout(
    yaxis_title="Impulse [(kg·m)/s]",
    title="Impulse per Trial and Average",
    xaxis=dict(tickangle=30),
    bargap=0.4,
    yaxis=dict(range=[0, max(bars) + std_dev + 25])
)

# Mostra in Streamlit
st.plotly_chart(fig)


st.header("Flight Time plot")
# Extract height_OC values from the first three columns
height_values = focused_df.loc["Flight_time_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepare values for plotting
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['skyblue']
alphas = [1] * 3 + [0.5]  # Lower opacity for the average bar

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bar_containers = []

# Plot each bar individually to apply individual alpha
for i in range(len(bars)):
    bar = ax.bar(i, bars[i], color=colors[i], alpha=alphas[i], width=0.6)
    bar_containers.append(bar)

# Add standard deviation whiskers on top of the average bar only
for i, val in enumerate(bars):
    ax.text(i, val + 0.025, f"{val:.2f}", ha='center', va='bottom',
            fontweight='bold', color='skyblue')
    # Add whiskers for std on the average bar only
    if i < 3:  # Only first three bars get whiskers
        ax.vlines(i, val, val + std_dev, colors='black', linewidth=2)
        ax.hlines(val + std_dev, i - 0.1, i + 0.1, colors='black', linewidth=2)

# Customize axes
ax.set_ylabel("Time [s]")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30)
ax.set_ylim(0, max(bars) + std_dev + 0.05)
ax.set_title("Flight-Time per Trial and Average")

plt.tight_layout()
st.pyplot(fig)


st.subheader("Interactive Flight Time plot")
# Estrai i valori
height_values = focused_df.loc["Flight_time_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_height = height_values.mean()
std_dev = height_values.std()

# Prepara i dati
bars = list(height_values) + [average_height]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors = ['skyblue'] * 3 + ['lightblue']
alphas = [1] * 3 + [0.5]

# Crea la figura Plotly
fig = go.Figure()

# Aggiungi barre
for i, val in enumerate(bars):
    fig.add_trace(go.Bar(
        x=[labels[i]],
        y=[val],
        name=labels[i],
        marker=dict(color=colors[i]),
        opacity=alphas[i],
        text=f"{val:.2f}",
        textposition="outside"
    ))

# Aggiungi whiskers della deviazione standard (solo ai primi 3 trials)
for i in range(3):
    fig.add_trace(go.Scatter(
        x=[labels[i], labels[i]],
        y=[bars[i], bars[i] + std_dev],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[labels[i]],
        y=[bars[i] + std_dev],
        mode="markers",
        marker=dict(symbol="line-ns-open", color="black", size=12),
        showlegend=False
    ))

# Layout del grafico
fig.update_layout(
    yaxis_title="Time [s]",
    title="Flight-Time per Trial and Average",
    xaxis=dict(tickangle=30),
    bargap=0.4,
    yaxis=dict(range=[0, max(bars) + std_dev + 0.05])
)

# Mostra il grafico in Streamlit
st.plotly_chart(fig)


st.header("COMPARISON - Flight Time plot")

# First subject
height_values_1 = focused_df.loc["Flight_time_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
average_1 = height_values_1.mean()
std_dev_1 = height_values_1.std()

bars_1 = list(height_values_1) + [average_1]
labels = ["Trial 1", "Trial 2", "Trial 3", "Average"]
colors_1 = ['skyblue'] * 3 + ['lightblue']
alphas_1 = [1] * 3 + [0.5]

# Initialize figure
fig = go.Figure()

# Plot first subject bars
for i, val in enumerate(bars_1):
    fig.add_trace(go.Bar(
        x=[labels[i]],
        y=[val],
        name=f"Subject 1 - {labels[i]}",
        marker=dict(color=colors_1[i]),
        opacity=alphas_1[i],
        width=0.5,  # Thicker bar
        #offsetgroup=str(i),
        text=f"{val:.2f}",
        textposition="outside"
    ))

# Add whiskers for subject 1 (only trials)
for i in range(3):
    fig.add_trace(go.Scatter(
        x=[labels[i], labels[i]],
        y=[bars_1[i], bars_1[i] + std_dev_1],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[labels[i]],
        y=[bars_1[i] + std_dev_1],
        mode="markers",
        marker=dict(symbol="line-ns-open", color="black", size=12),
        showlegend=False
    ))

# Add toggle for comparison
compare = st.toggle("COMPARISON with another player")

if compare:
    # Selectbox (into the sidebar)
    # Object-like approach
    compare_input= st.selectbox(
        "Comparison player",
        ('Subject1','Subject2','Subject3','Subject4','Subject5','Subject6',
        'Subject7','Subject8','Subject9', 'Subject10','Subject11','Subject12',
        'Subject13','Subject14','Subject15')
        )
    
    # Check if the subject exists in the dictionary
    if compare_input in subject_data:
        # Extract the DataFrame
        ext_df = subject_data[compare_input]

        # Display the extracted DataFrame
        print(f"Data for {compare_input} founded!")
    else:
        print(f"Subject '{compare_input}' not found in the dataset.")

    # List of desired row labels to extract
    selected_metrics = [
        "height_OC",
        "GRF_maxforce_OC",
        "timeRSI_OC",
        "max_power_OC",
        "I_OC",
        "Flight_time_OC",
        "Vel_takeoff_OC"
    ]

    # Filter the extracted DataFrame
    filt_df = ext_df.loc[ext_df.index.isin(selected_metrics)]

    # New column names in desired order
    new_column_names = ["SJ1", "SJ2", "SJ3", "DJ1", "DJ2", "DJ3"]

    # Replace old column names with new ones
    # Assumes filtered_df has exactly 6 columns in order: Sheet1, Sheet2, ..., Sheet6
    filt_df.columns = new_column_names

    # Validate input and filter columns
    if focus_type in ["SJ", "DJ"]:
        # Build list of relevant columns
        selected_columns = [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]

        # Filter the DataFrame
        compare_df = filt_df[selected_columns].copy()

        # Display result
        print(f"\nData focusing on {focus_type} extracted!")
    else:
        print("Invalid input. Please enter 'SJ' or 'DJ'.")

    # Remove columns that are entirely NaN
    compare_df = compare_df.dropna(axis=1, how='all')

    # Display the cleaned DataFrame
    #print("DataFrame after removing columns filled with NaN realised!")
    #compare_df

    # Calculate statistics
    compare_df["Average"] = compare_df.mean(axis=1)
    compare_df["Max"] = compare_df.max(axis=1)
    compare_df["Min"] = compare_df.min(axis=1)
    compare_df["Std"] = compare_df.std(axis=1)

    # Calculate % change: (col2 - col1)/col1 * 100 and (col3 - col2)/col2 * 100
    compare_df["% Δ 2 vs 1"] = ((compare_df[f"{focus_type}2"] - focused_df[f"{focus_type}1"]) / focused_df[f"{focus_type}1"]) * 100
    compare_df["% Δ 3 vs 2"] = ((compare_df[f"{focus_type}3"] - focused_df[f"{focus_type}2"]) / focused_df[f"{focus_type}2"]) * 100

    #compare_df

    # Replace this with the actual second subject data (e.g., another row from a different df)
    height_values_2 = compare_df.loc["Flight_time_OC", [f"{focus_type}1", f"{focus_type}2", f"{focus_type}3"]]
    average_2 = height_values_2.mean()

    bars_2 = list(height_values_2) + [average_2]
    colors_2 = ['lightcoral'] * 3 + ['lightsalmon']
    alphas_2 = [1] * 3 + [0.5]

    # Plot second subject bars
    for i, val in enumerate(bars_2):
        fig.add_trace(go.Bar(
            x=[labels[i]],
            y=[val],
            name=f"Subject 2 - {labels[i]}",
            marker=dict(color=colors_2[i]),
            opacity=alphas_2[i],
            width=0.5,  # Slightly narrower for visual layering
            #offsetgroup=str(i),
            base=None,
            text=f"{val:.2f}",
            textposition="outside"
        ))

# Update layout
fig.update_layout(
    yaxis_title="Time [s]",
    title="Flight-Time per Trial and Average",
    xaxis=dict(tickangle=30),
    bargap=0.2,
    barmode='group',  # Show grouped bars for comparison
    yaxis=dict(range=[0, max(bars_1 + (bars_2 if compare else [])) + std_dev_1 + 0.05])
)

st.plotly_chart(fig)