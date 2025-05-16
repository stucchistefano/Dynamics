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
plt.show()