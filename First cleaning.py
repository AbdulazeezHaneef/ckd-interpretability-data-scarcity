import pandas as pd
import numpy as np

# Load the combined NHANES dataset
file_path = "nhanes_raw_combined_2007_2018.csv"
df = pd.read_csv(file_path, low_memory=False)

# Display the column names and their data types
print(f"Total Columns: {len(df.columns)}\n")
print(df.dtypes.to_string())

# Identify a sample of object columns that logically should be numeric or categorical codes
target_columns = ['SDDSRVYR', 'DMDHHSIZ',
                  'DMDFMSIZ', 'DIQ010', 'BPQ020', 'BMIWT']

print("--- Inspecting Unique Values in Object Columns ---")
for col in target_columns:
    if col in df.columns:
        # Get unique values, dropping completely empty cells to see actual text content
        unique_vals = df[col].dropna().unique()
        # Show the first 10 unique values to keep the screen clean
        print(f"\nColumn: {col}")
        print(f"Sample unique values: {unique_vals[:10]}")


# 1. Map the text string to a clean number, leaving other values as they are
mapping = {"7 or more people in the Household": 7,
           "7 or more people in the Family": 7}

df['DMDHHSIZ'] = df['DMDHHSIZ'].replace(mapping)
df['DMDFMSIZ'] = df['DMDFMSIZ'].replace(mapping)

# 2. Convert the columns to numeric (float allows handling of missing values safely)
df['DMDHHSIZ'] = pd.to_numeric(df['DMDHHSIZ'], errors='coerce')
df['DMDFMSIZ'] = pd.to_numeric(df['DMDFMSIZ'], errors='coerce')

# 3. Verify the fix worked
print("Updated DMDHHSIZ unique values:", df['DMDHHSIZ'].unique())
print("Updated DMDFMSIZ unique values:", df['DMDFMSIZ'].unique())
print("\nNew Data Types:")
print(df[['DMDHHSIZ', 'DMDFMSIZ']].dtypes)


# Define the explicit mapping for diabetes (DIQ010)
diabetes_mapping = {
    'Yes': 1,
    'No': 0,
    'Borderline': 2,
    "Don't know": -1,
    'Refused': -1
}

# Define the explicit mapping for blood pressure (BPQ020)
bp_mapping = {
    'Yes': 1,
    'No': 0,
    "Don't know": -1
}

# Apply mappings and fill any pre-existing blank spaces (NaN) with -1
df['DIQ010'] = df['DIQ010'].map(diabetes_mapping).fillna(-1).astype(int)
df['BPQ020'] = df['BPQ020'].map(bp_mapping).fillna(-1).astype(int)

# Verify the changes
print("DIQ010 (Diabetes) counts:")
print(df['DIQ010'].value_counts())

print("\nBPQ020 (High BP) counts:")
print(df['BPQ020'].value_counts())

# Check unique values in the primary demographic text columns
demo_cols = ['RIAGENDR', 'RIDRETH1']

print("--- Inspecting Demographic Text Content ---")
for col in demo_cols:
    if col in df.columns:
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False))

# Check for any non-numeric entries or structural issues in our calculation columns
print("--- Gender Distribution Check ---")
print(df['RIAGENDR'].value_counts(dropna=False))

print("\n--- Summary of Core Calculation Columns ---")
calculation_cols = ['RIDAGEYR', 'LBXSCR', 'URXUMA', 'URXUCR']
print(df[calculation_cols].describe())


print(f"Starting Row Count: {len(df)}")

# Filter 1: Keep only adults (Age >= 18)
df_filtered = df[df['RIDAGEYR'] >= 18].copy()
print(f"Rows after removing minors (under 18): {len(df_filtered)}")

# Filter 2: Drop questionnaire refusals (-1) from Diabetes and High BP
df_filtered = df_filtered[(df_filtered['DIQ010'] != -1)
                          & (df_filtered['BPQ020'] != -1)]
print(f"Rows after removing questionnaire refusals (-1): {len(df_filtered)}")

# Filter 3: Drop rows with completely blank values (NaN) in our core calculation labs
essential_labs = ['LBXSCR', 'URXUMA', 'URXUCR']
df_filtered = df_filtered.dropna(subset=essential_labs)
print(
    f"Final Rows remaining after dropping missing vital labs (NaN): {len(df_filtered)}")

# ==========================================
# Export the cleaned dataset to a new CSV file
# ==========================================
output_file_path = "Cleaned_nhanes_raw_combined_2007_2018.csv"

# export directly to your working directory without adding an extra index column
df_filtered.to_csv(output_file_path, index=False)

print("\n--- Export Finished Successfully ---")
print(f"Cleaned dataset saved as: {output_file_path}")
print(f"Final clean rows exported: {len(df_filtered)}")
