import pandas as pd
import numpy as np

# 1. Load the dataset (assumes it is in the same directory as this script)
filename = 'Cleaned_nhanes_raw_combined_2007_2018.csv'
print(f"Loading {filename}...")
df = pd.read_csv(filename, low_memory=False)

# 2. Define a function to compute CKD-EPI 2021 eGFR vectorially


def calculate_ckd_epi_2021(row):
    scr = row['LBXSCR']
    age = row['RIDAGEYR']
    gender = str(row['RIAGENDR']).strip().capitalize()

    # Assign constants based on gender
    if gender == 'Female':
        kappa = 0.7
        alpha = -0.241
        gender_multiplier = 1.012
    elif gender == 'Male':
        kappa = 0.9
        alpha = -0.302
        gender_multiplier = 1.0
    else:
        # Fallback/Error handling just in case of unexpected strings
        return np.nan

    # Calculate components
    min_term = min(scr / kappa, 1) ** alpha
    max_term = max(scr / kappa, 1) ** (-1.200)
    age_term = 0.9938 ** age

    # Combine into final eGFR
    egfr = 142 * min_term * max_term * age_term * gender_multiplier
    return egfr


# 3. Apply the function to the dataset
print("Calculating eGFR using CKD-EPI 2021 (race-free) equation...")
df['eGFR'] = df.apply(calculate_ckd_epi_2021, axis=1)

# 4. Quick sanity check: Print descriptive stats and a small preview
print("\n--- eGFR Calculation Summary ---")
print(df['eGFR'].describe())

print("\n--- Preview of Results ---")
print(df[['RIAGENDR', 'RIDAGEYR', 'LBXSCR', 'eGFR']].head(10))

# 1. ACR

# 2. Recompute ACR (Urine Albumin / Urine Creatinine ratio multiplied by 100)
print("\nRecomputing ACR from URXUMA and URXUCR...")
df['ACR'] = (df['URXUMA'] / df['URXUCR']) * 100

# 3. Quick summary statistics for ACR
print("\n--- ACR Calculation Summary ---")
print(df['ACR'].describe())

# 4. Preview both recomputed values together
print("\n--- Preview of Recomputed eGFR and ACR ---")
print(df[['RIAGENDR', 'RIDAGEYR', 'eGFR', 'ACR']].head(10))

# 1. Build the CKD label based on recomputed thresholds
print("\nRebuilding CKD labels from recomputed eGFR and ACR thresholds...")
df['has_CKD'] = np.where((df['eGFR'] < 60) | (df['ACR'] >= 30), 1, 0)

# 2. Summary stats for the CKD column to see the prevalence
print("\n--- CKD Label Summary ---")
print(df['has_CKD'].value_counts())
print(f"Prevalence: {df['has_CKD'].mean() * 100:.2f}%")

# 3. Final preview of the columns side by side
print("\n--- Final Preview of Recomputed Metrics ---")
print(df[['RIAGENDR', 'RIDAGEYR', 'eGFR', 'ACR', 'has_CKD']].head(10))

# 4. Save the single comprehensive file with all three new columns included
final_filename = 'Cleaned_nhanes_with_recomputed_CKD.csv'
df.to_csv(final_filename, index=False)
print(
    f"\nAll steps complete! Your single final file is saved as '{final_filename}'.")
