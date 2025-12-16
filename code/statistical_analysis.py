# HDPE Radiation Protection Study - Statistical Analysis
import pandas as pd
import numpy as np
from scipy import stats

# Load the data
df = pd.read_csv('../data/raw/sun_geiger_data.csv')

print("HDPE Radiation Protection Study - Statistical Analysis")
print("=" * 60)
print(f"Data loaded: {len(df)} measurements")
print(f"Detectors: SpcS POD1-01, Cntrl POD1-02, HDPE POD1-03")

# Basic statistics
detectors = ['SpcS POD1-01', 'Cntrl POD1-02', 'HDPE POD1-03']
for detector in detectors:
    mean_val = df[detector].mean()
    std_val = df[detector].std()
    print(f"{detector}: Mean = {mean_val:.1f} ± {std_val:.1f}")

# Paired t-test: Control vs HDPE
t_stat, p_val = stats.ttest_rel(df['Cntrl POD1-02'], df['HDPE POD1-03'])
print(f"\nControl vs HDPE t-test: t = {t_stat:.3f}, p = {p_val:.6f}")

if p_val < 0.05:
    print("Result: Statistically significant difference!")
else:
    print("Result: No significant difference")