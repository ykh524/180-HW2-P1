# hw2
#Kaihan Yang

import pandas as pd, numpy as np

df = pd.read_csv("adult.csv")          

def dp_histogram(series, bins, eps):
    """Return (noisy_counts, bin_edges)."""
    counts, edges = np.histogram(series, bins=bins)
    noisy = counts + np.random.laplace(scale=1/eps, size=len(counts))
    return noisy, edges

def categorical_dp_hist(series, eps):
    levels = series.unique()
    counts = series.value_counts().reindex(levels, fill_value=0).values
    noisy  = counts + np.random.laplace(scale=1/eps, size=len(counts)) #Laplace noise with scale 1/ε (global sensitivity = 1)
    return noisy, levels

EPS_TOTAL = 1.0         
bins_age  = np.linspace(17, 91, 11)  

eps_age_i = EPS_TOTAL         
eps_wc_i  = EPS_TOTAL
eps_ed_i  = EPS_TOTAL

age_noisy_i, _   = dp_histogram(df["age"], bins_age, eps_age_i)
wc_noisy_i, lab_w= categorical_dp_hist(df["workclass"], eps_wc_i)
ed_noisy_i, lab_e= categorical_dp_hist(df["education"], eps_ed_i)

print("Scenario (i) — Laplace scale b = 1")
print("Noisy age histogram:      ", age_noisy_i.astype(int))
print("Noisy workclass counts:   ", dict(zip(lab_w, wc_noisy_i.astype(int))))
print("Noisy education counts:   ", dict(zip(lab_e, ed_noisy_i.astype(int))))
print()

eps_each = EPS_TOTAL / 3        # split ε
age_noisy_ii, _ = dp_histogram(df["age"], bins_age, eps_each)
wc_noisy_ii,_   = categorical_dp_hist(df["workclass"], eps_each)
ed_noisy_ii,_   = categorical_dp_hist(df["education"], eps_each)

print("Scenario (ii) — Laplace scale b = 3")
print("Noisy age histogram:      ", age_noisy_ii.astype(int))
print("Noisy workclass counts:   ", dict(zip(lab_w, wc_noisy_ii.astype(int))))
print("Noisy education counts:   ", dict(zip(lab_e, ed_noisy_ii.astype(int))))

# Explain/justify your choices
# I set ε = 1 and δ = 10⁻⁵ for Gaussian noise because this is the standard moderate privacy budget used in many demos; δ < 1/n.
# For numeric queries the global sensitivity equals (max − min)/n, for histogram counts it is 1, so the Laplace scale is b = 1/ε.
# What is a meaningful comparison?
# keep ε identical and δ identical.
# run the code ≥ 1 000 times and record the distribution of errors if you want an empirical plot.
