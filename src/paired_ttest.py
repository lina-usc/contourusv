import pandas as pd
from scipy.stats import ttest_rel

# Load the CSV files into pandas DataFrames
contourusv_df = pd.read_csv(
    '/Users/evana_anis/Desktop/VSCode/mouseapp_test/results/final_results/table2/Evaluation_ContourUSV_usvseg_data_gerbil_Ground_Truth_Annotations.csv', sep='\t')
deepsqueak_df = pd.read_csv(
    '/Users/evana_anis/Desktop/VSCode/mouseapp_test/results/final_results/table2/Evaluation_DeepSqueak_usvseg_data_gerbil_Ground_Truth_Annotations.csv', sep='\t')
josephthemouse_df = pd.read_csv(
    '/Users/evana_anis/Desktop/VSCode/mouseapp_test/results/final_results/table2/Evaluation_Joseph_usvseg_data_gerbil_Ground_Truth_Annotations.csv', sep='\t')

# Exclude the last row from each DataFrame
contourusv_df = contourusv_df.iloc[:-1]
deepsqueak_df = deepsqueak_df.iloc[:-1]
josephthemouse_df = josephthemouse_df.iloc[:-1]

# Convert the relevant columns to numeric, coercing errors to NaN
contourusv_precision = pd.to_numeric(
    contourusv_df['Precision'], errors='coerce').values
deepsqueak_precision = pd.to_numeric(
    deepsqueak_df['Precision'], errors='coerce').values
josephthemouse_precision = pd.to_numeric(
    josephthemouse_df['Precision'], errors='coerce').values

contourusv_recall = pd.to_numeric(
    contourusv_df['Recall'], errors='coerce').values
deepsqueak_recall = pd.to_numeric(
    deepsqueak_df['Recall'], errors='coerce').values
josephthemouse_recall = pd.to_numeric(
    josephthemouse_df['Recall'], errors='coerce').values

contourusv_f1 = pd.to_numeric(
    contourusv_df['F1 Score'], errors='coerce').values
deepsqueak_f1 = pd.to_numeric(
    deepsqueak_df['F1 Score'], errors='coerce').values
josephthemouse_f1 = pd.to_numeric(
    josephthemouse_df['F1 Score'], errors='coerce').values

contourusv_specificity = pd.to_numeric(
    contourusv_df['Specificity'], errors='coerce').values
deepsqueak_specificity = pd.to_numeric(
    deepsqueak_df['Specificity'], errors='coerce').values
josephthemouse_specificity = pd.to_numeric(
    josephthemouse_df['Specificity'], errors='coerce').values

# # Extract the relevant columns
# contourusv_precision = contourusv_df['Precision'].values
# deepsqueak_precision = deepsqueak_df['Precision'].values
# josephthemouse_precision = josephthemouse_df['Precision'].values

# contourusv_recall = contourusv_df['Recall'].values
# deepsqueak_recall = deepsqueak_df['Recall'].values
# josephthemouse_recall = josephthemouse_df['Recall'].values

# contourusv_f1 = contourusv_df['F1 Score'].values
# deepsqueak_f1 = deepsqueak_df['F1 Score'].values
# josephthemouse_f1 = josephthemouse_df['F1 Score'].values

# contourusv_specificity = contourusv_df['Specificity'].values
# deepsqueak_specificity = deepsqueak_df['Specificity'].values
# josephthemouse_specificity = josephthemouse_df['Specificity'].values

# Perform paired t-tests using the actual values
# Precision
t_stat_cd, p_value_cd = ttest_rel(contourusv_precision, deepsqueak_precision)
t_stat_cj, p_value_cj = ttest_rel(
    contourusv_precision, josephthemouse_precision)
t_stat_dj, p_value_dj = ttest_rel(
    deepsqueak_precision, josephthemouse_precision)

# Recall
t_stat_cr, p_value_cr = ttest_rel(contourusv_recall, deepsqueak_recall)
t_stat_cjr, p_value_cjr = ttest_rel(contourusv_recall, josephthemouse_recall)
t_stat_djr, p_value_djr = ttest_rel(deepsqueak_recall, josephthemouse_recall)

# F1 Score
t_stat_cf, p_value_cf = ttest_rel(contourusv_f1, deepsqueak_f1)
t_stat_cjf, p_value_cjf = ttest_rel(contourusv_f1, josephthemouse_f1)
t_stat_djf, p_value_djf = ttest_rel(deepsqueak_f1, josephthemouse_f1)

# Specificity
t_stat_cs, p_value_cs = ttest_rel(
    contourusv_specificity, deepsqueak_specificity)
t_stat_cjs, p_value_cjs = ttest_rel(
    contourusv_specificity, josephthemouse_specificity)
t_stat_djs, p_value_djs = ttest_rel(
    deepsqueak_specificity, josephthemouse_specificity)

# Print the results
print(
    f"Precision: ContourUSV vs. DeepSqueak: T={t_stat_cd:.2f}, P-value={p_value_cd:.2e}")
print(
    f"Precision: ContourUSV vs. JosephTheMoUSE: T={t_stat_cj:.2f}, P-value={p_value_cj:.2e}")
print(
    f"Precision: DeepSqueak vs. JosephTheMoUSE: T={t_stat_dj:.2f}, P-value={p_value_dj:.2e}")

print(
    f"Recall: ContourUSV vs. DeepSqueak: T={t_stat_cr:.2f}, P-value={p_value_cr:.2e}")
print(
    f"Recall: ContourUSV vs. JosephTheMoUSE: T={t_stat_cjr:.2f}, P-value={p_value_cjr:.2e}")
print(
    f"Recall: DeepSqueak vs. JosephTheMoUSE: T={t_stat_djr:.2f}, P-value={p_value_djr:.2e}")

print(
    f"F1 Score: ContourUSV vs. DeepSqueak: T={t_stat_cf:.2f}, P-value={p_value_cf:.2e}")
print(
    f"F1 Score: ContourUSV vs. JosephTheMoUSE: T={t_stat_cjf:.2f}, P-value={p_value_cjf:.2e}")
print(
    f"F1 Score: DeepSqueak vs. JosephTheMoUSE: T={t_stat_djf:.2f}, P-value={p_value_djf:.2e}")

print(
    f"Specificity: ContourUSV vs. DeepSqueak: T={t_stat_cs:.2f}, P-value={p_value_cs:.2e}")
print(
    f"Specificity: ContourUSV vs. JosephTheMoUSE: T={t_stat_cjs:.2f}, P-value={p_value_cjs:.2e}")
print(
    f"Specificity: DeepSqueak vs. JosephTheMoUSE: T={t_stat_djs:.2f}, P-value={p_value_djs:.2e}")
