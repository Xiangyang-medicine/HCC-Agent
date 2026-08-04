"""
TCGA-LIHC Analysis Figures for ACM TIST Publication
Visualization of LLM Agent System Results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import seaborn as sns
import json

# ============================================
# Load all experimental data
# ============================================
print("Loading experimental data...")

# Evaluation metrics
with open('F:/ACM/experiments/evaluation_metrics_20260709_113012.json', 'r') as f:
    metrics = json.load(f)

# Model comparison
model_comp = pd.read_csv('F:/ACM/experiments/model_comparison_20260709_113012.csv')

# Risk factors
risk_factors = pd.read_csv('F:/ACM/experiments/risk_factors.csv')

# Test predictions
test_preds = pd.read_csv('F:/ACM/experiments/test_predictions_20260709_113012.csv')

# Agent predictions
agent_preds = pd.read_csv('F:/ACM/experiments/agent_predictions_20260709_113012.csv')

# Data summary
with open('F:/ACM/experiments/data_summary.json', 'r') as f:
    data_summary = json.load(f)

# Load TCGA-LIHC data
tcga_df = pd.read_parquet('F:/ACM/data/tcga_lihc_validated.parquet')

print(f"Total samples: {len(tcga_df)}")
print(f"Test samples: {len(test_preds)}")

# ============================================
# Figure 1: Model Performance Comparison
# ============================================
print("\nGenerating Figure 1: Model Performance Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: C-index comparison
ax = axes[0]
models = ['Simple (LR)', 'Cox PH', 'DeepSurv', 'LLM Agent']
c_indices = [metrics[m]['c_index'] for m in models]
ci_lowers = [metrics[m]['c_index_ci_low'] for m in models]
ci_uppers = [metrics[m]['c_index_ci_high'] for m in models]
c_errors = [[c - l for c, l in zip(c_indices, ci_lowers)],
            [u - c for c, u in zip(c_indices, ci_uppers)]]

colors = ['#95a5a6', '#e74c3c', '#3498db', '#27ae60']
x_pos = np.arange(len(models))

bars = ax.bar(x_pos, c_indices, yerr=c_errors, color=colors, edgecolor='black',
              linewidth=1.2, capsize=5, error_kw={'linewidth': 1.5})

for bar, val in zip(bars, c_indices):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random')
ax.set_ylabel('C-index', fontsize=12)
ax.set_title('A. Model Discrimination', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10, rotation=15, ha='right')
ax.set_ylim(0, 1.0)
ax.grid(True, axis='y', alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

# Panel B: AUC at different time points
ax = axes[1]
time_points = ['1yr', '3yr', '5yr']
auc_data = {
    'Simple (LR)': [metrics['Simple (LR)']['auc_1yr'], metrics['Simple (LR)']['auc_3yr'], metrics['Simple (LR)']['auc_5yr']],
    'Cox PH': [metrics['Cox PH']['auc_1yr'], metrics['Cox PH']['auc_3yr'], metrics['Cox PH']['auc_5yr']],
    'DeepSurv': [metrics['DeepSurv']['auc_1yr'], metrics['DeepSurv']['auc_3yr'], metrics['DeepSurv']['auc_5yr']],
    'LLM Agent': [metrics['LLM Agent']['auc_1yr'], metrics['LLM Agent']['auc_3yr'], metrics['LLM Agent']['auc_5yr']]
}

x = np.arange(len(time_points))
width = 0.2
for i, (model, color) in enumerate(zip(models, colors)):
    vals = auc_data[model]
    ax.bar(x + i*width - 1.5*width, vals, width, label=model, color=color, edgecolor='black', linewidth=0.5)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_ylabel('AUC', fontsize=12)
ax.set_xlabel('Time Point', fontsize=12)
ax.set_title('B. Time-Dependent AUC', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(time_points, fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(True, axis='y', alpha=0.3)

# Panel C: Brier Score (lower is better)
ax = axes[2]
brier_scores = [metrics[m]['brier_score'] for m in models]

bars = ax.bar(x_pos, brier_scores, color=colors, edgecolor='black', linewidth=1.2)

for bar, val in zip(bars, brier_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Brier Score', fontsize=12)
ax.set_title('C. Prediction Accuracy (Brier Score)', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10, rotation=15, ha='right')
ax.set_ylim(0, 0.7)
ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('TCGA-LIHC Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('F:/ACM/tcga_figure1_model_comparison.pdf', bbox_inches='tight')
plt.close()
print('Saved: tcga_figure1_model_comparison.pdf')

# ============================================
# Figure 2: Risk Factor Importance (Forest Plot)
# ============================================
print("\nGenerating Figure 2: Risk Factor Importance...")

fig, ax = plt.subplots(figsize=(10, 8))

# Sort by absolute coefficient
risk_factors_sorted = risk_factors.sort_values('abs_coef', ascending=True)

y_pos = np.arange(len(risk_factors_sorted))
coef = risk_factors_sorted['coefficient'].values
colors_bar = ['#e74c3c' if c > 0 else '#27ae60' for c in coef]

# Horizontal bar chart
bars = ax.barh(y_pos, coef, color=colors_bar, edgecolor='black', linewidth=0.8, height=0.7)

# Add value labels
for bar, val in zip(bars, coef):
    x_offset = 2 if val >= 0 else -2
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + x_offset, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', ha=ha, fontsize=9)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(risk_factors_sorted['feature'].values, fontsize=10)
ax.set_xlabel('Cox Model Coefficient', fontsize=12)
ax.set_title('Risk Factor Importance (Cox PH Coefficients)', fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Higher Risk'),
                   Patch(facecolor='#27ae60', label='Lower Risk')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('F:/ACM/tcga_figure2_risk_factors.pdf', bbox_inches='tight')
plt.close()
print('Saved: tcga_figure2_risk_factors.pdf')

# ============================================
# Figure 3: Kaplan-Meier Curves by LLM Agent Risk Groups
# ============================================
print("\nGenerating Figure 3: Kaplan-Meier Curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.30, wspace=0.25)

panel_labels = ['A', 'B', 'C', 'D']
risk_colors = {'very_high': '#c0392b', 'high': '#e74c3c', 'intermediate': '#f39c12',
               'low': '#27ae60'}

# Risk group distributions
risk_groups = ['very_high', 'high', 'intermediate', 'low']
group_labels = ['Very High Risk', 'High Risk', 'Intermediate Risk', 'Low Risk']

# Merge agent predictions with TCGA data for duration/event
tcga_test = tcga_df[tcga_df['patient_id'].isin(agent_preds['patient_id'])].copy()
tcga_test = tcga_test.merge(agent_preds[['patient_id', 'risk_level', 'risk_score']],
                            left_on='patient_id', right_on='patient_id', how='left')

# Rename for consistency and convert event to binary
tcga_test = tcga_test.rename(columns={'survival_months': 'duration', 'vital_status': 'event'})
tcga_test['event'] = (tcga_test['event'] == 'Dead').astype(int)

for idx, (risk_group, group_label) in enumerate(zip(risk_groups, group_labels)):
    ax = axes[idx // 2, idx % 2]

    group_data = tcga_test[tcga_test['risk_level'] == risk_group]

    if len(group_data) == 0:
        ax.text(0.5, 0.5, f'No samples in {group_label}', ha='center', va='center', fontsize=12)
        ax.set_title(f'{group_label} (n=0)', fontsize=11, fontweight='bold')
        continue

    kmf = KaplanMeierFitter()
    kmf.fit(group_data['duration'].values, group_data['event'].values, label=group_label)

    color = risk_colors[risk_group]
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.5)

    # Add median survival if available
    try:
        median_surv = kmf.median_survival_time_
        if not pd.isna(median_surv) and median_surv != np.inf:
            ax.axvline(x=median_surv, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    except:
        pass

    ax.text(0.02, 0.98, panel_labels[idx], transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')
    ax.text(0.98, 0.98, f'n={len(group_data)}', transform=ax.transAxes, fontsize=10,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           edgecolor='gray', alpha=0.9))

    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title(f'{group_label} (n={len(group_data)})', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle('Kaplan-Meier Survival by LLM Agent Risk Stratification (TCGA-LIHC)', fontsize=14, fontweight='bold', y=0.98)
plt.savefig('F:/ACM/tcga_figure3_km_by_risk.pdf', bbox_inches='tight')
plt.close()
print('Saved: tcga_figure3_km_by_risk.pdf')

# ============================================
# Figure 4: Combined KM Comparison (High vs Low)
# ============================================
print("\nGenerating Figure 4: Combined KM Comparison...")

fig, ax = plt.subplots(figsize=(10, 8))

# Split into high vs low risk based on median score
median_score = tcga_test['risk_score'].median()
high_risk = tcga_test[tcga_test['risk_score'] > median_score]
low_risk = tcga_test[tcga_test['risk_score'] <= median_score]

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

kmf_high.fit(high_risk['duration'].values, high_risk['event'].values, label=f'High Risk (n={len(high_risk)})')
kmf_low.fit(low_risk['duration'].values, low_risk['event'].values, label=f'Low Risk (n={len(low_risk)})')

kmf_high.plot_survival_function(ax=ax, ci_show=True, color='#e74c3c', linewidth=2.5)
kmf_low.plot_survival_function(ax=ax, ci_show=True, color='#27ae60', linewidth=2.5)

# Log-rank test
results_lr = logrank_test(high_risk['duration'].values, low_risk['duration'].values,
                          high_risk['event'].values, low_risk['event'].values)

# Calculate HR
cph = CoxPHFitter()
temp_df = tcga_test[['duration', 'event', 'risk_score']].copy()
temp_df['risk_group'] = (temp_df['risk_score'] > median_score).astype(int)
cph.fit(temp_df[['duration', 'event', 'risk_group']], duration_col='duration', event_col='event')
hr = np.exp(cph.params_['risk_group'])
ci_l = np.exp(cph.confidence_intervals_.loc['risk_group', '95% lower-bound'])
ci_u = np.exp(cph.confidence_intervals_.loc['risk_group', '95% upper-bound'])

# Calculate C-index
cidx = concordance_index(tcga_test['duration'].values, -tcga_test['risk_score'].values,
                         tcga_test['event'].values)

hr_text = f'HR={hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f})\nLog-rank p={results_lr.p_value:.4f}\nC-index={cidx:.3f}'
ax.text(0.98, 0.98, hr_text, transform=ax.transAxes, fontsize=10,
        va='top', ha='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                       edgecolor='gray', alpha=0.9))

ax.set_xlabel('Time (months)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('LLM Agent Risk Stratification (High vs Low)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 160)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left', fontsize=11)

plt.tight_layout()
plt.savefig('F:/ACM/tcga_figure4_km_combined.pdf', bbox_inches='tight')
plt.close()
print('Saved: tcga_figure4_km_combined.pdf')

# ============================================
# Figure 5: Calibration Curves
# ============================================
print("\nGenerating Figure 5: Calibration Curves...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, model in enumerate(models):
    ax = axes[idx // 2, idx % 2]

    # Get predictions
    if model == 'LLM Agent':
        preds = agent_preds['risk_score'].values
    elif model == 'Simple (LR)':
        preds = test_preds['Simple (LR)'].values
    elif model == 'Cox PH':
        # Normalize Cox PH predictions
        cox_raw = test_preds['Cox PH'].values
        preds = (cox_raw - cox_raw.min()) / (cox_raw.max() - cox_raw.min())
    else:  # DeepSurv
        preds = (test_preds['DeepSurv'].values - test_preds['DeepSurv'].values.min()) / \
                (test_preds['DeepSurv'].values.max() - test_preds['DeepSurv'].values.min())

    # Get actual outcomes
    events = test_preds['event'].values
    durations = test_preds['time'].values

    # Define 3-year outcome
    y_true = ((durations <= 36) & (events == 1)).astype(int)

    # Calculate calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, preds, n_bins=5, strategy='uniform'
        )

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', color=colors[idx],
                linewidth=2, markersize=8, label=f' {model}')

        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Calibration data\nnot available', ha='center', va='center',
                fontsize=11, transform=ax.transAxes)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')

plt.suptitle('Calibration Curves (3-Year Outcome)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('F:/ACM/tcga_figure5_calibration.pdf', bbox_inches='tight')
plt.close()
print('Saved: tcga_figure5_calibration.pdf')

# ============================================
# Figure 6: Data Distribution Overview
# ============================================
print("\nGenerating Figure 6: Data Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Stage distribution
ax = axes[0, 0]
stages = ['Stage I', 'Stage II', 'Stage IIIA', 'Stage IIIB', 'Stage IV']
stage_counts = [data_summary['stage_distribution'].get(s, 0) for s in stages]
colors_stage = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stages)))

ax.bar(stages, stage_counts, color=colors_stage, edgecolor='black', linewidth=1)
for i, v in enumerate(stage_counts):
    ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Patients', fontsize=11)
ax.set_title('A. Cancer Stage Distribution', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.grid(True, axis='y', alpha=0.3)

# Panel B: Event rate
ax = axes[0, 1]
labels = ['Training\n(n=296)', 'Test\n(n=75)']
events = [data_summary['train_events'], data_summary['test_events']]
samples = [data_summary['n_train'], data_summary['n_test']]
rates = [e/s*100 for e, s in zip(events, samples)]

bars = ax.bar(labels, rates, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1)
for bar, rate, event in zip(bars, rates, events):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.1f}%\n({event} events)', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Event Rate (%)', fontsize=11)
ax.set_title('B. Survival Event Rate', fontsize=12, fontweight='bold')
ax.set_ylim(0, 50)
ax.grid(True, axis='y', alpha=0.3)

# Panel C: Risk score distribution
ax = axes[1, 0]
risk_levels = ['very_high', 'high', 'intermediate', 'low']
level_counts = [sum(agent_preds['risk_level'] == r) for r in risk_levels]
level_labels = ['Very High', 'High', 'Intermediate', 'Low']
colors_risk = ['#c0392b', '#e74c3c', '#f39c12', '#27ae60']

ax.bar(level_labels, level_counts, color=colors_risk, edgecolor='black', linewidth=1)
for i, v in enumerate(level_counts):
    ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Patients', fontsize=11)
ax.set_title('C. LLM Agent Risk Stratification', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

# Panel D: Feature importance heatmap
ax = axes[1, 1]
top_features = risk_factors.nlargest(10, 'abs_coef')
feature_names = top_features['feature'].values
gene_features = ['HK2', 'PKM', 'LDHA', 'LDHB', 'GPI', 'PFKL', 'GLS', 'GLUD1', 'FASN', 'SCD']
clinical_features = ['stage', 'grade', 'gender', 'age', 'afp_level', 'albumin', 'bilirubin']

feature_types = ['Gene' if f in gene_features else 'Clinical' for f in feature_names]
type_colors = ['#3498db' if t == 'Gene' else '#e74c3c' for t in feature_types]

y_pos = np.arange(len(feature_names))
ax.barh(y_pos, top_features['abs_coef'].values, color=type_colors, edgecolor='black', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names, fontsize=9)
ax.set_xlabel('|Coefficient|', fontsize=11)
ax.set_title('D. Top 10 Risk Factors', fontsize=12, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', label='Metabolic Gene'),
                   Patch(facecolor='#e74c3c', label='Clinical Feature')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.suptitle(f'TCGA-LIHC Dataset Overview (N={data_summary["n_patients"]})', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('F:/ACM/tcga_figure6_data_overview.pdf', bbox_inches='tight')
plt.close()
print('Saved: tcga_figure6_data_overview.pdf')

# ============================================
# Figure 7: Heatmap of Predictions vs Actuals
# ============================================
print("\nGenerating Figure 7: Prediction Heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sort by actual survival
sorted_idx = tcga_test.sort_values('duration')['patient_id'].values
sorted_agent = agent_preds.set_index('patient_id').loc[sorted_idx]

# Panel A: LLM Agent predictions
ax = axes[0]
im = ax.imshow(sorted_agent['risk_score'].values.reshape(1, -1), aspect='auto',
               cmap='RdYlGn_r', vmin=0, vmax=1)
ax.set_xlabel('Patients (sorted by survival time)', fontsize=11)
ax.set_yticks([])
ax.set_title('A. LLM Agent Risk Scores', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Risk Score', orientation='vertical', pad=0.02)

# Add event markers
events = sorted_agent['actual_event'].values
event_x = np.where(events == 1)[0]
ax.scatter(event_x, [0]*len(event_x), marker='|', color='black', s=100, linewidths=2, label='Death')
ax.legend(loc='upper right', fontsize=9)

# Panel B: Comparison of all models
ax = axes[1]

# Get predictions in order of test_preds
lr_norm = (test_preds['Simple (LR)'] - test_preds['Simple (LR)'].min()) / \
          (test_preds['Simple (LR)'].max() - test_preds['Simple (LR)'].min())
cox_norm = (test_preds['Cox PH'] - test_preds['Cox PH'].min()) / \
           (test_preds['Cox PH'].max() - test_preds['Cox PH'].min())
deepsurv_norm = (test_preds['DeepSurv'] - test_preds['DeepSurv'].min()) / \
                (test_preds['DeepSurv'].max() - test_preds['DeepSurv'].min())

# Sort test_preds by Cox PH (or any model) for consistent ordering
sort_idx = np.argsort(cox_norm.values)
sorted_lr = lr_norm.values[sort_idx]
sorted_cox = cox_norm.values[sort_idx]
sorted_deepsurv = deepsurv_norm.values[sort_idx]

comparison_data = pd.DataFrame({
    'Simple LR': sorted_lr,
    'Cox PH': sorted_cox,
    'DeepSurv': sorted_deepsurv
}).T

sns.heatmap(comparison_data, cmap='RdYlGn_r', vmin=0, vmax=1, ax=ax,
            xticklabels=False, cbar_kws={'label': 'Normalized Risk Score'})
ax.set_title('B. Model Prediction Comparison', fontsize=12, fontweight='bold')
ax.set_xlabel('Patients (sorted by Cox PH score)', fontsize=11)
ax.set_ylabel('')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.suptitle('Risk Score Heatmaps', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('F:/ACM/tcga_figure7_prediction_heatmap.pdf', bbox_inches='tight')
plt.close()
print('Saved: tcga_figure7_prediction_heatmap.pdf')

# ============================================
# Summary
# ============================================
print("\n" + "="*60)
print("TCGA-LIHC Figures Generated Successfully")
print("="*60)
print("\nGenerated Figures:")
print("  1. tcga_figure1_model_comparison.pdf - Model Performance (C-index, AUC, Brier)")
print("  2. tcga_figure2_risk_factors.pdf - Risk Factor Importance (Forest Plot)")
print("  3. tcga_figure3_km_by_risk.pdf - KM Curves by Risk Group")
print("  4. tcga_figure4_km_combined.pdf - Combined KM (High vs Low)")
print("  5. tcga_figure5_calibration.pdf - Calibration Curves")
print("  6. tcga_figure6_data_overview.pdf - Dataset Overview")
print("  7. tcga_figure7_prediction_heatmap.pdf - Prediction Heatmaps")
print("\nKey Results:")
print(f"  LLM Agent C-index: {metrics['LLM Agent']['c_index']:.3f} (95% CI: {metrics['LLM Agent']['c_index_ci_low']:.3f}-{metrics['LLM Agent']['c_index_ci_high']:.3f})")
print(f"  LLM Agent HR: {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f})")
print(f"  Dataset: {data_summary['n_patients']} patients ({data_summary['n_train']} train, {data_summary['n_test']} test)")
print("\nAll figures saved with Arial font!")
