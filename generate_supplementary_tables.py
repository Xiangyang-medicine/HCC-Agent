"""
Generate Supplementary Tables for ACM TIST Publication
"""
import pandas as pd
import numpy as np
import json

# Load all data
with open('F:/ACM/experiments/evaluation_metrics_20260709_113012.json', 'r') as f:
    metrics = json.load(f)

with open('F:/ACM/experiments/data_summary.json', 'r') as f:
    data_summary = json.load(f)

risk_factors = pd.read_csv('F:/ACM/experiments/risk_factors.csv')
test_preds = pd.read_csv('F:/ACM/experiments/test_predictions_20260709_113012.csv')
agent_preds = pd.read_csv('F:/ACM/experiments/agent_predictions_20260709_113012.csv')

# Load TCGA-LIHC data
tcga_df = pd.read_parquet('F:/ACM/data/tcga_lihc_validated.parquet')

# External validation results
gse116174_results = pd.read_csv('F:/ACM/GSE116174/model_comparison_results.csv')
gse14520_results = pd.read_csv('F:/ACM/GSE14520/model_comparison_results.csv')

# ============================================
# Supplementary Table 1: Model Performance Metrics
# ============================================
print("Generating Supplementary Table 1: Model Performance...")

table1 = []
for model in ['Simple (LR)', 'Cox PH', 'DeepSurv', 'LLM Agent']:
    m = metrics[model]
    table1.append({
        'Model': model,
        'C-index': f"{m['c_index']:.3f}",
        'C-index 95% CI': f"[{m['c_index_ci_low']:.3f}, {m['c_index_ci_high']:.3f}]",
        'AUC (1-year)': f"{m['auc_1yr']:.3f}",
        'AUC (3-year)': f"{m['auc_3yr']:.3f}",
        'AUC (5-year)': f"{m['auc_5yr']:.3f}",
        'Calibration Slope': f"{m['calibration_slope']:.3f}",
        'Calibration Intercept': f"{m['calibration_intercept']:.3f}",
        'Brier Score': f"{m['brier_score']:.3f}",
        'N': m['n_samples'],
        'Events': m['n_events']
    })

table1_df = pd.DataFrame(table1)
table1_df.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable1_Model_Performance.csv', index=False)

# ============================================
# Supplementary Table 2: Risk Factor Coefficients
# ============================================
print("Generating Supplementary Table 2: Risk Factors...")

table2 = risk_factors.copy()
table2.columns = ['Feature', 'Coefficient', 'Abs_Coefficient']
table2['HR'] = np.exp(table2['Coefficient'])
table2['Direction'] = table2['Coefficient'].apply(lambda x: 'Risk' if x > 0 else 'Protective')
table2 = table2[['Feature', 'Coefficient', 'HR', 'Direction', 'Abs_Coefficient']]
table2 = table2.sort_values('Abs_Coefficient', ascending=False)
table2.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable2_Risk_Factors.csv', index=False)

# ============================================
# Supplementary Table 3: Dataset Characteristics
# ============================================
print("Generating Supplementary Table 3: Dataset Characteristics...")

table3_data = {
    'Parameter': [
        'Total Patients', 'Training Set Size', 'Test Set Size',
        'Training Events', 'Test Events', 'Training Event Rate (%)',
        'Test Event Rate (%)', 'Number of Features', 'Number of Gene Features',
        'Age Range', 'Male (%)', 'Female (%)'
    ],
    'TCGA-LIHC': [
        data_summary['n_patients'],
        data_summary['n_train'],
        data_summary['n_test'],
        data_summary['train_events'],
        data_summary['test_events'],
        f"{data_summary['train_events']/data_summary['n_train']*100:.1f}",
        f"{data_summary['test_events']/data_summary['n_test']*100:.1f}",
        data_summary['n_features'],
        data_summary['n_gene_features'],
        'N/A',
        f"{len(tcga_df[tcga_df['gender']=='Male'])/len(tcga_df)*100:.1f}",
        f"{len(tcga_df[tcga_df['gender']=='Female'])/len(tcga_df)*100:.1f}"
    ]
}

# Add gene features
for gene in ['HK2', 'PKM', 'LDHA', 'LDHB', 'GPI', 'PFKL', 'GLS', 'GLUD1', 'FASN', 'SCD']:
    if gene in data_summary['feature_names']:
        table3_data['Parameter'].append(f'Metabolic Gene: {gene}')
        table3_data['TCGA-LIHC'].append('Included')

# Stage distribution
for stage, count in data_summary['stage_distribution'].items():
    table3_data['Parameter'].append(f'{stage}')
    table3_data['TCGA-LIHC'].append(f'{count} ({count/data_summary["n_patients"]*100:.1f}%)')

table3_df = pd.DataFrame(table3_data)
table3_df.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable3_Dataset_Characteristics.csv', index=False)

# ============================================
# Supplementary Table 4: External Validation Results
# ============================================
print("Generating Supplementary Table 4: External Validation...")

# GSE116174
gse116174_table = gse116174_results.copy()
gse116174_table['Dataset'] = 'GSE116174'
gse116174_table['N'] = len(pd.read_csv('F:/ACM/GSE116174/all_model_risk_scores.csv'))

# GSE14520
gse14520_table = gse14520_results.copy()
gse14520_table['Dataset'] = 'GSE14520'
gse14520_table['N'] = len(pd.read_csv('F:/ACM/GSE14520/all_model_risk_scores.csv'))

table4_df = pd.concat([gse116174_table, gse14520_table], ignore_index=True)
table4_df = table4_df[['Dataset', 'Model', 'C-index', 'HR', 'CI_lower', 'CI_upper', 'p_value']]
table4_df.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable4_External_Validation.csv', index=False)

# ============================================
# Supplementary Table 5: LLM Agent Predictions
# ============================================
print("Generating Supplementary Table 5: LLM Agent Predictions...")

# Merge with TCGA data for full details
tcga_test = tcga_df[tcga_df['patient_id'].isin(agent_preds['patient_id'])].copy()
tcga_test = tcga_test.merge(agent_preds, left_on='patient_id', right_on='patient_id', how='left')

table5_cols = ['patient_id', 'risk_level', 'risk_score', 'predicted_survival',
               'actual_survival', 'actual_event', 'age', 'gender', 'stage', 'grade']
table5_df = tcga_test[table5_cols].copy()
table5_df.columns = ['Patient_ID', 'Risk_Level', 'Risk_Score', 'Predicted_Survival_Months',
                     'Actual_Survival_Months', 'Death_Event', 'Age', 'Gender', 'Stage', 'Grade']
table5_df = table5_df.sort_values('Risk_Score', ascending=False)
table5_df.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable5_LLM_Agent_Predictions.csv', index=False)

# ============================================
# Supplementary Table 6: Metabolic Gene Expression Statistics
# ============================================
print("Generating Supplementary Table 6: Gene Expression...")

metabolic_genes = ['HK2', 'PKM', 'LDHA', 'LDHB', 'GPI', 'PFKL', 'GLS', 'GLUD1', 'FASN', 'SCD',
                   'CA9', 'VEGFA', 'HIF1A', 'MYC', 'CTNNB1']

gene_stats = []
for gene in metabolic_genes:
    if gene in tcga_df.columns:
        expr = tcga_df[gene].dropna()
        gene_stats.append({
            'Gene': gene,
            'N': len(expr),
            'Mean': f"{expr.mean():.3f}",
            'Std': f"{expr.std():.3f}",
            'Min': f"{expr.min():.3f}",
            'Q1': f"{expr.quantile(0.25):.3f}",
            'Median': f"{expr.median():.3f}",
            'Q3': f"{expr.quantile(0.75):.3f}",
            'Max': f"{expr.max():.3f}",
            'Available': 'Yes'
        })
    else:
        gene_stats.append({'Gene': gene, 'Available': 'No'})

table6_df = pd.DataFrame(gene_stats)
table6_df.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable6_Gene_Expression_Statistics.csv', index=False)

# ============================================
# Supplementary Table 7: Cross-Model Predictions
# ============================================
print("Generating Supplementary Table 7: Model Predictions Comparison...")

table7_df = test_preds.copy()
table7_df.columns = ['Simple_LR', 'Cox_PH', 'DeepSurv', 'Time', 'Event']
table7_df.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable7_Model_Predictions.csv', index=False)

# ============================================
# Supplementary Table 8: Kaplan-Meier Statistics
# ============================================
print("Generating Supplementary Table 8: KM Statistics...")

from lifelines import KaplanMeierFitter

# Calculate KM statistics by risk group
risk_groups = ['very_high', 'high', 'intermediate', 'low']
km_stats = []

for risk_level in risk_groups:
    group = agent_preds[agent_preds['risk_level'] == risk_level]
    if len(group) > 0:
        # Merge with tcga for survival data
        merged = group.merge(tcga_df[['patient_id', 'survival_months', 'vital_status']],
                            left_on='patient_id', right_on='patient_id')
        if len(merged) > 0:
            event = (merged['vital_status'] == 'Dead').astype(int)
            duration = merged['survival_months']

            kmf = KaplanMeierFitter()
            kmf.fit(duration, event)

            try:
                median_surv = kmf.median_survival_time_
                if pd.isna(median_surv) or median_surv == np.inf:
                    median_surv = 'Not reached'
                else:
                    median_surv = f"{median_surv:.1f}"
            except:
                median_surv = 'Not available'

            km_stats.append({
                'Risk_Level': risk_level,
                'N': len(group),
                'Events': event.sum(),
                'Event_Rate_%': f"{event.sum()/len(group)*100:.1f}",
                'Median_Survival_Months': median_surv,
                'Mean_Risk_Score': f"{group['risk_score'].mean():.3f}",
                'SD_Risk_Score': f"{group['risk_score'].std():.3f}"
            })

table8_df = pd.DataFrame(km_stats)
table8_df.to_csv('F:/ACM/publication_figures/supplementary_tables/SuppTable8_KM_Statistics.csv', index=False)

# ============================================
# Summary
# ============================================
print("\n" + "="*60)
print("Supplementary Tables Generated Successfully!")
print("="*60)
print("\nGenerated Tables:")
for i in range(1, 9):
    print(f"  SuppTable{i}_*.csv")

print("\nAll tables saved to: F:/ACM/publication_figures/supplementary_tables/")
