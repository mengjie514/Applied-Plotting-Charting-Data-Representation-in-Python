import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Uncomment if using Google Colab
# from google.colab import drive
# drive.mount('/content/gdrive')

plt.style.use('seaborn-v0_8-whitegrid')

BASE_PATH = '/content/gdrive/My Drive/Colab Notebooks/HPI'

# ============================================================================
# LOAD DATA
# ============================================================================

mae_file = os.path.join(BASE_PATH, 'CrossModel_MAE.xlsx')

mae_eth = pd.read_excel(mae_file, sheet_name='Eth')
mae_ethgen = pd.read_excel(mae_file, sheet_name='EthGen')
mae_ethgenage = pd.read_excel(mae_file, sheet_name='EthGenAge')

# Convert GPT_Model to string
mae_eth['GPT_Model'] = mae_eth['GPT_Model'].astype(str)
mae_ethgen['GPT_Model'] = mae_ethgen['GPT_Model'].astype(str)
mae_ethgenage['GPT_Model'] = mae_ethgenage['GPT_Model'].astype(str)

models = ['3.5', '4', '4o', '4.1']
model_labels = ['GPT-3.5', 'GPT-4', 'GPT-4o', 'GPT-4.1']
ethnicities = ['EU', 'Maori', 'Pacific', 'Asian', 'MELAA']

# Consistent color scheme for ethnicities (used throughout all graphs)
colors = {'EU': '#1f77b4', 'Maori': '#ff7f0e', 'Pacific': '#2ca02c',
          'Asian': '#d62728', 'MELAA': '#9467bd'}

# Different line styles for gender (Graph 2)
gender_linestyles = {'Male': '-', 'Female': '--'}  # solid for Male, dashed for Female

# Different line styles for age groups (Graph 3)
age_linestyles = {'15-29': '-', '30-64': '--', '65+': ':'}  # solid, dashed, dotted

# ============================================================================
# GRAPH 1: Ethnicity x Model
# ============================================================================

print("Generating GRAPH 1: Ethnicity x Model...")
fig, ax = plt.subplots(figsize=(10, 4))

for eth in ethnicities:
    mae_values = []
    for model in models:
        val = mae_eth[(mae_eth['GPT_Model'] == model) &
                      (mae_eth['Ethnicity'] == eth)]['MAE'].mean()
        mae_values.append(val)
    ax.plot(model_labels, mae_values,
            marker='o', linewidth=2.5, label=eth, markersize=8, color=colors[eth])

ax.set_xlabel('Model', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
ax.set_title('MAE by Ethnicity Across Models', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 2: 5 Subgraphs (Ethnicity) with Male/Female - 5 columns, 1 row
# ============================================================================

print("Generating GRAPH 2: Ethnicity × Gender (5 subgraphs in 1 row)...")

# Calculate global y-axis max for consistent scaling
all_mae_values = []
for eth in ethnicities:
    for gender in ['Male', 'Female']:
        for model in models:
            val = mae_ethgen[(mae_ethgen['GPT_Model'] == model) &
                            (mae_ethgen['Ethnicity'] == eth) &
                            (mae_ethgen['Gender'] == gender)]['MAE'].mean()
            if not np.isnan(val):
                all_mae_values.append(val)

y_max = max(all_mae_values) * 1.1  # Add 10% padding
y_min = min(all_mae_values) * 0.9  # Subtract 10% padding

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for idx, eth in enumerate(ethnicities):
    ax = axes[idx]

    for gender in ['Male', 'Female']:
        mae_values = []
        for model in models:
            val = mae_ethgen[(mae_ethgen['GPT_Model'] == model) &
                            (mae_ethgen['Ethnicity'] == eth) &
                            (mae_ethgen['Gender'] == gender)]['MAE'].mean()
            mae_values.append(val)

        # Use same color (ethnicity), different line styles for gender
        ax.plot(model_labels, mae_values,
                linestyle=gender_linestyles[gender], linewidth=2.5,
                label=gender, marker='o', markersize=6, color=colors[eth])

    ax.set_title(eth, fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=10)
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylim(y_min, y_max)  # Consistent y-axis across all subplots
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 3: 10 Subgraphs (Ethnicity x Gender) with Age Groups
# ============================================================================

print("Generating GRAPH 3: Ethnicity × Gender × Age (10 subgraphs)...")

ages = ['15-29', '30-64', '65+']

# Calculate global y-axis max for consistent scaling
all_mae_values = []
for eth in ethnicities:
    for gender in ['Male', 'Female']:
        for age in ages:
            for model in models:
                val = mae_ethgenage[(mae_ethgenage['GPT_Model'] == model) &
                                   (mae_ethgenage['Ethnicity'] == eth) &
                                   (mae_ethgenage['Gender'] == gender) &
                                   (mae_ethgenage['Age'] == age)]['MAE'].mean()
                if not np.isnan(val):
                    all_mae_values.append(val)

y_max = max(all_mae_values) * 1.1  # Add 10% padding
y_min = min(all_mae_values) * 0.9  # Subtract 10% padding

fig, axes = plt.subplots(2, 5, figsize=(18, 8))

plot_idx = 0
for eth in ethnicities:
    for gender in ['Male', 'Female']:
        ax = axes[plot_idx // 5, plot_idx % 5]

        for age in ages:
            mae_values = []
            for model in models:
                val = mae_ethgenage[(mae_ethgenage['GPT_Model'] == model) &
                                   (mae_ethgenage['Ethnicity'] == eth) &
                                   (mae_ethgenage['Gender'] == gender) &
                                   (mae_ethgenage['Age'] == age)]['MAE'].mean()
                mae_values.append(val)

            # Use same color (ethnicity), different line styles for age groups
            ax.plot(model_labels, mae_values,
                    linestyle=age_linestyles[age], linewidth=2.5,
                    label=age, marker='o', markersize=5, color=colors[eth])

        ax.set_title(f'{eth} - {gender}', fontsize=10, fontweight='bold')
        ax.set_ylabel('MAE', fontsize=9)
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylim(y_min, y_max)  # Consistent y-axis across all subplots
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

        plot_idx += 1

plt.tight_layout()
plt.show()

print("\nAll graphs generated successfully!")
print(f"\nColor scheme for ethnicities:")
for eth, color in colors.items():
    print(f"  {eth}: {color}")
print(f"\nGender line styles (Graph 2): Male=solid (—), Female=dashed (- -)")
print(f"Age line styles (Graph 3): 15-29=solid (—), 30-64=dashed (- -), 65+=dotted (···)")
