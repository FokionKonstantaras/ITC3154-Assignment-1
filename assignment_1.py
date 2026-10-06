import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

"""TASK 1"""
"""Dataset and uploaded to OnlineGDB as a file"""

"""TASK 2"""
df = pd.read_csv("loan_approval_dataset.csv", sep=";")

# Clean dataset of leading and ending whitespaces.
df.columns = df.columns.str.strip() # Column names.
df = df.apply(lambda column: column.str.strip() if column.dtype == "object" else column) # Entries.

print(df.head(20).to_string(index=False)) # index=False since we already have load_id, no need to display row id too.

"""TASK 3"""
feature_table = pd.DataFrame({ # Manual feature table.
    "Feature": [
        "loan_id",
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ],
    
    # Manually assess in which category each feature belongs to.
    "Important": [
        0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1
    ],
    
    "In-between": [
        0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0
    ],
    
    "Not-important": [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    
    # As we do not need the reasoning for the heatmap, I included it in the report to reduce clutter in the code.
})

"""TASK 4"""
selected_features = feature_table.loc[feature_table["Not-important"] == 0, "Feature"].tolist() # Exclude any feature that is not important.
numeric_df = df[selected_features].select_dtypes(include=np.number) # Get only numeric features.

# Calculate the correlation matrix and create the heatmap.
corrmatrix = numeric_df.corr()
plt.figure(figsize=(12, 12))

# Plot the heatmap.
sns.heatmap(corrmatrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5) #I found these values online.

# PNG export because I am using OnlineGDB.
plt.title("Correlation heatmap for loans")
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

"""TASK 5"""
print("-----------------")
df["loan_status_numeric"] = df["loan_status"].map({"Approved": 1, "Rejected": 0}) # Convert loan status to numeric.

# Calculate correlations.
status_corr = df[numeric_df.columns.tolist() + ["loan_status_numeric"]].corr()
print(status_corr["loan_status_numeric"].sort_values(ascending=False))

print("-----------------")

# Education categorical feature.
education_status = pd.crosstab(df["education"], df["loan_status"], normalize="index") * 100
print(education_status)
