import pandas as pd

def generate_summary_text(df: pd.DataFrame) -> str:
    lines = []

    # Shape
    lines.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Missing values
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]
    if not missing.empty:
        lines.append("Missing values detected:")
        for col, pct in missing.items():
            lines.append(f"- {col}: {pct:.2f}% missing")
    else:
        lines.append("No missing values detected.")

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        corr = numeric_cols.corr().abs().mean().sort_values(ascending=False)
        top_corr = corr.index[0]
        lines.append(f"Numeric features show strongest overall correlation around '{top_corr}'.")

    # Categorical columns
    cat_cols = df.select_dtypes(include="object")
    if not cat_cols.empty:
        lines.append(f"There are {cat_cols.shape[1]} categorical columns.")

    return "\n".join(lines)
