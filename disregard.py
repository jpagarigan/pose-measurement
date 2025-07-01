import pandas as pd

# Load the Excel file
file_path = r"C:\Users\John Alfred\Documents\sample.xlsx"
df = pd.read_excel(file_path)

# Clean each column by removing blank cells (NaN or empty strings)
for col in df.columns:
    cleaned_col = df[col].replace('', pd.NA).dropna().reset_index(drop=True)
    df[col] = pd.Series(cleaned_col)

# Save the cleaned DataFrame to a new file
output_path = r"C:\Users\John Alfred\Documents\cleaned_sample.xlsx"
df.to_excel(output_path, index=False)

print(f"Cleaned Excel file saved to: {output_path}")