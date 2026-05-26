import pandas as pd
import os
import warnings
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

warnings.filterwarnings('ignore', category=UserWarning)

def main():
    folder_path = r'your_path_here'
    input_file = os.path.join(folder_path, 'formula_pretty.csv')
    output_file = os.path.join(folder_path, 'Oliynyk_Features_Output.csv')
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    try:
        # 2. Read Data
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file)
        if 'formula_pretty' not in df.columns:
            print("Error: The table must contain a column named 'formula_pretty'")
            return

        # 3. Initialize Feature Extractor
        ep = ElementProperty.from_preset(preset_name="magpie")
        print("Converting chemical formulas and extracting physicochemical features...")
        df['comp_obj'] = df['formula_pretty'].apply(lambda x: Composition(str(x).strip()))

        # 4. Perform Feature Transformation
        df_features = ep.featurize_dataframe(df, col_id='comp_obj')

        # 5. Clean and Save Results
        df_final = df_features.drop(columns=['comp_obj'])
        df_final.to_csv(output_file, index=False)

        print("-" * 30)
        print(f"Feature Extraction Succeeded!")
        print(f"Total Number of Generated Feature Columns: {len(df_final.columns)}")
        print(f"Key Features Generated：Mean AtomicRadius, Mean Electronegativity [cite: 76, 84, 178]")
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"An error occurred during program execution: {e}")

if __name__ == "__main__":
    main()
