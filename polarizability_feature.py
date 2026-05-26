import pandas as pd
import os
import warnings
import time
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

warnings.filterwarnings('ignore', category=UserWarning)

def main():
    folder_path = r'your_path_here'
    input_file = os.path.join(folder_path, 'formula_pretty.csv')
    output_filename = 'Final_Polarizability_Data.csv'
    output_file = os.path.join(folder_path, output_filename)

    if not os.path.exists(input_file):
        print(f"Error!")
        return

    try:
        print(f"Reading {input_file}")
        df = pd.read_csv(input_file)
        ep = ElementProperty(
            data_source="magpie",
            features=["Polarizability"], 
            stats=["minimum", "maximum", "range", "mean", "avg_dev", "mode"]
        )

        print("Calculating...")
        df['comp_obj'] = df['formula_pretty'].apply(lambda x: Composition(str(x).strip()))

        df_features = ep.featurize_dataframe(df, col_id='comp_obj')

        df_final = df_features.drop(columns=['comp_obj'])
        
        df_final.to_csv(output_file, index=False)

        time.sleep(1)

        if os.path.exists(output_file):
            print("-" * 50)
            print(f"file_path: {output_file}")
            print(f"{os.path.getsize(output_file) / 1024:.2f} KB")
            print("-" * 50)
            os.startfile(folder_path)
        else:
            print("[Warning] The save command was executed, but the file was not found by the system. Please check whether antivirus software blocked Python from writing the file.")

if __name__ == "__main__":
    main()
