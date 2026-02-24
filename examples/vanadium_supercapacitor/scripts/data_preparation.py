import os 
import pandas as pd


def prepare_data():
    """ 
    Process data to be ready for ML models in data/processed.csv
    """

    # Script absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    # Set working directory to the project root
    os.chdir(base_dir) 

    # Input path
    input_path = "data/raw.csv"
    index_cols = "Num_Data" 
    # Output path
    output_path = "data/processed.csv"

    df = pd.read_csv(input_path, index_col = index_cols)

    df = df.drop(["DOI (Reference)"], axis=1) # Drop reference column

    # Rename columns to appropiate zbbrevations
    df.rename(columns={"Current_Density (A/g)": "j", 
                       "Specific_Capacitance (Fg-1)": "C_s",
                       "Material (M)": "Mat",
                       "M_Density(g/cm3)": "rho_m",
                       "Current_Collector (CC)": "CC",
                       "E_H_Cation_Radius" : "r_cat",
                       "E_H_Anion_Radius" : "r_an",
                       "E_pH": "pH",
                       "E_Ionic_Conductivity": "sigma_ion",
                       "E_Bare_Cation_Radius": "r0_cat",
                       "E_Bare_Anion_Radius": "r0_an",
                       "Is_Binder": "B_flag",
                       "Binder_Type": "B_type",
                       "CC_Electrical_Conductivity": "sigma_cc",
                       "CC_Thermal_Conductivity": "kappa_cc",
                       "CC_Work_Function": "phi_cc",
                       "Potential_Window": "Delta_V",
                       "Morphology_Encoded": "Morph",
                       "Synthesis_Method": "Synth",
                       "Electrode_ID": "Group_ID",
                       }, inplace = True)
    
    # Remain only rows with writen input
    df.drop(df.tail(len(df["Group_ID"]) - 185).index, inplace = True)

    # Change to appropiate data type
    df['Group_ID'] = df['Group_ID'].astype('int64')
    df['Mat'] = df['Mat'].astype('int64')
    df['B_flag'] = df['B_flag'].astype('int64')
    df['CC'] = df['CC'].astype('int64')
    df['Morph'] = df['Morph'].astype('int64')
    df['B_type'] = df['B_type'].astype('int64')
    df['Synth'] = df['Synth'].astype('int64')

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    df.to_csv(output_path, encoding="utf-8-sig")
    print(f"Processed data saved to {output_path}")

    return
    

if __name__ == "__main__":
    prepare_data()