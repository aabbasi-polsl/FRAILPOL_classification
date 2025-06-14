"""
main.py file for calling the script files.

"""
# main.py

import pandas as pd
import numpy as np
from scripts.read_cvs_files import load_and_check_data
from scripts.gait_parameters import prepare_frailpol_data_for_gaitmap
from scripts.gait_parameters import main_gait_parameters
from scripts.binary_classification import binary_classification_main
from scripts.three_class_classification import three_class_classification_main

def main():
    ####################### Load CSV files ####################################
    data_folder = r'IMU_signals'
    
    print("📥 Loading CSV files data...")
    df = load_and_check_data(data_folder)
    
    ###################### Prepare the data for Gaitmap #######################
    prepared_data = prepare_frailpol_data_for_gaitmap(df)
    all_rows = []
    ###################### compute gait parameters ##########################
    all_rows = main_gait_parameters(prepared_data, all_rows)
    
    """
    # Save the gait parameters to CSV file
    final_output_csv = "gait_parameters_result.csv"
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(final_output_csv, index=False)
    print(f"\nAll results saved to {final_output_csv}")
    """
    #################Instructions for adding the Targets column ###############
    """Note:
        When gait parameters file is saved then add the binary or 3-class targets 
        with a column name "Targets". Targets can be added from a file "participants.xlsx".
    """
    
    """Following files can be called for classification tasks:
        # gait_parameters_2_class.csv (for binary frailty classification)
        # gait_parameters_3_class.csv (for three class frailty classification)
    """
    ###########################################################################
    
    #################### Two class frailty classification #####################
    binary_file_path = 'data:/gait_parameters_2_class.csv'
    binary_classification_main(binary_file_path)
    
    #################### Three class frailty classification ###################
    # Three class frailty classification
    tri_file_path = 'data:/gait_parameters_3_class.csv'
    three_class_classification_main(tri_file_path)

if __name__ == "__main__":
    main()
