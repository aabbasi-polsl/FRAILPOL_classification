"""
Description:
    This section of code read all the CSV files from a folder and return the data as dataframe.
"""
import os
import pandas as pd

# map the sensor mounting positions with integer value
position_map = {
    'left wrist': 0,
    'right wrist': 1,
    'right ankle': 2,
    'hips back': 3,
    'left ankle': 4
}

def load_and_check_data(data_folder):
    """
    Load the CSV files and check the missing values and missing sensor data.
    
    Parameters: 
    -----------
        file_path: Input the folder path to read all CSV files.
        return: final_df : dataframe
            Returns the data as dataframe
    """
    dataframes = []
    files_with_nans = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            parts = filename.replace('.csv', '').split('-')
            participant_id = int(parts[0])
            position = parts[1].replace('_', ' ')
            
            if position == 'sacrum back':
                position = 'hips back'
            
            sensor_position_id = position_map.get(position.lower())
            if sensor_position_id is None:
                print(f"Skipping {filename}: Unknown position '{position}'")
                continue
            
            filepath = os.path.join(data_folder, filename)
            df = pd.read_csv(filepath)

            # drop the column
            if 'SampleTimeFine' in df.columns:
                df = df.drop(columns=['SampleTimeFine'])
            # check if any missing value found in a CSV file
            if df.isnull().values.any():
                print(f"⚠️ Missing values found in file: {filename}")
                files_with_nans.append(filename)
            # add participant_id and sensor_position columns 
            df['participant_id'] = participant_id
            df['sensor_position'] = sensor_position_id

            dataframes.append(df)

    final_df = pd.concat(dataframes, ignore_index=True)

    # Summary
    print("\n✅ Data loading complete.")
    print(f"Total samples: {len(final_df)}")
    print(f"Columns: {final_df.columns.tolist()}")

    if files_with_nans:
        print("\n❌ Files containing NaN values:")
        for f in sorted(files_with_nans):
            print(f"- {f}")
    else:
        print("\n✅ No files contain missing values.")

    # Sensor completeness check
    expected_sensors = set(range(5))
    participants = final_df['participant_id'].unique()
    incomplete_participants = []

    print("\n🔍 Checking sensor completeness per participant:")
    for pid in participants:
        sensor_set = set(final_df[final_df['participant_id'] == pid]['sensor_position'].unique())
        if sensor_set != expected_sensors:
            print(f"❌ Participant {pid} missing sensors: {expected_sensors - sensor_set}")
            incomplete_participants.append(pid)

    if not incomplete_participants:
        print("✅ All participants have complete sensor data (0–4).")

    return final_df #retun the final df for further computation 

