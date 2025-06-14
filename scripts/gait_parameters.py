"""
Read the 'df' dataframe and caluclate the gait parameters
The gait parameters were calculated using Gaitmap library.
Read the Gaitmap library documentation for installation and usage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepare_frailpol_data_for_gaitmap(df):
    """
    Load the raw IMU signals dataframe and convert it according to Gaitmap.
    Parameters: 
    -----------
        df: Input the dataframe with all columns.
    return: participants
        returns the original signals data as a dictionary 
        (i.e., left_sensor: DataFrame, left_sensor: DataFrame, right_sensor: DataFrame,)
    """
    # Take only right and left ankle sensors data
    sensor_mapping = {2: "right_sensor", 4: "left_sensor"}
    # change the column names according to Gaitmap conventions
    required_cols_renamed = {
        "Acc_X": "acc_x", "Acc_Y": "acc_y", "Acc_Z": "acc_z",
        "Gyr_X": "gyr_x", "Gyr_Y": "gyr_y", "Gyr_Z": "gyr_z",
    }
    # group participants by their id and sensor positions
    participants = {}
    for (pid, sensor_pos), group in df.groupby(["participant_id", "sensor_position"]):
        sensor_key = sensor_mapping.get(sensor_pos)
        if sensor_key is None:
            continue

        imu_data = group.rename(columns=required_cols_renamed)
        imu_data = imu_data[list(required_cols_renamed.values())].reset_index(drop=True)

        if pid not in participants:
            participants[pid] = {}
        participants[pid][sensor_key] = imu_data
    return participants

def transform_to_fsf(data):
    """
    Load the raw IMU signals (acc_x, acc_y, acc_z) and (gyr_x, gyr_y, gyr_z) 
    and transform them inro Foot Sensor Frame (FSF).
    Parameters:
    -----------
        data: dict
            Input the IMU signals
        return: transformed : dict
            Returns the FSF transformed data
    """
    transformed = {}
    for sensor_name, df in data.items():
        df_copy = df.copy()

        # 1. Align right sensor to match left sensor (mirror fix)
        if sensor_name == "right_sensor":
            df_copy["acc_y"] = -df_copy["acc_y"]
            df_copy["acc_z"] = -df_copy["acc_z"]
            df_copy["gyr_y"] = -df_copy["gyr_y"]
            df_copy["gyr_z"] = -df_copy["gyr_z"]

        # 2. Rotate to match Foot Sensor Frame (FSF)
        df_fsf = df_copy.copy()
        df_fsf["acc_x"], df_fsf["acc_y"], df_fsf["acc_z"] = (
            df_copy["acc_y"],  # Forward = X
            df_copy["acc_z"],  # Left = Y
            df_copy["acc_x"],  # Up = Z
        )
        df_fsf["gyr_x"], df_fsf["gyr_y"], df_fsf["gyr_z"] = (
            df_copy["gyr_y"],
            df_copy["gyr_z"],
            df_copy["gyr_x"],
        )

        transformed[sensor_name] = df_fsf[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]]

    return transformed

def plot_original_vs_transformed(data, transformed_data, pid, sensor_name, save_path, dpi, transform):
    """
    Plots original vs transformed (FSF) accelerometer and gyroscope signals.
    
    Parameters:
    -----------
    transformed_data : dict
        Data with x/y/z axes (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
    transformed_data : dict
        Data with x/y/z axes (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
    pid : str
        Participant ID
    sensor_name : str
        "left_sensor" / "right_sensor"
    save_path : str (optional)
        Path to save the figure
    dpi : int
        Figure resolution (330 dpi)
    transform : str
        Transformation description for title (Original/FSF)
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Participant: {pid}-{sensor_name}: Original vs Transformed Signals {transform}", fontsize=16)

    axes = ['x', 'y', 'z']
    
    for i, axis in enumerate(axes):
        # Accelerometer plots
        axs[0, i].plot(data[sensor_name][f'acc_{axis}'], label='Original', alpha=0.7, color=['r','g','b'][i])
        axs[0, i].plot(transformed_data[sensor_name][f'acc_{axis}'], label='Transformed', linestyle='--', alpha=0.7, color='orange')
        axs[0, i].set_title(f'Accelerometer - {axis.upper()}')
        axs[0, i].legend()
        axs[0, i].grid(True)

        # Gyroscope plots
        axs[1, i].plot(data[sensor_name][f'gyr_{axis}'], label='Original', alpha=0.7, color=['#FF00FF','#50C878','#00FFFF'][i])
        axs[1, i].plot(transformed_data[sensor_name][f'gyr_{axis}'], label='Transformed', linestyle='--', alpha=0.7, color='orange')
        axs[1, i].set_title(f'Gyroscope - {axis.upper()}')
        axs[1, i].legend()
        axs[1,i].set_xlabel("Time (Samples)")
        axs[1,i].grid(True, linestyle=':')

    plt.tight_layout()
    '''
    if save_path:
        fig.savefig(save_path, dpi=dpi, format='tiff', bbox_inches='tight')'''
    plt.show()
    
def plot_fsf_vs_fbf(transformed_data, bf_data, pid, sensor_name, save_path, dpi, transform):
    """
    Plots original (x/y/z) vs transformed (FBF)(pa/ml/si) accelerometer and 
    gyroscope signals.
    
    Parameters:
    -----------
    transformed_data : dict
        Data with x/y/z axes (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
    bf_data : dict
        Data with pa/ml/si axes (acc_pa, acc_ml, acc_si, gyr_pa, gyr_ml, gyr_si)
    pid : str
        Participant ID
    sensor_name : str
        "left_sensor" or "right_sensor"
    save_path : str (optional)
        Path to save the figure
    dpi : int
        Figure resolution (330 dpi)
    transform : str
        Transformation description for title(FSF/FBF)
    """
    # Mapping between x/y/z and pa/ml/si
    axis_map = {
        'x': 'ml',  # Mediolateral
        'y': 'pa',  # Posterior-Anterior
        'z': 'si'   # Superior-Inferior
    }
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Participant {pid}-{sensor_name}: {transform}", fontsize=16, y=1.05)
    
    for i, axis in enumerate(['x', 'y', 'z']):
        # Accelerometer plots
        axs[0, i].plot(
            bf_data[sensor_name][f'acc_{axis_map[axis]}'], 
            label='Transformed (FBF)',
            linestyle='--', 
            alpha=0.7, 
            color='orange'
        )
        axs[0, i].plot(
            transformed_data[sensor_name][f'acc_{axis}'], 
            label='FSF', 
            alpha=0.7,
            color=['r','g','b'][i]
        )
        axs[0, i].set_title(f'Accelerometer - {axis.upper()}')
        axs[0, i].legend()
        axs[0, i].grid(True)
        axs[0, i].set_ylabel("Acceleration [m/s²]")
        
        # Gyroscope plots
        axs[1, i].plot(
            bf_data[sensor_name][f'gyr_{axis_map[axis]}'], 
            label='Transformed (FBF)',
            linestyle='--', 
            alpha=0.7, 
            color='orange'
        )
        axs[1, i].plot(
            transformed_data[sensor_name][f'gyr_{axis}'], 
            label='FSF', 
            alpha=0.7, 
            color=['#FF00FF','#50C878','#00FFFF'][i]
        )
        axs[1, i].set_title(f'Gyroscope - {axis.upper()}')
        axs[1, i].legend()
        axs[1, i].set_xlabel("Time [samples]")
        axs[1, i].set_ylabel("Angular Velocity [deg/s]")
        axs[1, i].grid(True, linestyle=':')
    
    plt.tight_layout()
    '''
    if save_path:
        fig.savefig(save_path, dpi=dpi, format='tiff', bbox_inches='tight')'''
    plt.show()

def plot_original_imu_signals(prepared_data, pid, sensor, save_path, dpi):
    """
    Plots original accelerometer and gyroscope signals from prepared_data.
    
    Parameters:
    -----------
    prepared_data : dict
        Output from prepare_frailpol_data_for_gaitmap() 
        Structure: {pid: {"left_sensor": DataFrame, "right_sensor": DataFrame}}
    pid : str
        Participant ID
    sensor : str
        "left_sensor" / "right_sensor"
    save_path : str (optional)
        Path to save TIFF file (e.g., "output.tiff")
    dpi : int
        Figure resolution (330 dpi)
    """
    try:
        sensor_data = prepared_data[pid][sensor]
    except KeyError:
        print(f"Error: No data found for participant {pid} with sensor {sensor}")
        return

    # Create figure with 2 rows (acc, gyr) and 3 columns (x,y,z)
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Participant {pid}-{sensor}: Raw IMU Signals", fontsize=16)
    
    # Plot accelerometer data (top row)
    for i, axis in enumerate(['x', 'y', 'z']):
        axs[0,i].plot(sensor_data[f'acc_{axis}'], color=['r','g','b'][i])
        axs[0,i].set_title(f"Accelerometer {axis.upper()}")
        axs[0,i].set_ylabel("Acceleration (m/s²)")
        axs[0,i].grid(True, linestyle=':')
    
    # Plot gyroscope data (bottom row)
    for i, axis in enumerate(['x', 'y', 'z']):
        axs[1,i].plot(sensor_data[f'gyr_{axis}'], color=['#FF00FF','#50C878','#00FFFF'][i])
        axs[1,i].set_title(f"Gyroscope {axis.upper()}")
        axs[1,i].set_ylabel("Angular velocity (deg/s)")
        axs[1,i].set_xlabel("Time (Samples)")
        axs[1,i].grid(True, linestyle=':')
    
    plt.tight_layout()
    '''
    if save_path:
        fig.savefig(save_path, dpi=dpi, format='tiff', bbox_inches='tight')'''
    plt.show()

from gaitmap.stride_segmentation import DtwTemplate
from gaitmap_mad.stride_segmentation.dtw._dtw_templates.templates import FixedScaler

def stride_template(bf_data):
    """
    Computer customized template for the stride segmentation.
    
    Parameters:
    -----------
    bf_data : dict
        Data with pa/ml/si axes (acc_pa, acc_ml, acc_si, gyr_pa, gyr_ml, gyr_si)
    return: right_template
        Returns the left or right sensor template for computing BarthDtw
    """
    # Define stride window (gyr_ml signal was used as template)
    #left_stride = bf_data["left_sensor"].loc[0:250, ["gyr_ml", "gyr_pa", "gyr_si"]]
    right_stride = bf_data["right_sensor"].loc[0:230, ["gyr_ml"]]
    
    # Optional scaler
    scaler = FixedScaler(offset=0, scale=100)
    '''
    # Create templates for each sensor
    left_template = DtwTemplate(
        data=left_stride,
        sampling_rate_hz=100,
        scaling=scaler,
        use_cols=["gyr_ml", "gyr_pa", "gyr_si"]
    )'''
    
    right_template = DtwTemplate(
        data=right_stride,
        sampling_rate_hz=100,
        scaling=scaler,
        use_cols=["gyr_ml"]
    )
    return right_template


def plot_stride_template(template, title="Stride Template (gyr_ml)"):
    """
    Plots the stride template's gyr_ml signal.
    
    Parameters:
    -----------
    template : DtwTemplate
        The DTW template object containing the stride data.
    title : str (optional)
        Title of the plot. Default: "Stride Template (gyr_ml)"
    """
    # Extract the template data
    template_data = template.data["gyr_ml"]
    
    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.plot(template_data, label="gyr_ml", linewidth=2)
    
    # Add labels and title
    plt.title(title, fontsize=14)
    plt.xlabel("Time [samples]", fontsize=12)
    plt.ylabel("Angular Velocity [deg/s]", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    '''
    # Save the plot as TIFF (330 DPI)
    plt.savefig("stride_template.tiff", dpi=330, format="tiff")'''
    plt.show()

def plot_stride_segmentation(dtw, strides, pid, sensor, save_path, dpi=330):
    """
    Plots the segmented strides on the top row, cost function of the DTW is 
    plotted in second row, and the accumulated cost matrix in the third row.
    
    Parameters:
    -----------
    dtw : computed BarthDtw
        The BarthDtw computed data.
    strides : list
        Stride list
    pid : str
        Participant ID (specify the pid to plot)
    sensor : str
        "left_sensor" / "right_sensor"
    save_path : str (optional)
        Path to save TIFF file (e.g., "output.tiff")
    dpi : int
        Figure resolution (330 dpi)
    """

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
    fig.suptitle(f"Participant {pid}-{sensor}: Stride Segmentation", fontsize=16)
    dtw.data[sensor]["gyr_ml"].reset_index(drop=True).plot(ax=axs[0])
    axs[0].set_ylabel("gyr_ml [deg/s]")
    axs[1].plot(dtw.cost_function_[sensor])
    axs[1].set_ylabel("dtw cost [a.u.]")
    axs[1].axhline(dtw.max_cost, color="k", linestyle="--")
    axs[2].imshow(dtw.acc_cost_mat_[sensor], aspect="auto")
    axs[2].set_ylabel("template position [#]")
    for p in dtw.paths_[sensor]:
        axs[2].plot(p.T[1], p.T[0])
    for s in dtw.matches_start_end_original_[sensor]:
        axs[1].axvspan(*s, alpha=0.3, color="g")
    for _, s in dtw.stride_list_[sensor][["start", "end"]].iterrows():
        axs[0].axvspan(*s, alpha=0.3, color="g")
    
    axs[0].set_xlim(0, 800)
    axs[0].set_xlabel("time [#]")
    fig.tight_layout()
    '''
    # Save the figure as TIFF with 330 DPI
    if save_path:
        fig.savefig(save_path, dpi=dpi, format='tiff', bbox_inches='tight')'''
    plt.show()

def main_gait_parameters(prepared_data, all_rows):
    """
    A function to drive all sub-functions
    
    Parameters:
    -----------
    prepared_data : dict
        Original IMU accelerometer and groscope signals in Gaitmap format
    all_rows : list
        Computer and store the gait parameters of each parameter in a list (declared in main.py)
    return: all_rows
        Compute gait parameters and merged into all_rows
    """
    ##################### Plot original IMU signals (x/y/z) ###################
    # Plot original IMU signals for participant 9, right sensor
    plot_original_imu_signals(
        prepared_data=prepared_data,
        pid=list(prepared_data.keys())[8],  # Or specific participant ID
        sensor="right_sensor",
        save_path="participant_9_original_right_sensor.tiff",
        dpi=300
    )
    # Plot original IMU signals for participant 9, left sensor
    plot_original_imu_signals(
        prepared_data=prepared_data,
        pid=list(prepared_data.keys())[8],  # Or specific participant ID
        sensor="left_sensor",
        save_path="participant_9_original_left_sensor.tiff",
        dpi=300
    )
    
    for idx, (pid, sensor_data) in enumerate(list(prepared_data.items())):
        
        ############ Transform original signals to FSF (x/y/z) ################
        # transform the original IMU signals into FSF 
        transformed_data = transform_to_fsf(sensor_data)
        
        '''
        # # Optional: # Plot original VS transformed (FSF) signals for right sensor
        plot_original_vs_transformed(sensor_data, transformed_data, pid,
                                     sensor_name="right_sensor",
                                     save_path="participant_9_original_Vs_transformed_FSF_right_sensor.tiff",
                                     dpi=300, transform="FSF"
                                     )
        '''
        '''
        # Optional: # Plot original VS transformed (FSF) signals for left sensor
        plot_original_vs_transformed(sensor_data, transformed_data, pid,
                                     sensor_name="left_sensor",
                                     save_path="participant_9_original_Vs_transformed_FSF_left_sensor.tiff",
                                     dpi=300, transform="FSF"
                                     )
        '''
        ############ Transform FSF signals to FBF (pa/ml/si axes) #############
        from gaitmap.utils.coordinate_conversion import convert_to_fbf
    
        # Convert fsf to Foot-Based Frame
        bf_data = convert_to_fbf(
                transformed_data,
                left_like="left_sensor",
                right_like="right_sensor"
        )
        '''
        # Optional: # Plot FSF Vs FBF right sensor
        plot_fsf_vs_fbf(transformed_data, bf_data, pid,
                                     sensor_name="right_sensor",
                                     save_path="participant_9_FSF_Vs_FBF_right_sensor.tiff",
                                     dpi=300, transform="FSF VS FBF transformed signals"
                                     )
        
        # Optional: # Plot FSF Vs FBF left sensor
        plot_fsf_vs_fbf(transformed_data, bf_data, pid,
                                     sensor_name="left_sensor",
                                     save_path="participant_9_FSF_Vs_FBF_left_sensor.tiff",
                                     dpi=300, transform="FSF VS FBF transformed signals"
                                     )'''
        
        ##################### Load and plot Stride template ###################
        import pickle 
        # Load cunstomized stride template
        with open(r"scripts\universal_dtw_template.pkl", "rb") as f:
            right_template = pickle.load(f)
        
        '''
        # Optional: # Plot the template for gyr_ml
        plot_stride_template(right_template)
        '''
        
        ######################### Stride Segmentation #########################
        
        from gaitmap.stride_segmentation import BarthDtw
    
        # Use parameters optimized for ankle placement
        dtw = BarthDtw(
        template=right_template,
    
        )
        dtw.segment(bf_data, sampling_rate_hz=100)
        
        # Get strides
        strides = dtw.stride_list_
       
        '''
        # Optional: # Plot the strides segmentations for right sensor
        plot_stride_segmentation(dtw, strides, pid, sensor="right_sensor",
                                 save_path="participant_9_left_sensor_stride_segmentation.tiff",
                                 dpi=330)
        # Optional: # Plot the strides segmentations for left sensor
        plot_stride_segmentation(dtw, strides, pid, sensor="left_sensor",
                                 save_path="participant_9_right_sensor_stride_segmentation.tiff",
                                 dpi=330)
        '''
        
        ######################### Gait Event detetction #######################
        
        from gaitmap.event_detection import HerzerEventDetection
        # Event detection
        
        hd = HerzerEventDetection(
        min_vel_search_win_size_ms=130,
        mid_swing_peak_prominence=0.03,
        mid_swing_n_considered_peaks=2,
        )
    
        ed = hd.detect(data=bf_data, stride_list=strides, sampling_rate_hz=100.0)
        min_vel_events_left = ed.min_vel_event_list_["left_sensor"]
        #print(f"Gait events for {len(min_vel_events_left)} min_vel strides were detected.")
        min_vel_events_left.head()
        
        from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation
        from gaitmap.trajectory_reconstruction import StrideLevelTrajectory
        
        # Stride level Trajectory
        trajectory = StrideLevelTrajectory()
        trajectory = trajectory.estimate(
            data=transformed_data, stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=100.0
            )
        
        ######################### Temporal parameters #########################
        # Temporal parameters calculations
        tp = TemporalParameterCalculation()
        temporal_paras = tp.calculate(
            stride_event_list=ed.min_vel_event_list_,
            sampling_rate_hz=100.0
            ) 
        #print("Temporal Parameters....")
        #print(temporal_paras.parameters_pretty_)
        
        ######################### Spatial parameters ##########################
        #Spatial parameters calculations
        sp = SpatialParameterCalculation()
        spatial_paras = sp.calculate(
            stride_event_list=ed.min_vel_event_list_,
            positions=trajectory.position_,
            orientations=trajectory.orientation_,
            sampling_rate_hz=100.0,
            )
        
        ############## Participant-wise gait parameters aggregation ###########
        participant_row = {"participant_id": pid}
        for sensor in ["left_sensor", "right_sensor"]:
            try:
                stride_events = ed.min_vel_event_list_.get(sensor)
                if stride_events is None or len(stride_events) == 0:
                    #print(f"[Info] No gait events detected for {sensor} of participant {pid}.")
                    continue

                stride_count = len(strides[sensor])
                gait_event_count = len(stride_events)

                temporal = temporal_paras.parameters_pretty_.get(sensor, pd.DataFrame())
                spatial = spatial_paras.parameters_pretty_.get(sensor, pd.DataFrame())
                
                temporal_dicts = temporal.to_dict(orient="records") if isinstance(temporal, pd.DataFrame) else []
                temporal_str = temporal_dicts[0] if temporal_dicts else {}
                
                # Safely convert spatial dataframe to dict
                spatial_dicts = spatial.to_dict(orient="records") if isinstance(spatial, pd.DataFrame) else []
                spatial_str = spatial_dicts[0] if spatial_dicts else {}
                
                side = sensor.split("_")[0]
                participant_row[f"stride_count_{side}"] = stride_count
                participant_row[f"gait_event_count_{side}"] = gait_event_count
                
                for key, val in temporal_str.items():
                    participant_row[f"temporal_{side}_{key}"] = val
                for key, val in spatial_str.items():
                    participant_row[f"spatial_{side}_{key}"] = val
                
            except Exception as e:
                print(f"[Warning] Sensor {sensor} of participant {pid} caused an error: {e}")
            
        all_rows.append(participant_row)
    return all_rows