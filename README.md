# FRAILPOL_classification
FRAILPOL Database for Prevalence and Determinants of Frailty and Pre-frailty in Elderly People with Quantifying Functional Mobility

# Frailty Sensor Data Processing
This repository provides a modular Python pipeline to process raw IMU data from wearable sensors, extract gait parameters, and perform frailty classification (binary and three-class). It is designed for use in research projects involving elderly mobility and health assessment. The dataset consists of CSV files collected from multiple participants wearing IMU sensors at body position i.e., both wrists, ankles and back of sacrum.

# Database
[write the link of database here]

# Description
Frailty classification framework used the stride-based spatio-temporal features, extracted from the FRAILPOL database. This strategy allowed us to evaluate the effectiveness of the retrieved stride-segmentation-based spatio-temporal gait parameters. Five supervised ML algorithms, including Linear Support Vector Classifier (Linear SVC), Random Forest (RF), XGBoost, AdaBoost, and Multilayer Perceptron (MLP), were evaluated in both binary (robust vs. frail) and multi-class (robust, pre-frail, or frail) classification approaches.

🔧 Key Features
1.📥 CSV Reader: Imports and validates raw IMU sensor data (5 positions).
2.🔍 Data Checker: Detects missing sensor files or NaNs per participant.
3.🧮 Gait Parameter Extraction: Computes features using gaitmap library.
4.🧠 Frailty Classification:
    Binary (e.g., frail vs. non-frail)
    Three-class (e.g., robust, pre-frail, frail)
5.💾 Modular Structure: Each processing step is handled by separate script files for clarity and reuse.

📝 Notes
Before running classification models, add a Targets column (labels) to the gait parameters CSV file (if saved).
Example label files:
    gait_parameters_2_class.csv
    gait_parameters_3_class.csv

## 📁 Project Structure
│
├── data/
│ └── IMU_signals/ # all the raw IMU signal files
│ └── gait_parameters_2_class.csv # Gait parameters for binary (robust/frail) frailty classification
│ └── gait_parameters_3_class.csv # Gait parameters for three-class (robust/pre-frail/frail) frailty classification
│
├── scripts/
│ └── read_csv_files.py # Reads and preprocesses CSV files
│ └── gait_parameters.py # Compute gait parameters (i.e., spatio-temporal) using stride segmentation
│ └── binary_classification.py # for two class frailty classification (robust/frail)
│ └── three_class_classification.py # for three class frailty classification (robust/pre-frail/frail)
│ └── universal_dtw_template.pkl # Customized stride DTW template for stride segmentation
│
├── main.py # Main script to run the entire pipeline
└── README.md # Project description

