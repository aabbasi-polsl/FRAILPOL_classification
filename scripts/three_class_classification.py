"""
Description:
    This section of code classify the frailty into three classes (robust, pre-frail or frail)
    MLP models include: LinearSVC, RF, AdaBoost, and MLP.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score, f1_score

# Load and check data
def load_and_check_data(file_path):
    """
    Load the gait parameters files (as three class Targets).
    
    Args:
        file_path: Input the CSV file path to read.
    """
    df = pd.read_csv(file_path)
    print("Data shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    return df

def plot_combined_class_distribution(df):
    """
    Plot the class distribution for visualization.
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
    """
    total_participants = df['participant_id'].nunique()
    class_counts = df['Targets'].value_counts().sort_index()
    class_percentages = df['Targets'].value_counts(normalize=True).sort_index() * 100

    class_labels = ['Robust','Pre-frail', 'Frail']
    colors = sns.color_palette('coolwarm', 3)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(class_labels, class_counts.values, color=colors)

    # Annotate bars with counts and percentages
    for i, bar in enumerate(bars):
        count = class_counts.iloc[i]
        percent = class_percentages.iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count} ({percent:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    ax.set_title("Frailty Class Distribution (3-Class)", fontsize=16, y=1.05)
    ax.set_xlabel("Frailty Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_ylim(0, max(class_counts.values) + 5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Subtle participant info in the corner
    ax.text(0.99, 1.0, f'Total participants: {total_participants}',
            transform=ax.transAxes, ha='right', fontsize=10, style='italic', alpha=0.8)

    sns.despine()
    plt.tight_layout()
    # Save the plot as TIFF (330 DPI)
    #plt.savefig("3_Class_frailty_class_distribution.tiff", dpi=330, format="tiff")
    plt.show()

def print_summary_statistics(df):
    """
    Print the gait parameters statistics for better understading. 
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
    """
    total_participants = df['participant_id'].nunique()
    print(f"Total Participants: {total_participants}")

    class_counts = df['Targets'].value_counts().sort_index()
    class_percentages = df['Targets'].value_counts(normalize=True).sort_index() * 100

    summary_df = pd.DataFrame({
        'Class': ['Robust','Pre-frail', 'Frail'],
        'Count': class_counts.values,
        'Percentage (%)': class_percentages.values.round(2)
    })

    print("\nClass Distribution:")
    print(summary_df.to_string(index=False))

def plot_stride_time_boxplots(df):
    """
    Box plot to visualize the deviation of stride time feature from it's mean.
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    fig.suptitle("Stride Time by Frailty Class (3-Class)", fontsize=16, fontweight='bold')

    # Left Ankle IMU
    sns.boxplot(ax=axs[0], data=df, x='Targets', y='temporal_left_stride time [s]', palette='coolwarm')
    axs[0].set_title("Left Ankle IMU", fontsize=14)
    axs[0].set_xlabel("Frailty Class")
    axs[0].set_ylabel("Stride Time (s)")
    axs[0].set_xticks([0, 1, 2])
    axs[0].set_xticklabels(['Robust', 'Pre-frail', 'Frail'])
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # Right Ankle IMU
    sns.boxplot(ax=axs[1], data=df, x='Targets', y='temporal_right_stride time [s]', palette='coolwarm')
    axs[1].set_title("Right Ankle IMU", fontsize=14)
    axs[1].set_xlabel("Frailty Class")
    axs[1].set_ylabel("Stride Time (s)")
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticklabels(['Robust','Pre-frail', 'Frail'])
    axs[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.savefig("3_class_Stride_time_box_plot_left_right_sensors.tiff", format='tiff', dpi=330, bbox_inches='tight')
    plt.show()
    
def plot_no_stride_boxplots(df):
    """
    Box plot to visualize the deviation of stride count of each participant from it's mean.
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    fig.suptitle("Stride Count by Frailty Class (3-class)", fontsize=16, fontweight='bold')

    # Left Ankle IMU
    sns.boxplot(ax=axs[0], data=df, x='Targets', y='stride_count_left', palette='coolwarm')
    axs[0].set_title("Left Ankle IMU", fontsize=14)
    axs[0].set_xlabel("Frailty Class")
    axs[0].set_ylabel("Stride Count")
    axs[0].set_xticks([0, 1, 2])
    axs[0].set_xticklabels(['Robust','Pre-frail', 'Frail'])
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # Right Ankle IMU
    sns.boxplot(ax=axs[1], data=df, x='Targets', y='stride_count_right', palette='coolwarm')
    axs[1].set_title("Right Ankle IMU", fontsize=14)
    axs[1].set_xlabel("Frailty Class")
    axs[1].set_ylabel("Stride Count")
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticklabels(['Robust','Pre-frail', 'Frail'])
    axs[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.savefig("Binary_Stride_count_box_plot_left_right_sensors.tiff", format='tiff', dpi=330, bbox_inches='tight')
    plt.show()

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

def classify_frailty_svm_cv(df, target_column='Targets', n_splits=5):
    """
    Simple LinearSVC classifier for frailty (3-class).
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
        target_column (str): Name of the target column (default='Targets').
        n_splits (int): Number of CV folds (default=5).
    """
    
    X = df.drop(columns=['participant_id','Targets'])
    y = df[target_column]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    
    cm_list = [] 
    report_list = [] 
    acc_scores = []
    f1_scores = []
    target_names = ['Robust','Pre-frail', 'Frail']

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = LinearSVC(
            C=0.003,      
            multi_class   ='ovr',           # Controls regularization; higher = less regularization
            loss='squared_hinge', 
            class_weight='balanced',  # Useful if class distribution is uneven
            max_iter=1000,            # Ensures convergence
            random_state=42
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        acc_scores.append(acc)
        f1_scores.append(f1)

        print(f"\nFold {fold}")
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_list.append(report_df)
        
        print(report_df)


        cm = confusion_matrix(y_val, y_pred)
        cm_list.append(cm)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"SVC-Confusion Matrix - Fold {fold}", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        #plt.savefig(f"CM_SVC_{fold}.tiff", dpi=330, format="tiff")
        plt.show()
    
    
    # Combine all reports into a single DataFrame and average
    avg_report_df = pd.concat(report_list).groupby(level=0).mean()
    
    # Round and print neatly
    print("\n=== Averaged Classification Report (5-Fold CV) ===")
    print(avg_report_df.round(2))
    
    # Average the 5 confusion matrices
    
    # Compute sum of confusion matrices
    cm_sum = np.sum(cm_list, axis=0)
    
    # Normalize (row-wise) to get average confusion matrix
    cm_avg = cm_sum.astype('float') / cm_sum.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("LinearSVC - (5-Fold CV) Average Confusion Matrix", fontsize=15)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    #plt.savefig("SVC_CM_Average.tiff", dpi=330, format="tiff")
    plt.show()
    
    print("\n=== Final Summary ===")
    print(f"Average Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Average Macro F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        
from sklearn.ensemble import RandomForestClassifier

def classify_frailty_rf(df, target_column='Targets', n_splits=5):
    """
    Simple RF classifier for frailty (3-class).
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
        target_column (str): Name of the target column (default='Targets').
        n_splits (int): Number of CV folds (default=5).
    """
    X = df.drop(columns=['participant_id','Targets'])
    y = df[target_column]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    
    cm_list = [] 
    report_list = [] 
    acc_scores = []
    f1_scores = []
    target_names = ['Robust','Pre-frail','Frail']

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        

        clf = RandomForestClassifier(
            n_estimators=150,            # More trees for stability
            max_depth=10,                # Limits overfitting; tune as needed
            min_samples_split=3,         # Prevents splits on very small subsets
            min_samples_leaf=3,          # Ensures each leaf has enough samples
            class_weight='balanced',     # Key for handling class imbalance
            random_state=42,
            n_jobs=-1                    # Parallel training
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        acc_scores.append(acc)
        f1_scores.append(f1)

        print(f"\nFold {fold}")
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_list.append(report_df)
        
        print(report_df)


        cm = confusion_matrix(y_val, y_pred)
        cm_list.append(cm)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"RF-Confusion Matrix - Fold {fold}", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        #plt.savefig(f"CM_RF_{fold}.tiff", dpi=330, format="tiff")
        plt.show()
    
    # Combine all reports into a single DataFrame and average
    avg_report_df = pd.concat(report_list).groupby(level=0).mean()
    
    # Round and print neatly
    print("\n=== Averaged Classification Report (5-Fold CV) ===")
    print(avg_report_df.round(2))
    
    # Average the 5 confusion matrices
    
    # Compute sum of confusion matrices
    cm_sum = np.sum(cm_list, axis=0)
    
    # Normalize (row-wise) to get average confusion matrix
    cm_avg = cm_sum.astype('float') / cm_sum.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("RF - (5-Fold CV) Average Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    #plt.savefig("RF_CM_Average.tiff", dpi=330, format="tiff")
    plt.show()
    
    print("\n=== Final Summary ===")
    print(f"Average Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Average Macro F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


from xgboost import XGBClassifier

def classify_frailty_xgb(df,target_column='Targets', n_splits=5):
    """
    XGBoost classifier for frailty (3-class).
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
        target_column (str): Name of the target column (default='Targets').
        n_splits (int): Number of CV folds (default=5).
    """
    X = df.drop(columns=['participant_id','Targets'])
    y = df[target_column]
    
    X.columns = X.columns.astype(str).str.replace(r'[\[\]<>]', '', regex=True)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    
    cm_list = [] 
    report_list = [] 
    acc_scores = []
    f1_scores = []
    target_names = ['Robust','Pre-frail','Frail']

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            learning_rate=0.099,            # Lower LR for stability
            max_depth=2,                   # Prevent overfitting
            n_estimators=100,              # Sufficient estimators
            subsample=0.2,                 # Adds randomness
            colsample_bytree=0.5,          # Random feature sampling
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        acc_scores.append(acc)
        f1_scores.append(f1)

        print(f"\nFold {fold}")
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_list.append(report_df)
        
        print(report_df)


        cm = confusion_matrix(y_val, y_pred)
        cm_list.append(cm)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"XGBoost-Confusion Matrix - Fold {fold}", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        #plt.savefig(f"CM_XGBoost_{fold}.tiff", dpi=330, format="tiff")
        plt.show()
    
    # Combine all reports into a single DataFrame and average
    avg_report_df = pd.concat(report_list).groupby(level=0).mean()
    
    # Round and print neatly
    print("\n=== Averaged Classification Report (5-Fold CV) ===")
    print(avg_report_df.round(2))
    
    # Compute sum of confusion matrices
    cm_sum = np.sum(cm_list, axis=0)
    
    # Normalize (row-wise) to get average confusion matrix
    cm_avg = cm_sum.astype('float') / cm_sum.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("XGBoost - (5-Fold CV) Average Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    #plt.savefig("XGBoost_CM_Average.tiff", dpi=330, format="tiff")
    plt.show()
    
    print("\n=== Final Summary ===")
    print(f"Average Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Average Macro F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def classify_frailty_adaboost(df,target_column='Targets', n_splits=5):
    """
    AdaBoost classifier for frailty (3-class).
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
        target_column (str): Name of the target column (default='Targets').
        n_splits (int): Number of CV folds (default=5).
    """
    
    X = df.drop(columns=['participant_id','Targets'])
    y = df[target_column]
    
    X.columns = X.columns.astype(str).str.replace(r'[\[\]<>]', '', regex=True)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    
    cm_list = [] 
    report_list = [] 
    acc_scores = []
    f1_scores = []
    target_names = ['Robust','Pre-frail','Frail']

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Decision stump with built-in class weighting
        base_estimator = DecisionTreeClassifier(
            max_depth=3,
            class_weight='balanced'   # Key for imbalance handling
        )
        
        clf = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=200,          # More rounds
            learning_rate=0.7,         # Moderate pace
            algorithm='SAMME',       # Probabilistic boosting
            random_state=42
        )
        
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        acc_scores.append(acc)
        f1_scores.append(f1)

        print(f"\nFold {fold}")
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_list.append(report_df)
        print(report_df)

        cm = confusion_matrix(y_val, y_pred)
        cm_list.append(cm)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"AdaBoost-Confusion Matrix - Fold {fold}", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        #plt.savefig(f"CM_AdaBoost_{fold}.tiff", dpi=330, format="tiff")
        plt.show()
    
    # Combine all reports into a single DataFrame and average
    avg_report_df = pd.concat(report_list).groupby(level=0).mean()
    
    # Round and print neatly
    print("\n=== Averaged Classification Report (5-Fold CV) ===")
    print(avg_report_df.round(2))
    
    # Compute sum of confusion matrices
    cm_sum = np.sum(cm_list, axis=0)
    
    # Normalize (row-wise) to get average confusion matrix
    cm_avg = cm_sum.astype('float') / cm_sum.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("AdaBoost-(5-Fold CV) Average Confusion Matrix", fontsize=15)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("AdaBoost_CM_Average.tiff", dpi=330, format="tiff")
    plt.show()
    
    print("\n=== Final Summary ===")
    print(f"Average Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Average Macro F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

def classify_frailty_mlp(df, target_column='Targets', n_splits=5):
    """
    Simple MLP classifier for frailty (3-class).
    
    Args:
        df (pd.DataFrame): Input data with features and targets.
        target_column (str): Name of the target column (default='Targets').
        n_splits (int): Number of CV folds (default=5).
    """
    X = df.drop(columns=['participant_id', target_column])
    y = df[target_column]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    
    cm_list = [] 
    report_list = [] 
    acc_scores = []
    f1_scores = []
    target_names = ['Robust','Pre-frail','Frail']

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply SMOTE only on training data
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),   # Two hidden layers
            activation='relu',              # Best for most deep models
            solver='adam',                  # Adaptive optimizer
            learning_rate_init=0.009,       # Good starting point
            max_iter=2000,                   # Enough iterations to converge
            random_state=42,
            early_stopping=True,            # Avoid overfitting
            validation_fraction=0.1,        # Use 10% of training for validation
        )
        
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        acc_scores.append(acc)
        f1_scores.append(f1)

        print(f"\nFold {fold}")
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_list.append(report_df)
        
        print(report_df)


        cm = confusion_matrix(y_val, y_pred)
        cm_list.append(cm)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"MLP-Confusion Matrix - Fold {fold}", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        #plt.savefig(f"CM_MLP_{fold}.tiff", dpi=330, format="tiff")
        plt.show()
    
    # Combine all reports into a single DataFrame and average
    avg_report_df = pd.concat(report_list).groupby(level=0).mean()
    
    # Round and print neatly
    print("\n=== Averaged Classification Report (5-Fold CV) ===")
    print(avg_report_df.round(2))
    
    # Compute sum of confusion matrices
    cm_sum = np.sum(cm_list, axis=0)
    
    # Normalize (row-wise) to get average confusion matrix
    cm_avg = cm_sum.astype('float') / cm_sum.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("MLP-(5-Fold CV) Average Confusion Matrix", fontsize=15)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    #plt.savefig("MLP_CM_Average.tiff", dpi=330, format="tiff")
    plt.show()
    
    print("\n=== Final Summary ===")
    print(f"Average Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Average Macro F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


# Main function
def three_class_classification_main(file_path):
    
    df = load_and_check_data(file_path)
    #print(df.head())

    # CLass distribution visualizations
    plot_combined_class_distribution(df)
    
    # Print class distribution and participants statistics
    print_summary_statistics(df)
    
    # Plot the boxplot for the stride count
    plot_stride_time_boxplots(df)
    # Plot the boxplot for the number of stride
    plot_no_stride_boxplots(df)
    
    # training and evaluation of ML models with plots
    classify_frailty_svm_cv(df)
    classify_frailty_rf(df)
    classify_frailty_xgb(df)
    classify_frailty_adaboost(df)
    classify_frailty_mlp(df)