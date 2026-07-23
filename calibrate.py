import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

def calibrate_phase3(csv_file="evaluation_error_analysis_phase3.csv", output_img="det_curve_phase3_optimized.png"):
    print(f"Reading error analysis logs from {csv_file}...")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please ensure Phase 3 evaluation completed.")
        return

    df = pd.read_csv(csv_file)
    
    # Map the exported text labels back to binary targets
    y_true = (df['True_Label'] == 'Spoof').astype(int).values
    y_scores = df['Predicted_Prob'].values
    
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr # False Negative Rate is the inverse of True Positive Rate
    
    # Locate the optimal crossover point (Equal Error Rate)
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    optimal_threshold = thresholds[eer_idx]
    eer_val = fpr[eer_idx] * 100
    
    # Calculate standard 0.50 default EER for comparison
    default_idx = np.nanargmin(np.absolute(thresholds - 0.50))
    default_eer = ((fpr[default_idx] + fnr[default_idx]) / 2) * 100
    
    print("\n" + "="*50)
    print(" PHASE 3 MATHEMATICAL CALIBRATION RESULTS")
    print("="*50)
    print(f"Default Threshold (0.50) EER  : {default_eer:.2f}%")
    print(f"Optimal Decision Threshold    : {optimal_threshold:.4f}")
    print(f"Optimized Crossover EER       : {eer_val:.2f}%")
    print("="*50)
    
    # Plotting the DET Curve
    plt.figure(figsize=(10, 8))
    
    # Phase 3 Fine-Tuned Curve
    plt.plot(fpr * 100, fnr * 100, label=f'Phase 3 Acoustic (EER: {eer_val:.2f}%)', color='darkorange', lw=2.5)
    plt.scatter([eer_val], [eer_val], color='red', zorder=5, 
                label=f'New Optimal Threshold: {optimal_threshold:.4f}')
    
    # 50/50 Random Guessing Line for reference
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle=':', alpha=0.5)
    
    # Focus the graph on the relevant performance quadrant
    plt.xlim([0.0, 30.0])
    plt.ylim([0.0, 30.0])
    plt.xlabel('False Alarm Rate / False Positive (%)', fontsize=12)
    plt.ylabel('Miss Probability / False Negative (%)', fontsize=12)
    plt.title('Phase 3 Decision Boundary Calibration (Zero-Day Attacks)', fontsize=14, pad=15)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Export the final visualization
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\nOptimization graph exported successfully as '{output_img}'.")

if __name__ == "__main__":
    calibrate_phase3()