import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpoofDataset, parse_asvspoof_protocol
from model import DualSSL_AASIST
from metrics import compute_eer, compute_accuracy, compute_macro_f1

def evaluate_and_plot(eval_dir, eval_proto, checkpoint_path="dual_ssl_aasist_best.pt", output_csv="evaluation_results_dual_ssl.csv", output_det="det_curve_dual_ssl.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Dual-SSL Final Evaluation on Device: {device} ---")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found. Ensure train.py finished successfully.")

    # Parse Evaluation Protocol
    eval_paths, eval_labels = parse_asvspoof_protocol(eval_proto, eval_dir)
    print(f"Loaded {len(eval_paths)} evaluation files.")
    
    eval_dataset = SpoofDataset(eval_paths, eval_labels)
    
    # FIXED: Batch size reduced to 4 to prevent VRAM overflow and 19-minute slowdowns
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load Dual-SSL Model Architecture
    model = DualSSL_AASIST(w2v2_path="facebook/wav2vec2-base", wavlm_path="microsoft/wavlm-base-plus").to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded. Evaluating using RAW LOGITS to prevent float32 saturation...")
    
    all_paths = eval_paths
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating Dual-SSL", dynamic_ncols=True)
        for waveforms, labels in pbar:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(waveforms)
                
            # Use RAW LOGITS instead of sigmoids to preserve mathematical precision
            scores = logits.detach().cpu().numpy()
            scores = np.nan_to_num(scores, nan=0.0)
            
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(scores)
            
    # Compute Evaluation Metrics directly on logits
    # This will return the optimal logit threshold (e.g., -2.5 or 3.1), not a probability
    global_eer, opt_thresh_eval = compute_eer(all_labels, all_logits)
    
    # Calculate accuracy and F1 using the newly found logit threshold
    global_acc = compute_accuracy(all_labels, all_logits, threshold=opt_thresh_eval)
    global_f1 = compute_macro_f1(all_labels, all_logits, threshold=opt_thresh_eval)
    
    print("\n" + "="*50)
    print(" DUAL-SSL EVALUATION METRICS SUMMARY (RAW LOGITS) ")
    print("="*50)
    print(f"Global Equal Error Rate (EER) : {global_eer*100:.2f}%")
    print(f"Global Accuracy (@ Eval Thresh): {global_acc*100:.2f}%")
    print(f"Global Macro F1               : {global_f1:.4f}")
    print(f"Optimal Eval Threshold (Logit): {opt_thresh_eval:.4f}")
    print("="*50)
    
    # Export Error Analysis CSV
    print(f"\nExporting detailed predictions to {output_csv}...")
    results_export = []
    for i in range(len(all_labels)):
        path = all_paths[i]
        true_lbl = "Spoof" if all_labels[i] == 1.0 else "Bonafide"
        
        # Convert logit back to probability just for the human-readable CSV column
        pred_prob = 1.0 / (1.0 + np.exp(-all_logits[i]))
        
        is_fp = (all_logits[i] >= opt_thresh_eval) and (all_labels[i] == 0.0)
        is_fn = (all_logits[i] < opt_thresh_eval) and (all_labels[i] == 1.0)
        
        results_export.append({
            "File_Path": path,
            "True_Label": true_lbl,
            "Predicted_Logit": round(float(all_logits[i]), 4),
            "Predicted_Prob": round(float(pred_prob), 6),
            "False_Positive": is_fp,
            "False_Negative": is_fn
        })
        
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["File_Path", "True_Label", "Predicted_Logit", "Predicted_Prob", "False_Positive", "False_Negative"])
        writer.writeheader()
        writer.writerows(results_export)
        
    # Generate and Export DET Curve Plot using Logits
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_logits, pos_label=1)
    fnr = 1.0 - tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr * 100, fnr * 100, color='darkorange', lw=2.5, label=f'Dual-SSL (EER: {global_eer*100:.2f}%)')
    plt.scatter([global_eer*100], [global_eer*100], color='red', zorder=5, label=f'EER Point ({global_eer*100:.2f}%)')
    plt.plot([0, 100], [0, 100], color='navy', linestyle=':', alpha=0.5)
    plt.xlim([0.0, 30.0])
    plt.ylim([0.0, 30.0])
    plt.xlabel('False Positive / False Alarm Rate (%)', fontsize=12)
    plt.ylabel('False Negative / Miss Rate (%)', fontsize=12)
    plt.title('Dual-SSL DET Curve (Raw Logit Precision)', fontsize=14)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_det, dpi=300, bbox_inches='tight')
    print(f"DET Curve plot successfully saved as '{output_det}'.")

if __name__ == "__main__":
    base_dir = "/home/guest/Desktop/DeepFake_Dataset/ALL Deepfake Data/AsvSpoof2019_LA/LA"
    eval_dir = os.path.join(base_dir, "ASVspoof2019_LA_eval/flac")
    eval_proto = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    
    evaluate_and_plot(eval_dir, eval_proto)
