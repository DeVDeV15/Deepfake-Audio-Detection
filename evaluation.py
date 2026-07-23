import os
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SpoofDiarizationDataset, collate_diarization_batch
from model import Wav2Vec2AASIST_Diarizer
from metrics import compute_frame_eer, compute_spoof_jaccard_error_rate, compute_accuracy, compute_macro_f1

def evaluate_model(eval_paths, eval_labels, checkpoint_path="w2v2_aasist_diarizer_phase3_best.pt", batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Final Phase 3 Evaluation on {device} ---")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Run train_phase3.py first.")

    model = Wav2Vec2AASIST_Diarizer().to(device)
    print(f"Loading Phase 3 optimal weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    eval_dataset = SpoofDiarizationDataset(eval_paths, eval_labels)
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=collate_diarization_batch
    )

    total_acc, total_eer, total_jer, total_f1 = 0.0, 0.0, 0.0, 0.0
    batches = len(eval_loader)
    results_export = []
    
    OPTIMAL_THRESHOLD = 0.9466
    print(f"Applying Optimized Classification Threshold: {OPTIMAL_THRESHOLD}")
    
    print("\nExecuting forward pass on Evaluation Partition (Unseen Zero-Day Attacks)...")
    with torch.no_grad(): 
        pbar = tqdm(eval_loader, desc="Evaluating Phase 3", dynamic_ncols=True)
        
        for batch_idx, (frames, batch_labels) in enumerate(pbar):
            frames = frames.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).float()
            
            with torch.amp.autocast('cuda'):
                logits = model(frames)
                
            preds_float = torch.sigmoid(logits.detach().float())
            
            acc = compute_accuracy(preds_float, batch_labels, threshold=OPTIMAL_THRESHOLD)
            eer = compute_frame_eer(preds_float, batch_labels) 
            jer = compute_spoof_jaccard_error_rate(preds_float, batch_labels, threshold=OPTIMAL_THRESHOLD)
            macro_f1 = compute_macro_f1(preds_float, batch_labels, threshold=OPTIMAL_THRESHOLD)
            
            total_acc += acc
            total_eer += eer
            total_jer += jer
            total_f1 += macro_f1
            
            # Extract raw data for error analysis export
            for i in range(len(batch_labels)):
                valid_mask = batch_labels[i] != -1
                if valid_mask.sum() == 0: continue
                
                file_prob = preds_float[i][valid_mask].mean().item()
                file_label = batch_labels[i][valid_mask].mean().item()
                error_margin = abs(file_label - file_prob)
                
                is_false_positive = (file_prob >= OPTIMAL_THRESHOLD) and (file_label == 0.0)
                is_false_negative = (file_prob < OPTIMAL_THRESHOLD) and (file_label == 1.0)
                
                results_export.append({
                    "File_Index": (batch_idx * batch_size) + i, 
                    "True_Label": "Spoof" if file_label == 1.0 else "Bonafide",
                    "Predicted_Prob": round(file_prob, 4),
                    "Error_Margin": round(error_margin, 4),
                    "False_Positive": is_false_positive,
                    "False_Negative": is_false_negative
                })
                
            pbar.set_postfix(Acc=f"{acc*100:.1f}%", F1=f"{macro_f1:.3f}")
            
    print("\n--- Final Phase 3 Evaluation Metrics (Zero-Day Attacks) ---")
    print(f"Global Accuracy : {(total_acc/batches)*100:.2f}%")
    print(f"Global Macro F1 : {total_f1/batches:.4f}")
    print(f"Global EER      : {total_eer/batches:.4f}")
    print(f"Global JER      : {total_jer/batches:.4f}")

    csv_file = "evaluation_error_analysis_phase3.csv"
    print(f"\nExporting detailed predictions to {csv_file} ...")
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["File_Index", "True_Label", "Predicted_Prob", "Error_Margin", "False_Positive", "False_Negative"])
        writer.writeheader()
        results_export.sort(key=lambda x: x["Error_Margin"], reverse=True)
        writer.writerows(results_export)
        
    print("Export complete. Phase 3 validation finished.")

if __name__ == "__main__":
    base_dir = "/home/guest/Desktop/DeepFake_Dataset/ALL Deepfake Data/AsvSpoof2019_LA/LA"
    eval_flac_dir = os.path.join(base_dir, "ASVspoof2019_LA_eval/flac")
    eval_protocol = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    
    def parse_asvspoof_eval_protocol(protocol_path, flac_dir):
        audio_paths, labels = [], []
        print(f"Parsing EVALUATION protocol map: {protocol_path}")
        with open(protocol_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                file_id = parts[1]
                label_str = parts[4]
                file_path = os.path.join(flac_dir, f"{file_id}.flac")
                if not os.path.exists(file_path): continue
                
                if label_str == "bonafide": labels.append(torch.tensor([0.0])) 
                elif label_str == "spoof": labels.append(torch.tensor([1.0]))
                audio_paths.append(file_path)
        return audio_paths, labels

    eval_paths, eval_labels = parse_asvspoof_eval_protocol(eval_protocol, eval_flac_dir)
    print(f"Successfully loaded {len(eval_paths)} evaluation files.")
    evaluate_model(eval_paths, eval_labels)