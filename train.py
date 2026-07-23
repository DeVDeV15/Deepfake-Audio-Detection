import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import SpoofDataset, parse_asvspoof_protocol
from model import DualSSL_AASIST
from metrics import compute_eer, compute_accuracy, compute_macro_f1


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", dynamic_ncols=True)
    for waveforms, labels in pbar:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits = model(waveforms)
            loss = criterion(logits, labels)
            
        # Numerical Safety Guard against AMP float16 overflow
        if torch.isnan(loss) or torch.isinf(loss):
            scaler.update() # Skip step to clear scaler state
            continue
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        valid_batches += 1
        pbar.set_postfix(Loss=f"{loss.item():.4f}")
        
    return total_loss / max(1, valid_batches)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", dynamic_ncols=True)
        for waveforms, labels in pbar:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(waveforms)
                loss = criterion(logits, labels)
                
            probs = torch.sigmoid(logits)
            # Replace any stray NaNs with 0.5 to prevent evaluation crashes
            probs = torch.nan_to_num(probs, nan=0.5).cpu().numpy()
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs)
            
    avg_loss = total_loss / len(dataloader)
    eer, opt_thresh = compute_eer(all_labels, all_scores)
    acc = compute_accuracy(all_labels, all_scores, threshold=opt_thresh)
    macro_f1 = compute_macro_f1(all_labels, all_scores, threshold=opt_thresh)
    
    return avg_loss, eer, acc, macro_f1, opt_thresh


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Launching STABILIZED DUAL-SSL Training on device: {device} ---")
    
    base_dir = "/home/guest/Desktop/DeepFake_Dataset/ALL Deepfake Data/AsvSpoof2019_LA/LA"
    train_dir = os.path.join(base_dir, "ASVspoof2019_LA_train/flac")
    train_proto = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    
    dev_dir = os.path.join(base_dir, "ASVspoof2019_LA_dev/flac")
    dev_proto = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    
    print("Parsing dataset protocols...")
    train_paths, train_labels = parse_asvspoof_protocol(train_proto, train_dir)
    dev_paths, dev_labels = parse_asvspoof_protocol(dev_proto, dev_dir)
    print(f"Dataset successfully loaded: {len(train_paths)} Train samples | {len(dev_paths)} Dev samples.")
    
    train_dataset = SpoofDataset(train_paths, train_labels)
    dev_dataset = SpoofDataset(dev_paths, dev_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    model = DualSSL_AASIST(w2v2_path="facebook/wav2vec2-base", wavlm_path="microsoft/wavlm-base-plus").to(device)
    
    # Phase 1: Freeze both backbones
    print("\n--- STAGE 1: Training Cross-Attention & Classification Head ---")
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
    for param in model.wavlm.parameters():
        param.requires_grad = False
        
    # FIX: Reduced learning rate from 1e-3 to 1e-4 for attention stability
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    stage1_epochs = 3
    for epoch in range(stage1_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, eer, acc, macro_f1, thresh = evaluate(model, dev_loader, criterion, device)
        print(f"Stage 1 Epoch {epoch+1}/{stage1_epochs} -> Train Loss: {train_loss:.4f} | Dev Loss: {val_loss:.4f} | Dev EER: {eer*100:.2f}% | Dev Acc: {acc*100:.2f}%")
        
    # Phase 2: Unfreeze top 4 transformer layers of both backbones
    print("\n--- STAGE 2: Fine-Tuning Dual Encoder Top Layers ---")
    for param in model.wav2vec2.encoder.layers[-4:].parameters():
        param.requires_grad = True
    for param in model.wavlm.encoder.layers[-4:].parameters():
        param.requires_grad = True
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total active trainable parameters for Stage 2: {trainable_params:,}")
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    stage2_epochs = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-7)
    
    best_dev_eer = float('inf')
    best_checkpoint_path = "dual_ssl_aasist_best.pt"
    
    for epoch in range(stage2_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, eer, acc, macro_f1, thresh = evaluate(model, dev_loader, criterion, device)
        scheduler.step()
        
        print(f"Stage 2 Epoch {epoch+1}/{stage2_epochs} -> Train Loss: {train_loss:.4f} | Dev Loss: {val_loss:.4f} | Dev EER: {eer*100:.2f}% | Dev F1: {macro_f1:.4f} | Thresh: {thresh:.4f}")
        
        if eer < best_dev_eer:
            best_dev_eer = eer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dev_eer': best_dev_eer,
                'optimal_threshold': thresh
            }, best_checkpoint_path)
            print(f"Successfully saved new optimal Dual-SSL checkpoint (Lowest Dev EER: {best_dev_eer*100:.2f}%).")


if __name__ == "__main__":
    main()
