import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SpoofDiarizationDataset, collate_diarization_batch
from model import Wav2Vec2AASIST_Diarizer
from metrics import compute_frame_eer, compute_spoof_jaccard_error_rate, compute_accuracy

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        focal_loss = alpha_factor * modulating_factor * bce_loss
        return focal_loss.mean()

def train_diarizer(train_paths, train_labels, val_paths=None, val_labels=None, 
                    batch_size=4, accumulation_steps=4, epochs=15, lr=1e-5, checkpoint_path=None, patience=3):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training pipeline on: {device}")
    
    train_dataset = SpoofDiarizationDataset(train_paths, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, prefetch_factor=2,
        collate_fn=collate_diarization_batch
    )
    
    val_loader = None
    if val_paths and val_labels:
        val_dataset = SpoofDiarizationDataset(val_paths, val_labels)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True, collate_fn=collate_diarization_batch
        )
    
    model = Wav2Vec2AASIST_Diarizer().to(device)
    
    # Use Focal Loss for class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-7
    )
    
    start_epoch, best_val_loss, patience_counter = 0, float('inf'), 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint. Recovering states from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed successfully. Continuing from Epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, epoch_acc, epoch_eer, epoch_jer = 0.0, 0.0, 0.0, 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs} (Current LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        pbar = tqdm(train_loader, desc="Training", leave=False, dynamic_ncols=True)
        
        optimizer.zero_grad()
        
        for batch_idx, (frames, batch_labels) in enumerate(pbar):
            frames = frames.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).float()
            
            with torch.amp.autocast('cuda'):
                logits = model(frames)
                valid_mask = batch_labels != -1
                if valid_mask.sum() == 0:
                    continue
                loss = criterion(logits[valid_mask], batch_labels[valid_mask]) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            preds_float = torch.sigmoid(logits.detach().float())
            loss_val = loss.item() * accumulation_steps
            
            acc = compute_accuracy(preds_float, batch_labels)
            eer = compute_frame_eer(preds_float, batch_labels)
            jer = compute_spoof_jaccard_error_rate(preds_float, batch_labels)
            
            epoch_loss += loss_val
            epoch_acc += acc
            epoch_eer += eer
            epoch_jer += jer
            
            pbar.set_postfix(Loss=f"{loss_val:.3f}", Acc=f"{acc*100:.1f}%", EER=f"{eer:.3f}")
        
        pbar.close()
        
        batches = len(train_loader)
        print(f"Train Summary -> Loss: {epoch_loss/batches:.4f} | Acc: {(epoch_acc/batches)*100:.2f}% | EER: {epoch_eer/batches:.4f} | JER: {epoch_jer/batches:.4f}")
        
        current_val_loss = epoch_loss / batches
        if val_loader:
            model.eval()
            val_loss, val_acc, val_eer, val_jer = 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)
                for frames, batch_labels in val_pbar:
                    frames = frames.to(device, non_blocking=True)
                    batch_labels = batch_labels.to(device, non_blocking=True).float()
                    with torch.amp.autocast('cuda'):
                        logits = model(frames)
                        valid_mask = batch_labels != -1
                        if valid_mask.sum() == 0:
                            continue
                        v_loss = criterion(logits[valid_mask], batch_labels[valid_mask]).item()
                        preds_float = torch.sigmoid(logits.detach().float())
                        v_acc = compute_accuracy(preds_float, batch_labels)
                        v_eer = compute_frame_eer(preds_float, batch_labels)
                        v_jer = compute_spoof_jaccard_error_rate(preds_float, batch_labels)
                        val_loss += v_loss
                        val_acc += v_acc
                        val_eer += v_eer
                        val_jer += v_jer
                val_pbar.close()
            v_batches = len(val_loader)
            current_val_loss = val_loss / v_batches
            print(f"Val Summary   -> Loss: {current_val_loss:.4f} | Acc: {(val_acc/v_batches)*100:.2f}% | EER: {val_eer/v_batches:.4f} | JER: {val_jer/v_batches:.4f}")
            
        scheduler.step()
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            checkpoint_meta = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_loss': best_val_loss}
            torch.save(checkpoint_meta, "w2v2_aasist_diarizer_best.pt")
            print("Successfully saved new optimal structural checkpoint.")
        else:
            patience_counter += 1
            print(f"Early Stopping Alert: Validation metrics stagnated. Count: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early Stopping criteria triggered. Terminating pipeline execution loop at Epoch {epoch+1}.")
            break

if __name__ == "__main__":
    base_dir = "/home/guest/Desktop/DeepFake_Dataset/ALL Deepfake Data/AsvSpoof2019_LA/LA"
    train_flac_dir = os.path.join(base_dir, "ASVspoof2019_LA_train/flac")
    train_protocol = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    val_flac_dir = os.path.join(base_dir, "ASVspoof2019_LA_dev/flac")
    val_protocol = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    
    def parse_asvspoof_protocol(protocol_path, flac_dir):
        audio_paths, labels = [], []
        with open(protocol_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            file_id = parts[1]
            label_str = parts[4]
            file_path = os.path.join(flac_dir, f"{file_id}.flac")
            if not os.path.exists(file_path): continue
            labels.append(torch.tensor([0.0]) if label_str == "bonafide" else torch.tensor([1.0]))
            audio_paths.append(file_path)
        return audio_paths, labels

    print("--- Initializing Data Pipelines ---")
    train_paths, train_labels = parse_asvspoof_protocol(train_protocol, train_flac_dir)
    val_paths, val_labels = parse_asvspoof_protocol(val_protocol, val_flac_dir)
    
    train_diarizer(
        train_paths=train_paths, 
        train_labels=train_labels,
        val_paths=val_paths,    
        val_labels=val_labels,
        batch_size=4,             
        accumulation_steps=4,     
        epochs=15,          
        checkpoint_path="w2v2_aasist_diarizer_best.pt", 
        patience=3                
    )