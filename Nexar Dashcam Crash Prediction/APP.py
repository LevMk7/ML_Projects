"""
Accident Anticipation Pipeline
Modules:
 1. video_to_frames: extract RGB frames from video
 2. extract_features: count YOLO detections per frame
 3. preprocess_train/test: build datasets
 4. AccidentPredictor: LSTM + attention model
 5. train_and_validate: training loop with AUC-ROC logging
 6. inference: predict probabilities for test set
 7. visualize_detections: display one frame with YOLO boxes
 8. visualize_frame_vehicles: display one frame with vehicle detections only
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Configuration
FPS = 2                 # frames per second to sample
CLIP_DURATION = 4.0     # seconds of video to use
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 1. Frame Extraction

def video_to_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    fps: int = FPS
) -> np.ndarray:
    """
    Inputs:
      video_path: path to .mp4 file
      start_time, end_time: seconds defining clip window
      fps: frames per second to sample
    Output:
      frames: np.ndarray of shape (T, H, W, 3), dtype uint8 (RGB)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    # Read video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Validate properties
    if not np.isfinite(video_fps) or video_fps <= 0:
        video_fps = 30.0
    if not np.isfinite(total_frames) or total_frames <= 0:
        total_frames = video_fps * (end_time - start_time)
    # Sanitize time window
    start_time = 0.0 if not np.isfinite(start_time) or start_time < 0 else start_time
    end_time = total_frames / video_fps if not np.isfinite(end_time) or end_time < start_time else end_time
        # Compute frame indices for a fixed-length clip
    start_frame = int(start_time * video_fps)
    end_frame = int(min(end_time * video_fps, total_frames - 1))
    # Always use CLIP_DURATION to determine number of frames (constant T)
    num_frames = int(CLIP_DURATION * fps)
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(
            f"No frames extracted from {video_path} [{start_time}-{end_time}]"
        )
    # Pad last frame to reach fixed T
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames)

# 2. Feature Extraction

def extract_features(
    frames: np.ndarray,
    yolo_model: YOLO
) -> np.ndarray:
    """
    Inputs:
      frames: np.ndarray (T, H, W, 3)
      yolo_model: loaded YOLOv8 model on DEVICE
    Output:
      features: np.ndarray of shape (T, 4) containing counts [car, truck, person, traffic_light]
    """
    results = yolo_model.predict(
        source=list(frames),
        batch=len(frames),
        device=DEVICE,
        verbose=False,
        show=False
    )
    class_names = ['car', 'truck', 'person', 'traffic light']
    features = []
    for res in results:
        counts = dict.fromkeys(class_names, 0)
        for box in res.boxes:
            name = yolo_model.names[int(box.cls[0])]
            if name in counts:
                counts[name] += 1
        features.append([counts[c] for c in class_names])
    return np.array(features, dtype=np.float32)

# 3. Dataset Preprocessing

def preprocess_train(
    df: pd.DataFrame,
    video_dir: str,
    yolo_model: YOLO,
    fps: int = FPS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
      df: DataFrame with columns ['id', 'time_of_event', 'target']
      video_dir: folder containing videos named id.mp4
      yolo_model: YOLOv8 model
      fps: frames per second
    Outputs:
      X: np.ndarray shape (N, T, 4)
      y: np.ndarray shape (N,)
    """
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Preprocessing train'):
        vid = str(row['id']).zfill(5)
        path = os.path.join(video_dir, f"{vid}.mp4")
        end = row['time_of_event']
        start = max(0.0, end - CLIP_DURATION)
        frames = video_to_frames(path, start, end, fps)
        feats = extract_features(frames, yolo_model)
        X.append(feats)
        y.append(row['target'])
    return np.array(X), np.array(y)


def preprocess_test(
    df: pd.DataFrame,
    video_dir: str,
    yolo_model: YOLO,
    fps: int = FPS
) -> tuple[np.ndarray, list[str]]:
    """
    Inputs:
      df: DataFrame with column ['id']
      video_dir: folder containing videos
      yolo_model: YOLOv8 model
      fps: frames per second
    Outputs:
      X_test: np.ndarray shape (N, T, 4)
      ids: list of video ids
    """
    X, ids = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Preprocessing test'):
        vid = str(row['id']).zfill(5)
        path = os.path.join(video_dir, f"{vid}.mp4")
        cap = cv2.VideoCapture(path)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        start = max(0.0, dur - CLIP_DURATION)
        frames = video_to_frames(path, start, dur, fps)
        feats = extract_features(frames, yolo_model)
        X.append(feats)
        ids.append(vid)
    return np.array(X), ids

# 4. Model Definition
class AccidentPredictor(nn.Module):
    """
    Inputs:
      x: Tensor shape (B, T, 4)
    Output:
      Tensor shape (B,) with raw logits
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_dir = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim*self.num_dir, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_dim*self.num_dir, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)               # (B, T, H*num_dir)
        weights = self.attn(out)            # (B, T, 1)
        context = torch.sum(weights * out, dim=1)  # (B, H*num_dir)
        return self.fc(context).squeeze(1)   # (B,)

# 5. Training & Validation
def train_and_validate(
    X: np.ndarray,
    y: np.ndarray,
    device: str = DEVICE
) -> AccidentPredictor:
    """
    Inputs:
      X: (N, T, 4), y: (N,)
    Output:
      Trained AccidentPredictor
    """
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    n_val = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val])
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = AccidentPredictor().to(device)
    torch.backends.cudnn.benchmark = True
    neg, pos = (y==0).sum(), (y==1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos, device=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses, val_aucs = [], [], []
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * xb.size(0)
        train_losses.append(total_loss / len(train_ld.dataset))

        model.eval()
        val_loss, preds, truths = 0, [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device).float()
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds.append(logits.cpu())
                truths.append(yb.cpu())
        val_losses.append(val_loss / len(val_ld.dataset))
        preds = torch.cat(preds).numpy()
        truths = torch.cat(truths).numpy()
        auc = roc_auc_score(truths, preds)
        val_aucs.append(auc)
        print(f"Epoch {epoch} train_loss {train_losses[-1]:.3f} val_loss {val_losses[-1]:.3f} AUC {auc:.3f}")

    # Plot
    epochs = range(1, len(train_losses)+1)
    plt.figure(); plt.plot(epochs, train_losses, label='train_loss'); plt.plot(epochs, val_losses, label='val_loss'); plt.legend(); plt.show()
    plt.figure(); plt.plot(epochs, val_aucs, label='val_AUC'); plt.legend(); plt.show()

    torch.save(model.state_dict(), 'accident_model.pth')
    return model

# 6. Inference
def inference(
    model: AccidentPredictor,
    X_test: np.ndarray,
    ids: list[str],
    device: str = DEVICE
) -> None:
    """
    Inputs:
      model: trained AccidentPredictor
      X_test: (N, T, 4)
      ids: list of video ids
    Output:
      Writes submission.csv with columns ['id','probability']
    """
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb in DataLoader(torch.from_numpy(X_test), batch_size=16):
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            all_probs.extend(probs.tolist())
    pd.DataFrame({'id': ids, 'probability': all_probs}).to_csv('submission.csv', index=False)
    print('submission.csv saved')

# 7. Visualization Helper
def visualize_detections(
    video_path: str,
    yolo_model: YOLO,
    fps: int = FPS,
    clip_len: float = CLIP_DURATION
) -> None:
    """
    Inputs:
      video_path: path to video
      yolo_model: YOLOv8 model
      fps, clip_len: sampling parameters
    Output:
      Displays last frame with bounding boxes
    """
    cap = cv2.VideoCapture(video_path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    start = max(0.0, dur - clip_len)
    frames = video_to_frames(video_path, start, dur, fps)
    frame = frames[-1]
    res = yolo_model.predict(source=[frame], device=DEVICE, verbose=False, show=False)[0]
    img = frame.copy()
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = yolo_model.names[int(box.cls[0])]
        if cls in ['car','truck','person','traffic light']:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, cls, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    plt.figure(figsize=(8,6)); plt.imshow(img); plt.axis('off'); plt.show()

# 8. Vehicle-only Visualization

def visualize_frame_vehicles(
    frame: np.ndarray,
    yolo_model: YOLO
) -> None:
    """
    Inputs:
      frame: single RGB image np.ndarray (H, W, 3)
      yolo_model: YOLOv8 model
    Output:
      Displays the frame with bounding boxes around vehicles (car, truck)
    """
    # Run detection on single frame
    res = yolo_model.predict(source=[frame], device=DEVICE, verbose=False, show=False)[0]
    img = frame.copy()
    for box in res.boxes:
        cls = yolo_model.names[int(box.cls[0])]
        if cls in ['car', 'truck']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, cls, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    plt.figure(figsize=(8,6)); plt.imshow(img); plt.axis('off'); plt.show()

# 9. Main
if __name__ == '__main__':
    # Load data and model
    train_df = pd.read_csv('train.csv', dtype={'id': str, 'time_of_event': float, 'target': int})
    test_df = pd.read_csv('test.csv', dtype={'id': str})
    yolo_model = YOLO('yolov8n.pt').to(DEVICE)

    # Process and train
    X_train, y_train = preprocess_train(train_df, 'train', yolo_model)
    model = train_and_validate(X_train, y_train)

    # Inference
    X_test, ids = preprocess_test(test_df, 'test', yolo_model)
    inference(model, X_test, ids)

    # Example usage of vehicle-only visualization
    sample_frames = video_to_frames('train/00000.mp4', 36.0, 40.0, fps=FPS)
    visualize_frame_vehicles(sample_frames[-1], yolo_model)
