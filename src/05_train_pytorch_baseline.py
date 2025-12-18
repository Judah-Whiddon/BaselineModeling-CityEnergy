from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise SystemExit("PyTorch is not installed. Install with: pip install torch") from e

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"

DATA = OUT / "train_ready_2020_regression.csv"
METRICS_PATH = OUT / "pytorch_baseline_metrics.json"
MODEL_PATH   = OUT / "pytorch_baseline_model.pt"

SEED = 42

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))

def main():
    set_seed(SEED)

    df = pd.read_csv(DATA)

    # Features (numeric only)
    feature_cols = ["site_eui", "energy_star_score", "gross_floor_area_sqft"]
    if "gross_floor_area_sqft" in df.columns:
        df["gross_floor_area_sqft"] = df["gross_floor_area_sqft"].fillna(
            df["gross_floor_area_sqft"].median()
        )

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y_log_ghg"].to_numpy(dtype=np.float32).reshape(-1, 1)

    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    # Standardize features using train stats
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0

    X_train = (X_train - mu) / sd
    X_val   = (X_val   - mu) / sd
    X_test  = (X_test  - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = MLP(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 🔹 CHANGED: Huber loss for robust baseline training
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, 21):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test metrics
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy().reshape(-1)

    y_true_log = y_test.reshape(-1)
    y_pred_log = preds

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    def median_ae(a, b) -> float:
        return float(np.median(np.abs(a - b)))

    def mape(a, b) -> float:
        eps = 1e-6
        return float(np.mean(np.abs(a - b) / (np.abs(a) + eps)))

    print("\nAdditional diagnostics (raw scale):")
    print("  median_abs_error:", median_ae(y_true, y_pred))
    print("  mape:", mape(y_true, y_pred))

    abs_err = np.abs(y_true - y_pred)
    worst = np.argsort(-abs_err)[:10]

    print("\nWorst 10 absolute errors (true, pred):")
    for i in worst:
        print(f"  true={y_true[i]:,.0f}  pred={y_pred[i]:,.0f}")

    results = {
        "n_rows_total": int(n),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "features": feature_cols,
        "target": "total_ghg_mtco2e (log1p during training)",
        "train_loss": "HuberLoss(delta=1.0)",
        "test_rmse": rmse(y_true, y_pred),
        "test_mae": mae(y_true, y_pred),
        "test_rmse_log": rmse(y_true_log, y_pred_log),
        "test_mae_log": mae(y_true_log, y_pred_log),
        "device": str(device),
        "seed": SEED,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    torch.save(
        {"state_dict": model.state_dict(), "mu": mu, "sd": sd, "feature_cols": feature_cols},
        MODEL_PATH
    )

    print("\nSaved:")
    print(" -", METRICS_PATH)
    print(" -", MODEL_PATH)

    print("\nMetrics:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
