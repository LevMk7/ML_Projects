"""
Hybrid Linear-Grid Stereo Model — 26 trainable parameters
Pure numpy + pygame. No neural nets, no scipy.

ZONES (normalized coords  un=(u−CX)/CX,  vn=(v−CY)/CY)
  base = [k·un+c_x, k·vn+c_y, 1]           (linear with optical-centre offsets)
  Γ is a RELATIVE distortion multiplier (Γ=1 ⇒ pure linear, preserves c1/c2)
  Center u ∈ [810,1782]       v = base                             (α=0)
  Edges  u<810 or u>1782      v = base · [g,g,1]                   (α=1, per-camera Γ)
  Blend  α-smooth 40-px       v = base · ((1−α) + α·g)  (x,y only; z=1)
  Inner grid columns (1,2)    FROZEN at Γ=1 — only outer edge nodes train
  Baseline d = [dx, dy, 0]    dx ≥ 0.02 m

PARAMS (26):  GL[4] + GR[4] + Γ(4×4)=16 + d[2]

LOSS:  L = L_1D + L_Y   (ray-space disparity + y-epipolar — no Z² blow-up)
  disp_pred = vL[0] − vR[0];   disp_true = dx / Z_true
  L_1D   = ½(disp_pred − disp_true)²
  L_Y    = ½((vL[1] − vR[1]) − dy/Z_true)²
PARAMS (66):  ΓL(4×4×2)=32 + ΓR(4×4×2)=32 + d[2]    (no rotation)
  Anisotropic γ: separate γx, γy per node — captures fx≠fy and asymmetric distortion.
  Rotation removed entirely (R = I always).
  dy is now trainable.
LR:   lr_g=1e-1, lr_dx=1e-3, lr_dy=1e-3

INFERENCE:   A=[vL,−vR];  x = np.linalg.solve(AᵀA, Aᵀd);  Z = x[0]

UI: left-click = pick, wheel = zoom, right-drag = pan.
    K = add cal pt (terminal-prompted Z), C = reset, R = clear meas, T = swap, Esc = quit.
    After ≥15 cal pts, train() fires automatically and saves stereo_calibration.json.
"""

import json, math, os, sys
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────
IMG_W, IMG_H          = 2592, 1944
CX,    CY             = IMG_W / 2.0, IMG_H / 2.0        # 1296, 972
CENTER_LO, CENTER_HI  = 810, 1782
BLEND                 = 40
GRID_ROWS, GRID_COLS  = 3, 3
CELL_W                = IMG_W / (GRID_COLS - 1)         # 864.0
CELL_H                = IMG_H / (GRID_ROWS - 1)         # 648.0
CALIB_PATH            = "stereo_calibration.json"
CAL_PTS_PATH          = "cal_pts_clean.json"
MIN_CAL_PTS           = 15
F_INIT                = 2880.0                          # init focal length (px) ≈ 4.05 × 710
DX_MIN                = 0.02                            # physical baseline floor
K_MIN                 = 1e-6
G_MIN, G_MAX          = 1e-6, 10.0
DISP_W, DISP_H        = 800,  600                       # display panel size (4:3)

# ─── REGULARIZATION & VAL-SPLIT ───────────────────────────────
LAMBDA_ANCHOR  = 1e-3        # pulls Γ toward G0_INIT (physical focal prior)
LAMBDA_SMOOTH  = 1e-3        # 4-neighbor Laplacian on Γ
FREEZE_INNER   = False       # freeze Γ inner cols 1,2 at G0_INIT (soft anchor handles this)
VAL_FRAC       = 0.0         # held-out fraction (0 disables — needs N>~50 to be net-positive)
VAL_SEED       = 0

# ─── DEFAULT PARAMETERS  (Γ-only; Γ node = CX/f_x_effective at that grid node)
def defaults():
    g0     = CX / F_INIT                                    # ≈ 0.4507
    # Anisotropic γ: shape (rows, cols, 2)  →  [...,0]=γx, [...,1]=γy
    GammaL = np.full((GRID_ROWS, GRID_COLS, 2), g0, dtype=np.float64)
    GammaR = np.full((GRID_ROWS, GRID_COLS, 2), g0, dtype=np.float64)
    d      = np.array([0.10, 0.0, 0.0], dtype=np.float64)
    return GammaL, GammaR, d

G0_INIT = CX / F_INIT                                       # ≈ 0.4507 (nominal Γ)

# ─── BILINEAR GAMMA LOOKUP ────────────────────────────────────
def bilinear(u, v, Gamma):
    fx, fy = u / CELL_W, v / CELL_H
    ix = min(max(int(fx), 0), GRID_COLS - 2)
    iy = min(max(int(fy), 0), GRID_ROWS - 2)
    tx, ty = fx - ix, fy - iy
    w   = ((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty)
    idx = ((iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1))
    g   = w[0]*Gamma[idx[0]] + w[1]*Gamma[idx[1]] + w[2]*Gamma[idx[2]] + w[3]*Gamma[idx[3]]
    return float(g), w, idx

# ─── RAY BUILDER (anisotropic Γ)  v = [un·γx, vn·γy, 1] ───────────────────
def build_ray(u, v, Gamma):
    """Anisotropic Γ: separate γx, γy per node. Gamma shape = (R, C, 2)."""
    un  = (u - CX) / CX
    vn  = (v - CY) / CY
    gx, w, idx = bilinear(u, v, Gamma[..., 0])
    gy, _, _   = bilinear(u, v, Gamma[..., 1])
    info = {"u": u, "v": v, "un": un, "vn": vn,
            "gx": gx, "gy": gy, "w": w, "idx": idx}
    return np.array([un * gx, vn * gy, 1.0]), info

# ─── OLS TRIANGULATION   A = [vL, −vR] ,   x = (AᵀA)⁻¹ Aᵀd ,   Z = x[0]
def triangulate(uL, vL, uR, vR, GammaL, GammaR, d):
    if uL <= uR:
        return float("nan"), None, None, f"bad disparity uL={uL} ≤ uR={uR}"
    vL_d, _ = build_ray(uL, vL, GammaL)
    vR_d, _ = build_ray(uR, vR, GammaR)
    denom   = float(vL_d[0]) - float(vR_d[0])
    if abs(denom) < 1e-9:
        return float("nan"), vL_d, vR_d, "zero x-disparity (rays parallel)"
    A   = np.column_stack([vL_d, -vR_d])
    AtA = A.T @ A
    Atd = A.T @ d
    try:
        x = np.linalg.solve(AtA, Atd)
        Z = float(x[0])
    except np.linalg.LinAlgError:
        Z = float(d[0]) / denom                    # near-singular fallback
    if Z <= 0:
        return float("nan"), vL_d, vR_d, f"negative depth Z = {Z:.2f} m"
    return Z, vL_d, vR_d, None

# ─── FORWARD + ANALYTICAL GRADIENTS  (Γ-only model, no rotation) ─────────────
def forward_grads(uL, vL, uR, vR, Z_true, GammaL, GammaR, d):
    """Pure Γ-grid ray model. R = identity (rotation removed entirely).
       Loss = L_1D (relative disparity) + L_Y (y-epipolar), Huber-clipped."""
    vL_d, iL = build_ray(uL, vL, GammaL)
    vR_d, iR = build_ray(uR, vR, GammaR)

    denom = float(vL_d[0]) - float(vR_d[0])
    if abs(denom) < 1e-12:
        return (0.0,
                np.zeros((GRID_ROWS, GRID_COLS, 2)),
                np.zeros((GRID_ROWS, GRID_COLS, 2)),
                np.zeros(3))
    dx = max(float(d[0]), DX_MIN)

    # ── L_1D (Relative Percentage Disparity) ────────────────────────────────
    disp_true = dx / Z_true
    rel_err   = (denom / disp_true) - 1.0
    loss_1d   = 0.5 * rel_err * rel_err

    dL_ddenom = rel_err * (Z_true / dx)
    dL_dvLx   =  dL_ddenom
    dL_dvRx   = -dL_ddenom

    # ── L_Y (Ray-space y-epipolar) — dy now trainable ───────────────────────
    e_y_ray = (float(vL_d[1]) - float(vR_d[1])) - (float(d[1]) / Z_true)
    loss_y  = 0.5 * e_y_ray * e_y_ray
    dL_dvLy =  e_y_ray
    dL_dvRy = -e_y_ray

    # ── Huber loss: replace 0.5e² with linear tails beyond |e|≥δ ────────────
    #     L_h(e) = 0.5e²              if |e| ≤ δ      (∂L/∂e = e)
    #     L_h(e) = δ(|e| - 0.5δ)      if |e| >  δ     (∂L/∂e = δ·sign(e))
    HUBER_D = 0.05      # 5% relative-disparity threshold; same scale for L_Y
    if abs(rel_err) > HUBER_D:
        loss_1d   = HUBER_D * (abs(rel_err) - 0.5 * HUBER_D)
        dL_drel   = HUBER_D if rel_err > 0 else -HUBER_D
        dL_ddenom = dL_drel * (Z_true / dx)
        dL_dvLx, dL_dvRx =  dL_ddenom, -dL_ddenom
    if abs(e_y_ray) > HUBER_D:
        loss_y  = HUBER_D * (abs(e_y_ray) - 0.5 * HUBER_D)
        dL_dey  = HUBER_D if e_y_ray > 0 else -HUBER_D
        dL_dvLy, dL_dvRy =  dL_dey, -dL_dey

    loss      = loss_1d + loss_y
    grad_d    = np.zeros(3)
    grad_d[0] = rel_err * (-denom * Z_true / (dx * dx)) if abs(rel_err) <= HUBER_D \
                else (HUBER_D * (1 if rel_err>0 else -1)) * (-denom * Z_true / (dx * dx))
    if abs(e_y_ray) <= HUBER_D:
        grad_d[1] = -e_y_ray / Z_true
    else:
        grad_d[1] = -(HUBER_D if e_y_ray > 0 else -HUBER_D) / Z_true
    grad_d[2] = 0.0

    # ── Chain rule → anisotropic Γ grids  (Gamma[...,0]=γx, [...,1]=γy)
    #     ∂v_x/∂γx_node = un · w_node      (γy doesn't enter v_x)
    #     ∂v_y/∂γy_node = vn · w_node      (γx doesn't enter v_y)
    gGamL = np.zeros((GRID_ROWS, GRID_COLS, 2))
    sLx = dL_dvLx * iL["un"]
    sLy = dL_dvLy * iL["vn"]
    for k in range(4):
        gGamL[iL["idx"][k][0], iL["idx"][k][1], 0] += sLx * iL["w"][k]
        gGamL[iL["idx"][k][0], iL["idx"][k][1], 1] += sLy * iL["w"][k]
    gGamR = np.zeros((GRID_ROWS, GRID_COLS, 2))
    sRx = dL_dvRx * iR["un"]
    sRy = dL_dvRy * iR["vn"]
    for k in range(4):
        gGamR[iR["idx"][k][0], iR["idx"][k][1], 0] += sRx * iR["w"][k]
        gGamR[iR["idx"][k][0], iR["idx"][k][1], 1] += sRy * iR["w"][k]

    return loss, gGamL, gGamR, grad_d

# ─── TRAINING — trains Γ grids + baseline d (no k/c any more) ──────────────
def _laplacian(G):
    """4-neighbor Laplacian via symmetric edge replication (no roll-around)."""
    P = np.pad(G, 1, mode="edge")
    return 4 * G - P[:-2, 1:-1] - P[2:, 1:-1] - P[1:-1, :-2] - P[1:-1, 2:]

def train(cal_pts, epochs=20000, verbose=300):
    N_total = len(cal_pts)
    rng     = np.random.default_rng(VAL_SEED)
    perm    = rng.permutation(N_total)
    n_val   = int(round(VAL_FRAC * N_total)) if (VAL_FRAC > 0 and N_total >= 10) else 0
    val_idx   = set(perm[:n_val].tolist())
    train_set = [s for i, s in enumerate(cal_pts) if i not in val_idx]
    val_set   = [s for i, s in enumerate(cal_pts) if i in val_idx]
    N = len(train_set)
    print(f"\n[TRAIN] N_total={N_total}  train={N}  val={len(val_set)}  epochs={epochs}")

    GammaL, GammaR, d = defaults()
    disps   = [abs(s[0] - s[2]) for s in train_set]
    mean_dn = float(np.mean(disps)) * (G0_INIT / CX)
    mean_Z  = float(np.mean([s[4] for s in train_set]))
    d[0]    = max(mean_Z * mean_dn, DX_MIN)
    d[1]    = 0.0                                              # dy initialized at 0, will train
    print(f"[TRAIN] init  dx={d[0]:.4f}m  dy={d[1]:.4f}m  mean_Z={mean_Z:.2f}m  Γ_init={G0_INIT:.4f}  "
          f"λ_anchor={LAMBDA_ANCHOR}  λ_smooth={LAMBDA_SMOOTH}  freeze_inner={FREEZE_INNER}")

    lr_g, lr_dx, lr_dy = 1e-1, 1e-3, 1e-3

    # ── Adam state (β1, β2, ε) — separate buffers per param tensor ──
    ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
    m_GL = np.zeros_like(GammaL); v_GL = np.zeros_like(GammaL)
    m_GR = np.zeros_like(GammaR); v_GR = np.zeros_like(GammaR)
    m_d  = np.zeros(3);           v_d  = np.zeros(3)
    adam_t = 0

    # ── ReduceLROnPlateau: scale all three LRs by FACTOR after PATIENCE
    #    epochs without improvement of >MIN_DELTA on the tracked metric.
    PLATEAU_FACTOR   = 0.5
    PLATEAU_PATIENCE = 800
    PLATEAU_MIN_DELTA = 1e-5
    PLATEAU_COOLDOWN = 200
    PLATEAU_LR_FLOOR = 1e-4
    plateau_best     = float("inf")
    plateau_bad      = 0
    plateau_cool     = 0

    # ── Early stopping: bail out if MAE hasn't improved for this many epochs
    EARLY_STOP_PATIENCE = 3000
    early_bad = 0

    def eval_mae(samples):
        if not samples: return float("nan")
        tot = 0.0; n = 0
        for s in samples:
            Zp, *_ = triangulate(*s[:4], GammaL, GammaR, d)
            if not math.isnan(Zp):
                tot += abs(s[4] - Zp); n += 1
        return tot / n if n else float("nan")

    best_metric, best_state = float("inf"), None
    for ep in range(epochs):
        aGmL = np.zeros((GRID_ROWS, GRID_COLS, 2))
        aGmR = np.zeros((GRID_ROWS, GRID_COLS, 2))
        ad   = np.zeros(3)
        total = abs_err = 0.0;  n_valid = 0

        for s in train_set:
            loss, gGmL, gGmR, gd = forward_grads(*s, GammaL, GammaR, d)
            total += loss
            aGmL  += gGmL; aGmR += gGmR; ad += gd
            Zp, *_ = triangulate(*s[:4], GammaL, GammaR, d)
            if not math.isnan(Zp):
                abs_err += abs(s[4] - Zp); n_valid += 1

        aGmL /= N; aGmR /= N; ad /= N

        # ── regularization (per-axis): anchor + 4-neighbor smoothness on each γ-plane
        if LAMBDA_ANCHOR > 0:
            aGmL += LAMBDA_ANCHOR * (GammaL - G0_INIT)
            aGmR += LAMBDA_ANCHOR * (GammaR - G0_INIT)
        if LAMBDA_SMOOTH > 0:
            aGmL[..., 0] += LAMBDA_SMOOTH * _laplacian(GammaL[..., 0])
            aGmL[..., 1] += LAMBDA_SMOOTH * _laplacian(GammaL[..., 1])
            aGmR[..., 0] += LAMBDA_SMOOTH * _laplacian(GammaR[..., 0])
            aGmR[..., 1] += LAMBDA_SMOOTH * _laplacian(GammaR[..., 1])

        # ── freeze inner columns at G0_INIT (zero their gradient) ──
        if FREEZE_INNER:
            aGmL[:, 1:3, :] = 0
            aGmR[:, 1:3, :] = 0

        for g_vec in (aGmL, aGmR, ad):
            nrm = float(np.linalg.norm(g_vec))
            if nrm > 1e3:
                g_vec *= 1e3 / nrm

        # ── Adam update ────────────────────────────────────────────
        adam_t += 1
        bc1 = 1.0 - ADAM_B1 ** adam_t
        bc2 = 1.0 - ADAM_B2 ** adam_t

        m_GL = ADAM_B1 * m_GL + (1 - ADAM_B1) * aGmL
        v_GL = ADAM_B2 * v_GL + (1 - ADAM_B2) * (aGmL * aGmL)
        GammaL -= lr_g * (m_GL / bc1) / (np.sqrt(v_GL / bc2) + ADAM_EPS)

        m_GR = ADAM_B1 * m_GR + (1 - ADAM_B1) * aGmR
        v_GR = ADAM_B2 * v_GR + (1 - ADAM_B2) * (aGmR * aGmR)
        GammaR -= lr_g * (m_GR / bc1) / (np.sqrt(v_GR / bc2) + ADAM_EPS)

        m_d = ADAM_B1 * m_d + (1 - ADAM_B1) * ad
        v_d = ADAM_B2 * v_d + (1 - ADAM_B2) * (ad * ad)
        m_d_hat = m_d / bc1
        v_d_hat = v_d / bc2
        d[0] -= lr_dx * m_d_hat[0] / (math.sqrt(v_d_hat[0]) + ADAM_EPS)
        d[1] -= lr_dy * m_d_hat[1] / (math.sqrt(v_d_hat[1]) + ADAM_EPS)
        d[2]  = 0.0

        d[0] = max(d[0], DX_MIN)
        np.clip(GammaL, 0.05, 5.0, out=GammaL)
        np.clip(GammaR, 0.05, 5.0, out=GammaR)
        if FREEZE_INNER:
            GammaL[:, 1:3, :] = G0_INIT
            GammaR[:, 1:3, :] = G0_INIT

        # ── best-checkpoint selection: val MAE if available, else train MAE ──
        train_mae_now = abs_err / n_valid if n_valid > 0 else float("nan")
        if val_set:
            metric = eval_mae(val_set)
            if math.isnan(metric):
                metric = train_mae_now if not math.isnan(train_mae_now) else total
        else:
            metric = train_mae_now if not math.isnan(train_mae_now) else total
        if metric < best_metric:
            best_metric = metric
            best_state  = (GammaL.copy(), GammaR.copy(), d.copy())
            early_bad = 0
        else:
            early_bad += 1

        # ── ReduceLROnPlateau on the same metric ──────────────────
        if plateau_cool > 0:
            plateau_cool -= 1
        elif metric < plateau_best - PLATEAU_MIN_DELTA:
            plateau_best = metric
            plateau_bad  = 0
        else:
            plateau_bad += 1
            if plateau_bad >= PLATEAU_PATIENCE:
                new_lr_g  = max(lr_g  * PLATEAU_FACTOR, PLATEAU_LR_FLOOR * 1e2)
                new_lr_dx = max(lr_dx * PLATEAU_FACTOR, PLATEAU_LR_FLOOR)
                new_lr_dy = max(lr_dy * PLATEAU_FACTOR, PLATEAU_LR_FLOOR)
                if (new_lr_g, new_lr_dx, new_lr_dy) != (lr_g, lr_dx, lr_dy):
                    print(f"  [PLATEAU] ep {ep:5d}  no improvement for "
                          f"{PLATEAU_PATIENCE} ep → lr_g {lr_g:.2e}→{new_lr_g:.2e}  "
                          f"lr_dx {lr_dx:.2e}→{new_lr_dx:.2e}  "
                          f"lr_dy {lr_dy:.2e}→{new_lr_dy:.2e}")
                    lr_g, lr_dx, lr_dy = new_lr_g, new_lr_dx, new_lr_dy
                plateau_bad  = 0
                plateau_cool = PLATEAU_COOLDOWN
                plateau_best = metric

        if early_bad >= EARLY_STOP_PATIENCE:
            print(f"  [EARLY-STOP] ep {ep:5d}  no MAE improvement for "
                  f"{EARLY_STOP_PATIENCE} ep  (best MAE = {best_metric:.4f})")
            break

        if ep % verbose == 0 or ep == epochs - 1:
            train_mae = abs_err / n_valid if n_valid > 0 else float("nan")
            val_mae   = eval_mae(val_set) if val_set else float("nan")
            print(f"  ep {ep:5d}  L={total:8.4f}  trainMAE={train_mae:.3f}m  "
                  f"valMAE={val_mae:.3f}m  dx={d[0]:+.4f}  dy={d[1]:+.5f}  "
                  f"‖ΓL‖={np.linalg.norm(GammaL):.3f}  ‖ΓR‖={np.linalg.norm(GammaR):.3f}")

    GammaL, GammaR, d = best_state
    print(f"[TRAIN] best MAE ({'val' if val_set else 'train'}) = {best_metric:.4f}m")

    def _dead(G):
        out = []
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                if abs(G[i, j, 0] - G0_INIT) < 1e-6 and abs(G[i, j, 1] - G0_INIT) < 1e-6:
                    out.append((i, j))
        return out
    deadL, deadR = _dead(GammaL), _dead(GammaR)
    print(f"[TRAIN] ΓL untouched nodes ({len(deadL)}/16): {deadL}")
    print(f"[TRAIN] ΓR untouched nodes ({len(deadR)}/16): {deadR}")
    return GammaL, GammaR, d

# ─── PERSISTENCE ──────────────────────────────────────────────
def save_calibration(GammaL, GammaR, d, path=CALIB_PATH, postcorr=None):
    """postcorr: optional (a, b, c) for Z_corr = a + b·Z + c·Z²"""
    out = {"coords": "raw pixels",
           "model": f"anisotropic_gamma_{GRID_ROWS}x{GRID_COLS}x2",
           "grid_rows": GRID_ROWS, "grid_cols": GRID_COLS,
           "gamma_L_grid": GammaL.tolist(),
           "gamma_R_grid": GammaR.tolist(),
           "baseline": {"dx": float(d[0]), "dy": float(d[1]), "dz": 0.0}}
    if postcorr is not None:
        a, b, c = postcorr
        out["postcorr_z"] = {"a": float(a), "b": float(b), "c": float(c)}
    with open(path, "w") as f: json.dump(out, f, indent=2, default=float)
    print(f"[SAVE] {path}")

def save_cal_pts(cal_pts, path=CAL_PTS_PATH):
    with open(path, "w") as f:
        json.dump([list(s) for s in cal_pts], f, indent=2, default=float)

def load_cal_pts(path=CAL_PTS_PATH):
    if not os.path.exists(path): return []
    with open(path) as f:
        content = f.read()
    if not content.strip(): return []
    return [tuple(s) for s in json.loads(content)]

def load_calibration(path=CALIB_PATH):
    with open(path) as f:
        content = f.read()
    if not content.strip():
        raise ValueError(f"{path} is empty")
    p = json.loads(content)
    # New generic grid format
    if "gamma_L_grid" in p:
        GammaL = np.array(p["gamma_L_grid"], dtype=np.float64)
        GammaR = np.array(p["gamma_R_grid"], dtype=np.float64)
    elif "gamma_L_4x4x2" in p:
        GammaL = np.array(p["gamma_L_4x4x2"], dtype=np.float64)
        GammaR = np.array(p["gamma_R_4x4x2"], dtype=np.float64)
    else:
        # Back-compat: legacy isotropic NxN → broadcast to (N,N,2)
        GL2 = np.array(p["gamma_L_4x4"], dtype=np.float64)
        GR2 = np.array(p["gamma_R_4x4"], dtype=np.float64)
        GammaL = np.stack([GL2, GL2], axis=-1)
        GammaR = np.stack([GR2, GR2], axis=-1)
    d = np.array([p["baseline"]["dx"], p["baseline"].get("dy", 0.0), 0.0])
    pc = p.get("postcorr_z")
    postcorr = (pc["a"], pc["b"], pc["c"]) if pc else None
    return GammaL, GammaR, d, postcorr

# ─── LOCAL FUNDAMENTAL MATRIX  (Step 1: per-click, K_L from γ at click) ──────
def _K_inv(gx, gy):
    """Anisotropic K⁻¹ s.t. [u,v,1] → [(u-cx)·gx/cx, (v-cy)·gy/cy, 1]."""
    return np.array([[gx/CX, 0.0,   -gx],
                     [0.0,   gy/CY, -gy],
                     [0.0,   0.0,    1.0]])

def fundamental_matrix_local(uL, vL, GammaL, GammaR, d):
    """K_L sampled at (uL,vL); K_R from MEAN ΓR (right pixel unknown).
       R = I. E = [d]×.  F = K_R⁻ᵀ · E · K_L⁻¹.   Then unit-Frobenius-normalize."""
    gLx, _, _ = bilinear(uL, vL, GammaL[..., 0])
    gLy, _, _ = bilinear(uL, vL, GammaL[..., 1])
    gRx_mean = float(np.mean(GammaR[..., 0]))
    gRy_mean = float(np.mean(GammaR[..., 1]))
    KL_inv = _K_inv(gLx, gLy)
    KR_inv = _K_inv(gRx_mean, gRy_mean)
    dx_, dy_ = float(d[0]), float(d[1])
    dz_ = float(d[2]) if len(d) > 2 else 0.0
    E = np.array([[0.0, -dz_,  dy_],
                  [dz_,  0.0, -dx_],
                  [-dy_, dx_,  0.0]])
    F = KR_inv.T @ E @ KL_inv
    n = float(np.linalg.norm(F))
    return F / n if n > 1e-12 else F

def epipolar_line(F_local, uL, vL):
    """l = F·[uL,vL,1]ᵀ → (a, b, c) for ax + by + c = 0 in the right image."""
    line = F_local @ np.array([float(uL), float(vL), 1.0])
    return float(line[0]), float(line[1]), float(line[2])

# ─── 1-D EPIPOLAR-GUIDED NCC MATCH  (Step 3) ─────────────────────────────────
def _bilinear_patch(img, cx, cy, patch):
    """patch×patch grid sampled from `img` at sub-pixel (cx,cy). Vectorized."""
    h = patch // 2
    H, W = img.shape
    ys = cy + np.arange(-h, h + 1, dtype=np.float64)[:, None]
    xs = cx + np.arange(-h, h + 1, dtype=np.float64)[None, :]
    if xs.min() < 0 or ys.min() < 0 or xs.max() > W - 1 or ys.max() > H - 1:
        return None
    x0 = np.floor(xs).astype(np.int32); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(ys).astype(np.int32); y1 = np.clip(y0 + 1, 0, H - 1)
    tx = xs - x0; ty = ys - y0
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    return ((1-tx)*(1-ty)*Ia + tx*(1-ty)*Ib +
            (1-tx)*ty   *Ic + tx*ty   *Id)

def epipolar_guided_match(left_gray, right_gray, uL, vL, F_local,
                          patch=15, search_range=300):
    """Sweep uR ∈ [uL−search_range, uL] in 1-px steps; vR computed strictly on
       line a·uR + b·vR + c = 0. Zero-mean NCC at each sub-pixel position.
       After the integer peak, parabolic-refine uR to sub-pixel and re-score.
       Returns (uR_hint, vR_hint, score) or (None, None, -1.0)."""
    a, b, c = epipolar_line(F_local, uL, vL)
    if abs(b) < 1e-9:
        return None, None, -1.0

    h = patch // 2
    HL, WL = left_gray.shape
    uLi, vLi = int(round(uL)), int(round(vL))
    if uLi - h < 0 or vLi - h < 0 or uLi + h + 1 > WL or vLi + h + 1 > HL:
        return None, None, -1.0
    tmpl = left_gray[vLi-h:vLi+h+1, uLi-h:uLi+h+1].astype(np.float64)
    t = tmpl - tmpl.mean()
    t_norm = float(np.sqrt((t*t).sum())) + 1e-9

    HR, WR = right_gray.shape

    def score_at(uR_f):
        """ZNCC at sub-pixel uR (float); vR pinned on epi-line. None if OOB."""
        vR_f = -(a * uR_f + c) / b
        if vR_f < h or vR_f > HR - h - 1:
            return None, None
        p = _bilinear_patch(right_gray, float(uR_f), float(vR_f), patch)
        if p is None:
            return None, None
        pm = p - p.mean()
        pn = float(np.sqrt((pm*pm).sum())) + 1e-9
        return float((pm * t).sum() / (pn * t_norm)), float(vR_f)

    uR_lo = max(h, int(round(uL - search_range)))
    uR_hi = min(WR - h - 1, int(round(uL)))

    # ── 1-px sweep, keep scores so we can do parabolic refinement at the peak
    scores = {}
    best_score, best_uR_int, best_vR = -2.0, None, None
    for uR in range(uR_lo, uR_hi + 1):
        s, vR = score_at(float(uR))
        if s is None:
            continue
        scores[uR] = s
        if s > best_score:
            best_score = s
            best_uR_int = uR
            best_vR = vR
    if best_uR_int is None:
        return None, None, -1.0

    # ── Parabolic sub-pixel refinement on the NCC peak ─────────────────────
    s_m = scores.get(best_uR_int - 1)
    s_p = scores.get(best_uR_int + 1)
    s_0 = best_score
    if s_m is not None and s_p is not None:
        denom = 2.0 * (s_m - 2.0 * s_0 + s_p)
        if abs(denom) > 1e-9:
            delta = (s_m - s_p) / denom
            if -1.0 < delta < 1.0:
                uR_refined = float(best_uR_int) + delta
                s_ref, vR_ref = score_at(uR_refined)
                if s_ref is not None and s_ref >= s_0 - 1e-6:
                    return uR_refined, vR_ref, s_ref
    return float(best_uR_int), best_vR, best_score

def snap_to_epipolar(uR, vR, F_local, uL, vL):
    """Orthogonal projection of (uR,vR) onto the epipolar line for (uL,vL)."""
    a, b, c = epipolar_line(F_local, uL, vL)
    n2 = a*a + b*b
    if n2 < 1e-12:
        return float(uR), float(vR)
    t = (a*uR + b*vR + c) / n2
    return float(uR) - a*t, float(vR) - b*t

# ─── DEPTH-CIRCLE PIXEL PREDICTION  (Step 5: geometric, no iteration) ────────
def predict_right_pixel_at_Z(uL, vL, Z_target, GammaL, GammaR, d):
    """P_L = Z·v_L  →  v_R_ideal = (P_L − d)/Z  →  pixel via mean ΓR."""
    vL_d, _ = build_ray(uL, vL, GammaL)
    P_L = Z_target * vL_d
    X = P_L - d
    Zr = float(X[2])
    if Zr <= 1e-6:
        return None
    xn, yn = float(X[0]) / Zr, float(X[1]) / Zr
    gRx_mean = float(np.mean(GammaR[..., 0]))
    gRy_mean = float(np.mean(GammaR[..., 1]))
    if gRx_mean <= 1e-9 or gRy_mean <= 1e-9:
        return None
    return CX + xn * CX / gRx_mean, CY + yn * CY / gRy_mean

# ─── RESIDUAL POLYNOMIAL POST-CORRECTION ─────────────────────────────────────
# After geometric triangulation, fit Z_true = a + b·Z_pred + c·Z_pred²
# on the calibration set to absorb any residual systematic bias.  3 OLS params.
def fit_postcorrection(cal_pts, GammaL, GammaR, d):
    Zp_list, Zt_list = [], []
    for s in cal_pts:
        Zp, *_ = triangulate(*s[:4], GammaL, GammaR, d)
        if not math.isnan(Zp):
            Zp_list.append(Zp); Zt_list.append(s[4])
    if len(Zp_list) < 4:
        return (0.0, 1.0, 0.0)                  # identity (no correction)
    Zp = np.array(Zp_list); Zt = np.array(Zt_list)
    A  = np.column_stack([np.ones_like(Zp), Zp, Zp * Zp])
    coefs, *_ = np.linalg.lstsq(A, Zt, rcond=None)
    return tuple(float(c) for c in coefs)        # (a, b, c)

def apply_postcorrection(Z_pred, postcorr):
    if postcorr is None:
        return Z_pred
    a, b, c = postcorr
    return a + b * Z_pred + c * Z_pred * Z_pred

# ─── UNIFIED INTERACTIVE LOOP ─────────────────────────────────
def run(left_path, right_path):
    import pygame
    pygame.init(); pygame.font.init()

    # ── parameters ──────────────────────────────────────────────────────────
    GammaL, GammaR, d = defaults()
    postcorr   = None
    calibrated = False
    if os.path.exists(CALIB_PATH):
        try:
            GammaL, GammaR, d, postcorr = load_calibration()
            calibrated = True
            print(f"[INIT] loaded existing {CALIB_PATH}  (postcorr={postcorr})")
        except Exception as e:
            print(f"[INIT] load failed ({e}); starting uncalibrated")

    # ── window ───────────────────────────────────────────────────────────────
    HUD_H = 110
    WIN_W, WIN_H = 2*DISP_W, DISP_H + HUD_H
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("37-param hybrid stereo (Γ + d + R) | click=pick  K=add U=undo R=clear C=reset T=swap")

    # ── images (CLAHE applied once for NCC/SSD/LK stability) ────────────────
    def load_img(p):
        s = pygame.image.load(p).convert()
        if s.get_size() != (IMG_W, IMG_H):
            s = pygame.transform.scale(s, (IMG_W, IMG_H))
        return s
    left_img   = load_img(left_path)
    right_img  = load_img(right_path)

    def to_gray(surf):
        # pygame.surfarray.array3d returns (W, H, 3); transpose to (H, W, 3)
        a = pygame.surfarray.array3d(surf).transpose(1, 0, 2).astype(np.float32)
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    left_gray  = to_gray(left_img)
    right_gray = to_gray(right_img)

    font    = pygame.font.SysFont("consolas", 13)
    bigfont = pygame.font.SysFont("consolas", 17, bold=True)

    # ── state ────────────────────────────────────────────────────────────────
    cal_pts      = load_cal_pts()         # (uL,vL,uR,vR,Z) — persisted to cal_pts.json
    print(f"[INIT] loaded {len(cal_pts)} cal_pts from {CAL_PTS_PATH}")
    measurements = []                     # (uL,vL,uR,vR,Z)
    pending      = {"Z": None, "uL": None, "vL": None}
    mode         = "measure"              # "measure" | "calib_wait_L" | "calib_wait_R"
    current_L    = None
    current_R    = None                   # final R (after smart-snap / epi-snap)
    current_Z    = None
    current_err  = None
    F_local      = None                   # per-click F (Step 1)
    uR_hint      = None                   # NCC hint pixel on right (Step 3)
    vR_hint      = None
    hint_score   = 0.0
    SNAP_PX      = 8                      # Step 4 smart-snap radius

    zooms  = [1.0, 1.0]
    pans   = [[0, 0], [0, 0]]
    drag   = [False, False]
    drag_0 = (0, 0)
    drag_p = [[0, 0], [0, 0]]

    # ── helpers ──────────────────────────────────────────────────────────────
    SX = DISP_W / IMG_W    # image→display scale X
    SY = DISP_H / IMG_H    # image→display scale Y

    def panel_of(mx, my):
        if my >= DISP_H: return None
        return 0 if mx < DISP_W else 1

    def screen_to_img(mx, my, p):
        lx = mx - p*DISP_W
        return (lx - pans[p][0]) / (zooms[p] * SX), (my - pans[p][1]) / (zooms[p] * SY)

    def img_to_screen(ix, iy, p):
        return (ix * SX * zooms[p] + pans[p][0] + p*DISP_W,
                iy * SY * zooms[p] + pans[p][1])

    def clamp_pan(p):
        sw, sh = int(DISP_W*zooms[p]), int(DISP_H*zooms[p])
        pans[p][0] = min(0, max(pans[p][0], DISP_W - sw))
        pans[p][1] = min(0, max(pans[p][1], DISP_H - sh))

    def draw_panel(surf, p):
        s = pygame.transform.scale(surf, (int(DISP_W*zooms[p]), int(DISP_H*zooms[p])))
        screen.set_clip(pygame.Rect(p*DISP_W, 0, DISP_W, DISP_H))
        screen.blit(s, (p*DISP_W + pans[p][0], pans[p][1]))

    def recompute_depth():
        nonlocal current_Z, current_err
        if current_L is None or current_R is None:
            current_Z = current_err = None; return
        Z, _, _, err = triangulate(current_L[0], current_L[1],
                                    current_R[0], current_R[1],
                                    GammaL, GammaR, d)
        if not math.isnan(Z):
            Z = apply_postcorrection(Z, postcorr)
        current_Z   = None if math.isnan(Z) else Z
        current_err = err

    def maybe_auto_train():
        nonlocal GammaL, GammaR, d, postcorr, calibrated
        if len(cal_pts) < MIN_CAL_PTS: return
        # flash a training banner
        screen.fill((50, 25, 25), pygame.Rect(0, DISP_H, WIN_W, HUD_H))
        msg = bigfont.render(
            f"  TRAINING on {len(cal_pts)} samples — see terminal …",
            True, (255, 200, 100))
        screen.blit(msg, (10, DISP_H + 30))
        pygame.display.flip()
        GammaL, GammaR, d = train(cal_pts)
        postcorr = fit_postcorrection(cal_pts, GammaL, GammaR, d)
        save_calibration(GammaL, GammaR, d, postcorr=postcorr)
        calibrated = True

    def ask_distance_terminal():
        print(f"\n[K-MODE] calibration point {len(cal_pts)+1}"
              f"   (need {MIN_CAL_PTS} total)")
        try:
            raw = input("  Enter Real Distance (m): ").strip()
            Z = float(raw)
            if Z <= 0: raise ValueError
            return Z
        except Exception:
            print("  invalid distance — K-mode cancelled")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════
    clock   = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # ── KEYBOARD ─────────────────────────────────────────────────────
            elif ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_k:
                    Z = ask_distance_terminal()
                    if Z is not None:
                        pending.update({"Z": Z, "uL": None, "vL": None})
                        mode = "calib_wait_L"
                        print(f"  → click LEFT then RIGHT image for Z = {Z:.2f} m")
                elif ev.key == pygame.K_c:
                    cal_pts.clear()
                    save_cal_pts(cal_pts)
                    GammaL, GammaR, d = defaults()
                    postcorr = None
                    calibrated = False
                    current_L = current_R = current_Z = current_err = None
                    F_local = None; uR_hint = vR_hint = None; hint_score = 0.0
                    mode = "measure";  pending["Z"] = None
                    print("[C] cal_pts cleared; model reset to defaults")
                elif ev.key == pygame.K_r:
                    measurements.clear()
                    current_L = current_R = current_Z = current_err = None
                    print("[R] measurements cleared")
                elif ev.key == pygame.K_u:
                    if mode in ("calib_wait_L", "calib_wait_R"):
                        mode = "measure";  pending["Z"] = None
                        print("[U] cancelled pending K-mode entry")
                    elif cal_pts:
                        removed = cal_pts.pop()
                        save_cal_pts(cal_pts)
                        print(f"[U] removed cal_pt {len(cal_pts)+1}: "
                              f"L=({removed[0]},{removed[1]}) R=({removed[2]},{removed[3]}) Z={removed[4]:.2f}m  "
                              f"({len(cal_pts)} remaining)")
                elif ev.key == pygame.K_t:
                    left_img,  right_img  = right_img,  left_img
                    left_gray, right_gray = right_gray, left_gray
                    current_L = current_R = current_Z = current_err = None
                    print("[T] swapped L/R images")

            # ── WHEEL = zoom panel under cursor ──────────────────────────────
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                p = panel_of(mx, my)
                if p is not None:
                    old = zooms[p]
                    zooms[p] = max(1.0, min(8.0, old * (1.15 if ev.y > 0 else 1/1.15)))
                    lx = mx - p*DISP_W
                    pans[p][0] = int(lx - (lx - pans[p][0]) * zooms[p] / old)
                    pans[p][1] = int(my - (my - pans[p][1]) * zooms[p] / old)
                    clamp_pan(p)

            # ── MOUSE BUTTONS ────────────────────────────────────────────────
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                p = panel_of(mx, my)
                if p is None: continue
                if ev.button == 3:                                # right-drag pan
                    drag[p]    = True
                    drag_0     = (mx, my)
                    drag_p[p]  = pans[p].copy()
                elif ev.button == 1:                              # left-click → pick point
                    ix, iy = screen_to_img(mx, my, p)
                    ix, iy = int(round(ix)), int(round(iy))
                    if not (0 <= ix < IMG_W and 0 <= iy < IMG_H): continue
                    if p == 0:
                        # LEFT click: pick L, clear R, compute F_local + NCC hint
                        current_L = (ix, iy)
                        current_R = current_Z = current_err = None
                        if calibrated:
                            F_local = fundamental_matrix_local(ix, iy, GammaL, GammaR, d)
                            uR_hint, vR_hint, hint_score = epipolar_guided_match(
                                left_gray, right_gray, ix, iy, F_local,
                                patch=15, search_range=300)
                            if uR_hint is not None:
                                print(f"[PICK-L] L=({ix},{iy})  hint R≈"
                                      f"({uR_hint:.1f},{vR_hint:.1f})  NCC={hint_score:+.3f}")
                            else:
                                print(f"[PICK-L] L=({ix},{iy})  no NCC hint (line off-image)")
                        else:
                            F_local = None
                            uR_hint = vR_hint = None; hint_score = 0.0
                            print(f"[PICK-L] L=({ix},{iy})")
                    else:
                        # RIGHT click: pick R; needs L first
                        if current_L is None:
                            print("[PICK-R] click LEFT image first")
                            continue
                        uL_, vL_ = current_L
                        # Step 4: smart-snap (only when calibrated and we have a hint)
                        if calibrated and F_local is not None and uR_hint is not None:
                            dist = math.hypot(ix - uR_hint, iy - vR_hint)
                            if dist < SNAP_PX:
                                final_uR, final_vR = uR_hint, vR_hint
                                print(f"[PICK-R] R=({ix},{iy})  SMART-SNAP "
                                      f"({dist:.1f}px) → ({final_uR:.1f},{final_vR:.1f})")
                            else:
                                final_uR, final_vR = snap_to_epipolar(ix, iy, F_local, uL_, vL_)
                                print(f"[PICK-R] R=({ix},{iy})  manual ({dist:.1f}px) → "
                                      f"epi-snap ({final_uR:.1f},{final_vR:.1f})")
                        else:
                            final_uR, final_vR = float(ix), float(iy)
                            print(f"[PICK-R] R=({ix},{iy})  (uncalibrated, raw)")

                        current_R = (final_uR, final_vR)

                        if mode == "calib_wait_L" and pending["Z"] is not None:
                            Z_ = pending["Z"]
                            # Save sub-pixel correspondence (smart-snap / epi-snap output)
                            uR_save, vR_save = float(final_uR), float(final_vR)
                            if cal_pts and cal_pts[-1][:2] == (uL_, vL_) \
                                    and abs(cal_pts[-1][4] - Z_) < 1e-9:
                                cal_pts[-1] = (uL_, vL_, uR_save, vR_save, Z_)
                                save_cal_pts(cal_pts)
                                print(f"  [K] cal_pt R overridden → ({uR_save:.2f},{vR_save:.2f})")
                            else:
                                cal_pts.append((uL_, vL_, uR_save, vR_save, Z_))
                                save_cal_pts(cal_pts)
                                print(f"  [K] cal_pt added  L=({uL_},{vL_}) "
                                      f"R=({uR_save:.2f},{vR_save:.2f}) Z={Z_:.2f}m   "
                                      f"cal_pts={len(cal_pts)}/{MIN_CAL_PTS}")
                            mode = "measure";  pending["Z"] = None
                            maybe_auto_train()
                        elif calibrated:
                            recompute_depth()
                            if current_Z is not None:
                                measurements.append((uL_, vL_, final_uR, final_vR, current_Z))

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 3:
                    drag[0] = drag[1] = False

            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                for p in (0, 1):
                    if drag[p]:
                        pans[p][0] = drag_p[p][0] + (mx - drag_0[0])
                        pans[p][1] = drag_p[p][1] + (my - drag_0[1])
                        clamp_pan(p)

        # ══════════════════════════════════════════════════════════════════════
        # DRAW
        # ══════════════════════════════════════════════════════════════════════
        screen.fill((18, 18, 22))
        draw_panel(left_img,  0)
        draw_panel(right_img, 1)

        # ── LEFT panel overlays ──────────────────────────────────────────────
        screen.set_clip(pygame.Rect(0, 0, DISP_W, DISP_H))
        for u in (CENTER_LO, CENTER_HI):                     # zone boundaries
            sx, _ = img_to_screen(u, 0, 0)
            if 0 <= sx < DISP_W:
                pygame.draw.line(screen, (70, 70, 110), (int(sx), 0), (int(sx), DISP_H), 1)
        for (uL, vL, *_ ) in cal_pts:
            sx, sy = img_to_screen(uL, vL, 0)
            pygame.draw.circle(screen, (90, 150, 255), (int(sx), int(sy)), 3)
        if current_L is not None:
            sx, sy = img_to_screen(current_L[0], current_L[1], 0)
            pygame.draw.circle(screen, (50, 255, 90), (int(sx), int(sy)), 6, 2)
            pygame.draw.line(screen, (50, 255, 90), (int(sx)-12, int(sy)), (int(sx)+12, int(sy)), 1)
            pygame.draw.line(screen, (50, 255, 90), (int(sx), int(sy)-12), (int(sx), int(sy)+12), 1)

        # ── RIGHT panel overlays ─────────────────────────────────────────────
        screen.set_clip(pygame.Rect(DISP_W, 0, DISP_W, DISP_H))
        for u in (CENTER_LO, CENTER_HI):
            sx, _ = img_to_screen(u, 0, 1)
            if DISP_W <= sx < 2*DISP_W:
                pygame.draw.line(screen, (70, 70, 110), (int(sx), 0), (int(sx), DISP_H), 1)
        for (_, _, uR, vR, _) in cal_pts:
            sx, sy = img_to_screen(uR, vR, 1)
            pygame.draw.circle(screen, (90, 150, 255), (int(sx), int(sy)), 3)

        # Step 2: straight epipolar line ax+by+c=0 in cyan + Step 5 depth circles
        if current_L is not None:
            uL_, vL_ = current_L
            if calibrated and F_local is not None:
                a_, b_, c_ = epipolar_line(F_local, uL_, vL_)
                if abs(b_) > 1e-9:
                    pts = []
                    for u_img in (0.0, float(IMG_W - 1)):
                        v_img = -(a_ * u_img + c_) / b_
                        sx, sy = img_to_screen(u_img, v_img, 1)
                        pts.append((int(sx), int(sy)))
                    pygame.draw.line(screen, (0, 220, 220), pts[0], pts[1], 1)
                elif abs(a_) > 1e-9:
                    u_img = -c_ / a_
                    sx, _ = img_to_screen(u_img, 0, 1)
                    pygame.draw.line(screen, (0, 220, 220),
                                     (int(sx), 0), (int(sx), DISP_H), 1)

                # Step 3: NCC hint as a green crosshair
                if uR_hint is not None:
                    sx, sy = img_to_screen(uR_hint, vR_hint, 1)
                    pygame.draw.line(screen, (50, 255, 90),
                                     (int(sx)-14, int(sy)), (int(sx)+14, int(sy)), 2)
                    pygame.draw.line(screen, (50, 255, 90),
                                     (int(sx), int(sy)-14), (int(sx), int(sy)+14), 2)
                    pygame.draw.circle(screen, (50, 255, 90), (int(sx), int(sy)), 4, 1)
            else:
                _, sy = img_to_screen(0, vL_, 1)
                pygame.draw.line(screen, (70, 160, 90),
                                 (DISP_W, int(sy)), (2*DISP_W, int(sy)), 1)

        # Final triangulation point (yellow ok / red error)
        if current_R is not None:
            sx, sy = img_to_screen(current_R[0], current_R[1], 1)
            col = (255, 60, 60) if current_err else (255, 255, 60)
            pygame.draw.circle(screen, col, (int(sx), int(sy)), 6, 2)
            pygame.draw.line(screen, col, (int(sx)-12, int(sy)), (int(sx)+12, int(sy)), 1)
            pygame.draw.line(screen, col, (int(sx), int(sy)-12), (int(sx), int(sy)+12), 1)

        screen.set_clip(None)
        pygame.draw.line(screen, (60, 60, 70), (DISP_W, 0), (DISP_W, DISP_H), 1)

        # ── HUD ──────────────────────────────────────────────────────────────
        screen.fill((12, 12, 18), pygame.Rect(0, DISP_H, WIN_W, HUD_H))
        status = "CALIBRATED" if calibrated else f"UNCALIBRATED ({len(cal_pts)}/{MIN_CAL_PTS})"
        hud = [f"[{status}]  mode={mode}   d=[{d[0]:+.4f},{d[1]:+.4f},0]   "
               f"zoomL={zooms[0]:.2f}× zoomR={zooms[1]:.2f}×"]
        if current_L is not None:
            z_ = "center" if CENTER_LO <= current_L[0] <= CENTER_HI else "edge"
            hud.append(f"L=({current_L[0]:3d},{current_L[1]:3d}) zone={z_}")
        if current_R is not None:
            z_ = "center" if CENTER_LO <= current_R[0] <= CENTER_HI else "edge"
            hud.append(f"R=({current_R[0]:7.2f},{current_R[1]:7.2f}) zone={z_}")
        if current_Z is not None:
            hud.append(f">>> Z = {current_Z:.3f} m   ({len(measurements)} stored)")
        elif current_err:
            hud.append(f"triangulate: {current_err}")
        if   mode == "calib_wait_L": hud.append(f"K-mode → click LEFT then RIGHT for Z={pending['Z']:.2f}m")
        hud.append("L-click=pick  K=add-cal  U=undo-cal  C=clear+reset  R=clear-meas  T=swap-L/R  wheel=zoom  rclick-drag=pan  Esc=quit")
        for i, line in enumerate(hud[:6]):
            screen.blit(font.render(line, True, (220, 220, 225)), (8, DISP_H + 6 + i*16))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

# ─── ENTRY POINT ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_math.py <left_image> <right_image>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
