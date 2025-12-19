#%%
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import random
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from src.encoders_decoders import *
from src.losses import *
from src.useful_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# This script contains the main training loops.
# It saves a dictionary of N_trials models trained with different target rates, together with the training history
# The saved models can then be analyzed.
# %%
def train_Rt(enc,dec,q,x_data,x_test,opt,Rt,N_EPOCHS=500,lr_b = 0.1):
    # Train parameters of VAE (enc + dec) stored in opt at a given target rate
    # by minimizing -ELBO (D+R).
    history = { "loss" : [],
                "distortion" : [],
                "rate" : [],
                "beta" : [1] ,
                "distortion_test" : [],
                "rate_test" : []}

    for e in range(N_EPOCHS):
        lav = dav = rav = 0
        beta = history["beta"][-1]
        for x_ in x_data:
            rate = q(enc,x_)
            distortion = distortion_analytical_linear(x_,enc,dec,q.r_all)
            loss =  distortion +beta*rate
            opt.zero_grad()
            loss.backward()
            opt.step()
            lav += distortion + rate
            dav += distortion
            rav += rate
            if torch.isnan(loss):
                break;
        history["loss"].append(lav.item()/len(x_data))
        history["rate"].append(rav.item()/len(x_data))
        history["distortion"].append(dav.item()/len(x_data))
        
        #Update constraint
        beta += lr_b*(history["rate"][-1]-Rt)
        beta = beta if beta>0 else 0
        history["beta"].append(beta)
        if not e % 100:
            # Test decoder on larger set of stimuli
            history["distortion_test"].append(distortion_analytical_linear(x_test,enc,dec,q.r_all).mean().item())
            history["rate_test"].append(q(enc,x_test).mean().item())
        print(f'Epoch: {e} ||Rate: {history["rate"][-1]}||',
            f'ELBO:{history["loss"][-1]}||',
            f'Distortion: {history["distortion"][-1]}||Beta = {history["beta"][-1]}')
    history["beta"].pop()
    return history

def vary_R(RtVec, x_test, p_x, optimizer_name="adamw"):
    resume = {}

    # ✅ 샘플링은 반드시 결과를 변수에 받기
    with torch.no_grad():
        x_samples = p_x.sample((N_SAMPLES,))[:, None].to(device)

    x_sorted, _ = x_samples.sort(dim=0)

    # 🔑 NumPy/encoder용은 CPU로
    x_sorted_cpu = x_sorted.detach().cpu()

    x_min, x_max = x_sorted_cpu[0].item(), x_sorted_cpu[-1].item()

    x_data = DataLoader(x_samples, batch_size=BATCH_SIZE, shuffle=True)

    for Rt in RtVec:
        lr = 1e-4 if Rt < 1.5 else 1e-3
        print(f"Rate = {Rt} || lr = {lr}")
        enc = BernoulliEncoder(
                N,
                x_min - 1,
                x_max + 1,
                x_sorted_cpu,   # 🔑 CPU tensor
                w=2
            ).to(device)

        #enc = BernoulliEncoder(N, x_min-1, x_max+1, x_sorted, w=2).to(device)
        dec = MLPDecoder(N, M).to(device)
        q = rate_ising(N).to(device)

        # 🔥 핵심: r_all 같은 상수 텐서도 같이 이동
        if hasattr(q, "r_all"):
            q.r_all = q.r_all.to(device)

        q.J.register_hook(lambda grad: grad.fill_diagonal_(0))

        #q = rate_ising(N).to(device)
        #q.J.register_hook(lambda grad: grad.fill_diagonal_(0))

        params = list(enc.parameters()) + list(dec.parameters()) + list(q.parameters())
        opt = make_optimizer(optimizer_name, params, lr)
        print("Optimizer:", optimizer_name, "| class:", opt.__class__)

        history = train_Rt(enc, dec, q, x_data, x_test, opt, Rt,
                           N_EPOCHS=N_EPOCHS, lr_b=0.1)

        resume[Rt] = {
            'encode': enc.state_dict(),
            'decoder': dec.state_dict(),
            'q': q.state_dict(),
            'history': history,
            'lr': lr
        }
    return resume

def make_optimizer(opt_name, params, lr, weight_decay=0.0):
    opt_name = opt_name.lower()

    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # "AdamGrad"는 보통 AMSGrad를 의미하는 경우가 많아서 이렇게 구현
    if opt_name in ["adamgrad", "amsgrad"]:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)

    if opt_name == "sgd":
        # (권장) momentum 사용. nesterov는 취향.
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)

    if opt_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, momentum=0.0, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {opt_name}")

# ---------------------------
# Load
# ---------------------------
def load_results(path):
    obj = torch.load(path, map_location="cpu")
    # 기대: dict(optimizer -> list(trials))
    return obj

# ---------------------------
# Helpers
# ---------------------------
def _sorted_Rts_from_results(all_results, optimizer=None):
    # Rt key들이 float일 수도 있고 numpy float일 수도 있음 -> float로 통일
    if optimizer is None:
        optimizer = list(all_results.keys())[0]
    trial0 = all_results[optimizer][0]
    rts = list(trial0.keys())
    rts = [float(r) for r in rts]
    return sorted(rts)

def _get_history(all_results, optimizer, trial_idx, Rt):
    # Rt float 매칭이 애매할 수 있으니 "가장 가까운 키"를 찾는다
    trial = all_results[optimizer][trial_idx]
    keys = [float(k) for k in trial.keys()]
    Rt = float(Rt)
    closest = min(keys, key=lambda x: abs(x - Rt))
    entry = trial[closest]
    return entry["history"]

def _history_to_array(hist, metric):
    # hist[metric] is list; beta는 보통 epoch+1 길이였는데 코드에서 pop 처리했으니 epoch와 동일 길이 기대
    arr = np.array(hist[metric], dtype=float)
    return arr

def aggregate_metric(all_results, optimizer, Rt, metric):
    """Return epochs, mean, std across trials for a metric at a given Rt."""
    n_trials = len(all_results[optimizer])
    series = []
    min_len = None

    for t in range(n_trials):
        hist = _get_history(all_results, optimizer, t, Rt)
        arr = _history_to_array(hist, metric)
        if min_len is None:
            min_len = len(arr)
        else:
            min_len = min(min_len, len(arr))
        series.append(arr)

    # 길이가 다르면 최소 길이에 맞춰 자름
    series = np.stack([s[:min_len] for s in series], axis=0)  # (trials, epochs)
    mean = series.mean(axis=0)
    std = series.std(axis=0)
    epochs = np.arange(min_len)
    return epochs, mean, std

# ---------------------------
# Plot 1: "한 Rt"에서 optimizer 비교 (epoch별 비교 플롯)
# ---------------------------
def plot_epochwise_compare(all_results, Rt, metrics=("loss", "distortion", "rate"), 
                           optimizers=None, shade=True, title_prefix=""):
    if optimizers is None:
        optimizers = list(all_results.keys())

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for opt in optimizers:
            epochs, mean, std = aggregate_metric(all_results, opt, Rt, metric)
            plt.plot(epochs, mean, label=f"{opt}")
            if shade:
                plt.fill_between(epochs, mean-std, mean+std, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{title_prefix}Rt={float(Rt):.3g} | metric={metric}")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ---------------------------
# Plot 2: optimizer별로 Rt 여러 개를 한 번에 (각 metric별로 Rt curve 여러 개)
# ---------------------------
def plot_optimizer_with_multiple_Rts(all_results, optimizer, Rts=None,
                                     metrics=("loss", "distortion", "rate"),
                                     max_Rts=6, shade=False):
    if Rts is None:
        Rts = _sorted_Rts_from_results(all_results, optimizer)

    # 너무 많으면 일부만 뽑기
    if len(Rts) > max_Rts:
        idx = np.linspace(0, len(Rts)-1, max_Rts).astype(int)
        Rts = [Rts[i] for i in idx]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for Rt in Rts:
            epochs, mean, std = aggregate_metric(all_results, optimizer, Rt, metric)
            plt.plot(epochs, mean, label=f"Rt={float(Rt):.2g}")
            if shade:
                plt.fill_between(epochs, mean-std, mean+std, alpha=0.15)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{optimizer} | metric={metric} | (multiple Rt)")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ---------------------------
# Plot 3: "수렴 속도"를 rate-constraint 관점으로 보기: |rate - Rt|
# ---------------------------
def plot_rate_constraint_gap(all_results, Rt, optimizers=None, shade=True):
    if optimizers is None:
        optimizers = list(all_results.keys())

    plt.figure(figsize=(8, 5))
    for opt in optimizers:
        epochs, mean_rate, std_rate = aggregate_metric(all_results, opt, Rt, "rate")
        gap = np.abs(mean_rate - float(Rt))
        plt.plot(epochs, gap, label=f"{opt}")
        if shade:
            # std를 gap에 정교하게 반영하려면 trial별로 |.| 계산해야하지만,
            # 간단 비교용으로 rate의 std를 그대로 gap에 사용(근사)
            plt.fill_between(epochs, np.abs((mean_rate-std_rate) - float(Rt)),
                             np.abs((mean_rate+std_rate) - float(Rt)),
                             alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("|rate - Rt|")
    plt.title(f"Constraint satisfaction speed | Rt={float(Rt):.3g}")
    plt.yscale("log")  # 보통 수렴 비교가 더 잘 보임 (원하면 주석)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# Convenience: 한 Rt에 대해 "비교 플롯 세트"를 한 번에
# ---------------------------
def plot_all_for_Rt(all_results, Rt, optimizers=None):
    if optimizers is None:
        optimizers = list(all_results.keys())

    # (1) loss/distortion/rate epochwise compare
    plot_epochwise_compare(all_results, Rt, metrics=("loss","distortion","rate","beta"), optimizers=optimizers, shade=True)

    # (2) constraint gap
    plot_rate_constraint_gap(all_results, Rt, optimizers=optimizers, shade=True)

#%%
OPTIMIZERS = ["adam", "adamgrad", "adamw","sgd","rmsprop"]  # 비교 5종
RtVec = np.linspace(0.6, 3.3, num=4)        # <- RtVec 줄이기 예시 (10 -> 4)
N_TRIALS = 2                                 # <- trial 줄이기 예시 (16 -> 2)

all_results = {}
for opt_name in OPTIMIZERS:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    results_for_opt = [
        vary_R(RtVec, x_test, p_x, optimizer_name=opt_name)
        for _ in range(N_TRIALS)
    ]
    all_results[opt_name] = results_for_opt

PATH = os.getcwd() + f"/data/compare_opts_samples={N_SAMPLES}_N=12_q=Ising.pt"
torch.save(all_results, PATH)

#%%
# Draw the plots for a given Rt
all_results = load_results(PATH)

# 저장된 Rt 목록 확인
print("optimizers:", list(all_results.keys()))
print("example Rts:", _sorted_Rts_from_results(all_results))

# 원하는 Rt 하나 선택해서 optimizer 3개 비교
Rt = 0.6
plot_all_for_Rt(all_results, Rt)

Rt = 1.5
plot_all_for_Rt(all_results, Rt)

Rt = 2.4
plot_all_for_Rt(all_results, Rt)

Rt = 3.2
plot_all_for_Rt(all_results, Rt)

# optimizer 하나 고르고 Rt 여러 개 곡선 보기
plot_optimizer_with_multiple_Rts(all_results, optimizer="adamw", max_Rts=6, shade=False)
plot_optimizer_with_multiple_Rts(all_results, optimizer="adam", max_Rts=6, shade=False)
plot_optimizer_with_multiple_Rts(all_results, optimizer="adamgrad", max_Rts=6, shade=False)
plot_optimizer_with_multiple_Rts(all_results, optimizer="sgd", max_Rts=6, shade=False)
plot_optimizer_with_multiple_Rts(all_results, optimizer="rmsprop", max_Rts=6, shade=False)





# %%
