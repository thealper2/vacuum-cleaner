"""
Akıllı Temizlik Robotu - Pekiştirmeli Öğrenme Simülasyonu
=========================================================
DQN ve DDQN tabanlı ajan ile 7x7 grid ortamında temizlik robotu simülasyonu.
"""

import os
import sys
import csv
import json
import time
import copy
import random
import itertools
import threading
import traceback
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Tuple, List, Dict, Any, Deque
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk, ImageDraw

# ─────────────────────────────────────────────
# SABITLER
# ─────────────────────────────────────────────

GRID_SIZE: int = 7
CHARGE_STATION: Tuple[int, int] = (3, 3)
CELL_PX: int = 64
MAX_STEPS: int = 250
MAX_BATTERY: int = 16
RETURN_SAFETY_MARGIN: int = 6

ACTIONS: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "UP":    (-1,  0),
    "DOWN":  ( 1,  0),
    "LEFT":  ( 0, -1),
    "RIGHT": ( 0,  1),
}

THEME = {
    "bg_dark":      "#f7f9fc",
    "bg_mid":       "#eef3f9",
    "bg_panel":     "#ffffff",
    "bg_card":      "#f1f5fb",
    "accent":       "#007f5f",
    "accent2":      "#d95f02",
    "accent3":      "#1f6feb",
    "text":         "#1f2937",
    "text_dim":     "#5f6b7a",
    "border":       "#c9d2df",
    "success":      "#2e9b44",
    "warning":      "#b7791f",
    "danger":       "#d14343",
    "charge":       "#b8860b",
}

DEFAULT_HP: Dict[str, Any] = {
    "n_episodes":         1000,
    "learning_rate":      5e-4,
    "gamma":              0.99,
    "batch_size":         64,
    "buffer_size":        50000,
    "min_replay_size":    1000,
    "hidden1":            128,
    "hidden2":            64,
    "target_update_freq": 500,
    "train_freq":         1,
    "clip_grad":          5.0,
    "epsilon_start":      1.0,
    "epsilon_min":        0.05,
    "epsilon_decay":      0.997,
    "reward_scale":       1.0,
    "stop_reward":        500.0,
    "goal_reward":        300.0,
    "use_cuda":           False,
}

LOG_DIR:   Path = Path("rl_logs")
MODEL_DIR: Path = Path("rl_models")
GIF_DIR:   Path = Path("rl_gifs")
for _d in [LOG_DIR, MODEL_DIR, GIF_DIR]:
    _d.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# GÖRSEL YARDIMCILAR
# ─────────────────────────────────────────────

def _make_tile(color, size=CELL_PX):
    img = Image.new("RGBA", (size, size), (*color, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size-1, size-1], outline=(255,255,255,30), width=1)
    return img

def _make_robot_img(size=CELL_PX):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    m = size // 8
    draw.rounded_rectangle([m, m+4, size-m, size-m], radius=6, fill=(0,180,255,230), outline=(0,220,255,255), width=2)
    ey = size // 3
    for ex in [size//3, 2*size//3]:
        draw.ellipse([ex-4, ey-4, ex+4, ey+4], fill=(255,255,255,255))
        draw.ellipse([ex-2, ey-2, ex+2, ey+2], fill=(0,0,0,255))
    draw.line([size//2, m, size//2, m-8], fill=(0,220,255,200), width=2)
    draw.ellipse([size//2-3, m-11, size//2+3, m-5], fill=(0,255,200,255))
    return img

def _make_dirt_img(size=CELL_PX):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2,2,size-2,size-2], fill=(45,35,20,220))
    rng = random.Random(42)
    for _ in range(12):
        x,y,r = rng.randint(8,size-8), rng.randint(8,size-8), rng.randint(3,7)
        draw.ellipse([x-r,y-r,x+r,y+r], fill=(80,55,30,200))
    return img

def _make_clean_img(size=CELL_PX):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2,2,size-2,size-2], fill=(20,30,25,200))
    draw.ellipse([size//2-6,size//2-6,size//2+6,size//2+6], fill=(0,220,150,40))
    return img

def _make_charge_img(size=CELL_PX):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2,2,size-2,size-2], fill=(60,50,10,220))
    pts = [(size//2+4,4),(size//2-4,size//2),(size//2+2,size//2),(size//2-4,size-4),(size//2+6,size//2+2),(size//2+2,size//2+2)]
    draw.polygon(pts, fill=(255,215,0,255), outline=(255,255,100,200))
    return img

def _make_obstacle_img(size=CELL_PX):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2,2,size-2,size-2], fill=(80,10,10,230))
    draw.line([10,10,size-10,size-10], fill=(255,60,60,255), width=4)
    draw.line([10,size-10,size-10,10], fill=(255,60,60,255), width=4)
    return img

_TILE_CACHE: Dict[str, ImageTk.PhotoImage] = {}

def get_tile(name):
    if name not in _TILE_CACHE:
        if   name == "robot":        img = _make_robot_img()
        elif name == "dirt":         img = _make_dirt_img()
        elif name == "clean":        img = _make_clean_img()
        elif name == "charge":       img = _make_charge_img()
        elif name == "obstacle":     img = _make_obstacle_img()
        elif name == "robot_charge":
            base = _make_charge_img(); robot = _make_robot_img(); base.paste(robot,(0,0),robot); img = base
        else:                        img = _make_tile((30,30,30))
        _TILE_CACHE[name] = ImageTk.PhotoImage(img)
    return _TILE_CACHE[name]


# ─────────────────────────────────────────────
# ORTAM  ← BÜYÜK DÜZELTME
# ─────────────────────────────────────────────

class CleaningEnv:
    """
    DÜZELTİLMİŞ ORTAM:
    - State vektörü artık grid'in 2D dirty-map'ini + robot konumunu + bataryayı içeriyor
    - Ödül şekillendirmesi düzgün yapılmış
    - Batarya yönetimi mantıklı
    - MAX_BATTERY=16 ile çözülebilir: şarj istasyonundan en uzak hücre 6 adım,
      robot turlar halinde (git-temizle-dön-şarj) çalışmalı
    """

    def __init__(self, grid_size=7, charge_pos=(3,3), max_battery=MAX_BATTERY,
                 max_steps=MAX_STEPS, obstacles=None):
        self.grid_size   = grid_size
        self.charge_pos  = charge_pos
        self.max_battery = max_battery
        self.max_steps   = max_steps
        self.obstacles   = set(map(tuple, obstacles or []))

        # Temizlenebilir hücre sayısı: şarj istasyonu ve engeller hariç
        self.cleanable_count = (
            grid_size * grid_size - 1 - len(self.obstacles)
        )
        self.reset()

    # ── RESET ──────────────────────────────────
    def reset(self):
        self.robot_pos      = self.charge_pos
        self.battery        = self.max_battery
        self.cleaned        = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.dirty_cells    = {
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) != self.charge_pos and (i, j) not in self.obstacles
        }
        self.visited_count  = 0
        self.charge_visits  = 0
        self.total_energy   = 0
        self.step_count     = 0
        self.done           = False
        self.episode_reward = 0.0
        return self._get_state()

    def _nearest_dirty_dist(self, r: int, c: int) -> float:
        if not self.dirty_cells:
            return float(self.grid_size * 2)
        return float(min(abs(r-i) + abs(c-j) for (i, j) in self.dirty_cells))

    # ── STATE (DÜZELTİLMİŞ) ──────────────────
    # Eski: sadece 6 sayı → robot grid durumunu göremiyordu!
    # Yeni: robot pozisyonu (2) + batarya (1) + dirty grid (49) + özet istatistikler (3)
    # Toplam: 55 boyut
    def _get_state(self):
        r, c = self.robot_pos
        cr, cc = self.charge_pos

        # 1. Robot pozisyonu (normalize)
        pos = np.array([r / (self.grid_size-1), c / (self.grid_size-1)], dtype=np.float32)

        # Şarj istasyonuna mesafe hesabı
        real_dist_to_charge = abs(r-cr) + abs(c-cc)
        dist_to_charge = real_dist_to_charge / (self.grid_size*2)

        # 2. Batarya (normalize + kritik bayrak)
        bat_norm = self.battery / self.max_battery
        bat_crit = 1.0 if self.battery <= real_dist_to_charge + RETURN_SAFETY_MARGIN else 0.0

        # 3. Dirty grid (7x7=49 hücre, flatten)
        # 0=kirli, 1=temiz veya engel
        dirty_map = self.cleaned.copy()
        for obs in self.obstacles:
            dirty_map[obs[0], obs[1]] = 1.0  # engeller "temizlenmiş" gibi
        dirty_flat = dirty_map.flatten()  # 49 boyut

        # 4. Şarj istasyonuna mesafe (Zaten hesaplandı)

        # 5. En yakın kirli hücreye mesafe (manhattan)
        min_dist = self._nearest_dirty_dist(r, c)
        nearest_dirty_norm = min_dist / (self.grid_size*2)
        # Ekstra: Durum değişimi için eski min_dist'i bir yere kaydetmiyoruz, ama 
        # robot bu sayede kirlinin nerede olduğunu doğrudan sayısal olarak da hissedebilir.

        # 6. Temizlenme oranı
        clean_ratio = self.visited_count / max(1, self.cleanable_count)

        return_need = 1.0 if self.battery <= real_dist_to_charge + RETURN_SAFETY_MARGIN else 0.0

        state = np.concatenate([
            pos,                             # 2
            [bat_norm, bat_crit],            # 2
            dirty_flat,                      # 49
            [dist_to_charge, nearest_dirty_norm, clean_ratio, return_need],  # 4
        ]).astype(np.float32)

        assert len(state) == self.state_dim, f"State dim mismatch: {len(state)} != {self.state_dim}"
        return state

    @property
    def state_dim(self):
        return 2 + 2 + 49 + 4  # = 57

    @property
    def action_dim(self):
        return 4

    # ── STEP (DÜZELTİLMİŞ) ───────────────────
    def step(self, action: int):
        if self.done:
            return self._get_state(), 0.0, True, {}

        old_r, old_c = self.robot_pos
        prev_dist_charge = abs(old_r - self.charge_pos[0]) + abs(old_c - self.charge_pos[1])

        action_name = ACTIONS[action]
        dr, dc = ACTION_DELTAS[action_name]
        nr, nc = old_r + dr, old_c + dc

        # Düşük bataryada şarja dönme davranışını zorlayıcı ödülle güçlendir.
        # Güvenli dönüş için mesafeye bağlı dinamik bir sınır belirle.
        return_required = self.battery <= prev_dist_charge + RETURN_SAFETY_MARGIN

        # Duvar veya engel
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size) \
                or (nr,nc) in self.obstacles:
            # Geçersiz hamle de zaman/enerji tüketir; aksi halde testte adım sabit kalıp
            # ajan aynı duvara vurarak "takılmış" gibi görünür.
            self.battery     -= 1
            self.total_energy += 1
            self.step_count   += 1
            reward = -2.0
            if return_required:
                reward -= 1.0
            if self.battery <= 0:
                reward = -120.0
                self.done = True
                self.episode_reward += reward
                return self._get_state(), reward, True, {"blocked": True, "battery_dead": True}
            if self.step_count >= self.max_steps:
                self.done = True
                self.episode_reward += reward
                return self._get_state(), reward, True, {"blocked": True, "step_limit": True}
            self.episode_reward += reward
            return self._get_state(), reward, False, {"blocked": True}

        # Hareket
        self.robot_pos    = (nr, nc)
        self.battery     -= 1
        self.total_energy += 1
        self.step_count   += 1

        reward = -0.5  # küçük adım maliyeti (her adımı mutlaka cezalandır)

        # Hareket öncesi min_dist (kirlilere) hesapla (ödül şekillendirme için)
        prev_min_dist = self._nearest_dirty_dist(old_r, old_c)

        # Yeni temiz hücre
        if (nr, nc) in self.dirty_cells:
            self.dirty_cells.remove((nr, nc))
            self.cleaned[nr,nc] = 1.0
            self.visited_count  += 1
            reward += 20.0  # temizlik ödülü
        elif self.cleaned[nr, nc] == 1 and (nr, nc) != self.charge_pos:
            # Aynı temiz hücrelerde gezinmeyi caydır (loop cezası)
            reward -= 1.5
        elif self.dirty_cells and not return_required:
            # Temizlenmemiş yer varsa ve şarja acil dönülmesi GEREKMİYORSA,
            # en yakın kirliye yaklaşıp uzaklaşmayı cezalandır/ödüllendir
            new_min_dist = self._nearest_dirty_dist(nr, nc)
            
            if new_min_dist < prev_min_dist:
                reward += 2.0  # Kirliye doğru bir adım
            elif new_min_dist > prev_min_dist:
                reward -= 3.0  # Kirliden uzaklaşıyor (zaman israfı)

        dist_charge = abs(nr - self.charge_pos[0]) + abs(nc - self.charge_pos[1])

        # Şarj istasyonu
        at_charge = (nr,nc) == self.charge_pos
        if at_charge:
            old_battery   = self.battery
            self.battery  = self.max_battery
            self.charge_visits += 1

            # Tüm hücreler temizlendi mi? → BAŞARI
            if self.visited_count == self.cleanable_count:
                reward      += 1000.0
                self.done    = True
                self.episode_reward += reward
                return self._get_state(), reward, True, {"success": True}

            # Sadece gerekli şarjı ödüllendir; gereksiz şarj spam'ini sert cezalandır.
            # Not: at_charge anında dist_charge=0 olur; karar için hareket öncesi mesafe kullanılır.
            if old_battery < prev_dist_charge + RETURN_SAFETY_MARGIN:
                reward += 2.0
            else:
                reward -= 5.0

        # EĞER her yer temizlendiyse VE henüz şarja gitmemişse,
        # şarja gitmesi HER ŞEYDEN çok daha önemlidir!
        if self.visited_count == self.cleanable_count and not at_charge:
            if dist_charge < prev_dist_charge:
                reward += 10.0
            elif dist_charge > prev_dist_charge:
                reward -= 10.0
            else:
                reward -= 2.0
        # Normal batarya kritik kuralı: dönüş güvenlik marjı altındaysa
        elif return_required and not at_charge:
            if dist_charge < prev_dist_charge:
                reward += 2.5
            elif dist_charge > prev_dist_charge:
                reward -= 5.0
            else:
                reward -= 1.5

        # Batarya bitti
        if self.battery <= 0:
            reward      = -120.0
            self.done    = True
            self.episode_reward += reward
            return self._get_state(), reward, True, {"battery_dead": True}

        if self.step_count >= self.max_steps:
            self.done = True
            self.episode_reward += reward
            return self._get_state(), reward, True, {"step_limit": True}

        self.episode_reward += reward
        return self._get_state(), reward, False, {}

    def get_grid_info(self):
        return {
            "robot_pos":       self.robot_pos,
            "cleaned":         self.cleaned.copy(),
            "battery":         self.battery,
            "max_battery":     self.max_battery,
            "charge_pos":      self.charge_pos,
            "obstacles":       list(self.obstacles),
            "visited_count":   self.visited_count,
            "cleanable_count": self.cleanable_count,
            "step_count":      self.step_count,
            "charge_visits":   self.charge_visits,
            "total_energy":    self.total_energy,
            "episode_reward":  self.episode_reward,
        }


# ─────────────────────────────────────────────
# SİNİR AĞI  ← DÜZELTİLMİŞ
# ─────────────────────────────────────────────

def build_network(state_dim, action_dim, hidden1, hidden2):
    """
    Dueling DQN mimarisi: avantaj ve değer akışları ayrı.
    56 boyutlu state için daha güçlü temsil.
    """
    class DuelingNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Ortak özellik çıkarıcı
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden1),
                nn.LayerNorm(hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.LayerNorm(hidden2),
                nn.ReLU(),
            )
            # Değer akışı (V)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden2, hidden2 // 2),
                nn.ReLU(),
                nn.Linear(hidden2 // 2, 1),
            )
            # Avantaj akışı (A)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden2, hidden2 // 2),
                nn.ReLU(),
                nn.Linear(hidden2 // 2, action_dim),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            feat = self.shared(x)
            v    = self.value_stream(feat)
            a    = self.advantage_stream(feat)
            # Q = V + (A - mean(A))  →  daha stabil öğrenme
            return v + (a - a.mean(dim=1, keepdim=True))

    return DuelingNet()


# ─────────────────────────────────────────────
# REPLAY BUFFER (Öncelikli Deneyim Tekrarlama)
# ─────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Öncelikli Deneyim Tekrarlama:
    Yüksek TD-hatalı geçişleri daha sık örnekler → daha hızlı öğrenme.
    MAX_BATTERY=16 gibi seyrek ödüllü ortamlarda kritik.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity     = capacity
        self.alpha        = alpha      # öncelik üssü (0=düzgün, 1=tam öncelikli)
        self.beta_start   = beta_start
        self.beta_frames  = beta_frames
        self.frame        = 1

        self.buffer    = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)

    def sample(self, batch_size):
        prios  = np.array(self.priorities, dtype=np.float32)
        probs  = prios ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in idxs]

        beta = min(1.0, self.beta_start + self.frame * (1.0-self.beta_start) / self.beta_frames)
        self.frame += 1

        weights = (len(self.buffer) * probs[idxs]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            idxs,
            np.array(weights,     dtype=np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            self.priorities[idx] = abs(err) + 1e-6  # küçük epsilon: sıfır öncelik olmasın

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# DQN / DDQN AJAN  ← DÜZELTİLMİŞ
# ─────────────────────────────────────────────

class DQNAgent:
    def __init__(self, state_dim, action_dim, hp, use_ddqn=True, device=None):
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.hp          = hp
        self.use_ddqn    = use_ddqn  # varsayılan True (DDQN daha stabil)
        use_cuda = bool(hp.get("use_cuda", False))
        self.device      = torch.device(
            device if device else ("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"))

        self.q_net      = build_network(state_dim, action_dim, hp["hidden1"], hp["hidden2"]).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=hp["learning_rate"], weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=hp.get("n_episodes", 1000), eta_min=1e-5)

        buf_cap = int(hp.get("buffer_size", 50000))
        self.replay      = PrioritizedReplayBuffer(buf_cap)
        self.epsilon     = float(hp.get("epsilon_start", 1.0))
        self.train_steps = 0
        self.optim_steps = 0
        self.episode_count = 0
        self.losses: List[float] = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax().item())

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self):
        hp = self.hp
        if len(self.replay) < int(hp["min_replay_size"]):
            return None

        # PER örnekleme
        states, actions, rewards, next_states, dones, idxs, weights = \
            self.replay.sample(int(hp["batch_size"]))

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)
        w  = torch.FloatTensor(weights).to(self.device)

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_ddqn:
                next_actions = self.q_net(ns).argmax(1)
                next_q = self.target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(ns).max(1)[0]
            target = r + float(hp["gamma"]) * next_q * (1-d)

        td_errors = (q_vals - target).detach().cpu().numpy()
        self.replay.update_priorities(idxs, td_errors)

        loss = (w * nn.SmoothL1Loss(reduction='none')(q_vals, target)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), float(hp.get("clip_grad", 5.0)))
        self.optimizer.step()
        self.optim_steps += 1

        self.train_steps += 1
        target_freq = int(hp.get("target_update_freq", 200))
        if self.train_steps % target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def decay_epsilon(self):
        self.epsilon = max(
            float(self.hp.get("epsilon_min", 0.05)),
            self.epsilon * float(self.hp["epsilon_decay"]),
        )

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
            "train_steps": self.train_steps, "optim_steps": self.optim_steps,
            "hp": self.hp, "use_ddqn": self.use_ddqn,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ck["q_net"])
        self.target_net.load_state_dict(ck["target_net"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.epsilon     = ck.get("epsilon", float(self.hp.get("epsilon_min", 0.05)))
        self.train_steps = ck.get("train_steps", 0)
        self.optim_steps = ck.get("optim_steps", self.train_steps)


# ─────────────────────────────────────────────
# LOGLAMA
# ─────────────────────────────────────────────

def make_run_id():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_episode_log(run_id, episode_data):
    path = LOG_DIR / f"run_{run_id}_episodes.csv"
    if not episode_data: return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_data[0].keys()))
        writer.writeheader(); writer.writerows(episode_data)
    return path

def save_hp_search_log(results):
    path = LOG_DIR / f"hp_search_{make_run_id()}.csv"
    if not results: return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)
    return path


# ─────────────────────────────────────────────
# EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────

def select_policy_action(agent, env, state, force_greedy=False):
    """
    Eğitim ve testte aynı aksiyon politikasını kullan.
    force_greedy=True iken ağdan epsilon=0 ile aksiyon seçilir.
    """
    if force_greedy:
        orig = agent.epsilon
        agent.epsilon = 0.0
        action = agent.select_action(state)
        agent.epsilon = orig
    else:
        action = agent.select_action(state)

    # Kritik bataryada, şarja uzaklaştıran adımları engelle.
    rr, rc = env.robot_pos
    cur_dist = abs(rr - env.charge_pos[0]) + abs(rc - env.charge_pos[1])
    return_required = env.battery <= cur_dist + RETURN_SAFETY_MARGIN

    valid_actions = []
    for a_idx, a_name in enumerate(ACTIONS):
        dr, dc = ACTION_DELTAS[a_name]
        nr, nc = rr + dr, rc + dc
        if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size and (nr, nc) not in env.obstacles:
            valid_actions.append(a_idx)

    # Geçersiz aksiyon seçimini engelle: duvar/engele çarpıp batarya yakmasın.
    if valid_actions and action not in valid_actions:
        if force_greedy:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_vals = agent.q_net(s).squeeze(0).detach().cpu().numpy()
            action = max(valid_actions, key=lambda a: q_vals[a])
        else:
            action = random.choice(valid_actions)

    # Çok erken şarja dönmeyi engellemek için sadece gerçek kritik bölgede zorla:
    # kalan batarya, şarja kalan mesafeye eşit/az ise güvenli aksiyon dayat.
    if return_required:
        better_actions = []
        safe_actions = []
        for a_idx, a_name in enumerate(ACTIONS):
            dr, dc = ACTION_DELTAS[a_name]
            nr, nc = rr + dr, rc + dc
            if not (0 <= nr < env.grid_size and 0 <= nc < env.grid_size):
                continue
            if (nr, nc) in env.obstacles:
                continue
            new_dist = abs(nr - env.charge_pos[0]) + abs(nc - env.charge_pos[1])
            safe_actions.append((a_idx, new_dist))
            if new_dist < cur_dist:
                better_actions.append(a_idx)
        if better_actions:
            action = random.choice(better_actions)
        elif safe_actions:
            best_dist = min(d for _, d in safe_actions)
            best_actions = [a for a, d in safe_actions if d == best_dist]
            action = random.choice(best_actions)

    return action

def run_episodes(agent, env, n_episodes, hp, callback=None, stop_flag=None,
                 train_mode=True, episode_start=0, max_total_steps=None,
                 reward_early_stop=True):
    episode_data   = []
    total_steps    = 0
    
    # Debug training accumulators
    debug_success_count = 0
    debug_battery_dead_count = 0
    debug_blocked_count = 0

    for ep in range(n_episodes):
        if stop_flag and stop_flag[0]: break
        if max_total_steps and total_steps >= max_total_steps: break

        state          = env.reset()
        total_reward   = 0.0
        steps          = 0
        local_stop     = False
        
        ep_blocked = 0
        ep_battery_dead = False

        while not env.done and steps < env.max_steps:
            if stop_flag and stop_flag[0]: local_stop = True; break
            if max_total_steps and total_steps >= max_total_steps: local_stop = True; break

            action = select_policy_action(agent, env, state, force_greedy=False)

            next_state, reward, done, info = env.step(action)
            
            if info.get("blocked"):
                ep_blocked += 1
            if info.get("battery_dead"):
                ep_battery_dead = True

            if train_mode:
                rs = float(hp.get("reward_scale", 1.0))
                agent.store(state, action, reward*rs, next_state, done)
                tf = int(hp.get("train_freq", 1))
                if steps % tf == 0:
                    agent.train_step()

            state         = next_state
            total_reward += reward
            steps        += 1
            total_steps  += 1

        if local_stop: break

        if train_mode:
            agent.decay_epsilon()
            agent.episode_count += 1
            # PyTorch uyarısını önlemek için: scheduler sadece optimizer en az bir kez çalıştıysa.
            if hasattr(agent, 'scheduler') and getattr(agent, 'optim_steps', 0) > 0:
                agent.scheduler.step()

        success = int(env.visited_count == env.cleanable_count)
        
        # Update debug aggregations
        debug_success_count += success
        if ep_battery_dead:
            debug_battery_dead_count += 1
        debug_blocked_count += ep_blocked

        # Debug print every 100 episodes
        if train_mode and (ep + 1) % 100 == 0:
            print(f"[DEBUG TRAIN] Ep {ep+1:04d} | Success: {debug_success_count:3d}/100 | "
                  f"Battery Dead: {debug_battery_dead_count:3d}/100 | Blocked Hits Avg: {debug_blocked_count/100:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            debug_success_count = 0
            debug_battery_dead_count = 0
            debug_blocked_count = 0

        info_dict = {
            "episode":       episode_start + ep + 1,
            "reward":        round(total_reward, 3),
            "steps":         steps,
            "energy":        env.total_energy,
            "charge_visits": env.charge_visits,
            "cleaned_cells": env.visited_count,
            "success":       success,
            "epsilon":       round(agent.epsilon, 4),
            "blocked":       ep_blocked,
            "battery_dead":  ep_battery_dead,
        }
        episode_data.append(info_dict)

        if reward_early_stop and total_reward >= hp["stop_reward"] and train_mode and success:
            if callback: callback(ep, info_dict, env.get_grid_info())
            break

        if callback: callback(ep, info_dict, env.get_grid_info())

    return episode_data


# ─────────────────────────────────────────────
# ENGEL HARİTALARI
# ─────────────────────────────────────────────

OBSTACLE_MAPS = {
    "Yok":      [],
    "Duvarlar": [(1,1),(1,2),(1,3),(5,3),(5,4),(5,5),(3,1),(3,5)],
    "Dağınık":  [(0,2),(2,0),(4,6),(6,4),(1,5),(5,1),(2,4),(4,2)],
    "Labirent": [(1,1),(1,2),(1,4),(1,5),(2,3),(3,1),(3,5),(4,3),(5,2),(5,4)],
    "Rastgele": [],
}

RANDOM_OBSTACLE_FILE = LOG_DIR / "random_obstacle_map.json"

def _generate_random_obstacle_map(count=6):
    all_cells = [(r,c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r,c)!=CHARGE_STATION]
    return random.Random().sample(all_cells, count)

def _load_or_create_persistent_random_map(count=6):
    try:
        if RANDOM_OBSTACLE_FILE.exists():
            with open(RANDOM_OBSTACLE_FILE) as f:
                raw = json.load(f)
            parsed = [(item[0],item[1]) for item in raw if len(item)==2]
            if len(parsed)==count: return parsed
    except Exception:
        pass
    generated = _generate_random_obstacle_map(count)
    try:
        with open(RANDOM_OBSTACLE_FILE,"w") as f: json.dump(generated, f)
    except Exception:
        pass
    return generated

def get_obstacle_map(name):
    if name == "Rastgele": return _load_or_create_persistent_random_map()
    return list(OBSTACLE_MAPS.get(name, []))


# ─────────────────────────────────────────────
# GRAFİK YÖNETİCİSİ
# ─────────────────────────────────────────────

class ChartManager:
    CHART_DEFS = [
        ("reward",        "Ödül",           "Toplam Ödül",     "#00d4aa"),
        ("success",       "Başarı (0/1)",    "Başarı",          "#3fb950"),
        ("steps",         "Adım Sayısı",     "Adım",            "#7c5cbf"),
        ("energy",        "Enerji",          "Enerji",          "#ff6b35"),
        ("charge_visits", "Şarj Dönüşleri",  "Şarj Ziyareti",   "#ffd700"),
    ]

    def __init__(self):
        self.data    = {k: [] for k, *_ in self.CHART_DEFS}
        self.data["episode"] = []
        self.frames  = {k: [] for k, *_ in self.CHART_DEFS}
        self._figures  = {}
        self._axes     = {}
        self._canvases = {}

    def reset(self):
        for k in self.data: self.data[k] = []
        for k in self.frames: self.frames[k] = []

    def add_episode(self, info):
        self.data["episode"].append(info["episode"])
        for key, *_ in self.CHART_DEFS:
            self.data[key].append(info.get(key, 0))

    def create_figure(self, key, parent):
        fig = Figure(figsize=(5,2.8), dpi=96, facecolor=THEME["bg_mid"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(THEME["bg_dark"])
        ax.tick_params(colors=THEME["text_dim"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(THEME["border"])
        self._figures[key] = fig
        self._axes[key]    = ax
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        self._canvases[key] = canvas
        return canvas

    def update_chart(self, key, run_id="", capture_frame=False):
        if key not in self._axes: return
        meta = {k:(t,y,c) for k,t,y,c in self.CHART_DEFS}
        if key not in meta: return
        title, ylabel, color = meta[key]
        ax = self._axes[key]
        ax.clear()
        ax.set_facecolor(THEME["bg_dark"])
        ax.tick_params(colors=THEME["text_dim"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(THEME["border"])
        eps  = self.data["episode"]
        vals = self.data[key]
        if eps:
            ax.plot(eps, vals, color=color, linewidth=1.2, alpha=0.8)
            if len(vals) >= 20:
                smooth = np.convolve(vals, np.ones(20)/20, mode='valid')
                ax.plot(eps[19:], smooth, color="#374151", linewidth=1.8, linestyle="--", alpha=0.7, label="MA-20")
        ax.set_title(title, color=THEME["text"], fontsize=8, pad=3)
        ax.set_xlabel("Episode", color=THEME["text_dim"], fontsize=7)
        ax.set_ylabel(ylabel, color=THEME["text_dim"], fontsize=7)
        ax.grid(True, alpha=0.15, color=THEME["border"])
        self._figures[key].tight_layout(pad=0.5)
        self._canvases[key].draw_idle()
        if capture_frame and run_id and eps:
            try:
                bp = GIF_DIR / f"tmp_{run_id}_{key}.png"
                self._figures[key].savefig(str(bp), dpi=72, bbox_inches='tight', facecolor=THEME["bg_mid"])
                img = Image.open(str(bp))
                self.frames[key].append(img.copy())
            except Exception:
                pass

    def save_gifs(self, run_id):
        for key in self.frames:
            frames = self.frames[key]
            if len(frames) < 2: continue
            try:
                out = GIF_DIR / f"{run_id}_{key}.gif"
                frames[0].save(str(out), save_all=True, append_images=frames[1:], duration=100, loop=0)
            except Exception as e:
                print(f"GIF hatası ({key}): {e}")

    def update_all(self, run_id="", capture_frame=False):
        for key, *_ in self.CHART_DEFS:
            self.update_chart(key, run_id, capture_frame=capture_frame)


# ─────────────────────────────────────────────
# HİPERPARAMETRE ARAMALARI
# ─────────────────────────────────────────────

HP_SEARCH_SPACE = {
    "learning_rate":      [5e-4, 1e-3],
    "gamma":              [0.99],
    "hidden1":            [128, 256],
    "hidden2":            [64, 128],
    "batch_size":         [64, 128],
    "epsilon_decay":      [0.95, 0.99],  # Daha hizli decay, ki epsilon cabuk dusup ogrendigini kullanabilsin
    "target_update_freq": [100, 500],    # Daha agresif target guncelleme
    "buffer_size":        [10_000, 20_000],
    "epsilon_min":        [0.05],
    "epsilon_start":      [1.0],
    "train_freq":         [1, 2],
    "clip_grad":          [5.0],
    "reward_scale":       [1.0],
}

def hp_search(base_hp, env_kwargs, use_ddqn, training_budget_steps=50000,
              eval_window=100, progress_cb=None, stop_flag=None):
    keys   = list(HP_SEARCH_SPACE.keys())
    combos = list(itertools.product(*[HP_SEARCH_SPACE[k] for k in keys]))
    random.shuffle(combos)

    all_results  = []
    best_score   = -float("inf")
    best_hp      = base_hp.copy()

    for i, combo in enumerate(combos):
        if stop_flag and stop_flag[0]: break
        hp = base_hp.copy()
        for k,v in zip(keys,combo): hp[k] = v
        hp["stop_reward"]    = float("inf")
        hp["min_replay_size"] = min(hp["min_replay_size"], 500)

        env         = CleaningEnv(**env_kwargs)
        agent       = DQNAgent(env.state_dim, env.action_dim, hp, use_ddqn=use_ddqn)
        local_stop  = [False]
        success_hist = []

        def eval_cb(_ep, info, _g):
            if stop_flag and stop_flag[0]: local_stop[0]=True; return
            success_hist.append(info["success"])
            if len(success_hist)>=eval_window and np.mean(success_hist[-eval_window:])>0.9:
                local_stop[0]=True

        data = run_episodes(agent, env, 100000, hp, callback=eval_cb,
                            stop_flag=local_stop, train_mode=True,
                            max_total_steps=training_budget_steps, reward_early_stop=False)
        if not data: continue

        wd = data[-min(eval_window,len(data)):]
        rewards = [d["reward"] for d in wd]; successes = [d["success"] for d in wd]
        suc_eps = [d for d in wd if d["success"]==1]
        avg_r   = float(np.mean(rewards)) if rewards else 0.0
        suc_r   = float(np.mean(successes)) if successes else 0.0
        avg_e   = float(np.mean([d["energy"] for d in wd])) if wd else 0.0
        score   = suc_r*100.0 - avg_e*0.5 + avg_r*0.1

        result = {**{k:v for k,v in zip(keys,combo)}, "use_ddqn":use_ddqn,
                  "avg_reward":round(avg_r,3), "success_rate":round(suc_r,3),
                  "avg_energy":round(avg_e,3), "score":round(score,3), "n_episodes":len(data)}
        all_results.append(result)

        if score > best_score:
            best_score = score; best_hp = hp.copy(); best_hp["use_ddqn"] = use_ddqn
            agent.save(str(MODEL_DIR/"best_hp_search_model.pt"))

        if progress_cb: progress_cb(i+1, len(combos), result)

    save_hp_search_log(all_results)
    return best_hp, all_results


# ─────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Akıllı Temizlik Robotu — RL Simülasyonu")
        self.root.configure(bg=THEME["bg_dark"])
        self.root.minsize(1280, 800)

        self.agent         = None
        self.env           = None
        self.run_id        = ""
        self.stop_flag     = [False]
        self.episode_data  = []
        self.is_training   = False
        self.is_testing    = False
        self.train_thread  = None
        self.hp_thread     = None
        self.chart_mgr     = ChartManager()
        self.grid_images   = []
        self.summary_totals = {
            "episodes": 0,
            "success": 0.0,
            "reward": 0.0,
            "steps": 0.0,
            "energy": 0.0,
            "charge_visits": 0.0,
        }

        self.hp_vars      = {}
        self.model_var    = tk.StringVar(value="DDQN")  # varsayılan DDQN
        self.obstacle_var = tk.StringVar(value="Yok")
        self.use_cuda_var = tk.BooleanVar(value=False)
        self.render_grid_train_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.obstacle_var.trace_add("write", self._on_obstacle_map_changed)
        self._initialize_env_agent()
        self._update_control_states()

    # ── UI ────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0, minsize=300)
        self.root.columnconfigure(1, weight=0, minsize=GRID_SIZE*CELL_PX+20)
        self.root.columnconfigure(2, weight=1, minsize=380)
        self.root.rowconfigure(0, weight=1)
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    def _section(self, parent, title, pady=4):
        outer = tk.Frame(parent, bg=THEME["bg_panel"],
                         highlightbackground=THEME["border"], highlightthickness=1)
        outer.pack(fill="x", padx=8, pady=(pady,0))
        tk.Frame(outer, bg=THEME["accent"], height=2).pack(fill="x")
        tk.Label(outer, text=title, bg=THEME["bg_panel"], fg=THEME["accent"],
                 font=("Consolas",9,"bold")).pack(anchor="w", padx=8, pady=(4,2))
        content = tk.Frame(outer, bg=THEME["bg_panel"])
        content.pack(fill="x", padx=6, pady=(0,6))
        return content

    def _labeled_entry(self, parent, label, key, default, is_float=False):
        row = tk.Frame(parent, bg=THEME["bg_panel"])
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, bg=THEME["bg_panel"], fg=THEME["text_dim"],
                 font=("Consolas",8), width=22, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=default) if is_float else tk.IntVar(value=int(default))
        self.hp_vars[key] = var
        e = tk.Entry(row, textvariable=var, width=9, bg=THEME["bg_card"], fg=THEME["text"],
                     insertbackground=THEME["accent"], relief="flat", font=("Consolas",8),
                     highlightthickness=1, highlightbackground=THEME["border"],
                     highlightcolor=THEME["accent"])
        e.pack(side="right", padx=2)

    def _build_left_panel(self):
        left = tk.Frame(self.root, bg=THEME["bg_mid"], width=300)
        left.grid(row=0, column=0, sticky="nsew")
        left.pack_propagate(False)
        tk.Label(left, text="⚡ RL KONTROL PANELİ", bg=THEME["bg_mid"], fg=THEME["accent"],
                 font=("Consolas",11,"bold")).pack(pady=(14,4))

        cv = tk.Canvas(left, bg=THEME["bg_mid"], highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); cv.pack(side="left", fill="both", expand=True)
        sf = tk.Frame(cv, bg=THEME["bg_mid"])
        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0,0), window=sf, anchor="nw")
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(-1*(e.delta//120),"units"))

        self._build_env_info(sf); self._build_hp_section(sf)
        self._build_options_section(sf); self._build_buttons(sf)

        self.status_var = tk.StringVar(value="⏳ Hazır")
        tk.Label(left, textvariable=self.status_var, bg=THEME["bg_dark"], fg=THEME["accent"],
                 font=("Consolas",8), wraplength=280).pack(fill="x", padx=4, pady=4)

    def _build_env_info(self, parent):
        sec = self._section(parent, "📊 ORTAM BİLGİSİ")
        self.info_labels = {}
        for key, lbl in [("episode","Episode:"),("reward","Son Ödül:"),("steps","Adım:"),
                          ("energy","Enerji:"),("charge_visits","Şarj Dönüşü:"),
                          ("success_rate","Başarı Oranı:"),("epsilon","Epsilon:"),
                          ("battery","Batarya:"),("cleaned","Temizlenen:")]:
            row = tk.Frame(sec, bg=THEME["bg_panel"]); row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, bg=THEME["bg_panel"], fg=THEME["text_dim"],
                     font=("Consolas",8), width=15, anchor="w").pack(side="left")
            v = tk.Label(row, text="—", bg=THEME["bg_panel"], fg=THEME["text"],
                         font=("Consolas",8,"bold"), anchor="e")
            v.pack(side="right"); self.info_labels[key] = v

    def _build_hp_section(self, parent):
        sec = self._section(parent, "⚙️ HİPERPARAMETRELER", pady=6)
        for lbl, key, default, is_float in [
            ("N Episodes","n_episodes",DEFAULT_HP["n_episodes"],False),
            ("Learning Rate","learning_rate",DEFAULT_HP["learning_rate"],True),
            ("Gamma","gamma",DEFAULT_HP["gamma"],True),
            ("Batch Size","batch_size",DEFAULT_HP["batch_size"],False),
            ("Buffer Size","buffer_size",DEFAULT_HP["buffer_size"],False),
            ("Min Replay Size","min_replay_size",DEFAULT_HP["min_replay_size"],False),
            ("Hidden Layer 1","hidden1",DEFAULT_HP["hidden1"],False),
            ("Hidden Layer 2","hidden2",DEFAULT_HP["hidden2"],False),
            ("Target Update Fr","target_update_freq",DEFAULT_HP["target_update_freq"],False),
            ("Train Freq","train_freq",DEFAULT_HP["train_freq"],False),
            ("Clip Grad","clip_grad",DEFAULT_HP["clip_grad"],True),
            ("Epsilon Start","epsilon_start",DEFAULT_HP["epsilon_start"],True),
            ("Epsilon Min","epsilon_min",DEFAULT_HP["epsilon_min"],True),
            ("Epsilon Decay","epsilon_decay",DEFAULT_HP["epsilon_decay"],True),
            ("Reward Scale","reward_scale",DEFAULT_HP["reward_scale"],True),
            ("Stop Reward","stop_reward",DEFAULT_HP["stop_reward"],True),
        ]:
            self._labeled_entry(sec, lbl, key, default, is_float)

    def _build_options_section(self, parent):
        sec = self._section(parent, "🔧 SEÇENEKLER", pady=6)
        tk.Label(sec, text="Model Mimarisi:", bg=THEME["bg_panel"],
                 fg=THEME["text_dim"], font=("Consolas",8)).pack(anchor="w")
        for val in ["DQN","DDQN"]:
            tk.Radiobutton(sec, text=val, variable=self.model_var, value=val,
                           bg=THEME["bg_panel"], fg=THEME["text"],
                           selectcolor=THEME["bg_card"], activebackground=THEME["bg_panel"],
                           font=("Consolas",8)).pack(anchor="w", padx=8)
        tk.Label(sec, text="Engel Haritası:", bg=THEME["bg_panel"],
                 fg=THEME["text_dim"], font=("Consolas",8)).pack(anchor="w", pady=(6,0))
        for name in OBSTACLE_MAPS:
            tk.Radiobutton(sec, text=name, variable=self.obstacle_var, value=name,
                           bg=THEME["bg_panel"], fg=THEME["text"],
                           selectcolor=THEME["bg_card"], activebackground=THEME["bg_panel"],
                           font=("Consolas",8)).pack(anchor="w", padx=8)

        tk.Checkbutton(
            sec,
            text="CUDA Kullan (küçük modelde yavaş olabilir)",
            variable=self.use_cuda_var,
            bg=THEME["bg_panel"],
            fg=THEME["text"],
            selectcolor=THEME["bg_card"],
            activebackground=THEME["bg_panel"],
            font=("Consolas",8),
        ).pack(anchor="w", padx=8, pady=(6, 0))

        tk.Checkbutton(
            sec,
            text="Eğitimde Grid Çiz",
            variable=self.render_grid_train_var,
            bg=THEME["bg_panel"],
            fg=THEME["text"],
            selectcolor=THEME["bg_card"],
            activebackground=THEME["bg_panel"],
            font=("Consolas",8),
        ).pack(anchor="w", padx=8, pady=(6, 0))

    def _build_buttons(self, parent):
        sec = self._section(parent, "🎮 KONTROLLER", pady=6)
        btns = [
            ("🔧 Initialize",        self._on_initialize,   THEME["accent"]),
            ("▶ Train",              self._on_train,         THEME["success"]),
            ("⏩ N Episode Run",      self._on_n_episode_run, THEME["accent2"]),
            ("🧪 Test (Visualize)",   self._on_test,          THEME["accent3"]),
            ("⏹ Stop",               self._on_stop,          THEME["danger"]),
            ("🔄 Reset All",          self._on_reset_all,     THEME["warning"]),
            ("🔍 HP Search",          self._on_hp_search,     THEME["accent2"]),
        ]
        self.btn_refs = {}; self.btn_colors = {}
        for lbl, cmd, color in btns:
            b = tk.Button(sec, text=lbl, command=cmd, bg=THEME["bg_card"], fg=color,
                          font=("Consolas",9,"bold"), relief="flat", bd=0, pady=5,
                          cursor="hand2", activebackground=THEME["border"], activeforeground=color)
            b.pack(fill="x", padx=2, pady=2)
            self.btn_refs[lbl] = b; self.btn_colors[lbl] = color
            b.bind("<Enter>", lambda e,btn=b,c=color: btn.config(bg=THEME["border"])
                   if str(btn.cget("state"))=="normal" else None)
            b.bind("<Leave>", lambda e,btn=b: btn.config(
                bg=THEME["bg_card"] if str(btn.cget("state"))=="normal" else THEME["bg_mid"]))

    def _build_center_panel(self):
        center = tk.Frame(self.root, bg=THEME["bg_dark"])
        center.grid(row=0, column=1, sticky="ns", padx=4)
        tk.Label(center, text="🗺️ GRID DÜNYASI", bg=THEME["bg_dark"], fg=THEME["text"],
                 font=("Consolas",10,"bold")).pack(pady=(14,4))
        info_bar = tk.Frame(center, bg=THEME["bg_mid"],
                            highlightbackground=THEME["border"], highlightthickness=1)
        info_bar.pack(fill="x", padx=4, pady=(0,4))
        self.battery_bar_var = tk.StringVar(value=f"⚡ {MAX_BATTERY}/{MAX_BATTERY}")
        self.cleaned_bar_var = tk.StringVar(value="🧹 0/48")
        tk.Label(info_bar, textvariable=self.battery_bar_var, bg=THEME["bg_mid"],
                 fg=THEME["charge"], font=("Consolas",9,"bold")).pack(side="left",padx=8,pady=3)
        tk.Label(info_bar, textvariable=self.cleaned_bar_var, bg=THEME["bg_mid"],
                 fg=THEME["accent"], font=("Consolas",9,"bold")).pack(side="right",padx=8,pady=3)
        grid_px = GRID_SIZE*CELL_PX
        self.grid_canvas = tk.Canvas(center, width=grid_px, height=grid_px,
                                     bg=THEME["bg_dark"], highlightthickness=2,
                                     highlightbackground=THEME["border"])
        self.grid_canvas.pack(pady=4)
        self.step_label_var = tk.StringVar(value="Adım: 0  |  Episode: 0")
        tk.Label(center, textvariable=self.step_label_var, bg=THEME["bg_dark"],
                 fg=THEME["text_dim"], font=("Consolas",8)).pack(pady=(0,4))
        self._draw_grid_init()

    def _draw_grid_init(self):
        if self.env: self._draw_grid(self.env.get_grid_info())
        else:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    x,y = c*CELL_PX, r*CELL_PX
                    t = get_tile("charge" if (r,c)==CHARGE_STATION else "dirt")
                    self.grid_canvas.create_image(x+CELL_PX//2, y+CELL_PX//2, image=t)

    def _draw_grid(self, grid_info):
        self.grid_canvas.delete("all"); self.grid_images = []
        robot_pos  = grid_info["robot_pos"]
        cleaned    = grid_info["cleaned"]
        charge_pos = grid_info["charge_pos"]
        obstacle_pos = set(map(tuple, grid_info["obstacles"]))
        battery    = grid_info["battery"]
        max_bat    = grid_info["max_battery"]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x,y = c*CELL_PX, r*CELL_PX
                if   (r,c)==robot_pos:            tn = "robot_charge" if (r,c)==charge_pos else "robot"
                elif (r,c)==charge_pos:            tn = "charge"
                elif (r,c) in obstacle_pos:        tn = "obstacle"
                elif cleaned[r,c]==1:              tn = "clean"
                else:                              tn = "dirt"
                t = get_tile(tn); self.grid_images.append(t)
                self.grid_canvas.create_image(x+CELL_PX//2, y+CELL_PX//2, image=t)
        self.battery_bar_var.set(f"⚡ {battery}/{max_bat}")
        self.cleaned_bar_var.set(f"🧹 {grid_info['visited_count']}/{grid_info['cleanable_count']}")

    def _build_right_panel(self):
        right = tk.Frame(self.root, bg=THEME["bg_mid"])
        right.grid(row=0, column=2, sticky="nsew", padx=(0,4))
        right.rowconfigure(0,weight=0); right.rowconfigure(1,weight=1); right.columnconfigure(0,weight=1)
        tk.Label(right, text="📈 EĞİTİM GRAFİKLERİ", bg=THEME["bg_mid"], fg=THEME["text"],
                 font=("Consolas",10,"bold")).grid(row=0,column=0,pady=(14,4),sticky="ew")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Chart.TNotebook", background=THEME["bg_mid"], borderwidth=0)
        style.configure("Chart.TNotebook.Tab", background=THEME["bg_card"],
                        foreground=THEME["text_dim"], padding=[8,4], font=("Consolas",8))
        style.map("Chart.TNotebook.Tab",
                  background=[("selected",THEME["bg_dark"])],
                  foreground=[("selected",THEME["accent"])])
        nb = ttk.Notebook(right, style="Chart.TNotebook")
        nb.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0,4))
        self.chart_tabs = {}
        for key, label in [("reward","Ödül"),("success","Başarı"),("steps","Adım"),
                           ("energy","Enerji"),("charge_visits","Şarj")]:
            frame = tk.Frame(nb, bg=THEME["bg_dark"])
            nb.add(frame, text=f"  {label}  ")
            self.chart_tabs[key] = frame
            self.chart_mgr.create_figure(key, frame).get_tk_widget().pack(fill="both", expand=True)
        self.summary_frame = tk.Frame(right, bg=THEME["bg_panel"],
                                      highlightbackground=THEME["border"], highlightthickness=1)
        self.summary_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=(0,4))
        tk.Label(self.summary_frame, text="📋 ÖZET", bg=THEME["bg_panel"], fg=THEME["accent"],
                 font=("Consolas",8,"bold")).pack(anchor="w", padx=6, pady=(4,0))
        self.summary_text = tk.Text(self.summary_frame, bg=THEME["bg_dark"], fg=THEME["text"],
                                    font=("Consolas",7), height=6, relief="flat", state="disabled")
        self.summary_text.pack(fill="x", padx=4, pady=(0,4))
        right.rowconfigure(2, weight=0)

    # ── YARDIMCI METODLAR ─────────────────────

    def _get_hp(self):
        hp = {}
        for key, var in self.hp_vars.items():
            try: hp[key] = var.get()
            except tk.TclError: hp[key] = DEFAULT_HP.get(key, 0)
        return hp

    def _get_obstacles(self):
        return list(get_obstacle_map(self.obstacle_var.get()))

    def _on_obstacle_map_changed(self, *_args):
        if self.env and not self.is_training:
            self.env.obstacles = set(map(tuple, self._get_obstacles()))
            self.env.cleanable_count = (self.env.grid_size**2 - 1 - len(self.env.obstacles))
            self.env.reset(); self._draw_grid(self.env.get_grid_info())
        self._set_status(f"🧱 Engel haritası: {self.obstacle_var.get()}")

    def _update_info(self, info):
        for key, lbl in self.info_labels.items():
            val = info.get(key,"—")
            if isinstance(val, float): val = f"{val:.3f}"
            lbl.config(text=str(val))

    def _clear_info_panel(self):
        for lbl in self.info_labels.values(): lbl.config(text="—")

    def _clear_summary_panel(self):
        self.summary_text.config(state="normal"); self.summary_text.delete("1.0","end")
        self.summary_text.config(state="disabled")

    def _update_control_states(self):
        busy = self.is_training or self.is_testing
        for lbl, btn in self.btn_refs.items():
            enabled = True
            if busy:
                enabled = (lbl=="⏹ Stop")
            else:
                if lbl=="⏹ Stop": enabled=False
                if lbl=="🧪 Test (Visualize)" and (not self.agent or not self.env): enabled=False
            if enabled:
                btn.config(state="normal", bg=THEME["bg_card"],
                           fg=self.btn_colors.get(lbl,THEME["text"]), cursor="hand2")
            else:
                btn.config(state="disabled", bg=THEME["bg_mid"], fg=THEME["text_dim"], cursor="arrow")

    def _set_status(self, msg):
        self.status_var.set(msg); self.root.update_idletasks()

    def _update_summary(self):
        n = self.summary_totals["episodes"]
        if n == 0: return
        sr  = self.summary_totals["success"] / n
        ar  = self.summary_totals["reward"] / n
        ast = self.summary_totals["steps"] / n
        ae  = self.summary_totals["energy"] / n
        ac  = self.summary_totals["charge_visits"] / n
        text = (f"Toplam Episode: {n}\nBaşarı Oranı:   {sr*100:.1f}%\n"
                f"Ort. Ödül:      {ar:.2f}\nOrt. Adım:      {ast:.1f}\n"
                f"Ort. Enerji:    {ae:.1f}\nOrt. Şarj Dön.: {ac:.2f}\n")
        self.summary_text.config(state="normal"); self.summary_text.delete("1.0","end")
        self.summary_text.insert("end",text); self.summary_text.config(state="disabled")

    # ── BUTON KOMUTLARI ───────────────────────

    def _initialize_env_agent(self):
        hp          = self._get_hp()
        hp["use_cuda"] = bool(self.use_cuda_var.get())
        obstacles   = self._get_obstacles()
        use_ddqn    = self.model_var.get()=="DDQN"
        self.env    = CleaningEnv(obstacles=obstacles)
        self.agent  = DQNAgent(self.env.state_dim, self.env.action_dim, hp, use_ddqn=use_ddqn)
        self.run_id = make_run_id()
        self.episode_data = []
        self.summary_totals = {
            "episodes": 0,
            "success": 0.0,
            "reward": 0.0,
            "steps": 0.0,
            "energy": 0.0,
            "charge_visits": 0.0,
        }
        self.chart_mgr.reset()
        self.env.reset(); self._draw_grid(self.env.get_grid_info())
        self.step_label_var.set("Adım: 0  |  Episode: 0")
        self._clear_info_panel(); self._clear_summary_panel()
        self.chart_mgr.update_all(); self._update_control_states()
        self._set_status(
            f"✅ Hazır. Model: {self.model_var.get()} | Cihaz: {self.agent.device.type.upper()} | State dim: {self.env.state_dim}"
        )

    def _on_initialize(self):
        try: self._initialize_env_agent()
        except Exception as e: messagebox.showerror("Hata", str(e))

    def _on_train(self):
        if self.is_training:
            messagebox.showwarning("Uyarı","Eğitim devam ediyor!"); return
        if not self.agent or not self.env: self._initialize_env_agent()
        self.stop_flag  = [False]
        self.is_training = True
        self._update_control_states(); self._set_status("🏃 Eğitim başladı...")
        hp = self._get_hp(); n = int(hp.get("n_episodes",1000)); ep_start = len(self.episode_data)

        def callback(ep_idx, info, grid_info):
            self.root.after(0, lambda g=grid_info,i=info,e=ep_idx: self._on_episode_result(g,i,e))

        def train_thread():
            try:
                run_episodes(self.agent, self.env, n, hp, callback=callback,
                             stop_flag=self.stop_flag, train_mode=True, episode_start=ep_start)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Hata",str(e)))
            finally:
                self.root.after(0, self._on_train_done)

        self.train_thread = threading.Thread(target=train_thread, daemon=True)
        self.train_thread.start()

    def _on_episode_result(self, grid_info, info, ep_idx):
        self.episode_data.append(info)
        self.summary_totals["episodes"] += 1
        self.summary_totals["success"] += float(info.get("success", 0.0))
        self.summary_totals["reward"] += float(info.get("reward", 0.0))
        self.summary_totals["steps"] += float(info.get("steps", 0.0))
        self.summary_totals["energy"] += float(info.get("energy", 0.0))
        self.summary_totals["charge_visits"] += float(info.get("charge_visits", 0.0))
        self.chart_mgr.add_episode(info)
        self._update_train_ui(
            grid_info,
            info,
            ep_idx,
            draw_grid=bool(self.render_grid_train_var.get()),
            update_charts=True,
            update_summary=True,
            update_status=(ep_idx % 10 == 0),
        )

    def _update_train_ui(self, grid_info, info, ep_idx, update_charts=True,
                         update_summary=True, update_status=True, draw_grid=True):
        if draw_grid:
            self._draw_grid(grid_info)
        else:
            self.battery_bar_var.set(f"⚡ {grid_info['battery']}/{grid_info['max_battery']}")
            self.cleaned_bar_var.set(f"🧹 {grid_info['visited_count']}/{grid_info['cleanable_count']}")
        sr = np.mean([d["success"] for d in self.episode_data[-50:]]) if self.episode_data else 0
        info["success_rate"] = f"{sr*100:.1f}%"
        info["battery"]      = f"{grid_info['battery']}/{grid_info['max_battery']}"
        info["cleaned"]      = f"{grid_info['visited_count']}/{grid_info['cleanable_count']}"
        self._update_info(info)
        self.step_label_var.set(f"Adım: {info['steps']}  |  Episode: {info['episode']}")
        if update_charts:
            self.chart_mgr.update_all(self.run_id, capture_frame=(ep_idx%20==0))
        if update_summary:
            self._update_summary()
        if update_status:
            self._set_status(f"🏃 Eğitim devam ediyor... Episode {info['episode']}")

    def _on_train_done(self):
        self.is_training = False; self._update_control_states()
        self._set_status(f"✅ Eğitim tamamlandı! ({len(self.episode_data)} episode)")
        self._update_summary(); self.chart_mgr.update_all(self.run_id, capture_frame=True)
        try:
            save_episode_log(self.run_id, self.episode_data)
            mp = MODEL_DIR/f"model_{self.run_id}.pt"
            self.agent.save(str(mp)); self.chart_mgr.save_gifs(self.run_id)
            self._set_status(f"✅ Tamamlandı. Model: {mp.name}")
        except Exception as e: self._set_status(f"⚠️ Kayıt hatası: {e}")

    def _on_n_episode_run(self):
        if self.is_training: messagebox.showwarning("Uyarı","Eğitim devam ediyor!"); return
        if not self.agent or not self.env: self._initialize_env_agent()
        hp = self._get_hp(); n = min(int(hp.get("n_episodes",50)), 100); ep_start = len(self.episode_data)
        self.stop_flag=[False]; self.is_training=True
        self._update_control_states(); self._set_status(f"⏩ {n} episode...")

        def callback(ep_idx,info,grid_info):
            self.root.after(0,lambda g=grid_info,i=info,e=ep_idx: self._on_episode_result(g,i,e))
        def run_thread():
            try: run_episodes(self.agent,self.env,n,hp,callback=callback,
                              stop_flag=self.stop_flag,train_mode=True,episode_start=ep_start)
            finally: self.root.after(0,self._on_n_run_done)
        self.train_thread=threading.Thread(target=run_thread,daemon=True); self.train_thread.start()

    def _on_n_run_done(self):
        self.is_training=False; self._update_control_states(); self._set_status("✅ N Episode eğitimi tamamlandı.")

    def _on_test(self):
        if self.is_training or self.is_testing:
            messagebox.showwarning("Uyarı","Eğitim devam ediyor!"); return
        if not self.agent or not self.env:
            messagebox.showwarning("Uyarı","Önce Initialize edin!"); return
        self.stop_flag=[False]; self.is_testing=True
        self._update_control_states(); state=self.env.reset()
        self._draw_grid(self.env.get_grid_info()); self._set_status("🧪 Test görselleştirmesi...")

        def test_step(step_n,state):
            if self.stop_flag[0] or self.env.done or self.env.step_count >= self.env.max_steps:
                self.is_testing=False; self._update_control_states()
                self._set_status(f"✅ Test bitti. Temizlenen: {self.env.visited_count}/{self.env.cleanable_count}")
                return
            action=select_policy_action(self.agent, self.env, state, force_greedy=True)
            next_state,reward,done,info=self.env.step(action)
            self._draw_grid(self.env.get_grid_info())
            self.step_label_var.set(f"Adım: {self.env.step_count}  |  Ödül: {reward:.2f}")
            self.root.after(100, lambda s=next_state,n=step_n+1: test_step(n,s))

        test_step(0,state)

    def _on_stop(self):
        self.stop_flag[0]=True
        self._set_status("⏹ Durduruluyor..." if (self.is_training or self.is_testing) else "⏹ Durduruldu.")

    def _on_reset_all(self):
        if self.is_training or self.is_testing:
            self.stop_flag[0]=True
            messagebox.showwarning("Uyarı","Eğitim sürüyor. Önce Stop ile bekleyin."); return
        self.stop_flag[0]=True; self.agent=None; self.env=None; self.episode_data=[]
        self._clear_info_panel(); self.chart_mgr.reset(); self.chart_mgr.update_all()
        for key,var in self.hp_vars.items():
            try: var.set(DEFAULT_HP.get(key,0))
            except Exception: pass
        self._initialize_env_agent(); self._set_status("🔄 Tümü sıfırlandı.")

    def _on_hp_search(self):
        if self.is_training or self.is_testing:
            messagebox.showwarning("Uyarı","Eğitim devam ediyor!"); return
        if not messagebox.askyesno("HP Arama","Hiperparametre araması başlatılsın mı?"): return

        base_hp      = self._get_hp()
        base_hp["use_cuda"] = bool(self.use_cuda_var.get())
        use_ddqn     = self.model_var.get()=="DDQN"
        obstacles    = self._get_obstacles()
        env_kwargs   = {"obstacles": obstacles}
        self.stop_flag=[False]; self.is_training=True
        self._update_control_states(); self._set_status("🔍 HP araması başladı...")

        pw = tk.Toplevel(self.root); pw.title("HP Arama"); pw.configure(bg=THEME["bg_dark"]); pw.geometry("900x560")
        pl = tk.Label(pw,text="Başlatılıyor...",bg=THEME["bg_dark"],fg=THEME["text"],font=("Consolas",9),wraplength=860,justify="left")
        pl.pack(pady=10,padx=10)
        bv = tk.StringVar(value="En iyi: —")
        tk.Label(pw,textvariable=bv,bg=THEME["bg_dark"],fg=THEME["success"],font=("Consolas",9,"bold"),wraplength=860,justify="left").pack(pady=(0,6),padx=10,anchor="w")
        pb = ttk.Progressbar(pw,length=350,mode="determinate"); pb.pack(pady=5)
        rt = tk.Text(pw,bg=THEME["bg_card"],fg=THEME["text"],font=("Consolas",8),height=18,relief="flat")
        rt.pack(fill="both",expand=True,padx=10,pady=5)

        def progress_cb(done,total,result):
            self.root.after(0,lambda d=done,t=total,r=result: _apply(d,t,r))
        def _apply(done,total,result):
            pb["value"]=done/total*100; pl.config(text=f"Kombinasyon {done}/{total}")
            rt.insert("end",f"[{done}] score={result.get('score','—')} sr={result.get('success_rate','—')} lr={result.get('learning_rate','—')}\n"); rt.see("end")
            if float(result.get("score",-1e9)) > float(bv.get().split("=")[-1] if "=" in bv.get() else "-1e9"):
                bv.set(f"En iyi: score={result.get('score','—')} sr={result.get('success_rate','—')}")

        def search_thread():
            try:
                best_hp,all_results = hp_search(base_hp,env_kwargs,use_ddqn=use_ddqn,
                                                 training_budget_steps=10000,eval_window=100,
                                                 progress_cb=progress_cb,stop_flag=self.stop_flag)
                self.root.after(0,lambda: self._on_hp_search_done(best_hp,all_results,pw))
            except Exception as e:
                import traceback
                err_str = traceback.format_exc()
                self.root.after(0,lambda err=err_str: (messagebox.showerror("Hata", err),
                                           setattr(self,"is_training",False),
                                           self._update_control_states()))

        self.hp_thread=threading.Thread(target=search_thread,daemon=True); self.hp_thread.start()

    def _on_hp_search_done(self, best_hp, results, pw):
        self.is_training=False; self._update_control_states()
        for key,var in self.hp_vars.items():
            if key in best_hp:
                try: var.set(best_hp[key])
                except Exception: pass
        self.model_var.set("DDQN" if best_hp.get("use_ddqn",False) else "DQN")
        self._initialize_env_agent()
        best = max(results,key=lambda x:x.get("score",-1e9)) if results else {}
        messagebox.showinfo("HP Arama Tamamlandı",
                            f"En iyi skor: {best.get('score','—')}\nBaşarı: {best.get('success_rate','—')}\nToplam: {len(results)}")
        try: pw.destroy()
        except Exception: pass


# ─────────────────────────────────────────────
# GİRİŞ NOKTASI
# ─────────────────────────────────────────────

def main():
    torch.set_num_threads(2)
    root = tk.Tk()
    style = ttk.Style(root); style.theme_use("default")
    style.configure("TScrollbar", background=THEME["bg_card"],
                    troughcolor=THEME["bg_dark"], arrowcolor=THEME["text_dim"])
    app = App(root)
    root.update_idletasks()
    w,h = root.winfo_width(),root.winfo_height()
    sw,sh = root.winfo_screenwidth(),root.winfo_screenheight()
    root.geometry(f"{max(w,1280)}x{max(h,800)}+{(sw-max(w,1280))//2}+{(sh-max(h,800))//2}")
    root.mainloop()

if __name__ == "__main__":
    main()