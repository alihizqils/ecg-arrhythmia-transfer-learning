"""Head V4: NonlinearHead warisan + expert tabular tingkat window.

Pada head warisan, keputusan N/S/V dibuat oleh cabang beat lokal; fitur RR,
fisiologi, dan gelombang P tingkat window hanya masuk ke cabang global
(kelompok beat vs AF vs VT) sebagai 32 dimensi di samping 1.280 dimensi citra.
Analisis development fold 0 (sebelum holdout dibuka) menunjukkan embedding
ImageNet beku hampir tidak membawa informasi S, sedangkan 64 fitur tabular
global tersebut membawa sebagian besar informasinya.

Expert ini hanya membaca fitur tabular window (tanpa citra) dan menghasilkan
logit lima kelas yang dijumlahkan ke log-probabilitas head warisan (product of
experts). Lapisan terakhirnya diinisialisasi nol sehingga model awal identik
dengan head warisan. Expert juga diberi loss klasifikasi sendiri agar tetap
belajar walaupun jalur citra lebih cepat menurunkan loss gabungan (masalah
"greedy/modality imbalance" pada pembelajaran multimodal).
"""
from __future__ import annotations

import torch
from torch import nn

from tools.a5_physiology import GLOBAL_PHYSIOLOGY_NAMES
from tools.a6_pb_nonlinear_fusion import NonlinearHead
from tools.a6_wave_delineation import GLOBAL_WAVE_NAMES
from tools.a5_physiology import LOCAL_PHYSIOLOGY_NAMES
from tools.a6_wave_delineation import LOCAL_WAVE_NAMES
from v4_record_template import BEAT_RECORD_TEMPLATE_NAMES, DOMINANT_RR_NAMES, RECORD_TEMPLATE_NAMES
from v4_second_lead import SECOND_LEAD_NAMES
from v4_local_rhythm import BEAT_NAMES as LOCAL_RHYTHM_BEAT_NAMES, WINDOW_NAMES as LOCAL_RHYTHM_WINDOW_NAMES

# Sama dengan masukan cabang global_rr A6: 10 RR global + 12 fisiologi + 10 gelombang,
# masing-masing beserta penanda nilai hilang.
# Revisi 6: ditambah fitur morfologi relatif template rekaman (v4_record_template.py).
# Revisi 10: ditambah fitur RR antar-beat dominan (blok yang sama dengan template rekaman).
WINDOW_TABULAR_INPUTS = 20 + 2 * (len(GLOBAL_PHYSIOLOGY_NAMES) + len(GLOBAL_WAVE_NAMES) + len(RECORD_TEMPLATE_NAMES)
                                 + len(DOMINANT_RR_NAMES) + len(SECOND_LEAD_NAMES)  # Revisi 11: + lead kedua
                                 + len(LOCAL_RHYTHM_WINDOW_NAMES))  # Revisi 14: + ritme lokal E1/E3

# Masukan proyeksi tabular cabang beat A6 (14 RR lokal + fisiologi + gelombang, dengan
# penanda nilai hilang). Revisi 7: ditambah fitur template rekaman per beat.
# Revisi 14: + fitur ritme lokal E1/E3 per beat (blok yang sama dengan template rekaman per beat).
BEAT_TABULAR_INPUTS = 14 + 2 * (len(LOCAL_PHYSIOLOGY_NAMES) + len(LOCAL_WAVE_NAMES) + len(BEAT_RECORD_TEMPLATE_NAMES)
                               + len(LOCAL_RHYTHM_BEAT_NAMES))


class WindowTabularExpertHead(NonlinearHead):
    def __init__(self, dim=1280, dropout=0.2, hidden=64, expert_weight=1.0, vt_gate_power=1.0):
        super().__init__(dim, dropout)
        # Revisi 8: bobot expert dan ketajaman gate VT (metode_v4.yaml, bagian head).
        self.expert_weight, self.vt_gate_power = float(expert_weight), float(vt_gate_power)
        self.window_expert = nn.Sequential(
            nn.Linear(WINDOW_TABULAR_INPUTS, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 5),
        )
        nn.init.zeros_(self.window_expert[-1].weight)
        nn.init.zeros_(self.window_expert[-1].bias)
        # Revisi 7: proyeksi tabular cabang beat menerima fitur template per beat.
        self.local_rr = nn.Sequential(nn.Linear(BEAT_TABULAR_INPUTS, 32), nn.ReLU(), nn.LayerNorm(32))

    def forward(self, global_x, local_x, mask, global_rr, local_rr, rr_sequence,
                physiology_global, physiology_local, wave_global, wave_local, record_template,
                beat_record_template):
        # Fitur per beat digabung ke blok fitur lokal sehingga masuk proyeksi tabular cabang beat.
        output = super().forward(global_x, local_x, mask, global_rr, local_rr, rr_sequence,
                                 physiology_global, physiology_local, wave_global,
                                 torch.cat([wave_local, beat_record_template], dim=-1))
        expert = self.window_expert(torch.cat([global_rr, physiology_global, wave_global, record_template], dim=-1))
        # Gate VT: fitur tabular tidak dapat mengenali VT (pada pilot fold 1, expert
        # saja memberi F1 VT 0,29 dan mendorong window VT ke N), sedangkan jalur
        # citra warisan andal untuk VT (0,92). Kontribusi expert dikalikan
        # 1 - P(VT) jalur warisan (tanpa gradien) agar tidak menimpa keputusan VT.
        gate = (1.0 - output["logp"][:, 4:5].exp().detach()) ** self.vt_gate_power
        output["expert_logits"] = expert
        output["expert_gate"] = gate
        output["logp"] = (output["logp"] + self.expert_weight * gate * expert).log_softmax(-1)
        return output
