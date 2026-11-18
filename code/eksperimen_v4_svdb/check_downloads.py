"""Pemeriksaan kelengkapan dan keutuhan hasil uji tambahan yang diunduh dari Colab (tanpa membaca hold-out).

Untuk setiap backbone dan setiap id uji (colab_tests_setup.urutan) diperiksa:
  - berkas wajib ada (selesai.json, uji.json, log, kunci protokol, checkpoint final, CV dan seleksi untuk U3/U5,
    encoder acak untuk U5);
  - sha256 best.pt final = selesai.json; isi uji.json (kunci, hash protokol/metode, seed, E*, GPU, holdout_dibaca);
  - G0L4 identik dengan model resmi (sha, atau bila sha beda: semua tensor checkpoint sama, kecuali waktu tempuh); U4 S3 memakai sumber U4 S2 seed yang sama; U5 encoder_acak.pt = selesai.json;
  - E* dan seed final sesuai aturan; perangkat L4.
Tidak ada angka hold-out yang dihitung atau dicetak.

Pakai:  python check_downloads.py <folder hasil uji tambahan>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "colab"))
import colab_tests as U  # noqa: E402

ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]
PROTO, METODE = "9db60710fd60433663bab15e1bbe3f61ba0b97cdf0ef59f0fc6531d5d510735d", \
    "d7f73001bb35e08774efdbfb9bdb487fadf4dff0a7b3c5bcdc6df48966c71569"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def g0l4_beda(arch, path):
    """'' bila semua isi checkpoint G0L4 sama dengan model final S2 resmi kecuali waktu tempuh; selain itu uraian beda."""
    import torch
    a = torch.load(path, map_location="cpu", weights_only=False)
    b = torch.load(HERE / "outputs_colab_r15" / "final_models" / arch / "S2" / "final" / "best.pt",
                   map_location="cpu", weights_only=False)
    abaikan = {"elapsed", "elapsed_s"}

    def sama(x, y):
        if torch.is_tensor(x) or torch.is_tensor(y):
            return torch.is_tensor(x) and torch.is_tensor(y) and x.shape == y.shape and torch.equal(x, y)
        if isinstance(x, dict) and isinstance(y, dict):
            return set(x) - abaikan == set(y) - abaikan and all(sama(x[k], y[k]) for k in x if k not in abaikan)
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return len(x) == len(y) and all(sama(p, q) for p, q in zip(x, y))
        return type(x) is type(y) and repr(x) == repr(y)

    if set(a) != set(b):
        return f"kunci berbeda {sorted(set(a) ^ set(b))}"
    return ", ".join(k for k in a if k not in abaikan and not sama(a[k], b[k]))


def urutan(arch):
    out = [("G0L4", "S2", None, None, None), ("U2", "S2", None, None, None), ("U3", "S2", None, None, None),
           ("U5", "S2", None, None, None)]
    out += [("U4", sc, s, None, None) for s in U.SEEDS_U4 for sc in ("S1", "S2")]
    out += [("U4", "S3", s, None, None) for s in U.SEEDS_U4]
    if arch == "efficientnet_b0":
        out += [("U6", "S2", None, f, r) for r in U.U6_ULANG for f in U.U6_FRAKSI]
    return out


def periksa(base: Path):
    temuan, ringkas = [], []
    for arch in ARCHS:
        root_arch = base / arch / "TA_R15_uji_tambahan_hasil"
        ids = []
        for uji, sc, seed, fr, ul in urutan(arch):
            uid = U.uji_id(uji, arch, sc, seed, False, fr, ul); ids.append(uid)
            r = root_arch / uid; kurang = []
            sk = "TL_murni_S2" if uji == "U3" else sc
            fin = r / "final_models" / arch / sk / "final"
            wajib = [r / "selesai.json", r / "uji.json", r / "protocol_lock.json", r / "v4_method_lock.json",
                     r / "final_method_lock.json", r / "logs" / f"{arch}_{sk}.log"]
            wajib += [fin / n for n in ("best.pt", "last.pt", "result.json", "history.csv", "initialization_anchor.npz")]
            if sc == "S3":
                wajib += [r / "v4_s3_method_lock.json"]
                wajib += [r / "final_models" / arch / "S2" / "final" / n for n in ("best.pt", "result.json")]
            if uji in ("U3", "U5"):
                for k in range(5):
                    fd = r / "cv" / arch / sk / f"fold_{k}"
                    wajib += [fd / n for n in ("best.pt", "result.json", "history.csv")]
                wajib += [r / "selection" / arch / f"{sk}_final_epoch.json"]
            if uji == "U5":
                wajib += [r / "encoder_acak.pt", r / "encoder_acak.json"]
            kurang = [str(p.relative_to(r)) for p in wajib if not p.exists()]
            info = {"arch": arch, "uji": uid, "kurang": kurang}
            if not r.exists():
                temuan.append(f"{uid}: folder tidak ada"); ringkas.append(info); continue
            s = json.loads((r / "selesai.json").read_text()) if (r / "selesai.json").exists() else {}
            if not s:
                temuan.append(f"{uid}: selesai.json tidak ada (uji belum selesai atau tidak terunduh)")
            if (fin / "best.pt").exists() and s and sha(fin / "best.pt") != s["checkpoint_sha256"]:
                temuan.append(f"{uid}: sha best.pt final BERBEDA dari selesai.json (berkas rusak/terpotong)")
            if (fin / "result.json").exists() and s:
                rj = json.loads((fin / "result.json").read_text())
                if rj.get("checkpoint_sha256") != s["checkpoint_sha256"]:
                    temuan.append(f"{uid}: result.json final tidak cocok dengan selesai.json")
            if s:
                if "L4" not in s.get("device", ""):
                    temuan.append(f"{uid}: perangkat {s.get('device')} bukan L4")
                harap_seed = seed if seed else 142
                if s.get("seed_final") != harap_seed:
                    temuan.append(f"{uid}: seed final {s.get('seed_final')} != {harap_seed}")
                if uji in ("G0L4", "U2", "U4", "U6"):
                    e_res = (U.RESMI_S3[arch][0] if sc == "S3" else U.RESMI[(arch, sc)][0])
                    if s.get("E*") != e_res:
                        temuan.append(f"{uid}: E* {s.get('E*')} != E* resmi {e_res}")
                if s.get("epochs_run") != s.get("E*"):
                    temuan.append(f"{uid}: epochs_run {s.get('epochs_run')} != E* {s.get('E*')}")
                if uji == "G0L4" and not s.get("sha_sama_dengan_resmi") and (fin / "best.pt").exists():
                    # sha checkpoint ikut memuat waktu tempuh (elapsed, history.elapsed_s): bandingkan tensornya
                    beda = g0l4_beda(arch, fin / "best.pt")
                    if beda:
                        temuan.append(f"{uid}: G0L4 TIDAK identik dengan model resmi ({beda})")
                    else:
                        info["G0L4"] = "tensor identik; sha beda hanya karena waktu tempuh"
                if uji == "U5" and (r / "encoder_acak.pt").exists() and sha(r / "encoder_acak.pt") != s.get("encoder_acak_sha256"):
                    temuan.append(f"{uid}: encoder_acak.pt berbeda dari selesai.json")
                info.update({"E*": s.get("E*"), "sha": s["checkpoint_sha256"][:8], "menit": round(s.get("training_s", 0) / 60, 1)})
            if (r / "uji.json").exists():
                u = json.loads((r / "uji.json").read_text())
                if u.get("asap"):
                    temuan.append(f"{uid}: uji.json menandai asap=True")
                if u.get("holdout_dibaca") is not False:
                    temuan.append(f"{uid}: uji.json tidak mencatat holdout_dibaca=False")
                if u.get("hash", {}).get("data_protocol_hash") != PROTO or u.get("hash", {}).get("method_hash") != METODE:
                    temuan.append(f"{uid}: hash protokol/metode di uji.json berbeda")
                if sc == "S3" and u.get("hash", {}).get("s3_method_hash", U.S3_HASH) != U.S3_HASH:
                    temuan.append(f"{uid}: hash metode S3 berbeda")
                if sc == "S3":
                    src = root_arch / U.uji_id("U4", arch, "S2", seed, False)
                    s2 = json.loads((src / "selesai.json").read_text()) if (src / "selesai.json").exists() else {}
                    if u.get("sumber_S2_sha256") != s2.get("checkpoint_sha256"):
                        temuan.append(f"{uid}: sumber S2 ({str(u.get('sumber_S2_sha256'))[:8]}) != U4 S2 seed {seed}")
            if kurang:
                temuan.append(f"{uid}: {len(kurang)} berkas wajib tidak ada: {kurang[:6]}{' ...' if len(kurang) > 6 else ''}")
            ringkas.append(info)
        lebih = sorted(p.name for p in root_arch.iterdir() if p.is_dir() and p.name not in ids) if root_arch.exists() else []
        if lebih:
            temuan.append(f"{arch}: folder tambahan tak dikenal {lebih}")
    return temuan, ringkas


if __name__ == "__main__":
    base = Path(sys.argv[1])
    temuan, ringkas = periksa(base)
    for i in ringkas:
        print(f"{i['arch'][:12]:12s} {i['uji']:36s} E*={i.get('E*')!s:>3} sha={i.get('sha', '-'):8s} "
              f"menit={i.get('menit', '-')!s:>6} kurang={len(i['kurang'])}" + (f"  G0L4: {i['G0L4']}" if "G0L4" in i else ""))
    print(f"\n{len(ringkas)} uji diperiksa; {len(temuan)} temuan")
    for t in temuan:
        print(" -", t)
