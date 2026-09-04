"""Bandingkan hasil eval_perilaku.py antar rilis, berdampingan.

    .venv\\Scripts\\python.exe scripts\\banding_eval.py hasil_v4.json hasil_v5.json

Satu perubahan per rilis. Dengan disiplin itu, selisih kolom menunjuk langsung
ke perubahannya — cukup untuk memutuskan lanjut atau batal.
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URUT = ["persona", "kemampuan", "terbuka", "instruksi", "percakapan",
        "sapaan", "pengetahuan"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("hasil", nargs="+", help="berkas JSON dari --simpan")
    args = p.parse_args()

    data = []
    for f in args.hasil:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        d["nama"] = Path(d["ckpt"]).parent.name.replace("wicara-56m-chat-", "")
        data.append(d)

    set_eval = {d["set"] for d in data}
    if len(set_eval) > 1:
        print(f"PERINGATAN: set berbeda ({set_eval}), tidak sebanding")

    lebar = max(len(d["nama"]) for d in data) + 2
    print(f"\n  set {data[0]['set']} · {data[0]['n']} sampel/prompt · "
          f"seed {data[0]['seed']}\n")
    print("  " + "kategori".ljust(14) +
          "".join(d["nama"].rjust(lebar) for d in data) + "     Δ")
    print("  " + "-" * (14 + lebar * len(data) + 7))

    kat = [k for k in URUT if k in data[0]["skor"]]
    for k in kat:
        baris = "  " + k.ljust(14)
        nilai = []
        for d in data:
            s = d["skor"].get(k)
            v = s["lolos"] / s["total"] if s else None
            nilai.append(v)
            baris += (f"{v:>{lebar}.0%}" if v is not None else "-".rjust(lebar))
        if len(nilai) >= 2 and None not in (nilai[0], nilai[-1]):
            d_pp = (nilai[-1] - nilai[0]) * 100
            baris += f"  {d_pp:>+5.0f}"
        print(baris)

    print("  " + "-" * (14 + lebar * len(data) + 7))
    tot = []
    baris = "  " + "TOTAL".ljust(14)
    for d in data:
        n = sum(s["lolos"] for s in d["skor"].values())
        t = sum(s["total"] for s in d["skor"].values())
        tot.append(n / t)
        baris += f"{n / t:>{lebar}.0%}"
    baris += f"  {(tot[-1] - tot[0]) * 100:>+5.0f}"
    print(baris)

    # Rilis sebelum gema diperkenalkan tidak punya kuncinya.
    for nama, kunci in (("degenerasi", "degenerasi"), ("gema", "gema")):
        baris = "  " + nama.ljust(14)
        for d in data:
            t = sum(s["total"] for s in d["skor"].values())
            baris += (f"{d[kunci] / t:>{lebar}.0%}" if kunci in d
                      else "-".rjust(lebar))
        print(baris)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
