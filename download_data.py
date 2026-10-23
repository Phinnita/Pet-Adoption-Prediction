"""ดาวน์โหลดชุดข้อมูล PetFinder (mini) มาไว้ที่ petfinder-mini.csv (รันครั้งเดียว)"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
OUT = Path(__file__).parent / "petfinder-mini.csv"

print("กำลังดาวน์โหลด", URL)
raw = urllib.request.urlopen(URL, timeout=60).read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    OUT.write_bytes(z.read(name))
print("เสร็จแล้ว ->", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
