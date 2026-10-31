"""Build the portable Colab notebook and POSIX-member ZIP archives."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def cell(source, markdown=False):
    result = {"cell_type": "markdown" if markdown else "code", "id": hashlib.sha256(source.encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if not markdown:
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook():
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [
            cell("""# Hakem deneyleri: kontrollu yeniden egitim

99 is; 3 seed; orijinal birlesik-cikti egitimi + bagimsiz CE ogrenci.
GPU secin. Tek notebook tum deneyleri sirayla yonetir; tek Colab oturumunda
bitmesi garanti degildir. Her epoch Drive'a dogrulanarak kaydedilir.
Kesinti sonrasi ayni RUN_ID ile bastan kurulum hucrelerini calistirin.

Drive'da en az **85 GiB bos alan** planlayin; 15 GB hesap yeterli degildir.
Eski PTH dosyalarini silmeyin. Bu paket eski deneylere dokunmaz.

MyDrive/DR_Reviewer_Complete/ klasorune kod ZIP'i, veri ZIP'i ve resmi
`B. Disease Grading.zip` arsivini yukleyin. Yeni sayilar elde edilene kadar
eski makale sonuclarini yeni deney sonucu olarak kullanmayin.
""", True),
            cell("""from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
PROJECT = Path('/content/drive/MyDrive/DR_Reviewer_Complete')
RUN_ID = 'reviewer_complete_v1'
JOBS_THIS_SESSION = None  # None: all unfinished jobs; e.g. 3: stop after three jobs.
STORAGE_CONFIRMED = False  # Manually check Drive account quota, then set True.
if not STORAGE_CONFIRMED:
    raise RuntimeError('Drive hesabinda en az 85 GiB bos alan oldugunu kontrol edip STORAGE_CONFIRMED=True yapin.')
if not RUN_ID or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in RUN_ID):
    raise ValueError('Use a simple, new RUN_ID; do not reuse an old experiment.')
CODE_ZIP = PROJECT / 'DR_Reviewer_Complete_Colab.zip'
DATA_ZIP = PROJECT / 'DR_Reviewer_Complete_Data.zip'
IDRID_ZIP = PROJECT / 'B. Disease Grading.zip'
for path in (CODE_ZIP, DATA_ZIP, IDRID_ZIP):
    if not path.is_file():
        raise FileNotFoundError(path)
REMOTE = PROJECT / 'runs' / RUN_ID
LOCAL = Path('/content/reviewer_runs') / RUN_ID
REMOTE.mkdir(parents=True, exist_ok=True)
LOCAL.mkdir(parents=True, exist_ok=True)
"""),
            cell("""import hashlib, json, subprocess, sys, zipfile
from pathlib import PurePosixPath
WORK = Path('/content/reviewer_workspace')
WORK.mkdir(exist_ok=True)
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def unpack(path, destination):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts or chr(92) in name:
                raise ValueError('Unsafe/nonportable ZIP member: ' + name)
        archive.extractall(destination)
unpack(CODE_ZIP, WORK)
PACKAGE = WORK / 'colab_reviewer_complete'
manifest = json.loads((PACKAGE / 'MANIFEST.json').read_text())
for name, expected in manifest.items():
    if sha256(PACKAGE / name) != expected:
        raise ValueError('Package hash mismatch: ' + name)
DATA = WORK / 'APTOS2019'
if not (DATA / 'cache_224.npz').exists():
    unpack(DATA_ZIP, WORK)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PACKAGE / 'requirements.txt')], check=True)
import torch, torchvision, timm
print('Torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'timm:', timm.__version__)
if not torch.cuda.is_available():
    raise RuntimeError('Colab GPU runtime secilmeli.')
print('GPU:', torch.cuda.get_device_name(0))
def execute(script, *arguments):
    command = [sys.executable, str(PACKAGE / script), *map(str, arguments)]
    subprocess.run(command, cwd=PACKAGE, check=True)
COMMON = ['--data', DATA, '--output', LOCAL, '--drive', REMOTE]
"""),
            cell("""## Kontroller ve gercek GPU smoke testi
Bu kontroller deney sonucu degildir. Pretrained agirliklar yuklenir;
ileri/geri gecis, boyutlar, ogretmen gradient durdurma ve kayit yapisi kontrol edilir.
""", True),
            cell("""execute('test_experiments.py')
execute('smoke.py', '--pretrained')
execute('run.py', '--stage', 'preflight', *COMMON)
"""),
            cell("""## Egitim / kesintiden devam
Bu hucreyi yeniden calistirmak biten isleri atlar. Yeni bir Colab oturumunda
da once yukaridaki kurulum ve kontrolleri calistirin. Dogrulama QWK epoch
secimi icin kullanilir; test seti egitimde degerlendirilmez.
""", True),
            cell("""limit = [] if JOBS_THIS_SESSION is None else ['--limit', JOBS_THIS_SESSION]
execute('run.py', '--stage', 'train', *COMMON, *limit)
expected = json.loads((PACKAGE / 'MANIFEST.json').read_text())
sys.path.insert(0, str(PACKAGE))
from plan import matrix
completed = [r.key for r in matrix() if (LOCAL / r.key / 'complete.json').exists()]
print(f'Tamamlanan: {len(completed)}/99')
if len(completed) < 99:
    raise RuntimeError('Matris henuz bitmedi. Egitim hucresini tekrar calistirin; test/analiz asamasina gecmeyin.')
"""),
            cell("""## Sabit protokolle test ve analiz
Tum egitimler bittikten sonra: farkli ciktilar, mean/sample-SD,
validation-only temperature calibration, 10.000 eslenik bootstrap,
guvenilirlik grafikleri ve ogrenci/full confusion matrisleri.
""", True),
            cell("""execute('run.py', '--stage', 'evaluate', *COMMON)
execute('analyze.py', '--stage', 'analyze', *COMMON)
"""),
            cell("""## Ayni GPU'da gercek yol bazli hiz/bellek olcumu
FP32; batch 1/16; 20 isinma; 100 tekrar; uc seed.
Aktif yol disindaki modeller GPU'da tutulmaz. I/O ve preprocessing harictir.
""", True),
            cell("""execute('analyze.py', '--stage', 'benchmark', *COMMON)
"""),
            cell("""## IDRiD ve sabit secimli nitel Grad-CAM
103 resmi test goruntusu; egitim/kalibrasyon/esik ayari yapilmaz.
Seed 42 icin her gercek sinifin ilk APTOS/IDRiD test ornegi gosterilir.
Bu, lezyon yerellesimi dogrulamasi veya klinik onay degildir.
""", True),
            cell("""execute('external.py', '--zip', IDRID_ZIP, *COMMON)
required = ['analysis/status.json', 'analysis/mean_sd.json', 'analysis/calibration.json',
            'analysis/paired_qwk.json', 'benchmark/matched_gpu.json',
            'external/metrics.json', 'external/gradcam_manifest.json']
for name in required:
    if not (REMOTE / name).is_file():
        raise FileNotFoundError('Persistent output missing: ' + name)
print('Deney ve analiz dosyalari Drive uzerinde:', REMOTE)
print('Sonuclarin olumlu olmasi varsayilmadi; sayilari inceledikten sonra hakem cevaplarini guncelleyin.')
"""),
        ],
    }


def build(with_data=False):
    destination = ROOT.parent
    nb = notebook()
    # Compile every generated code cell before distributing it.
    for index, item in enumerate(nb["cells"]):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"notebook_cell_{index}", "exec")
    nbpath = destination / "DR_Reviewer_Complete_Colab.ipynb"
    nbpath.write_text(json.dumps(nb, indent=2, ensure_ascii=True), encoding="utf-8")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix not in {".pyc", ".zip"} and p.name != "MANIFEST.json")
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    (ROOT / "MANIFEST.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    package = destination / "DR_Reviewer_Complete_Colab.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [ROOT / "MANIFEST.json"]:
            archive.write(path, arcname=f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}")
    if with_data:
        with zipfile.ZipFile(destination / "DR_Reviewer_Complete_Data.zip", "w", compression=zipfile.ZIP_STORED) as archive:
            for name in ("train.csv", "cache_224.npz"):
                archive.write(destination / "APTOS2019" / name, arcname=f"APTOS2019/{name}")
    outputs = [nbpath, package]
    if with_data:
        outputs.append(destination / "DR_Reviewer_Complete_Data.zip")
    print(json.dumps({p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in outputs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-data", action="store_true")
    build(parser.parse_args().with_data)
