"""Repair legacy prediction IDs without unpickling or changing numeric arrays.

Run outside the frozen training package. Recover row identities from the exact
manifest order used by evaluate(), checking all row labels against train.csv.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair(path, ids, labels):
    ids = np.asarray(ids, dtype=str)
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(set(members)) != len(members) or "ids.npy" not in members or "labels.npy" not in members:
            raise ValueError(f"Unexpected NPZ members: {path}")
        with archive.open("ids.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported IDs NPY header: {version}")
        if shape != ids.shape:
            raise ValueError(f"Unexpected ID count: {path}")
        with np.load(path, allow_pickle=False) as predictions:
            if not np.array_equal(predictions["labels"], labels):
                raise ValueError(f"Manifest/CSV label order mismatch: {path}")
            for name in predictions.files:
                if name != "ids":
                    value = predictions[name]
                    if value.dtype.hasobject or value.shape[0] != len(ids):
                        raise ValueError(f"Unexpected numeric array {name}: {path}")
            if not dtype.hasobject:
                if not np.array_equal(predictions["ids"], ids):
                    raise ValueError(f"Existing string IDs differ from manifest: {path}")
                return False
        buffer = io.BytesIO()
        np.save(buffer, ids, allow_pickle=False)
        temporary = path.with_name(path.name + ".repairing")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for name in members:
                output.writestr(name, buffer.getvalue() if name == "ids.npy" else archive.read(name))
        with zipfile.ZipFile(temporary) as check:
            if check.testzip() is not None:
                raise OSError("Repaired archive CRC check failed.")
            for name in members:
                if name != "ids.npy" and check.read(name) != archive.read(name):
                    raise OSError(f"Numeric array changed: {path}/{name}")
        with np.load(temporary, allow_pickle=False) as check:
            for name in check.files:
                if check[name].dtype.hasobject:
                    raise ValueError("Repaired file still contains an object array.")
            np.testing.assert_array_equal(check["ids"], ids)
    temporary.replace(path)
    return True


def persist(source, target):
    if target.exists() and sha256(source) == sha256(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".repair-upload")
    shutil.copy2(source, temporary)
    if sha256(source) != sha256(temporary):
        raise OSError(f"Drive copy verification failed: {target}")
    temporary.replace(target)
    if sha256(source) != sha256(target):
        raise OSError(f"Final Drive verification failed: {target}")


def main(args):
    sys.path.insert(0, str(args.package))
    from plan import matrix
    manifest_path = args.package / "split_manifest.json"
    csv_path = args.data / "train.csv"
    protocol = json.loads((args.output / "protocol.json").read_text(encoding="utf-8"))
    for path, key in ((manifest_path, "manifest_sha256"), (csv_path, "csv_sha256")):
        if sha256(path) != protocol[key]:
            raise ValueError(f"Original experiment source hash mismatch: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["sample_ids"]
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels_by_id = {row["id_code"]: int(row["diagnosis"]) for row in rows}
    jobs = [(args.output / run.key / f"{split}_logits.npz", run.key, split)
            for run in matrix() for split in ("val", "test")]
    missing = [str(path) for path, _, _ in jobs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction archives; first: {missing[0]}")
    changed = 0
    for index, (path, key, split) in enumerate(jobs, 1):
        ids = np.asarray(manifest[split], dtype=str)
        labels = np.asarray([labels_by_id[value] for value in ids], dtype=np.int64)
        modified = repair(path, ids, labels)
        changed += modified
        if args.drive:
            persist(path, args.drive / key / path.name)
        print(f"{index}/{len(jobs)} {'REPAIRED' if modified else 'VERIFIED'} {key}/{path.name}", flush=True)
    report = {"archives_checked": len(jobs), "ids_repaired_this_invocation": changed,
              "pickle_enabled": False, "numeric_npy_members": "preserved byte-for-byte",
              "ids_source": "original frozen split manifest; exact source hashes and per-row labels checked"}
    report_path = args.output / "prediction_ids_repair.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.drive:
        persist(report_path, args.drive / report_path.name)
    print("REPAIR COMPLETE. Run analyze only; no retraining or evaluate rerun required.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive", type=Path)
    main(parser.parse_args())
