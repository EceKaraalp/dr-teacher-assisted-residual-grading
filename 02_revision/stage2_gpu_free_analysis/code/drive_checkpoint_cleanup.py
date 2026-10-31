"""Colab-only, opt-in cleanup of verified superseded checkpoint trash entries.

Keep this helper OUTSIDE the training package: it does not change its code lock.
It never deletes active files, folders, revisions or unverified trash entries.
"""
import hashlib
import json
from pathlib import Path
import threading
import time


DRIVE_ROOT_ID = "1ksmjYYF6EeoAYzIO_SdNSVNK-F_2ldDw"
TRASH_NAMES = ("last.pth", "last.pth.uploading", "best.pth", "best.pth.uploading")
FIELDS = "id,name,parents,trashed,mimeType,md5Checksum,size,modifiedTime"
FOLDER = "application/vnd.google-apps.folder"


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_files(service, query):
    token = None
    while True:
        result = service.files().list(
            q=query, spaces="drive", pageSize=100,
            fields=f"nextPageToken,files({FIELDS})", pageToken=token,
        ).execute(num_retries=5)
        yield from result.get("files", [])
        token = result.get("nextPageToken")
        if not token:
            break


class CheckpointTrashCleaner:
    def __init__(self, service_factory, local_root, run_keys, interval=120):
        self.service_factory = service_factory
        self.local_root = Path(local_root).resolve()
        self.run_keys = tuple(run_keys)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.hash_cache = {}
        self.audit_path = self.local_root / "checkpoint_cleanup_audit.jsonl"

    def children(self, service, parent):
        return list(list_files(service, f"'{parent}' in parents and trashed = false"))

    def verified_folders(self, service):
        root = service.files().get(fileId=DRIVE_ROOT_ID, fields=FIELDS).execute(num_retries=5)
        if root.get("trashed") or root.get("mimeType") != FOLDER or root.get("name") != "reviewer_complete_v1":
            raise RuntimeError("Configured Drive root must be the active reviewer_complete_v1 folder.")
        if self.local_root.name != "reviewer_complete_v1":
            raise RuntimeError("Unexpected local experiment folder.")
        root_children = self.children(service, DRIVE_ROOT_ID)
        protocols = [p for p in root_children if p["name"] == "protocol.json"]
        if len(protocols) != 1 or not self.matches_local(protocols[0], self.local_root / "protocol.json"):
            raise RuntimeError("Drive root protocol.json does not match the local experiment; cleanup refused.")
        contents = {DRIVE_ROOT_ID: root_children}
        paths = {}
        for key in self.run_keys:
            parts = Path(key).parts
            if not parts or any(p in {"..", "."} for p in parts) or Path(key).is_absolute():
                raise ValueError("Unsafe experiment key.")
            parent = DRIVE_ROOT_ID
            for part in parts:
                if parent not in contents:
                    contents[parent] = self.children(service, parent)
                found = [p for p in contents[parent] if p["name"] == part and p.get("mimeType") == FOLDER]
                if not found:
                    break
                if len(found) != 1:
                    raise RuntimeError(f"Ambiguous Drive experiment folder: {key}")
                parent = found[0]["id"]
            else:
                paths[parent] = self.local_root.joinpath(*parts)
        return paths, contents

    def matches_local(self, active, path):
        if active.get("trashed") or active.get("mimeType") == FOLDER or not active.get("md5Checksum"):
            return False
        if not path.is_file() or str(path.stat().st_size) != str(active.get("size")):
            return False
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hash_cache:
            self.hash_cache[key] = md5_file(path)
        return self.hash_cache[key] == active["md5Checksum"]

    def active_counterpart(self, service, candidate, paths):
        parents = candidate.get("parents", [])
        if not candidate.get("trashed") or candidate.get("name") not in TRASH_NAMES or len(parents) != 1:
            return None
        parent = parents[0]
        if parent not in paths or candidate.get("mimeType") == FOLDER:
            return None
        name = candidate["name"].removesuffix(".uploading")
        counterparts = [p for p in self.children(service, parent) if p["name"] == name]
        if len(counterparts) != 1:
            return None
        active = counterparts[0]
        if active["id"] == candidate["id"] or not self.matches_local(active, paths[parent] / name):
            return None
        return active

    def scan(self, delete=False, approved_ids=None):
        service = self.service_factory()
        paths, _ = self.verified_folders(service)
        names = " or ".join(f"name = '{name}'" for name in TRASH_NAMES)
        candidates = list(list_files(service, f"trashed = true and ({names})"))
        rows = []
        for candidate in candidates:
            if approved_ids is not None and candidate["id"] not in approved_ids:
                continue
            # Ignore account-wide entries not in the explicitly resolved experiment leaves.
            if len(candidate.get("parents", [])) != 1 or candidate["parents"][0] not in paths:
                continue
            active = self.active_counterpart(service, candidate, paths)
            if active is None:
                continue
            row = {"trash_id": candidate["id"], "name": candidate["name"],
                   "experiment": str(paths[candidate["parents"][0]].relative_to(self.local_root)),
                   "bytes": int(candidate.get("size", 0)), "active_id": active["id"]}
            if delete:
                # Revalidate metadata immediately before permanent deletion.
                current = service.files().get(fileId=candidate["id"], fields=FIELDS).execute(num_retries=5)
                if current.get("parents") != candidate.get("parents") or current.get("name") != candidate.get("name"):
                    raise RuntimeError("Trash candidate moved or renamed; cleanup stopped.")
                checked = self.active_counterpart(service, current, paths)
                if checked is None or checked["id"] != active["id"]:
                    raise RuntimeError("Active checkpoint changed or no longer verified; cleanup stopped.")
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as audit:
                    audit.write(json.dumps({"event": "delete_requested", "time": time.time(), **row}) + "\n")
                    audit.flush()
                service.files().delete(fileId=current["id"]).execute()
                with self.audit_path.open("a", encoding="utf-8") as audit:
                    audit.write(json.dumps({"event": "deleted", "time": time.time(), **row}) + "\n")
                print(f"CLEANUP: {row['experiment']} / trash {row['name']} permanently deleted", flush=True)
            rows.append(row)
        print(f"CLEANUP {'deleted' if delete else 'preview'}: {len(rows)} verified trash copies, "
              f"{sum(r['bytes'] for r in rows) / 2**30:.2f} GiB", flush=True)
        return rows

    def start(self, confirmation):
        if confirmation != "DELETE_VERIFIED_OLD_CHECKPOINT_TRASH":
            raise ValueError("Explicit permanent-deletion confirmation required.")
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Cleanup monitor is already running.")
        self.stop_event.clear()
        self.error = None
        def worker():
            while not self.stop_event.is_set():
                try:
                    self.scan(delete=True)
                    self.error = None
                except Exception as error:
                    self.error = error
                    status = getattr(getattr(error, "resp", None), "status", None)
                    if status in {429, 500, 502, 503, 504}:
                        print(f"CLEANUP RETRY: Drive HTTP {status}: {error}. "
                              f"Retrying in {self.interval} seconds; monitor quota meanwhile.", flush=True)
                    else:
                        self.stop_event.set()
                        print(f"CLEANUP STOPPED: {type(error).__name__}: {error}. "
                              "No further automatic deletion; check quota before continuing.", flush=True)
                        break
                self.stop_event.wait(self.interval)
        self.thread = threading.Thread(target=worker, name="verified-checkpoint-trash-cleanup", daemon=True)
        self.thread.start()
        print("Cleanup monitor started. It must be restarted after each Colab runtime reset.", flush=True)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=180)
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Cleanup request is still finishing; do not start a second monitor.")
        print("Cleanup monitor stopped.", flush=True)
