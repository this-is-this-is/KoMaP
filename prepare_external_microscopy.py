"""Download, verify and index the external microscopy pretraining sources."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


CROSS_URL = "https://zenodo.org/api/records/19851246/files/Dataset%20+%20Metadata.zip/content"
CROSS_MD5 = "4fc9b0567c218e9e5290985da4d62a74"
CROSS_BYTES = 301_907_090
NASA_COMMIT = "9b7c4abc1321e81eca7a68d548e5371676fa74fa"
NASA_URL = f"https://github.com/nasa/pretrained-microscopy-models/archive/{NASA_COMMIT}.tar.gz"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url, path):
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    start = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        # A server that ignores Range sent the full object; do not append it.
        append = start > 0 and response.status == 206
        mode = "ab" if append else "wb"
        if not append:
            start = 0
        expected = response.headers.get("Content-Length")
        expected = start + int(expected) if expected else None
        received, next_report = start, start + 64 * 1024 * 1024
        with temporary.open(mode) as f:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                f.write(block)
                received += len(block)
                if received >= next_report:
                    print(f"  downloaded {received / 1024**2:.0f} MiB"
                          + (f" / {expected / 1024**2:.0f} MiB" if expected else ""), flush=True)
                    next_report += 64 * 1024 * 1024
    temporary.replace(path)


def safe_unzip(archive, target):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as value:
        for member in value.infolist():
            destination = (target / member.filename).resolve()
            if target not in destination.parents and destination != target:
                raise ValueError(f"unsafe zip member: {member.filename}")
        value.extractall(target)


def safe_untar(archive, target):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as value:
        for member in value.getmembers():
            destination = (target / member.name).resolve()
            if target not in destination.parents and destination != target:
                raise ValueError(f"unsafe tar member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not accepted in source archive: {member.name}")
        value.extractall(target)


def _column(reference):
    letters = re.match(r"[A-Z]+", reference).group(0)
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def read_first_xlsx_sheet(path):
    """Read the simple metadata workbook using the standard library only."""
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as book:
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter(namespace + "t"))
                      for item in root.findall(namespace + "si")]
        sheet_name = sorted(name for name in book.namelist()
                            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))[0]
        sheet = ET.fromstring(book.read(sheet_name))
        table = []
        for row in sheet.iter(namespace + "row"):
            values = {}
            for cell in row.findall(namespace + "c"):
                index = _column(cell.attrib["r"])
                kind = cell.attrib.get("t")
                if kind == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(namespace + "t"))
                else:
                    node = cell.find(namespace + "v")
                    value = "" if node is None else (node.text or "")
                    if kind == "s" and value:
                        value = shared[int(value)]
                values[index] = value
            table.append([values.get(i, "") for i in range(max(values, default=-1) + 1)])
    header = [str(value).strip() for value in table[0]]
    return [{header[i]: (row[i] if i < len(row) else "") for i in range(len(header))}
            for row in table[1:] if any(str(value).strip() for value in row)]


def clean_integer(value):
    text = str(value).strip()
    return str(int(float(text))) if re.fullmatch(r"\d+(\.0+)?", text) else text


def make_cross_manifest(cross_root):
    raw = cross_root / "raw"
    workbook = next(raw.rglob("Dataset_Patches_Microscope_Metadata.xlsx"), None)
    if workbook is None:
        raise FileNotFoundError("Dataset_Patches_Microscope_Metadata.xlsx was not extracted")
    rows = read_first_xlsx_sheet(workbook)
    required = {"FileNameHarm", "ID", "SampleIDHarm", "Magnification", "MicroscopeID",
                "Phase", "PatchID"}
    if len(rows) != 5327 or not required.issubset(rows[0]):
        raise ValueError(f"unexpected cross-microscope metadata: {len(rows)} rows, {rows[0].keys()}")
    images = sorted(raw.rglob("*.png"))
    by_name = defaultdict(list)
    for path in images:
        by_name[path.name].append(path)
    pending, corrections = [], []
    for row in rows:
        filename = Path(str(row["FileNameHarm"])).name
        if not Path(filename).suffix:
            filename += ".png"
        candidates = by_name.get(filename, [])
        if len(candidates) != 1:
            identifier = clean_integer(row["ID"])
            patch = int(float(row["PatchID"]))
            sample = clean_integer(row["SampleIDHarm"])
            phase = str(row["Phase"]).strip()
            roi_directory = f"{sample}_{phase}_{patch}"
            candidates = [path for path in images
                          if path.parent.name == roi_directory
                          if path.name.startswith(identifier + "_")
                          and path.name.endswith(f"_{patch:02d}.png")]
            if len(candidates) != 1:
                raise ValueError(f"cannot resolve metadata image {filename}: {candidates}")
            corrections.append({"metadata": filename, "archive": candidates[0].name})
        path = candidates[0]
        sample = clean_integer(row["SampleIDHarm"])
        phase = str(row["Phase"]).strip()
        patch = clean_integer(row["PatchID"])
        roi_id = f"{sample}|{phase}|{patch}"
        pending.append({"path": os.path.relpath(path, cross_root), "roi_id": roi_id,
                        "sample": sample, "phase": phase, "patch_id": patch,
                        "microscope": clean_integer(row["MicroscopeID"]),
                        "magnification": clean_integer(row["Magnification"]),
                        "source_id": clean_integer(row["ID"])})
    # Hold out complete physical specimens.  ROI-only splitting would keep
    # registered variants together but still leak specimen-specific texture.
    samples = sorted({row["sample"] for row in pending},
                     key=lambda x: hashlib.sha256(x.encode()).digest())
    valid_samples = set(samples[:max(1, round(0.1 * len(samples)))])
    manifest = [dict(row, split="valid" if row["sample"] in valid_samples else "train")
                for row in pending]
    groups = defaultdict(list)
    for row in manifest:
        groups[row["roi_id"]].append(row)
    if len(images) != 5327 or len(manifest) != 5327 or len(groups) != 228:
        raise ValueError(f"cross dataset counts changed: images={len(images)}, rows={len(manifest)}, groups={len(groups)}")
    if any(len({x["microscope"] for x in values}) < 2 for values in groups.values()):
        raise ValueError("an ROI group has fewer than two microscopes")
    path = cross_root / "manifest.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    (cross_root / "source.json").write_text(json.dumps({
        "source": "Zenodo record 19851246", "doi": "10.5281/zenodo.19851246",
        "url": CROSS_URL, "archive_md5": CROSS_MD5, "license": "CC-BY-4.0",
        "images": len(images), "roi_groups": len(groups),
        "validation_samples": sorted(valid_samples),
        "split_groups": dict(Counter({row["roi_id"]: row["split"] for row in manifest}.values())),
        "filename_corrections": corrections,
    }, indent=2))
    return path


def prepare_cross(root, skip_download=False):
    target = root / "cross_microscope"
    archive = root / "downloads" / "Dataset+Metadata.zip"
    if not skip_download:
        if archive.exists() and (archive.stat().st_size != CROSS_BYTES or digest(archive, "md5") != CROSS_MD5):
            archive.unlink()
        download(CROSS_URL, archive)
    if not archive.is_file() or archive.stat().st_size != CROSS_BYTES or digest(archive, "md5") != CROSS_MD5:
        raise ValueError(f"cross-microscope archive failed size/MD5 verification: {archive}")
    if not (target / "raw").is_dir():
        safe_unzip(archive, target / "raw")
    manifest = make_cross_manifest(target)
    print(f"cross-microscope ready: {manifest}")


def prepare_nasa(root, skip_download=False):
    target = root / "nasa_pretrained_microscopy_models"
    source = target / "source"
    archive = root / "downloads" / f"nasa-{NASA_COMMIT}.tar.gz"
    if not source.is_dir():
        if not skip_download:
            download(NASA_URL, archive)
        if not archive.is_file():
            raise FileNotFoundError(archive)
        temporary = target / "extracting"
        if temporary.exists():
            shutil.rmtree(temporary)
        safe_untar(archive, temporary)
        children = [path for path in temporary.iterdir() if path.is_dir()]
        if len(children) != 1:
            raise ValueError(f"unexpected NASA archive layout: {children}")
        target.mkdir(parents=True, exist_ok=True)
        children[0].replace(source)
        temporary.rmdir()
    base = source / "benchmark_segmentation_data" / "Super1"
    counts = {split: len(list((base / split).glob("*.tif"))) for split in ("train", "val", "test")}
    if counts != {"train": 10, "val": 4, "test": 4}:
        raise ValueError(f"NASA Super1 split changed: {counts}")
    (target / "source.json").write_text(json.dumps({
        "source": "nasa/pretrained-microscopy-models", "commit": NASA_COMMIT,
        "url": NASA_URL, "archive_sha256": digest(archive) if archive.is_file() else None,
        "license": "MIT", "canonical_split": counts,
        "selection": "18 unique Super1 images train / 5 Super2 different_test images OOD validation",
        "refit": "all 23 unique image-mask pairs for the selected epoch count",
        "note": "Super2-4 otherwise repeat the same blobs under few-shot/OOD benchmark layouts.",
    }, indent=2))
    print(f"NASA Super ready: {source}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("external_data"))
    ap.add_argument("--sources", nargs="+", choices=["cross", "nasa"], default=["cross", "nasa"])
    ap.add_argument("--skip_download", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out).free < 3 * 1024**3:
        raise OSError("less than 3 GiB free; external data preparation stopped")
    if "cross" in args.sources:
        prepare_cross(args.out, args.skip_download)
    if "nasa" in args.sources:
        prepare_nasa(args.out, args.skip_download)
    if set(args.sources) == {"cross", "nasa"}:
        ready = {
            "cross_manifest": "cross_microscope/manifest.jsonl",
            "nasa_source": "nasa_pretrained_microscopy_models/source",
            "cross_archive_md5": CROSS_MD5,
            "nasa_commit": NASA_COMMIT,
        }
        (args.out / "READY.json").write_text(json.dumps(ready, indent=2))


if __name__ == "__main__":
    main()
