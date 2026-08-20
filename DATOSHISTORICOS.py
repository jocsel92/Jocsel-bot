import argparse
import csv
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<pair>[A-Z]{6})_(?P<timeframe>[A-Z0-9]+)_(?P<start>\d{12})_(?P<end>\d{12})\.csv$"
)

REQUIRED_FILES = {
    "EURUSD_H1_202001140000_202608200600.csv": 41065,
    "GBPUSD_H1_202001140000_202608200600.csv": 41064,
}

MIN_REQUIRED_COLUMNS = {"open", "high", "low", "close"}
EXPECTED_DELTA = timedelta(hours=1)


DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)


def parse_filename(name: str) -> dict:
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Nombre inválido: {name}")

    data = m.groupdict()
    data["start_dt"] = datetime.strptime(data["start"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    data["end_dt"] = datetime.strptime(data["end"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return data


def _parse_datetime_value(raw: str) -> datetime | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None

    # ISO handling first
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def _extract_datetimes(rows: list[dict], fieldnames_lower: dict) -> list[datetime | None]:
    if "datetime" in fieldnames_lower:
        col = fieldnames_lower["datetime"]
        return [_parse_datetime_value(r.get(col)) for r in rows]

    if "timestamp" in fieldnames_lower:
        col = fieldnames_lower["timestamp"]
        return [_parse_datetime_value(r.get(col)) for r in rows]

    if "date" in fieldnames_lower and "time" in fieldnames_lower:
        dcol = fieldnames_lower["date"]
        tcol = fieldnames_lower["time"]
        return [_parse_datetime_value(f"{r.get(dcol, '')} {r.get(tcol, '')}") for r in rows]

    raise ValueError("No se encontró columna datetime/timestamp ni combinación date+time")


def _read_csv_rows(path: Path) -> tuple[list[dict], dict]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV sin encabezados")

        rows = list(reader)
        fieldnames_lower = {name.lower().strip(): name for name in reader.fieldnames}
        return rows, fieldnames_lower


def _quality_checks(rows: list[dict], fieldnames_lower: dict, expected_rows: int) -> dict:
    missing_cols = sorted(MIN_REQUIRED_COLUMNS - set(fieldnames_lower.keys()))

    dt_series = _extract_datetimes(rows, fieldnames_lower)
    invalid_ts = sum(1 for dt in dt_series if dt is None)

    dt_valid = [dt for dt in dt_series if dt is not None]
    seen = set()
    duplicates = 0
    for dt in dt_valid:
        if dt in seen:
            duplicates += 1
        else:
            seen.add(dt)

    chrono_ok = all(dt_valid[i] > dt_valid[i - 1] for i in range(1, len(dt_valid)))

    dt_sorted = sorted(dt_valid)
    gaps_all = 0
    missing_bars_all = 0
    gaps_weekday = 0
    missing_bars_weekday = 0

    for i in range(1, len(dt_sorted)):
        prev_dt = dt_sorted[i - 1]
        curr_dt = dt_sorted[i]
        delta = curr_dt - prev_dt
        if delta > EXPECTED_DELTA:
            gaps_all += 1
            missing_bars_all += int(delta.total_seconds() // EXPECTED_DELTA.total_seconds()) - 1

            if prev_dt.weekday() < 5 and curr_dt.weekday() < 5:
                gaps_weekday += 1
                missing_bars_weekday += int(delta.total_seconds() // EXPECTED_DELTA.total_seconds()) - 1

    row_count = len(rows)

    return {
        "rows": row_count,
        "expected_rows": expected_rows,
        "rows_match_expected": row_count == expected_rows,
        "missing_columns": ",".join(missing_cols) if missing_cols else "",
        "invalid_timestamps": invalid_ts,
        "duplicate_timestamps": duplicates,
        "chronological_order_ok": chrono_ok,
        "gaps_gt_1h_all": gaps_all,
        "missing_bars_all": missing_bars_all,
        "gaps_gt_1h_weekday": gaps_weekday,
        "missing_bars_weekday": missing_bars_weekday,
    }


def analyze_file(path: Path, expected_rows: int) -> dict:
    meta = parse_filename(path.name)
    rows, fieldnames_lower = _read_csv_rows(path)
    checks = _quality_checks(rows, fieldnames_lower, expected_rows)

    return {
        "file": path.name,
        "pair": meta["pair"],
        "timeframe": meta["timeframe"],
        "start": meta["start_dt"].isoformat(),
        "end": meta["end_dt"].isoformat(),
        **checks,
    }


def validate_required_files(folder: Path) -> list[Path]:
    missing = [name for name in REQUIRED_FILES if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos requeridos:\n- " + "\n- ".join(missing))

    return [folder / name for name in REQUIRED_FILES]


def print_summary(results: list[dict]) -> None:
    print("\n=== RESUMEN ===")
    for r in results:
        status = "OK" if r["rows_match_expected"] else "ERROR_FILAS"
        start_human = r["start"].replace("T", " ").replace("+00:00", "")
        end_human = r["end"].replace("T", " ").replace("+00:00", "")
        print(
            f"- {r['file']} | {r['pair']} {r['timeframe']} | "
            f"{start_human} -> {end_human} | filas: {r['rows']} ({status})"
        )


def print_quality(results: list[dict]) -> None:
    print("\n=== CONTROL DE CALIDAD H1 ===")
    for r in results:
        missing_cols = r["missing_columns"] or "ninguna"
        print(
            f"- {r['file']} | columnas_faltantes: {missing_cols} | "
            f"timestamps_invalidos: {r['invalid_timestamps']} | duplicados: {r['duplicate_timestamps']} | "
            f"orden_cronologico: {r['chronological_order_ok']} | "
            f"huecos_1h_total: {r['gaps_gt_1h_all']} (faltantes: {r['missing_bars_all']}) | "
            f"huecos_1h_semana: {r['gaps_gt_1h_weekday']} (faltantes: {r['missing_bars_weekday']})"
        )


def save_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return

    headers = list(results[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida y resume únicamente 2 archivos H1 requeridos (EURUSD/GBPUSD)."
    )
    parser.add_argument("--folder", default=".", help="Carpeta donde están los CSV")
    parser.add_argument("--out", default="resumen_2_archivos.csv", help="CSV de salida")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Carpeta inválida: {folder}")
        return 1

    print(f"Buscando en: {folder}")

    try:
        files = validate_required_files(folder)
        results = [analyze_file(p, REQUIRED_FILES[p.name]) for p in files]
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print_summary(results)
    print_quality(results)

    save_csv(results, out_path)
    print(f"\nGuardado en: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
