import argparse
import re
import sys
from pathlib import Path

import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<pair>[A-Z]{6})_(?P<timeframe>[A-Z0-9]+)_(?P<start>\d{12})_(?P<end>\d{12})\.csv$"
)

REQUIRED_FILES = {
    "EURUSD_H1_202001140000_202608200600.csv": 41065,
    "GBPUSD_H1_202001140000_202608200600.csv": 41064,
}

MIN_REQUIRED_COLUMNS = {"open", "high", "low", "close"}
EXPECTED_DELTA = pd.Timedelta(hours=1)


def parse_filename(name: str) -> dict:
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Nombre inválido: {name}")

    data = m.groupdict()
    data["start_dt"] = pd.to_datetime(data["start"], format="%Y%m%d%H%M", utc=True)
    data["end_dt"] = pd.to_datetime(data["end"], format="%Y%m%d%H%M", utc=True)
    return data


def _extract_datetime(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower().strip(): c for c in df.columns}

    if "datetime" in cols:
        return pd.to_datetime(df[cols["datetime"]], errors="coerce", utc=True)

    if "timestamp" in cols:
        return pd.to_datetime(df[cols["timestamp"]], errors="coerce", utc=True)

    if "date" in cols and "time" in cols:
        merged = df[cols["date"]].astype(str).str.strip() + " " + df[cols["time"]].astype(str).str.strip()
        return pd.to_datetime(merged, errors="coerce", utc=True)

    raise ValueError("No se encontró columna datetime/timestamp ni combinación date+time")


def _quality_checks(df: pd.DataFrame, expected_rows: int) -> dict:
    df_cols = {c.lower().strip() for c in df.columns}
    missing_cols = sorted(MIN_REQUIRED_COLUMNS - df_cols)

    dt = _extract_datetime(df)
    invalid_ts = int(dt.isna().sum())

    dt_valid = dt.dropna()
    duplicates = int(dt_valid.duplicated().sum())

    chrono_ok = bool(dt_valid.is_monotonic_increasing)

    dt_sorted = dt_valid.sort_values()
    diffs = dt_sorted.diff().dropna()

    gaps_all = int((diffs > EXPECTED_DELTA).sum())
    missing_bars_all = int(((diffs[diffs > EXPECTED_DELTA] / EXPECTED_DELTA) - 1).sum()) if gaps_all else 0

    prev_ts = dt_sorted.shift(1)
    weekday_mask = (prev_ts.dt.dayofweek < 5) & (dt_sorted.dt.dayofweek < 5)
    diffs_weekday = diffs[weekday_mask.loc[diffs.index]]
    gaps_weekday = int((diffs_weekday > EXPECTED_DELTA).sum()) if not diffs_weekday.empty else 0
    missing_bars_weekday = (
        int(((diffs_weekday[diffs_weekday > EXPECTED_DELTA] / EXPECTED_DELTA) - 1).sum())
        if gaps_weekday
        else 0
    )

    row_count = len(df)
    rows_match_expected = row_count == expected_rows

    return {
        "rows": row_count,
        "expected_rows": expected_rows,
        "rows_match_expected": rows_match_expected,
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
    df = pd.read_csv(path)
    checks = _quality_checks(df, expected_rows)

    return {
        "file": path.name,
        "pair": meta["pair"],
        "timeframe": meta["timeframe"],
        "start": str(meta["start_dt"]),
        "end": str(meta["end_dt"]),
        **checks,
    }


def validate_required_files(folder: Path) -> list[Path]:
    missing = [name for name in REQUIRED_FILES if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos:\n- " + "\n- ".join(missing)
        )

    return [folder / name for name in REQUIRED_FILES]


def print_summary(results: list[dict]) -> None:
    print("\n=== RESUMEN ===")
    for r in results:
        status = "OK" if r["rows_match_expected"] else "ERROR_FILAS"
        print(
            f"- {r['file']} | {r['pair']} {r['timeframe']} | "
            f"{r['start']} -> {r['end']} | filas: {r['rows']} ({status})"
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
    pd.DataFrame(results).to_csv(out_path, index=False)


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
