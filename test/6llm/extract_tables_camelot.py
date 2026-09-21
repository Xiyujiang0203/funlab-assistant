"""Extract tables from GroupRules.pdf via camelot."""
from __future__ import annotations

import argparse
from pathlib import Path

import camelot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PDF = ROOT / "GroupRules.pdf"
DEFAULT_OUT = HERE / "output" / "grouprules_tables"


def extract(pdf: Path, out_dir: Path, flavor: str = "lattice") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = camelot.read_pdf(str(pdf), pages="all", flavor=flavor)
    print(f"flavor={flavor} tables={tables.n}")
    for i, t in enumerate(tables, 1):
        acc = getattr(t, "accuracy", None)
        print(f"  [{i}] page={t.page} shape={t.df.shape} accuracy={acc}")
        csv_path = out_dir / f"table_{i:02d}_p{t.page}_{flavor}.csv"
        md_path = out_dir / f"table_{i:02d}_p{t.page}_{flavor}.md"
        t.to_csv(str(csv_path))
        md_path.write_text(t.df.to_markdown(index=False), encoding="utf-8")
        print(f"    -> {csv_path.name}")
    summary = out_dir / f"summary_{flavor}.txt"
    summary.write_text(tables.n and str(tables[0].parsing_report) or "none", encoding="utf-8")
    # 合并预览
    preview = out_dir / f"all_tables_{flavor}.md"
    parts = [f"# GroupRules tables ({flavor})", f"count: {tables.n}", ""]
    for i, t in enumerate(tables, 1):
        parts.append(f"## Table {i} · page {t.page}")
        parts.append("")
        parts.append(t.df.to_markdown(index=False))
        parts.append("")
    preview.write_text("\n".join(parts), encoding="utf-8")
    print(f"preview: {preview}")
    return tables.n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path, nargs="?", default=DEFAULT_PDF)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--flavor", choices=["lattice", "stream", "both"], default="both")
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")
    flavors = ["lattice", "stream"] if args.flavor == "both" else [args.flavor]
    total = 0
    for f in flavors:
        try:
            total += extract(pdf, args.output_dir.resolve(), f)
        except Exception as e:
            print(f"flavor={f} failed: {e}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
