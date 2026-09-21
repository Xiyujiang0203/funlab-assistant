"""PDF -> Markdown via local opendataloader-pdf CLI jar (hybrid by default)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF_DIR = HERE / "pdf"
MD_DIR = HERE / "markdown"
DEFAULT_JAR = (
    ROOT
    / "open-source"
    / "opendataloader-pdf"
    / "java"
    / "opendataloader-pdf-cli"
    / "target"
    / "opendataloader-pdf-cli-0.0.0.jar"
)
DEFAULT_HYBRID_URL = "http://127.0.0.1:5002"


def convert_pdf(
    pdf: Path,
    output_dir: Path,
    jar: Path,
    *,
    hybrid: str = "docling-fast",
    hybrid_url: str = DEFAULT_HYBRID_URL,
    hybrid_mode: str = "auto",
    markdown_with_html: bool = True,
) -> Path:
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not jar.is_file():
        raise FileNotFoundError(f"JAR not found: {jar}. Build opendataloader-pdf-cli first.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-Djava.awt.headless=true",
        "-jar",
        str(jar),
        "--format",
        "markdown",
        "--output-dir",
        str(output_dir),
    ]
    if markdown_with_html:
        cmd.append("--markdown-with-html")
    if hybrid and hybrid != "off":
        cmd.extend(["--hybrid", hybrid, "--hybrid-url", hybrid_url, "--hybrid-mode", hybrid_mode])
    cmd.append(str(pdf))
    subprocess.run(cmd, check=True)
    md = output_dir / f"{pdf.stem}.md"
    if not md.is_file():
        matches = sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise RuntimeError(f"No markdown produced in {output_dir}")
        md = matches[0]
    return md


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown with opendataloader-pdf hybrid")
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="*",
        help="PDF path(s); default: all *.pdf under test/pdf2markdown/pdf",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=MD_DIR)
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--hybrid", default="docling-fast", help="off | docling-fast")
    parser.add_argument("--hybrid-url", default=DEFAULT_HYBRID_URL)
    parser.add_argument("--hybrid-mode", default="auto", choices=["auto", "full"])
    parser.add_argument("--no-html-tables", action="store_true")
    args = parser.parse_args()

    pdfs = [p.resolve() for p in args.pdf] if args.pdf else sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"未找到 PDF：{PDF_DIR}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve()
    jar = args.jar.resolve()
    for pdf in pdfs:
        md = convert_pdf(
            pdf,
            out_dir,
            jar,
            hybrid=args.hybrid,
            hybrid_url=args.hybrid_url,
            hybrid_mode=args.hybrid_mode,
            markdown_with_html=not args.no_html_tables,
        )
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
