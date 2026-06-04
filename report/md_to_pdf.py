"""Render the final report Markdown to a print-ready HTML, for headless-Chrome PDF.

Usage: python report/md_to_pdf.py
Produces report/CS455_SuSchedule-r_Report.html (then printed to PDF by Chrome).
"""
from pathlib import Path
import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "CS455_SuSchedule-r_Report.md"
OUT_HTML = HERE / "CS455_SuSchedule-r_Report.html"

CSS = """
@page { size: Letter; margin: 22mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Calibri", "Segoe UI", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.45; color: #1a1f2e; margin: 0;
}
h1 { font-size: 17pt; color: #002776; border-bottom: 2px solid #d8e0f0;
     padding-bottom: 3px; margin: 18px 0 8px; }
h2 { font-size: 13pt; color: #1c47a8; margin: 14px 0 6px; }
h3 { font-size: 11.5pt; color: #002776; margin: 10px 0 4px; }
h1:first-of-type { margin-top: 0; }
p { margin: 6px 0; text-align: justify; }
a { color: #1c47a8; text-decoration: none; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #c9d2e5; padding: 4px 8px; text-align: left;
         vertical-align: top; }
th { background: #d8e0f0; color: #002776; font-weight: bold; }
tr:nth-child(even) td { background: #f1f4fa; }
code { font-family: "Consolas", monospace; font-size: 9pt;
       background: #f1f4fa; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f8fc; border: 1px solid #d8e0f0; border-radius: 4px;
      padding: 8px 10px; overflow: hidden; }
pre code { font-size: 7.6pt; line-height: 1.2; background: none; padding: 0;
           white-space: pre; }
hr { border: none; border-top: 1px solid #d8e0f0; margin: 14px 0; }
ul, ol { margin: 6px 0 6px 0; padding-left: 22px; }
li { margin: 2px 0; }
strong { color: #0e1730; }
h1, h2, h3 { page-break-after: avoid; }
table, pre { page-break-inside: avoid; }
"""

def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
    )
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>SuSchedule-r Report</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML}")

if __name__ == "__main__":
    main()
