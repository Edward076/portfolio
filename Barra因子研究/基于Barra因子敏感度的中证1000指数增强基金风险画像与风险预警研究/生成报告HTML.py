from pathlib import Path

import markdown


def main() -> None:
    md_path = Path(__file__).with_name("项目报告_风险画像与预警.md")
    html_path = md_path.with_suffix(".html")

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>项目报告</title>
  <style>
    @page {{
      size: A4;
      margin: 16mm 12mm;
    }}
    body {{
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      max-width: 960px;
      margin: 40px auto;
      padding: 0 20px;
      line-height: 1.75;
      color: #222;
    }}
    h1, h2, h3 {{ line-height: 1.4; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 16px 0;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
    }}
    th {{ background: #f6f8fa; }}
    code {{
      background: #f5f5f5;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    blockquote {{
      margin: 12px 0;
      padding: 10px 14px;
      border-left: 4px solid #ddd;
      background: #fafafa;
    }}
    img {{
      display: block;
      max-width: 100%;
      width: auto;
      height: auto;
      margin: 12px auto;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    @media print {{
      body {{
        max-width: none;
        margin: 0;
        padding: 0;
      }}
      img {{
        max-width: 92%;
      }}
    }}
  </style>
</head>
<body>
{body}
<hr>
<p style="color:#666;font-size:13px;">由项目报告Markdown自动转换生成</p>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    print(html_path)


if __name__ == "__main__":
    main()
