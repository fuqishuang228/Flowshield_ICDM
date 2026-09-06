from pathlib import Path
import json
import re
from bs4 import BeautifulSoup

# ========== Config ==========
json_path = Path("/workspace/Llama4AML_250721/explainability/llama_prompts/gpt4_outputs_no_strategies.json")
html_dir = Path("/workspace/Llama4AML_250721/explainability/sankey_parallel_updated_html")
output_dir = Path("/workspace/Llama4AML_250721/explainability/sankey_parallel_updated_html_with_gpt4_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# ========== HTML Format Helpers ==========
def convert_markdown_to_html_blocks(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    summary_match = re.search(r'### 💸 Summary of Fund Flow\s*(.*?)\s*### 🚩 Red Flags', text, re.DOTALL)
    redflags_match = re.search(r'### 🚩 Red Flags\s*(.*)', text, re.DOTALL)



    summary_section = summary_match.group(1).strip() if summary_match else ""
    redflags_section = redflags_match.group(1).strip() if redflags_match else ""
    summary_section = re.sub(r'-{3,}', '', summary_section).strip()
    redflags_section = re.sub(r'-{3,}', '', redflags_section).strip()

    def blockify(section_text):
        lines = section_text.split("\n")
        html_lines = []
        ul_open = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("•"):
                if not ul_open:
                    html_lines.append("<ul>")
                    ul_open = True
                html_lines.append(f"<li>{line[1:].strip()}</li>")
            else:
                if ul_open:
                    html_lines.append("</ul>")
                    ul_open = False
                # html_lines.append(f"<p>{line}</p>")
                html_lines.append(f'<p style="margin-top: 5px; margin-bottom: 5px;">{line}</p>')

        if ul_open:
            html_lines.append("</ul>")
        return "\n".join(html_lines)

    return blockify(summary_section), blockify(redflags_section)

def make_final_divs(flow_html, summary_html, redflag_html):
    analysis_div = f"""
    <div style="width: 100%; padding: 30px; font-family: Calibri, sans-serif; display: flex; gap: 20px; justify-content: flex-start; flex-wrap: wrap; box-sizing: border-box;">

      <div style="flex: 1; min-width: 320px; max-height: 650px; overflow-y: auto; background-color: #DFF5E1; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); line-height: 1.6; font-size: 22px;">
        <h2 style="color: #2e7d32;">💸 Summary of Fund Flow</h2>
        {summary_html}
      </div>

      <div style="flex: 1; min-width: 320px; max-height: 650px; overflow-y: auto; background-color: #FFE1E1; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); line-height: 1.6;font-size: 22px;">
        <h2 style="color: #c62828;">🚩 Red Flags</h2>
        {redflag_html}
      </div>

    </div>
    </body>
    </html>
    """
    return flow_html.replace("</body>\n</html>", analysis_div)

# ========== Main ==========
with open(json_path, "r") as f:
    all_data = json.load(f)

for entry in all_data:
    filename = entry["file"].replace(".csv", ".html")
    html_path = html_dir / filename
    if not html_path.exists():
        continue

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    raw_html = str(soup)
    blocks = convert_markdown_to_html_blocks(entry["response"])
    if len(blocks) < 2:
        continue
    summary_html, redflag_html = blocks[0], blocks[1]
    combined_html = make_final_divs(raw_html, summary_html, redflag_html)

    with open(output_dir / filename, "w", encoding="utf-8") as f:
        f.write(combined_html)

