#!/usr/bin/env python3
"""
AI Learning Engine - Daily Email Generator

Flow:
  1. Load state/progress.json  -> current domain + concept index
  2. Parse concept from reference/knowledge_map/<domain>.md
  3. Call OpenAI gpt-4o-mini to generate structured learning content
  4. Encode today's Mermaid graph -> mermaid.ink image URL
  5. Build HTML email and send via Gmail SMTP
  6. Update state/progress.json + docs/index.html (knowledge graph page)
"""

import os
import json
import re
import smtplib
import base64
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
STATE_FILE = ROOT / "state" / "progress.json"
DOCS_FILE  = ROOT / "docs" / "index.html"
REF_DIR    = ROOT / "reference"
KM_DIR     = REF_DIR / "knowledge_map"

# ── Config from environment (set as GitHub Secrets) ───────────────────────────
OPENAI_API_KEY     = os.environ["OPENAI_API_KEY"]
GMAIL_ADDRESS      = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL    = os.environ["RECIPIENT_EMAIL"]
GITHUB_PAGES_URL   = os.environ.get("GITHUB_PAGES_URL", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Domain config ─────────────────────────────────────────────────────────────
DOMAIN_ORDER = [
    "Systems",
    "LLM Capabilities",
    "Applications",
    "Model Principles",
    "Math Foundations",
]
DOMAIN_FILES = {
    "Systems":           "systems.md",
    "LLM Capabilities":  "llm_capabilities.md",
    "Applications":      "applications.md",
    "Model Principles":  "model_principles.md",
    "Math Foundations":  "math_foundations.md",
}
DOMAIN_COLORS = {
    "Systems":           {"background": "#667eea", "border": "#4a5cd4"},
    "LLM Capabilities":  {"background": "#f093fb", "border": "#d070e0"},
    "Applications":      {"background": "#4facfe", "border": "#2d8fe0"},
    "Model Principles":  {"background": "#43e97b", "border": "#2cc45a"},
    "Math Foundations":  {"background": "#fa8231", "border": "#e06010"},
}


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "session_count": 0,
        "current_domain_index": 0,
        "current_concept_index": 0,
        "learned_concepts": [],
        "knowledge_graph": {
            "nodes": [
                {"id": "AI",               "label": "AI",               "group": "root",             "color": {"background": "#ffffff", "border": "#cccccc"}},
                {"id": "Systems",          "label": "Systems",          "group": "Systems",          "color": DOMAIN_COLORS["Systems"]},
                {"id": "LLM Capabilities", "label": "LLM Capabilities", "group": "LLM Capabilities", "color": DOMAIN_COLORS["LLM Capabilities"]},
                {"id": "Applications",     "label": "Applications",     "group": "Applications",     "color": DOMAIN_COLORS["Applications"]},
                {"id": "Model Principles", "label": "Model Principles", "group": "Model Principles", "color": DOMAIN_COLORS["Model Principles"]},
                {"id": "Math Foundations", "label": "Math Foundations", "group": "Math Foundations", "color": DOMAIN_COLORS["Math Foundations"]},
            ],
            "edges": [
                {"from": "AI", "to": "Systems",          "label": "includes"},
                {"from": "AI", "to": "LLM Capabilities", "label": "includes"},
                {"from": "AI", "to": "Applications",     "label": "includes"},
                {"from": "AI", "to": "Model Principles", "label": "includes"},
                {"from": "AI", "to": "Math Foundations", "label": "includes"},
            ],
        },
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Concept parsing ───────────────────────────────────────────────────────────

def parse_concepts_from_file(domain_name: str) -> list:
    """
    Parse concepts from a domain .md file.
    Returns list of {"name": str, "group": str}.
    Lines under ## headings that are plain text (not headers, not ---) are concepts.
    """
    filepath = KM_DIR / DOMAIN_FILES[domain_name]
    text = filepath.read_text(encoding="utf-8")

    concepts = []
    current_group = "General"
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "---":
            continue
        if line.startswith("## "):
            current_group = line[3:].strip()
        elif line.startswith("#"):
            continue
        else:
            concepts.append({"name": line, "group": current_group})
    return concepts


# ── OpenAI content generation ─────────────────────────────────────────────────

def generate_learning_content(concept: dict, domain: str, learned: list, graph: dict) -> dict:
    """
    Call gpt-4o-mini to generate structured learning content.
    Returns a dict matching the concept_template.md structure.
    """
    template_text = (REF_DIR / "concept_template.md").read_text(encoding="utf-8")
    rules_text    = (REF_DIR / "rules.md").read_text(encoding="utf-8")

    learned_summary = ", ".join(learned[-15:]) if learned else "none yet"
    existing_node_labels = [n["label"] for n in graph["nodes"]]

    system_prompt = f"""You are an AI learning engine that generates structured daily learning emails.
Follow the TEMPLATE and RULES exactly.

TEMPLATE:
{template_text}

RULES:
{rules_text}

OUTPUT FORMAT:
Return a single JSON object with these keys:
- "concept_name": string
- "email_subject": string  (engaging, e.g. "Day 3: How AI Agents Think and Act")
- "definition": string  (concise, 2-4 sentences)
- "key_terms": string  (bullet list using • character)
- "simple_example": string  (1-2 paragraph analogy or story)
- "applications": string  (bullet list of real use cases)
- "system_flow": string  (numbered steps describing workflow, or "N/A" if not applicable)
- "component_structure": string  (bullet list of components, or "N/A" if not applicable)
- "technical_extensions": string  (bullet list of related techniques)
- "current_status": string  (one of: widely adopted / emerging / experimental / being replaced — with brief explanation)
- "alternatives": string  (bullet list)
- "real_world_examples": string  (named real products/systems with 1-line description each)
- "references": string  (bullet list with clickable markdown links: [title](url))
- "mermaid_graph": string  (valid Mermaid diagram code showing today's concept and relationships to nearby concepts)
- "new_nodes": array of {{"id": string, "label": string, "group": string, "color": {{"background": string, "border": string}}}}
- "new_edges": array of {{"from": string, "to": string, "label": string}}

For new_nodes: use group = the domain name "{domain}". Color background/border from these:
{json.dumps(DOMAIN_COLORS[domain])}

For mermaid_graph: use "graph LR" style. Show the concept connecting to its domain root and 2-3 related concepts.
Keep it clean, 5-8 nodes max.

IMPORTANT: References must use real, working URLs to papers, docs, or reputable articles."""

    user_prompt = f"""Generate today's learning content:

Concept: {concept['name']}
Domain: {domain}
Group within domain: {concept['group']}

Already learned (do not re-explain, only reference): {learned_summary}
Existing knowledge graph nodes: {existing_node_labels}

Make the content educational, clear, and practical. Suitable for a developer learning AI systematically."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=3500,
    )
    return json.loads(response.choices[0].message.content)


# ── Mermaid → image URL ───────────────────────────────────────────────────────

def mermaid_to_image_url(mermaid_code: str) -> str:
    """Encode Mermaid code to a mermaid.ink image URL."""
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
    return f"https://mermaid.ink/img/{encoded}"


# ── HTML email builder ────────────────────────────────────────────────────────

def _section(title: str, body: str, icon: str = "") -> str:
    if not body or body.strip().upper() in ("N/A", ""):
        return ""
    body_html = body.replace("\n", "<br>")
    # Convert markdown links [text](url) -> <a href="url">text</a>
    body_html = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'<a href="\2">\1</a>', body_html)
    # Convert bullet character
    body_html = body_html.replace("• ", "&#8226; ")
    return f"""
    <div class="section">
      <h3>{icon} {title}</h3>
      <div class="content">{body_html}</div>
    </div>"""


def build_html_email(content: dict, domain: str, session_num: int,
                     mermaid_img_url: str, graph_url: str) -> str:
    concept_name = content.get("concept_name", "AI Concept")
    today = datetime.now().strftime("%Y-%m-%d")

    sections_html = "".join([
        _section("Concept Definition",   content.get("definition", ""),            "📖"),
        _section("Key Terminology",       content.get("key_terms", ""),             "🔑"),
        _section("Simple Example",        content.get("simple_example", ""),        "💡"),
        _section("Application Scenarios", content.get("applications", ""),          "🚀"),
        _section("System Flow",           content.get("system_flow", ""),           "⚙️"),
        _section("Component Structure",   content.get("component_structure", ""),   "🧩"),
        _section("Technical Extensions",  content.get("technical_extensions", ""),  "🔧"),
        _section("Current Status",        content.get("current_status", ""),        "📊"),
        _section("Alternative Approaches",content.get("alternatives", ""),          "🔀"),
        _section("Real-world Examples",   content.get("real_world_examples", ""),   "🌍"),
        _section("References",            content.get("references", ""),            "📚"),
    ])

    graph_link_html = ""
    if graph_url:
        graph_link_html = f'<p style="margin-top:12px;"><a href="{graph_url}" style="color:#667eea;font-weight:600;">View Full Knowledge Graph →</a></p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{concept_name}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f0f2f5;
      margin: 0;
      padding: 20px;
      color: #333;
    }}
    .container {{
      max-width: 680px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    }}
    .header {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 32px 36px 28px;
      color: #fff;
    }}
    .header .badge {{
      display: inline-block;
      background: rgba(255,255,255,0.2);
      border-radius: 20px;
      padding: 4px 14px;
      font-size: 12px;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 26px;
      font-weight: 700;
      line-height: 1.3;
    }}
    .body {{
      padding: 28px 36px;
    }}
    .section {{
      margin-bottom: 22px;
      padding-left: 14px;
      border-left: 3px solid #667eea;
    }}
    .section h3 {{
      margin: 0 0 8px 0;
      font-size: 14px;
      font-weight: 600;
      color: #667eea;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .content {{
      font-size: 15px;
      line-height: 1.75;
      color: #444;
    }}
    .graph-card {{
      background: #f8f9ff;
      border: 1px solid #e8eaff;
      border-radius: 12px;
      padding: 24px;
      text-align: center;
      margin: 24px 0;
    }}
    .graph-card h3 {{
      margin: 0 0 16px 0;
      font-size: 15px;
      color: #667eea;
    }}
    .graph-card img {{
      max-width: 100%;
      border-radius: 8px;
      border: 1px solid #e0e0e0;
    }}
    .footer {{
      background: #f8f9ff;
      padding: 18px 36px;
      font-size: 13px;
      color: #999;
      border-top: 1px solid #eee;
      text-align: center;
    }}
    a {{ color: #667eea; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="badge">Session #{session_num} &nbsp;|&nbsp; {domain} &nbsp;|&nbsp; {today}</div>
      <h1>{concept_name}</h1>
    </div>
    <div class="body">
      {sections_html}
      <div class="graph-card">
        <h3>Today's Knowledge Graph Update</h3>
        <img src="{mermaid_img_url}" alt="Knowledge Graph for {concept_name}" />
        {graph_link_html}
      </div>
    </div>
    <div class="footer">
      AI Learning Engine &nbsp;&bull;&nbsp; Building systematic AI knowledge, one concept a day.<br>
      See you tomorrow!
    </div>
  </div>
</body>
</html>"""


# ── Email sender ──────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


# ── GitHub Pages knowledge graph ──────────────────────────────────────────────

def update_graph_page(graph: dict, session_count: int):
    nodes_json = json.dumps(graph["nodes"], ensure_ascii=False)
    edges_json = json.dumps(graph["edges"], ensure_ascii=False)
    node_count = len(graph["nodes"])
    today = datetime.now().strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AI Learning Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    #header {{
      padding: 18px 28px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    #header h1 {{ font-size: 20px; font-weight: 700; }}
    #header .meta {{ font-size: 13px; opacity: 0.85; }}
    #legend {{
      display: flex;
      gap: 16px;
      padding: 10px 28px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      flex-wrap: wrap;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: #8b949e;
    }}
    .legend-dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }}
    #graph {{ width: 100%; height: calc(100vh - 100px); }}
  </style>
</head>
<body>
  <div id="header">
    <h1>AI Learning Knowledge Graph</h1>
    <div class="meta">Session #{session_count} &nbsp;|&nbsp; {node_count} concepts &nbsp;|&nbsp; Updated {today}</div>
  </div>
  <div id="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#667eea"></div> Systems</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f093fb"></div> LLM Capabilities</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4facfe"></div> Applications</div>
    <div class="legend-item"><div class="legend-dot" style="background:#43e97b"></div> Model Principles</div>
    <div class="legend-item"><div class="legend-dot" style="background:#fa8231"></div> Math Foundations</div>
  </div>
  <div id="graph"></div>
  <script>
    const nodesData = {nodes_json};
    const edgesData = {edges_json};

    const nodes = new vis.DataSet(nodesData);
    const edges = new vis.DataSet(edgesData);

    const options = {{
      nodes: {{
        shape: "dot",
        size: 20,
        font: {{ size: 13, color: "#e6edf3", face: "sans-serif" }},
        borderWidth: 2,
        shadow: true,
      }},
      edges: {{
        color: {{ color: "#484f58", opacity: 0.8, highlight: "#667eea" }},
        font: {{ size: 10, color: "#8b949e", align: "middle" }},
        smooth: {{ type: "continuous" }},
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
        width: 1.5,
      }},
      physics: {{
        stabilization: {{ iterations: 250, fit: true }},
        barnesHut: {{
          gravitationalConstant: -9000,
          springLength: 130,
          springConstant: 0.04,
          damping: 0.12,
        }},
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 150,
        navigationButtons: true,
        keyboard: true,
      }},
      layout: {{ improvedLayout: true }},
    }};

    new vis.Network(document.getElementById("graph"), {{ nodes, edges }}, options);
  </script>
</body>
</html>"""

    DOCS_FILE.parent.mkdir(exist_ok=True)
    DOCS_FILE.write_text(html, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    state = load_state()
    state["session_count"] += 1

    domain_idx  = state["current_domain_index"]
    concept_idx = state["current_concept_index"]

    # Wrap around when all domains complete
    if domain_idx >= len(DOMAIN_ORDER):
        print("All domains completed — restarting from the beginning.")
        state["current_domain_index"] = 0
        state["current_concept_index"] = 0
        domain_idx  = 0
        concept_idx = 0

    domain   = DOMAIN_ORDER[domain_idx]
    concepts = parse_concepts_from_file(domain)

    # Advance to next domain if current is exhausted
    if concept_idx >= len(concepts):
        state["current_domain_index"] += 1
        state["current_concept_index"] = 0
        domain_idx  = state["current_domain_index"]
        if domain_idx >= len(DOMAIN_ORDER):
            print("All domains completed.")
            state["current_domain_index"] = 0
            domain_idx = 0
        domain   = DOMAIN_ORDER[domain_idx]
        concepts = parse_concepts_from_file(domain)
        concept_idx = 0

    concept = concepts[concept_idx]
    print(f"[Session {state['session_count']}] {domain} / {concept['group']} / {concept['name']}")

    # Generate content
    content = generate_learning_content(
        concept=concept,
        domain=domain,
        learned=state["learned_concepts"],
        graph=state["knowledge_graph"],
    )

    # Update knowledge graph state
    existing_ids = {n["id"] for n in state["knowledge_graph"]["nodes"]}
    for node in content.get("new_nodes", []):
        if node.get("id") and node["id"] not in existing_ids:
            state["knowledge_graph"]["nodes"].append(node)
            existing_ids.add(node["id"])
    for edge in content.get("new_edges", []):
        state["knowledge_graph"]["edges"].append(edge)

    # Mermaid image
    mermaid_code    = content.get("mermaid_graph", f"graph LR\n  A[{concept['name']}] --> B[{domain}]")
    mermaid_img_url = mermaid_to_image_url(mermaid_code)

    # Build + send email
    html_body = build_html_email(
        content=content,
        domain=domain,
        session_num=state["session_count"],
        mermaid_img_url=mermaid_img_url,
        graph_url=GITHUB_PAGES_URL,
    )
    subject = content.get("email_subject", f"AI Learning #{state['session_count']}: {concept['name']}")
    send_email(subject, html_body)
    print(f"Email sent: {subject}")

    # Advance progress
    state["learned_concepts"].append(concept["name"])
    state["current_concept_index"] += 1

    # Persist
    save_state(state)
    update_graph_page(state["knowledge_graph"], state["session_count"])
    print("State and knowledge graph page updated.")


if __name__ == "__main__":
    main()
