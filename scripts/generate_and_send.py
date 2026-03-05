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
RECIPIENT_EMAILS   = [e.strip() for e in os.environ["RECIPIENT_EMAIL"].split(",") if e.strip()]
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
    current_group = None  # None means we haven't entered any ## section yet
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "---":
            continue
        if line.startswith("## "):
            current_group = line[3:].strip()
        elif line.startswith("#"):
            continue
        elif current_group is not None:
            # Only collect concept lines after the first ## heading
            concepts.append({"name": line, "group": current_group})
    return concepts


# ── OpenAI content generation ─────────────────────────────────────────────────

def generate_learning_content(concept: dict, domain: str, learned: list, graph: dict, session_count: int = 1) -> dict:
    """
    Call gpt-4o-mini to generate structured learning content.
    Returns a dict matching the concept_template.md structure.
    """
    template_text = (REF_DIR / "concept_template.md").read_text(encoding="utf-8")
    rules_text    = (REF_DIR / "rules.md").read_text(encoding="utf-8")

    learned_summary = ", ".join(learned[-15:]) if learned else "none yet"
    existing_node_labels = [n["label"] for n in graph["nodes"]]

    system_prompt = f"""你是一个AI学习引擎，负责生成每日深度AI学习邮件。
所有输出内容必须使用简体中文，内容面向系统学习AI的开发者，要求深入、细致、有实际价值。

【严格禁止】已学过的概念：{learned_summary}
对以上任何概念，禁止重新介绍或铺垫背景，哪怕一句话也不行。
如需引用，只能用"（参见：XXX）"简短提及，不得展开。

输出格式：返回一个JSON对象，包含以下键：

- "concept_name": 字符串（保留英文原名，加中文副标题，如"Multi-Agent Systems · 多智能体系统"）
- "definition": 字符串（深入定义，4-6句话，含技术本质、核心特征、与相关概念的区别）
- "key_terms": 字符串（6-10个核心术语，每条格式：• 英文术语名（中文译名）：2-3句详细中文解释；术语名必须保留英文）
- "simple_example": 字符串（2-3段生动类比或场景故事，具体有画面感）
- "applications": 字符串（6-8个真实应用场景，每条用 • 开头，含具体产品或公司名）
- "system_flow": 字符串（详细工作流程，每步2-3句说明；不适用填"N/A"）
- "system_flow_mermaid": 字符串（system_flow对应的Mermaid flowchart，graph TD风格，节点文字不超过8个中文字，节点id只用英文字母和下划线；不适用填"N/A"）
- "component_structure": 字符串（各组件详细说明，每条用 • 开头，含功能和组件间关系；不适用填"N/A"）
- "component_mermaid": 字符串（component_structure对应的Mermaid图，graph LR风格，节点文字不超过8个中文字，节点id只用英文字母和下划线；不适用填"N/A"）
- "technical_extensions": 字符串（6-8个相关技术/变体，每条用 • 开头，说明与本概念的关联）
- "current_status": 字符串（2-3句话，含技术现状、主要推动者、近期重要进展）
- "alternatives": 字符串（3-5个替代或互补方案，每条用 • 开头，说明优劣对比）
- "real_world_examples": 字符串（5-6个真实产品/系统，格式：• 产品名：1-2句说明）
- "references": 字符串（5-8条参考资料，格式：• [标题](url)，优先论文和官方文档）
- "mermaid_graph": 字符串（知识图谱用的Mermaid图，graph LR风格，5-8个节点）
- "new_nodes": 数组，每个元素：{{"id": string, "label": string, "group": string, "color": {{"background": string, "border": string}}}}
- "new_edges": 数组，每个元素：{{"from": string, "to": string, "label": string}}

new_nodes/new_edges规则：
- node id 必须是 snake_case 英文，如 "multi_agent_systems"
- 域根节点的精确 id 是："{domain}"（大小写和空格必须与此完全一致，不得修改）
- 必须有一条边从域根节点连到新概念：{{"from": "{domain}", "to": "<新节点id>", "label": "包含"}}
- 颜色：{json.dumps(DOMAIN_COLORS[domain])}，group："{domain}"

Mermaid图规则：所有节点id只用英文字母和下划线，节点文字用中文但不超过8个字，语法必须正确。
references必须是真实可访问的URL。"""

    user_prompt = f"""请为以下AI概念生成今日学习内容：

概念：{concept['name']}
所属领域：{domain}
所属分组：{concept['group']}

知识图谱已有节点：{existing_node_labels}

要求：内容深入细致，每个部分信息量充足，适合认真系统学习。"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=5000,
    )
    return json.loads(response.choices[0].message.content)


# ── Mermaid → image URL ───────────────────────────────────────────────────────

def mermaid_to_image_url(mermaid_code: str) -> str:
    """Encode Mermaid code to a mermaid.ink image URL. Returns empty string if too long."""
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
    url = f"https://mermaid.ink/img/{encoded}"
    # mermaid.ink fails silently above ~2000 chars; skip rather than show broken image
    if len(url) > 2000:
        return ""
    return url


# ── HTML email builder ────────────────────────────────────────────────────────

def _section(title: str, body, icon: str = "") -> str:
    # Normalize: list -> newline-joined string
    if isinstance(body, list):
        body = "\n".join(str(item) for item in body)
    if not body or str(body).strip().upper() in ("N/A", ""):
        return ""
    body_html = str(body).replace("\n", "<br>")
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

    def _mermaid_section(title: str, mermaid_code, icon: str) -> str:
        if not mermaid_code or str(mermaid_code).strip().upper() == "N/A":
            return ""
        img_url = mermaid_to_image_url(str(mermaid_code))
        return (
            f'\n    <div class="section">'
            f'\n      <h3>{icon} {title}</h3>'
            f'\n      <div style="text-align:center;margin-top:10px;">'
            f'\n        <img src="{img_url}" alt="{title}" style="max-width:100%;border-radius:8px;border:1px solid #e8eaff;" />'
            f'\n      </div>'
            f'\n    </div>'
        )

    sections_html = "".join([
        _section("概念定义",   content.get("definition", ""),              "📖"),
        _section("核心术语",   content.get("key_terms", ""),               "🔑"),
        _section("直觉理解",   content.get("simple_example", ""),          "💡"),
        _section("应用场景",   content.get("applications", ""),            "🚀"),
        _section("工作流程",   content.get("system_flow", ""),             "⚙️"),
        _mermaid_section("流程图", content.get("system_flow_mermaid", ""), "📊"),
        _section("组件结构",   content.get("component_structure", ""),     "🧩"),
        _mermaid_section("结构图", content.get("component_mermaid", ""),   "🗂️"),
        _section("技术延伸",   content.get("technical_extensions", ""),    "🔧"),
        _section("技术现状",   content.get("current_status", ""),          "📈"),
        _section("替代方案",   content.get("alternatives", ""),            "🔀"),
        _section("真实案例",   content.get("real_world_examples", ""),     "🌍"),
        _section("参考资料",   content.get("references", ""),              "📚"),
    ])

    graph_link_html = ""
    if graph_url:
        graph_link_html = f'<p style="margin-top:12px;"><a href="{graph_url}" style="color:#667eea;font-weight:600;">查看完整知识图谱 →</a></p>'

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
        <h3>今日知识图谱更新</h3>
        <img src="{mermaid_img_url}" alt="Knowledge Graph for {concept_name}" />
        {graph_link_html}
      </div>
    </div>
    <div class="footer">
      AI 学习引擎 &nbsp;&bull;&nbsp; 系统化构建 AI 知识体系，每天一个概念。<br>
      明天继续，每天进步一点点！
    </div>
  </div>
</body>
</html>"""


# ── Email sender ──────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = ", ".join(RECIPIENT_EMAILS)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAILS, msg.as_string())


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

    // Use node.level set by Python (0=AI, 1=domain, 2=group, 3=concept)
    const SIZE_MAP = {{ 0: 36, 1: 26, 2: 20, 3: 14 }};
    const FONT_MAP  = {{ 0: 16, 1: 13, 2: 12, 3: 11 }};
    nodesData.forEach(n => {{
      const lv = (n.level !== undefined) ? n.level : 3;
      n.level = lv;
      n.size  = SIZE_MAP[lv] || 14;
      n.font  = {{ size: FONT_MAP[lv] || 11, color: "#e6edf3" }};
    }});

    const options = {{
      nodes: {{
        shape: "dot",
        borderWidth: 2,
        shadow: true,
      }},
      edges: {{
        color: {{ color: "#484f58", opacity: 0.9, highlight: "#667eea" }},
        font: {{ size: 10, color: "#8b949e", align: "middle" }},
        smooth: {{ type: "cubicBezier", forceDirection: "vertical", roundness: 0.4 }},
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
        width: 1.5,
      }},
      physics: {{ enabled: false }},
      layout: {{
        hierarchical: {{
          enabled: true,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: 120,
          nodeSpacing: 160,
          treeSpacing: 200,
          blockShifting: true,
          edgeMinimization: true,
          parentCentralization: true,
        }},
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 150,
        navigationButtons: true,
        keyboard: true,
        zoomView: true,
        dragView: true,
      }},
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
        session_count=state["session_count"],
    )

    # ── Update knowledge graph (code-managed, not model-generated) ────────────
    def _node_id(name: str) -> str:
        """Convert name to snake_case id."""
        import re as _re
        s = _re.sub(r'\(.*?\)', '', name)        # strip parentheses
        s = _re.sub(r'[^a-zA-Z0-9]+', '_', s)   # non-alphanumeric → _
        return s.strip('_').lower()

    existing_ids = {n["id"] for n in state["knowledge_graph"]["nodes"]}

    # Level 2: group node (e.g. "Core System Architectures")
    group_name = concept["group"]
    group_id   = _node_id(group_name)
    if group_id not in existing_ids:
        state["knowledge_graph"]["nodes"].append({
            "id": group_id, "label": group_name,
            "group": domain, "level": 2,
            "color": DOMAIN_COLORS[domain],
        })
        state["knowledge_graph"]["edges"].append(
            {"from": domain, "to": group_id, "label": "包含"}
        )
        existing_ids.add(group_id)

    # Level 3: concept node (keep English name)
    concept_id = _node_id(concept["name"])
    if concept_id not in existing_ids:
        state["knowledge_graph"]["nodes"].append({
            "id": concept_id, "label": concept["name"],
            "group": domain, "level": 3,
            "color": DOMAIN_COLORS[domain],
        })
        state["knowledge_graph"]["edges"].append(
            {"from": group_id, "to": concept_id, "label": "包含"}
        )
        existing_ids.add(concept_id)

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
    concept_name = content.get("concept_name", concept["name"])
    subject = f"Day {state['session_count']} {concept_name}"
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
