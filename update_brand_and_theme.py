import re

# 1. Update index.html
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Replace ZEUS-AI / Zeus-AI / ZEUS with VAJRA-AI / Vajra-AI / VAJRA
html = html.replace("ZEUS-AI", "VAJRA-AI")
html = html.replace("Zeus-AI", "Vajra-AI")
html = html.replace("ZEUS", "VAJRA")
html = html.replace("zeus", "vajra")

# Update title tag if needed
html = html.replace("<title>⚡ Lightning Nowcaster | SEVIR, GOES, NEXRAD & IMD Data Integration</title>",
                     "<title>⚡ VAJRA-AI // Atmospheric Nowcasting | Ministry of Earth Sciences & ISRO</title>")

# Add Light Theme CSS rules before </style>
light_theme_css = """
    /* --- Light Theme Mode Overrides --- */
    body.light-theme {
      --bg-dark: #f8fafc;
      --bg-card: rgba(255, 255, 255, 0.95);
      --bg-card-hover: rgba(241, 245, 249, 1);
      --border-color: #cbd5e1;
      --border-accent: #94a3b8;
      
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;

      background-color: #f8fafc !important;
      color: #0f172a !important;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(2, 132, 199, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(121, 40, 202, 0.05) 0%, transparent 40%) !important;
    }

    body.light-theme header {
      background: rgba(255, 255, 255, 0.95) !important;
      border-bottom: 1px solid #cbd5e1 !important;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
    }

    body.light-theme .brand-title {
      background: linear-gradient(135deg, #0f172a 30%, #0284c7 100%) !important;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
    }

    body.light-theme .brand-subtitle {
      color: #64748b !important;
    }

    body.light-theme div[style*="background: #050810"],
    body.light-theme div[style*="background:#050810"] {
      background: #f1f5f9 !important;
      border-bottom-color: #cbd5e1 !important;
      color: #475569 !important;
    }

    body.light-theme .tab-btn {
      background: #e2e8f0 !important;
      color: #334155 !important;
      border: 1px solid #cbd5e1 !important;
    }

    body.light-theme .tab-btn.active {
      background: linear-gradient(135deg, #0284c7, #2563eb) !important;
      color: #ffffff !important;
      border-color: #0284c7 !important;
      box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }

    body.light-theme .card, 
    body.light-theme .control-panel, 
    body.light-theme .pipeline-step,
    body.light-theme .dataset-card,
    body.light-theme div[style*="background: rgba(15, 23, 42"],
    body.light-theme div[style*="background: #0f172a"],
    body.light-theme div[style*="background:#0f172a"] {
      background: #ffffff !important;
      border-color: #cbd5e1 !important;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06) !important;
      color: #0f172a !important;
    }

    body.light-theme table th {
      background: #f1f5f9 !important;
      color: #475569 !important;
      border-bottom-color: #cbd5e1 !important;
    }

    body.light-theme table td {
      border-bottom-color: #e2e8f0 !important;
      color: #1e293b !important;
    }

    body.light-theme pre, 
    body.light-theme code,
    body.light-theme div[style*="background: #080c14"],
    body.light-theme div[style*="background:#080c14"] {
      background: #f1f5f9 !important;
      color: #0f172a !important;
      border-color: #cbd5e1 !important;
    }

    body.light-theme input, 
    body.light-theme select {
      background: #ffffff !important;
      color: #0f172a !important;
      border-color: #cbd5e1 !important;
    }

    body.light-theme .leaflet-container {
      background: #e2e8f0 !important;
    }
  </style>
"""

html = html.replace("</style>", light_theme_css + "\n</style>")

# Update Header HTML to include header actions with Theme Switcher
old_header_pattern = r'<header>\s*<div class="brand">.*?</div>\s*</div>\s*</header>'

new_header = """<header>
    <div class="brand">
      <div class="brand-icon">⚡</div>
      <div>
        <div class="brand-title" style="display: flex; align-items: center; gap: 0.5rem;">
          <span>VAJRA-AI // Atmospheric Nowcasting</span>
          <span style="background: rgba(0, 242, 254, 0.15); color: var(--cyan); border: 1px solid rgba(0, 242, 254, 0.3); font-size: 0.65rem; padding: 0.15rem 0.4rem; border-radius: 4px; font-family: var(--font-code);">LATENCY: 420ms</span>
        </div>
        <div class="brand-subtitle">SEVIR • GOES-16/18 • NEXRAD Doppler • IMD INSAT-3D/DWR • IITM LLN (Damini) • ISRO NRSC</div>
      </div>
    </div>

    <div class="header-actions" style="display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;">
      <div style="font-family: var(--font-code); font-size: 0.75rem; color: var(--cyan); background: rgba(0, 242, 254, 0.08); padding: 0.35rem 0.65rem; border-radius: 6px; border: 1px solid rgba(0, 242, 254, 0.2);" id="live-utc-clock">
        14:28:45 UTC (+05:30 IST)
      </div>
      <div class="badge badge-danger" style="background: rgba(239, 68, 68, 0.15); color: var(--rose); border: 1px solid rgba(239, 68, 68, 0.3); padding: 0.35rem 0.65rem; border-radius: 6px; font-family: var(--font-code); font-size: 0.7rem; font-weight: 700;">
        • THREAT: SEVERE // 4 CELLS
      </div>
      <button id="theme-toggle-btn" onclick="toggleTheme()" style="background: rgba(255, 255, 255, 0.1); color: var(--text-main); border: 1px solid var(--border-color); padding: 0.4rem 0.85rem; border-radius: 6px; font-family: var(--font-main); font-size: 0.75rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 0.4rem; transition: all 0.2s ease;">
        <span id="theme-toggle-icon">☀️</span> <span id="theme-toggle-text">Light Mode</span>
      </button>
      <button class="btn" style="background: linear-gradient(135deg, #ef4444, #dc2626); color: white; border: none; padding: 0.4rem 0.85rem; border-radius: 6px; font-weight: 700; font-size: 0.75rem; cursor: pointer;" onclick="triggerBroadcast()">
        ⚡ BROADCAST
      </button>
      <button class="btn" style="background: rgba(255, 255, 255, 0.05); color: var(--cyan); border: 1px solid rgba(0, 242, 254, 0.4); padding: 0.4rem 0.85rem; border-radius: 6px; font-weight: 600; font-size: 0.75rem; cursor: pointer;" onclick="runTestsModal()">
        🧪 Run Verification Tests
      </button>
    </div>
  </header>"""

html = re.sub(old_header_pattern, new_header, html, flags=re.DOTALL)

# Add toggleTheme JavaScript function before </body>
theme_js = """
  <script>
    function toggleTheme() {
      const body = document.body;
      const isLight = body.classList.toggle('light-theme');
      const toggleBtnText = document.getElementById('theme-toggle-text');
      const toggleBtnIcon = document.getElementById('theme-toggle-icon');
      
      if (isLight) {
        if (toggleBtnText) toggleBtnText.innerText = "Dark Mode";
        if (toggleBtnIcon) toggleBtnIcon.innerText = "🌙";
        localStorage.setItem('vajra_theme', 'light');
      } else {
        if (toggleBtnText) toggleBtnText.innerText = "Light Mode";
        if (toggleBtnIcon) toggleBtnIcon.innerText = "☀️";
        localStorage.setItem('vajra_theme', 'dark');
      }
    }

    // Check local storage on load
    document.addEventListener('DOMContentLoaded', () => {
      const savedTheme = localStorage.getItem('vajra_theme');
      if (savedTheme === 'light') {
        toggleTheme();
      }
    });
  </script>
</body>
"""

html = html.replace("</body>", theme_js)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Updated index.html branding to VAJRA-AI and added Light/Dark Theme toggle.")

# 2. Update server.py
with open("server.py", "r", encoding="utf-8") as f:
    server_code = f.read()

server_code = server_code.replace("ZEUS-AI", "VAJRA-AI")
server_code = server_code.replace("Zeus-AI", "Vajra-AI")
server_code = server_code.replace("ZEUS", "VAJRA")
server_code = server_code.replace("zeus", "vajra")

with open("server.py", "w", encoding="utf-8") as f:
    f.write(server_code)

print("Updated server.py branding to VAJRA-AI.")

# 3. Update run_tests.py
try:
    with open("run_tests.py", "r", encoding="utf-8") as f:
        tests_code = f.read()
    tests_code = tests_code.replace("ZEUS-AI", "VAJRA-AI")
    tests_code = tests_code.replace("Zeus-AI", "Vajra-AI")
    tests_code = tests_code.replace("ZEUS", "VAJRA")
    tests_code = tests_code.replace("zeus", "vajra")
    with open("run_tests.py", "w", encoding="utf-8") as f:
        f.write(tests_code)
    print("Updated run_tests.py branding to VAJRA-AI.")
except Exception as e:
    print(f"run_tests.py update notice: {e}")
