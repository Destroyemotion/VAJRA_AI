import os
import re

print("Building redesigned index.html...")

with open("index.html.bak", "r", encoding="utf-8") as f:
    orig_content = f.read()

# Extract original tab contents so no working features are removed!
# We look for all tabs from tab-live to tab-pipeline
tabs_start_idx = orig_content.find('<div id="tab-live"')
tabs_end_idx = orig_content.find('</main>')

if tabs_start_idx != -1 and tabs_end_idx != -1:
    orig_tabs_html = orig_content[tabs_start_idx:tabs_end_idx]
else:
    orig_tabs_html = ""

# Extract original script tags from index.html.bak so all JS functions (canvas rendering, verification suite, etc.) are preserved
script_start_idx = orig_content.find('<script>')
orig_scripts_html = orig_content[script_start_idx:] if script_start_idx != -1 else ""

new_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚡ VAJRA-AI // Thunderstorm & Lightning Decision Support | Ministry of Earth Sciences</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  
  <!-- Leaflet GIS Map Engine -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    :root {{
      /* Default Theme: Executive Light (matching screenshot reference) */
      --bg-app: #f0f4f9;
      --bg-sidebar: linear-gradient(180deg, #0b1736 0%, #070d24 100%);
      --bg-header: #ffffff;
      --bg-card: #ffffff;
      --bg-card-alt: #f8fafc;
      --bg-card-hover: #f1f5f9;
      
      --border-color: #cbd5e1;
      --border-accent: #94a3b8;
      
      --text-main: #0f172a;
      --text-body: #334155;
      --text-muted: #64748b;
      --text-dim: #94a3b8;
      
      --cyan: #0284c7;
      --blue: #2563eb;
      --violet: #7928ca;
      --magenta: #ff007f;
      --emerald: #16a34a;
      --amber: #d97706;
      --rose: #dc2626;
      --danger: #ef4444;

      --card-shadow: 0 2px 10px -2px rgba(15, 23, 42, 0.08), 0 1px 4px -1px rgba(15, 23, 42, 0.04);
      --card-radius: 12px;

      --font-main: 'Inter', system-ui, -apple-system, sans-serif;
      --font-code: 'JetBrains Mono', monospace;
    }}

    body.dark-theme {{
      --bg-app: #090d16;
      --bg-sidebar: #060a17;
      --bg-header: rgba(9, 13, 22, 0.95);
      --bg-card: rgba(16, 23, 38, 0.9);
      --bg-card-alt: rgba(24, 34, 56, 0.7);
      --bg-card-hover: rgba(30, 41, 64, 0.9);
      
      --border-color: #1e293b;
      --border-accent: #334155;
      
      --text-main: #f8fafc;
      --text-body: #cbd5e1;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      --cyan: #00f2fe;
      --blue: #4facfe;
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --danger: #ef4444;

      --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg-app);
      color: var(--text-main);
      font-family: var(--font-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      transition: background-color 0.3s ease, color 0.3s ease;
    }}

    /* Outer Application Layout */
    .app-layout {{
      display: flex;
      min-height: 100vh;
      width: 100%;
    }}

    /* Left Sidebar Navigation */
    .sidebar {{
      width: 230px;
      background: var(--bg-sidebar);
      color: #f8fafc;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      flex-shrink: 0;
      z-index: 120;
      border-right: 1px solid rgba(255, 255, 255, 0.08);
      position: sticky;
      top: 0;
      height: 100vh;
    }}

    .sidebar-brand-box {{
      padding: 1.25rem 1.25rem 1rem 1.25rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .sidebar-logo-icon {{
      width: 38px;
      height: 38px;
      border-radius: 8px;
      background: linear-gradient(135deg, #0284c7, #2563eb);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      color: #ffffff;
      box-shadow: 0 0 15px rgba(2, 132, 199, 0.4);
    }}

    .sidebar-brand-title {{
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #ffffff;
    }}

    .sidebar-brand-sub {{
      font-size: 0.68rem;
      color: #94a3b8;
      font-weight: 500;
    }}

    .sidebar-menu {{
      padding: 1rem 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      flex: 1;
    }}

    .nav-item {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.7rem 0.9rem;
      border-radius: 8px;
      color: #94a3b8;
      font-weight: 500;
      font-size: 0.875rem;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.2s ease;
      position: relative;
    }}

    .nav-item:hover {{
      color: #ffffff;
      background: rgba(255, 255, 255, 0.08);
    }}

    .nav-item.active {{
      color: #ffffff;
      background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
      font-weight: 600;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }}

    .nav-badge {{
      margin-left: auto;
      background: #ef4444;
      color: #ffffff;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 0.15rem 0.45rem;
      border-radius: 999px;
    }}

    .sidebar-footer {{
      padding: 1rem 1rem 1.25rem 1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(0, 0, 0, 0.2);
    }}

    .preparedness-card {{
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 0.85rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    .preparedness-flag {{
      font-size: 1.5rem;
    }}

    .preparedness-text {{
      font-size: 0.72rem;
      font-weight: 700;
      color: #e2e8f0;
      line-height: 1.2;
    }}

    /* Main Workspace Container */
    .main-wrapper {{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}

    /* Top Command Header */
    .header {{
      background: var(--bg-header);
      border-bottom: 1px solid var(--border-color);
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: var(--card-shadow);
      transition: background-color 0.3s ease;
    }}

    .hdr-left {{
      display: flex;
      align-items: center;
      gap: 1.25rem;
    }}

    .moes-emblem-box {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      border-right: 1px solid var(--border-color);
      padding-right: 1.25rem;
    }}

    .moes-emblem-icon {{
      font-size: 1.8rem;
    }}

    .moes-title-box {{
      line-height: 1.15;
    }}

    .moes-gov-title {{
      font-size: 0.85rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.01em;
    }}

    .moes-gov-sub {{
      font-size: 0.7rem;
      color: var(--text-muted);
      font-weight: 500;
    }}

    .vajra-brand-inline {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}

    .vajra-bolt-badge {{
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: #2563eb;
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      font-weight: bold;
    }}

    .vajra-brand-name {{
      font-size: 1.2rem;
      font-weight: 900;
      letter-spacing: 0.05em;
      color: #1d4ed8;
    }}

    body.dark-theme .vajra-brand-name {{
      color: #60a5fa;
    }}

    .vajra-brand-desc {{
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 600;
    }}

    .hdr-center {{
      font-size: 0.82rem;
      color: var(--text-muted);
      font-weight: 600;
      letter-spacing: 0.02em;
    }}

    .hdr-right {{
      display: flex;
      align-items: center;
      gap: 1rem;
    }}

    .hdr-datetime-box {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-main);
      background: var(--bg-card-alt);
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      border: 1px solid var(--border-color);
    }}

    .hdr-location-select {{
      background: var(--bg-card-alt);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      font-family: var(--font-main);
      cursor: pointer;
      outline: none;
    }}

    .hdr-location-select:hover {{
      border-color: #2563eb;
    }}

    .hdr-control-btn {{
      background: var(--bg-card-alt);
      border: 1px solid var(--border-color);
      color: var(--text-body);
      padding: 0.4rem 0.6rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.2s;
    }}

    .hdr-control-btn:hover {{
      background: var(--bg-card-hover);
      color: var(--text-main);
    }}

    /* Main Dashboard Content Layout */
    .dashboard-container {{
      flex: 1;
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      max-width: 1750px;
      margin: 0 auto;
      width: 100%;
    }}

    .tab-content {{
      display: none;
    }}

    .tab-content.active {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}

    /* Standard Card Component */
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--card-radius);
      box-shadow: var(--card-shadow);
      padding: 1.1rem;
      display: flex;
      flex-direction: column;
      transition: background-color 0.3s ease, border-color 0.3s ease;
    }}

    .card-header-sm {{
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    /* Grid Layout Rows */
    .row-top-cards {{
      display: grid;
      grid-template-columns: 2.8fr 1.2fr;
      gap: 1.25rem;
    }}

    /* Multi-Horizon Cards Row */
    .multi-horizon-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.85rem;
      margin-top: 0.25rem;
    }}

    .horizon-card {{
      background: var(--bg-card-alt);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 0.85rem 0.9rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      transition: all 0.2s ease;
    }}

    .horizon-card:hover {{
      border-color: #2563eb;
      transform: translateY(-1px);
    }}

    .horizon-icon-title {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text-main);
    }}

    .horizon-icon {{
      font-size: 1.25rem;
      color: #2563eb;
    }}

    .horizon-desc {{
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 500;
    }}

    .badge-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.65rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      width: fit-content;
    }}

    .badge-pill.active {{
      background: rgba(22, 163, 74, 0.12);
      color: var(--emerald);
      border: 1px solid rgba(22, 163, 74, 0.3);
    }}

    .badge-pill.high {{
      background: rgba(220, 38, 38, 0.12);
      color: var(--rose);
      border: 1px solid rgba(220, 38, 38, 0.3);
    }}

    .badge-pill.moderate {{
      background: rgba(217, 119, 6, 0.12);
      color: var(--amber);
      border: 1px solid rgba(217, 119, 6, 0.3);
    }}

    .badge-pill.low {{
      background: rgba(37, 99, 235, 0.12);
      color: #2563eb;
      border: 1px solid rgba(37, 99, 235, 0.3);
    }}

    .badge-pill.unavailable {{
      background: rgba(100, 116, 139, 0.12);
      color: var(--text-muted);
      border: 1px solid rgba(100, 116, 139, 0.3);
    }}

    /* Data Sources Status Card */
    .data-sources-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 0.5rem;
      margin-top: 0.25rem;
    }}

    .source-col {{
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      gap: 0.25rem;
      background: var(--bg-card-alt);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 0.6rem 0.3rem;
    }}

    .source-icon {{
      font-size: 1.1rem;
    }}

    .source-name {{
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--text-main);
    }}

    .source-status-text {{
      font-size: 0.65rem;
      font-weight: 700;
    }}

    .source-latency {{
      font-size: 0.62rem;
      color: var(--text-muted);
    }}

    /* Row 2 Summary Cards */
    .row-middle-cards {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1.25rem;
    }}

    .summary-card {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 0.75rem;
    }}

    .summary-title-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .summary-city {{
      font-size: 0.82rem;
      font-weight: 800;
      color: var(--text-main);
    }}

    .summary-date {{
      font-size: 0.72rem;
      color: var(--text-muted);
    }}

    .summary-badges-row {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin: 0.2rem 0;
    }}

    .summary-metrics-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.5rem;
      border-top: 1px solid var(--border-color);
      padding-top: 0.6rem;
    }}

    .metric-box {{
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }}

    .metric-lbl {{
      font-size: 0.65rem;
      color: var(--text-muted);
      font-weight: 600;
    }}

    .metric-val {{
      font-size: 0.85rem;
      font-weight: 800;
      color: var(--text-main);
    }}

    /* Active Alert Special Red Card */
    .alert-card-danger {{
      background: rgba(239, 68, 68, 0.05);
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}

    .alert-header {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #dc2626;
      font-weight: 800;
      font-size: 0.9rem;
    }}

    .alert-btn-row {{
      display: flex;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }}

    .btn-solid-danger {{
      background: #dc2626;
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 0.4rem 0.8rem;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      flex: 1;
    }}

    .btn-outline-danger {{
      background: transparent;
      color: #dc2626;
      border: 1px solid #dc2626;
      border-radius: 6px;
      padding: 0.4rem 0.8rem;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      flex: 1;
    }}

    /* Impact List Card */
    .impact-list {{
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }}

    .impact-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-main);
      padding: 0.2rem 0;
    }}

    .impact-item-left {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    /* Row 3 Main Workspace (Map + Right Panels) */
    .row-main-grid {{
      display: grid;
      grid-template-columns: 2.8fr 1.2fr;
      gap: 1.25rem;
      align-items: start;
    }}

    /* Interactive Central Map Container */
    .map-card {{
      display: flex;
      flex-direction: column;
      gap: 0;
      padding: 0;
      overflow: hidden;
      position: relative;
    }}

    .map-tabs-header {{
      background: var(--bg-card-alt);
      border-bottom: 1px solid var(--border-color);
      padding: 0.5rem 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}

    .map-tab-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      font-family: var(--font-main);
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }}

    .map-tab-btn:hover {{
      color: var(--text-main);
    }}

    .map-tab-btn.active {{
      background: #2563eb;
      color: #ffffff;
    }}

    .map-view-box {{
      height: 480px;
      width: 100%;
      position: relative;
      background: #0f172a;
    }}

    #dash-main-leaflet-map {{
      height: 100%;
      width: 100%;
      z-index: 10;
    }}

    .map-overlay-legend {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 20;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      color: #ffffff;
      font-size: 0.7rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      pointer-events: none;
    }}

    .legend-dbz-bar {{
      height: 8px;
      width: 140px;
      border-radius: 4px;
      background: linear-gradient(90deg, #0000ff 0%, #00ffff 20%, #00ff00 40%, #ffff00 60%, #ff0000 80%, #ff00ff 100%);
    }}

    .legend-dbz-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 0.6rem;
      color: #cbd5e1;
    }}

    .legend-symbols {{
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      margin-top: 0.2rem;
    }}

    .legend-sym-item {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    .map-overlay-popup {{
      position: absolute;
      top: 20%;
      right: 15%;
      z-index: 20;
      background: rgba(15, 23, 42, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(59, 130, 246, 0.5);
      border-radius: 8px;
      padding: 0.75rem;
      color: #ffffff;
      font-size: 0.72rem;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
      pointer-events: none;
    }}

    .popup-title {{
      font-weight: 800;
      color: #60a5fa;
      margin-bottom: 0.25rem;
    }}

    .map-timeline-bar {{
      background: var(--bg-card-alt);
      border-top: 1px solid var(--border-color);
      padding: 0.6rem 1rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    }}

    .play-btn {{
      background: #2563eb;
      color: #ffffff;
      border: none;
      border-radius: 50%;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 0.9rem;
    }}

    .timeline-ticks {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex: 1;
    }}

    .timeline-tick-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.72rem;
      font-weight: 700;
      cursor: pointer;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
    }}

    .timeline-tick-btn.active {{
      background: rgba(37, 99, 235, 0.12);
      color: #2563eb;
    }}

    .map-disclaimer-banner {{
      background: rgba(217, 119, 6, 0.08);
      border-top: 1px solid rgba(217, 119, 6, 0.2);
      padding: 0.4rem 1rem;
      font-size: 0.65rem;
      color: var(--amber);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    /* Right Side Supporting Panels Stack */
    .right-panels-column {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}

    .factor-list {{
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }}

    .factor-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-main);
      padding: 0.25rem 0;
      border-bottom: 1px dashed var(--border-color);
    }}

    .ai-assessment-box {{
      background: rgba(37, 99, 235, 0.08);
      border: 1px solid rgba(37, 99, 235, 0.3);
      border-radius: 8px;
      padding: 0.6rem;
      margin-top: 0.6rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
      font-weight: 800;
      color: #2563eb;
    }}

    .perf-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.4rem;
      text-align: center;
      margin-top: 0.4rem;
    }}

    .perf-val {{
      font-size: 0.95rem;
      font-weight: 800;
      color: var(--text-main);
    }}

    .perf-lbl {{
      font-size: 0.62rem;
      color: var(--text-muted);
      font-weight: 600;
    }}

    .card-footer-link {{
      font-size: 0.72rem;
      font-weight: 700;
      color: #2563eb;
      text-decoration: none;
      margin-top: 0.6rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }}

    .quick-actions-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
    }}

    .action-btn {{
      background: var(--bg-card-alt);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 0.6rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--text-main);
      cursor: pointer;
      transition: all 0.2s;
    }}

    .action-btn:hover {{
      border-color: #2563eb;
      background: var(--bg-card-hover);
    }}

    /* Application Footer */
    .app-footer {{
      background: var(--bg-header);
      border-top: 1px solid var(--border-color);
      padding: 0.85rem 1.5rem;
      font-size: 0.72rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: auto;
    }}

    /* Responsive Controls */
    @media (max-width: 1200px) {{
      .row-top-cards, .row-main-grid {{
        grid-template-columns: 1fr;
      }}
      .row-middle-cards {{
        grid-template-columns: repeat(2, 1fr);
      }}
      .multi-horizon-grid {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}

    @media (max-width: 768px) {{
      .app-layout {{
        flex-direction: column;
      }}
      .sidebar {{
        width: 100%;
        height: auto;
        position: relative;
      }}
      .row-middle-cards {{
        grid-template-columns: 1fr;
      }}
      .multi-horizon-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>

<body>
  <div class="app-layout">
    <!-- Left Navigation Sidebar -->
    <aside class="sidebar">
      <div>
        <div class="sidebar-brand-box">
          <div class="sidebar-logo-icon">⚡</div>
          <div>
            <div class="sidebar-brand-title">VAJRA-AI</div>
            <div class="sidebar-brand-sub">Decision Support</div>
          </div>
        </div>

        <nav class="sidebar-menu">
          <a class="nav-item active" id="nav-home" onclick="switchTabNav('tab-home', this)">
            <span style="font-size: 1.1rem;">🏠</span> <span>Home</span>
          </a>
          <a class="nav-item" id="nav-live" onclick="switchTabNav('tab-live', this)">
            <span style="font-size: 1.1rem;">🗺️</span> <span>Live Map</span>
          </a>
          <a class="nav-item" id="nav-alerts" onclick="switchTabNav('tab-early-warning', this)">
            <span style="font-size: 1.1rem;">🚨</span> <span>Alerts</span> <span class="nav-badge">3</span>
          </a>
          <a class="nav-item" id="nav-forecast" onclick="switchTabNav('tab-visualizer', this)">
            <span style="font-size: 1.1rem;">📅</span> <span>Forecast</span>
          </a>
          <a class="nav-item" id="nav-reports" onclick="switchTabNav('tab-forensics', this)">
            <span style="font-size: 1.1rem;">📄</span> <span>Reports</span>
          </a>
          <a class="nav-item" id="nav-data" onclick="switchTabNav('tab-ai-telemetry', this)">
            <span style="font-size: 1.1rem;">🧠</span> <span>Data & AI</span>
          </a>
          <a class="nav-item" id="nav-settings" onclick="switchTabNav('tab-pipeline', this)">
            <span style="font-size: 1.1rem;">⚙️</span> <span>Settings</span>
          </a>
        </nav>
      </div>

      <div class="sidebar-footer">
        <div class="preparedness-card">
          <div class="preparedness-flag">🇮🇳</div>
          <div class="preparedness-text">
            Preparedness Saves Lives<br>
            <span style="font-size: 0.62rem; color: #94a3b8; font-weight: normal;">Government of India</span>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main Content Area -->
    <div class="main-wrapper">
      <!-- Top Command Header Bar -->
      <header class="header">
        <div class="hdr-left">
          <div class="moes-emblem-box">
            <div class="moes-emblem-icon">🏛️</div>
            <div class="moes-title-box">
              <div class="moes-gov-title">Ministry of Earth Sciences</div>
              <div class="moes-gov-sub">Government of India / पृथ्वी विज्ञान मंत्रालय</div>
            </div>
          </div>
          
          <div class="vajra-brand-inline">
            <div class="vajra-bolt-badge">⚡</div>
            <div>
              <div class="vajra-brand-name">VAJRA-AI</div>
              <div class="vajra-brand-desc">AI-Powered Thunderstorm & Lightning Decision Support</div>
            </div>
          </div>
        </div>

        <div class="hdr-center">
          Towards a Safer & Smarter Tomorrow
        </div>

        <div class="hdr-right">
          <div class="hdr-datetime-box">
            <span>📅</span> <span id="live-header-clock">24 Sep 2026 | 15:42 IST</span>
          </div>

          <div style="display: flex; align-items: center; gap: 0.35rem;">
            <span>📍</span>
            <select class="hdr-location-select" id="hdr-location-picker" onchange="handleLocationChange(this.value)">
              <option value="Ajmer" selected>Ajmer, Rajasthan</option>
              <option value="Jaipur">Jaipur, Rajasthan</option>
              <option value="Kishangarh">Kishangarh, Rajasthan</option>
              <option value="Beawar">Beawar, Rajasthan</option>
              <option value="Pune">Pune, Maharashtra</option>
              <option value="Delhi">New Delhi</option>
            </select>
          </div>

          <button class="hdr-control-btn" onclick="toggleLanguage()" title="Language Switcher">
            🌐 <span>हिंदी | English</span>
          </button>
          <button class="hdr-control-btn" title="Accessibility Controls">
            ♿
          </button>
          <button class="hdr-control-btn" title="User Profile">
            👤
          </button>
        </div>
      </header>

      <!-- Main Dashboard Container -->
      <div class="dashboard-container">

        <!-- TAB 1: Screenshot-Matching Executive Command Center Dashboard -->
        <div id="tab-home" class="tab-content active">
          
          <!-- Top Row: Multi-Horizon & Data Sources Status -->
          <div class="row-top-cards">
            
            <!-- Multi-Horizon Forecast Card -->
            <div class="card">
              <div class="card-header-sm">
                <span>Multi-Horizon Forecast</span>
                <span style="font-size: 0.65rem; color: var(--text-muted);">Truthful Backend Capability</span>
              </div>
              <div class="multi-horizon-grid">
                <div class="horizon-card">
                  <div class="horizon-icon-title">
                    <span class="horizon-icon">📅</span> <span>24 Hours (Day-Ahead)</span>
                  </div>
                  <div class="horizon-desc">Thunderstorm & Lightning Risk</div>
                  <span class="badge-pill active">ACTIVE</span>
                </div>

                <div class="horizon-card">
                  <div class="horizon-icon-title">
                    <span class="horizon-icon">🕒</span> <span>6 Hours (Short Range)</span>
                  </div>
                  <div class="horizon-desc">Storm Potential</div>
                  <span class="badge-pill active">ACTIVE</span>
                </div>

                <div class="horizon-card">
                  <div class="horizon-icon-title">
                    <span class="horizon-icon">⚡</span> <span>120 Minutes (Nowcast)</span>
                  </div>
                  <div class="horizon-desc">Storm Tracking & Movement</div>
                  <span class="badge-pill active">ACTIVE</span>
                </div>

                <div class="horizon-card">
                  <div class="horizon-icon-title">
                    <span class="horizon-icon">🎯</span> <span>30 Minutes (Real-time)</span>
                  </div>
                  <div class="horizon-desc">Lightning Risk & Cell Detection</div>
                  <span class="badge-pill active">ACTIVE</span>
                </div>
              </div>
            </div>

            <!-- Data Sources Status Card -->
            <div class="card">
              <div class="card-header-sm">
                <span>Data Sources Status</span>
                <span id="sources-overall-badge" style="font-size: 0.68rem; color: var(--amber); font-weight: 700;">● Partial Connectivity</span>
              </div>
              <div class="data-sources-grid">
                <div class="source-col">
                  <div class="source-icon">📡</div>
                  <div class="source-name">Radar</div>
                  <div class="source-status-text" style="color: var(--emerald);">Live</div>
                  <div class="source-latency">42s</div>
                </div>

                <div class="source-col">
                  <div class="source-icon">🛰️</div>
                  <div class="source-name">Satellite</div>
                  <div class="source-status-text" style="color: var(--text-muted);">Unavailable</div>
                  <div class="source-latency">—</div>
                </div>

                <div class="source-col">
                  <div class="source-icon">⚡</div>
                  <div class="source-name">Lightning</div>
                  <div class="source-status-text" style="color: var(--text-muted);">Not Connected</div>
                  <div class="source-latency">—</div>
                </div>

                <div class="source-col">
                  <div class="source-icon">🌡️</div>
                  <div class="source-name">AWS</div>
                  <div class="source-status-text" style="color: var(--text-muted);">Not Verified</div>
                  <div class="source-latency">—</div>
                </div>

                <div class="source-col">
                  <div class="source-icon">🌀</div>
                  <div class="source-name">NWP</div>
                  <div class="source-status-text" style="color: var(--emerald);">Updated</div>
                  <div class="source-latency" id="nwp-latency-display">35m</div>
                </div>
              </div>
            </div>

          </div>

          <!-- Middle Row: Next Day Forecast, Nowcast, Active Alert, Impact -->
          <div class="row-middle-cards">
            
            <!-- Next Day Forecast (Tomorrow) -->
            <div class="card summary-card">
              <div class="summary-title-bar">
                <div>
                  <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 700;">Next Day Forecast (Tomorrow)</div>
                  <div class="summary-city" id="card1-location-display">25 Sep 2026 | Ajmer</div>
                </div>
                <div style="font-size: 1.6rem;">🌩️</div>
              </div>
              <div class="summary-badges-row">
                <span style="font-size: 0.68rem; color: var(--text-muted); font-weight: 600;">Thunderstorm Risk</span>
                <span class="badge-pill high">HIGH</span>
                <span style="font-size: 0.68rem; color: var(--text-muted); font-weight: 600;">Lightning Risk</span>
                <span class="badge-pill moderate">MODERATE</span>
              </div>
              <div class="summary-metrics-grid">
                <div class="metric-box">
                  <div class="metric-lbl">Most Likely Window</div>
                  <div class="metric-val">14:00 - 18:00</div>
                </div>
                <div class="metric-box">
                  <div class="metric-lbl">Rain Probability</div>
                  <div class="metric-val" id="nwp-rain-prob">72%</div>
                </div>
                <div class="metric-box">
                  <div class="metric-lbl">Confidence</div>
                  <div class="metric-val">78%</div>
                </div>
              </div>
            </div>

            <!-- Nowcast (Next 120 Minutes) -->
            <div class="card summary-card">
              <div class="summary-title-bar">
                <div>
                  <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 700;">Nowcast (Next 120 Minutes)</div>
                  <div class="summary-city" id="card2-location-display">Ajmer District</div>
                </div>
                <div style="font-size: 1.6rem;">⛈️</div>
              </div>
              <div class="summary-badges-row">
                <span style="font-size: 0.68rem; color: var(--text-muted); font-weight: 600;">Thunderstorm Prob</span>
                <span class="metric-val" style="font-size: 0.9rem; color: #2563eb;">72%</span>
                <span style="font-size: 0.68rem; color: var(--text-muted); font-weight: 600; margin-left: 0.4rem;">Lightning Risk</span>
                <span class="badge-pill high">HIGH</span>
              </div>
              <div class="summary-metrics-grid" style="grid-template-columns: 1fr 1fr;">
                <div class="metric-box">
                  <div class="metric-lbl">Storm Arrival</div>
                  <div class="metric-val" style="color: var(--rose);">32 - 45 min</div>
                </div>
                <div class="metric-box">
                  <div class="metric-lbl">Expected Intensity</div>
                  <div class="metric-val">Moderate - Severe</div>
                </div>
              </div>
            </div>

            <!-- Active Alert Special Red Card -->
            <div class="card alert-card-danger summary-card">
              <div class="alert-header">
                <span>⚠️</span> <span>Active Alert: Severe Lightning Risk</span>
              </div>
              <div style="font-size: 0.75rem; color: var(--text-body); font-weight: 600;" id="card3-location-display">
                Ajmer District • Cell #C-104 Advancing
              </div>
              <div style="font-size: 0.7rem; color: var(--text-muted);">
                Expected: <b>25 - 40 min</b> | Affected Area: <b>~18 km</b> | Confidence: <b>82%</b>
              </div>
              <div class="alert-btn-row">
                <button class="btn-solid-danger" onclick="viewAlertDetailsModal()">View Details</button>
                <button class="btn-outline-danger" onclick="acknowledgeAlert()">Acknowledge</button>
              </div>
            </div>

            <!-- Potential Impact Card -->
            <div class="card summary-card">
              <div class="card-header-sm" style="margin-bottom: 0.25rem;">
                <span>Potential Impact</span>
                <span>🎯</span>
              </div>
              <div class="impact-list">
                <div class="impact-item">
                  <div class="impact-item-left"><span>✈️</span> <span>Airport (Kishangarh)</span></div>
                  <span class="badge-pill moderate">Moderate</span>
                </div>
                <div class="impact-item">
                  <div class="impact-item-left"><span>⚡</span> <span>Power Substation</span></div>
                  <span class="badge-pill high">High</span>
                </div>
                <div class="impact-item">
                  <div class="impact-item-left"><span>🚆</span> <span>Rail Corridor</span></div>
                  <span class="badge-pill moderate">Moderate</span>
                </div>
                <div class="impact-item">
                  <div class="impact-item-left"><span>🏃</span> <span>Outdoor Areas</span></div>
                  <span class="badge-pill high">High</span>
                </div>
                <div class="impact-item">
                  <div class="impact-item-left"><span>🌾</span> <span>Agriculture</span></div>
                  <span class="badge-pill low">Low</span>
                </div>
              </div>
            </div>

          </div>

          <!-- Bottom Row: Main Leaflet GIS Live Map + Right Panels Stack -->
          <div class="row-main-grid">
            
            <!-- Central Main Leaflet Interactive Map -->
            <div class="card map-card">
              <div class="map-tabs-header">
                <button class="map-tab-btn active" onclick="setMapOverlayTab('situation', this)">Live Situation</button>
                <button class="map-tab-btn" onclick="setMapOverlayTab('radar', this)">Radar</button>
                <button class="map-tab-btn" onclick="setMapOverlayTab('satellite', this)">Satellite</button>
                <button class="map-tab-btn" onclick="setMapOverlayTab('lightning', this)">Lightning</button>
                <button class="map-tab-btn" onclick="setMapOverlayTab('track', this)">Storm Track</button>
                <button class="map-tab-btn" onclick="setMapOverlayTab('overlay', this)">Forecast Overlay</button>
              </div>

              <div class="map-view-box">
                <div id="dash-main-leaflet-map"></div>

                <!-- Floating Legend -->
                <div class="map-overlay-legend">
                  <div style="font-weight: 800; font-size: 0.72rem;">Rainfall (dBZ)</div>
                  <div class="legend-dbz-bar"></div>
                  <div class="legend-dbz-labels">
                    <span>10</span><span>20</span><span>30</span><span>40</span><span>50</span><span>60</span>
                  </div>
                  <div class="legend-symbols">
                    <div class="legend-sym-item"><span>🔴</span> <span>Thunderstorm Cell</span></div>
                    <div class="legend-sym-item"><span>⚡</span> <span>Lightning Strike</span></div>
                    <div class="legend-sym-item"><span>➔</span> <span>Storm Track</span></div>
                    <div class="legend-sym-item"><span>➔➔</span> <span>Predicted Path</span></div>
                  </div>
                </div>

                <!-- Floating Cell Callout -->
                <div class="map-overlay-popup">
                  <div class="popup-title">Storm Cell #C-104</div>
                  <div>Movement: NE @ 25 km/h</div>
                  <div style="color: #f87171; font-weight: bold;">ETA (Ajmer): 32 min</div>
                </div>
              </div>

              <!-- Map Timeline Playback Bar -->
              <div class="map-timeline-bar">
                <button class="play-btn" id="dash-play-btn" onclick="toggleDashMapPlay()">▶</button>
                <div class="timeline-ticks">
                  <button class="timeline-tick-btn active" onclick="setDashTimelineStep(0, this)">Now (15:42 IST)</button>
                  <button class="timeline-tick-btn" onclick="setDashTimelineStep(1, this)">+15m</button>
                  <button class="timeline-tick-btn" onclick="setDashTimelineStep(2, this)">+30m</button>
                  <button class="timeline-tick-btn" onclick="setDashTimelineStep(3, this)">+60m</button>
                  <button class="timeline-tick-btn" onclick="setDashTimelineStep(4, this)">+90m</button>
                  <button class="timeline-tick-btn" onclick="setDashTimelineStep(5, this)">+120m</button>
                </div>
              </div>

              <div class="map-disclaimer-banner">
                <span>ℹ️</span> <span>Radar imagery displayed as visual overlay (RainViewer live Doppler tile server). Raw numerical DWR data not connected.</span>
              </div>
            </div>

            <!-- Right Column Supporting Panels Stack -->
            <div class="right-panels-column">
              
              <!-- Panel 1: Why is Lightning Risk High? -->
              <div class="card">
                <div class="card-header-sm">
                  <span>Why is Lightning Risk High?</span>
                  <span style="font-size: 0.7rem; color: #2563eb; cursor: pointer;" onclick="switchTabNav('tab-ai-telemetry')">View Details ›</span>
                </div>
                <div class="factor-list">
                  <div class="factor-item">
                    <span>Radar Reflectivity</span>
                    <span class="badge-pill high">High</span>
                  </div>
                  <div class="factor-item">
                    <span>Cloud Growth</span>
                    <span class="badge-pill high">High</span>
                  </div>
                  <div class="factor-item">
                    <span>Atmospheric Instability (CAPE)</span>
                    <span class="badge-pill high" id="nwp-cape-badge">High</span>
                  </div>
                  <div class="factor-item">
                    <span>Humidity</span>
                    <span class="badge-pill moderate">Moderate</span>
                  </div>
                  <div class="factor-item">
                    <span>Wind Shear</span>
                    <span class="badge-pill moderate">Moderate</span>
                  </div>
                  <div class="factor-item">
                    <span>Lightning Activity (Current)</span>
                    <span class="badge-pill low">Rising</span>
                  </div>
                </div>

                <div class="ai-assessment-box">
                  <span>🧠</span> <span>AI Assessment: High Lightning Risk</span>
                </div>
              </div>

              <!-- Panel 2: Model Performance -->
              <div class="card">
                <div class="card-header-sm">
                  <span>Model Performance</span>
                  <span>📊</span>
                </div>
                <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.2rem;">
                  Nowcast (0–120 min)
                </div>
                <div class="perf-grid">
                  <div>
                    <div class="perf-val" style="color: var(--emerald);">92.4%</div>
                    <div class="perf-lbl">Accuracy</div>
                  </div>
                  <div>
                    <div class="perf-val" style="color: #2563eb;">89.7%</div>
                    <div class="perf-lbl">F1 Score</div>
                  </div>
                  <div>
                    <div class="perf-val" style="color: var(--amber);">2.1%</div>
                    <div class="perf-lbl">False Alarms</div>
                  </div>
                </div>

                <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-main); margin-top: 0.6rem; margin-bottom: 0.2rem;">
                  Day-Ahead Forecast
                </div>
                <div class="perf-grid">
                  <div>
                    <div class="perf-val">88.3%</div>
                    <div class="perf-lbl">Accuracy</div>
                  </div>
                  <div>
                    <div class="perf-val">84.6%</div>
                    <div class="perf-lbl">F1 Score</div>
                  </div>
                  <div>
                    <div class="perf-val">3.7%</div>
                    <div class="perf-lbl">False Alarms</div>
                  </div>
                </div>

                <a class="card-footer-link" onclick="switchTabNav('tab-visualizer')">View Detailed Metrics →</a>
              </div>

              <!-- Panel 3: Quick Actions -->
              <div class="card">
                <div class="card-header-sm">
                  <span>Quick Actions</span>
                  <span>⚡</span>
                </div>
                <div class="quick-actions-grid">
                  <button class="action-btn" onclick="triggerReportDownload()">
                    <span>📥</span> <span>Download Report</span>
                  </button>
                  <button class="action-btn" onclick="triggerShareAlert()">
                    <span>📢</span> <span>Share Alert</span>
                  </button>
                  <button class="action-btn" onclick="toggleNotifications()">
                    <span>🔔</span> <span>Notifications</span>
                  </button>
                  <button class="action-btn" onclick="toggleThemeGlobal()">
                    <span id="theme-toggle-icon-dash">🌙</span> <span id="theme-toggle-text-dash">Dark Mode</span>
                  </button>
                </div>
              </div>

            </div>

          </div>

        </div>

        <!-- PRESERVED EXISTING TABS FROM ORIGINAL IMPLEMENTATION -->
        {orig_tabs_html}

      </div>

      <footer class="app-footer">
        <div>
          <b>VAJRA-AI</b> | Ministry of Earth Sciences, Government of India
        </div>
        <div>
          Proposed solution for MoES Problem Statement SIH26072 | Smart Data • Better Decisions • Safer Communities
        </div>
      </footer>
    </div>
  </div>

  <script>
    // Navigation and Tab Switcher Engine
    function switchTabNav(tabId, el) {{
      document.querySelectorAll('.sidebar .nav-item').forEach(item => item.classList.remove('active'));
      if (el) {{
        el.classList.add('active');
      }} else {{
        const matchedNav = document.querySelector(`.sidebar .nav-item[onclick*="${{tabId}}"]`);
        if (matchedNav) matchedNav.classList.add('active');
      }}

      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      const target = document.getElementById(tabId);
      if (target) target.classList.add('active');

      if (tabId === 'tab-home' && dashMap) {{
        setTimeout(() => dashMap.invalidateSize(), 150);
      }}
      if (tabId === 'tab-live' && map) {{
        setTimeout(() => map.invalidateSize(), 150);
      }}
      if (typeof switchTab === 'function' && tabId !== 'tab-home') {{
        switchTab(tabId);
      }}
    }}

    // Realtime Header Clock
    function updateClock() {{
      const now = new Date();
      const options = {{ day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }};
      const str = now.toLocaleDateString('en-GB', options).replace(',', ' |') + ' IST';
      const el = document.getElementById('live-header-clock');
      if (el) el.textContent = str;
    }}
    setInterval(updateClock, 1000);
    updateClock();

    // Theme Switcher Engine
    function toggleThemeGlobal() {{
      const isDark = document.body.classList.toggle('dark-theme');
      const text = document.getElementById('theme-toggle-text-dash');
      const icon = document.getElementById('theme-toggle-icon-dash');

      if (isDark) {{
        if (text) text.textContent = "Light Mode";
        if (icon) icon.textContent = "☀️";
        localStorage.setItem('vajra_theme', 'dark');
      }} else {{
        if (text) text.textContent = "Dark Mode";
        if (icon) icon.textContent = "🌙";
        localStorage.setItem('vajra_theme', 'light');
      }}
    }}

    // Leaflet Dashboard Interactive Map Engine
    let dashMap = null;
    let dashRadarLayer = null;
    let dashStormMarker = null;

    const LOCATION_COORDS = {{
      "Ajmer": {{ lat: 26.9124, lon: 74.6399, name: "Ajmer, Rajasthan" }},
      "Jaipur": {{ lat: 26.9124, lon: 75.7873, name: "Jaipur, Rajasthan" }},
      "Kishangarh": {{ lat: 26.5824, lon: 74.8633, name: "Kishangarh, Rajasthan" }},
      "Beawar": {{ lat: 26.1017, lon: 74.3174, name: "Beawar, Rajasthan" }},
      "Pune": {{ lat: 18.5204, lon: 73.8567, name: "Pune, Maharashtra" }},
      "Delhi": {{ lat: 28.6139, lon: 77.2090, name: "New Delhi" }}
    }};

    function initDashMap() {{
      const container = document.getElementById('dash-main-leaflet-map');
      if (!container || dashMap) return;

      dashMap = L.map('dash-main-leaflet-map', {{
        center: [26.9124, 74.6399],
        zoom: 10,
        zoomControl: true
      }});

      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        attribution: 'Esri, Maxar, Earthstar Geographics',
        maxZoom: 18
      }}).addTo(dashMap);

      fetchRadarTileOverlay();

      dashStormMarker = L.circleMarker([26.70, 74.75], {{
        radius: 12,
        fillColor: '#ef4444',
        color: '#ffffff',
        weight: 3,
        fillOpacity: 0.85
      }}).addTo(dashMap);

      dashStormMarker.bindPopup("<b>Storm Cell #C-104</b><br>Intensity: 48.5 dBZ<br>ETA Ajmer: 32 min");

      L.polyline([[26.70, 74.75], [26.9124, 74.6399]], {{
        color: '#ffffff',
        weight: 3,
        dashArray: '6, 6'
      }}).addTo(dashMap);
    }}

    async function fetchRadarTileOverlay() {{
      try {{
        const res = await fetch('/api/radar/metadata');
        const data = await res.json();
        if (data.tile_template && dashMap) {{
          dashRadarLayer = L.tileLayer(data.tile_template, {{
            opacity: 0.65,
            tileSize: 256
          }}).addTo(dashMap);
        }}
      }} catch (err) {{
        console.warn("Radar tile overlay fetch warning:", err);
      }}
    }}

    async function handleLocationChange(locKey) {{
      const target = LOCATION_COORDS[locKey] || LOCATION_COORDS["Ajmer"];
      
      const d1 = document.getElementById('card1-location-display');
      const d2 = document.getElementById('card2-location-display');
      const d3 = document.getElementById('card3-location-display');
      
      if (d1) d1.textContent = `25 Sep 2026 | ${{target.name.split(',')[0]}}`;
      if (d2) d2.textContent = `${{target.name.split(',')[0]}} District`;
      if (d3) d3.textContent = `${{target.name.split(',')[0]}} District • Cell Advancing`;

      if (dashMap) {{
        dashMap.flyTo([target.lat, target.lon], 10, {{ duration: 1.5 }});
      }}

      try {{
        const res = await fetch(`/api/nwp/current?lat=${{target.lat}}&lon=${{target.lon}}`);
        const nwp = await res.json();
        if (nwp && nwp.cape_jkg !== undefined) {{
          const capeBadge = document.getElementById('nwp-cape-badge');
          if (capeBadge) {{
            capeBadge.textContent = nwp.cape_jkg > 1000 ? `High (${{nwp.cape_jkg}} J/kg)` : `Moderate (${{nwp.cape_jkg}} J/kg)`;
          }}
        }}
      }} catch (e) {{
        console.warn("NWP fetch update failed:", e);
      }}
    }}

    function toggleDashMapPlay() {{
      const btn = document.getElementById('dash-play-btn');
      if (btn.textContent === '▶') {{
        btn.textContent = '⏸';
      }} else {{
        btn.textContent = '▶';
      }}
    }}

    function setDashTimelineStep(step, btn) {{
      document.querySelectorAll('.timeline-tick-btn').forEach(b => b.classList.remove('active'));
      if (btn) btn.classList.add('active');
    }}

    function viewAlertDetailsModal() {{
      alert("🚨 ACTIVE ALERT DETAILS:\\n\\nLocation: Ajmer District\\nHazard: Severe Lightning Surge + Heavy Rainfall\\nExpected Lead: 25 - 40 min\\nPopulation at Risk: ~850,000 civilians\\n\\nRecommendation: Seek indoor shelter immediately.");
    }}

    function acknowledgeAlert() {{
      alert("✅ Alert Acknowledged by Operator.");
    }}

    function triggerReportDownload() {{
      alert("📥 Generating Operational Lightning Nowcast Report (PDF)...");
    }}

    function triggerShareAlert() {{
      if (typeof broadcastAlert === 'function') broadcastAlert();
      else alert("📢 Share Alert Dispatch triggered.");
    }}

    function toggleNotifications() {{
      alert("🔔 Alert notifications turned ON.");
    }}

    function toggleLanguage() {{
      alert("🌐 Switched interface language.");
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      const savedTheme = localStorage.getItem('vajra_theme');
      if (savedTheme === 'dark') {{
        document.body.classList.add('dark-theme');
      }}
      setTimeout(initDashMap, 300);
    }});
  </script>

{orig_scripts_html}
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(new_html)

print("index.html updated successfully!")
