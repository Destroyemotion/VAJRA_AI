"""
Script to build a clean, unified, perfectly-styled index.html for VAJRA-AI.
Fixes HTML structure, duplicate heads/bodies, and CSS conflicts.
"""
import os

with open(r'backup_before_deployment_prep/index.html', 'r', encoding='utf-8') as f:
    orig_html = f.read()

# Extract from TAB 0 to end of main in orig_html
tab_start_idx = orig_html.find('<!-- TAB 0: INTERACTIVE INDIA SATELLITE & LIGHTNING MAP -->')
main_end_idx = orig_html.find('</main>')
original_tabs_content = orig_html[tab_start_idx:main_end_idx]

# Extract original JS scripts
script_start_idx = orig_html.find('<script>')
original_scripts = orig_html[script_start_idx:]

clean_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚡ VAJRA-AI // Atmospheric Nowcasting | Ministry of Earth Sciences & ISRO</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  
  <!-- Leaflet GIS Map Engine -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    :root {
      --bg-dark: #070b14;
      --bg-sidebar: #091224;
      --bg-card: #0f192e;
      --bg-card-hover: #162442;
      --bg-header: #0b1528;
      --border-color: #1e2d4a;
      --border-accent: #2e436e;
      
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;

      --cyan: #00f2fe;
      --blue: #2563eb;
      --blue-light: #38bdf8;
      --violet: #7928ca;
      --magenta: #ff007f;
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --danger: #ef4444;

      --font-main: 'Inter', system-ui, -apple-system, sans-serif;
      --font-code: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: var(--font-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Top Main Header */
    header {
      background: var(--bg-header);
      border-bottom: 1px solid var(--border-color);
      padding: 0.65rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 1000;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 1.25rem;
    }

    .emblem-box {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      padding-right: 1.25rem;
      border-right: 1px solid var(--border-color);
    }

    .emblem-text {
      font-size: 0.68rem;
      color: var(--text-muted);
      line-height: 1.25;
      font-weight: 500;
    }

    .emblem-text span {
      display: block;
      color: var(--text-main);
      font-weight: 700;
      font-size: 0.72rem;
    }

    .brand-logo-area {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand-shield {
      width: 38px;
      height: 38px;
      border-radius: 9px;
      background: linear-gradient(135deg, #0284c7, #2563eb);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.35rem;
      color: #ffffff;
      box-shadow: 0 0 15px rgba(37, 99, 235, 0.4);
    }

    .brand-title-group {
      display: flex;
      flex-direction: column;
    }

    .brand-title {
      font-size: 1.35rem;
      font-weight: 900;
      letter-spacing: 0.02em;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .brand-subtitle {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 500;
    }

    .header-center-tagline {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--blue-light);
      font-style: italic;
      letter-spacing: 0.02em;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .header-widget {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.05);
      padding: 0.4rem 0.8rem;
      border-radius: 7px;
      border: 1px solid var(--border-color);
    }

    .header-widget-icon {
      color: var(--cyan);
    }

    .lang-toggle {
      display: flex;
      align-items: center;
      gap: 0.3rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
    }

    .lang-toggle span.active {
      color: var(--text-main);
      font-weight: 700;
    }

    .user-avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: linear-gradient(135deg, #1e3a8a, #0284c7);
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      border: 1px solid var(--border-accent);
    }

    /* Main App Layout Body */
    .app-body {
      display: flex;
      flex: 1;
      width: 100vw;
      overflow: hidden;
    }

    /* Left Navigation Sidebar */
    aside.sidebar {
      width: 220px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 1rem 0;
      flex-shrink: 0;
      z-index: 100;
    }

    .nav-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      padding: 0 0.75rem;
    }

    .nav-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.7rem 0.9rem;
      border-radius: 8px;
      color: var(--text-muted);
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
    }

    .nav-item-left {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .nav-item-icon {
      font-size: 1.1rem;
      width: 20px;
      text-align: center;
    }

    .nav-item:hover {
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-main);
    }

    .nav-item.active {
      background: linear-gradient(135deg, #1d4ed8, #2563eb);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    .nav-badge {
      background: var(--rose);
      color: #ffffff;
      font-size: 0.7rem;
      font-weight: 800;
      padding: 0.15rem 0.45rem;
      border-radius: 9999px;
      font-family: var(--font-code);
    }

    .sidebar-footer {
      padding: 1rem 1.25rem 0.5rem;
      border-top: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .preparedness-tag {
      font-size: 0.72rem;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.03em;
    }

    .tricolor-bar {
      height: 4px;
      width: 100%;
      border-radius: 2px;
      background: linear-gradient(to right, #ff9933 33%, #ffffff 33%, #ffffff 66%, #138808 66%);
    }

    /* Content Area */
    .content-area {
      flex: 1;
      overflow-y: auto;
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      background: var(--bg-dark);
    }

    /* Page View / Tab Containers */
    .page-view, .tab-content {
      display: none;
      flex-direction: column;
      gap: 1.25rem;
      width: 100%;
    }

    .page-view.active, .tab-content.active {
      display: flex;
    }

    /* Cards & Grids */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }

    .card-title-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-title {
      font-size: 0.95rem;
      font-weight: 800;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Top Dashboard Row */
    .top-dashboard-row {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.25rem;
    }

    .horizon-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.85rem;
    }

    .horizon-box {
      background: rgba(7, 11, 20, 0.6);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .horizon-box:hover, .horizon-box.active {
      border-color: var(--blue-light);
      background: rgba(14, 165, 233, 0.08);
    }

    .horizon-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .horizon-icon { font-size: 1.2rem; }
    .horizon-time { font-size: 0.85rem; font-weight: 800; color: #ffffff; }
    .horizon-desc { font-size: 0.72rem; color: var(--text-muted); }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.5rem;
      border-radius: 9999px;
      font-size: 0.65rem;
      font-weight: 800;
      text-transform: uppercase;
      width: fit-content;
      font-family: var(--font-code);
    }

    .status-badge.active-green {
      background: rgba(16, 185, 129, 0.15);
      color: var(--emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Data Sources Grid */
    .data-sources-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 0.4rem;
      text-align: center;
    }

    .data-source-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.25rem;
    }

    .ds-icon { font-size: 1.25rem; }
    .ds-name { font-size: 0.7rem; font-weight: 700; color: var(--text-main); }
    .ds-status { font-size: 0.65rem; color: var(--emerald); font-weight: 600; }
    .ds-time { font-size: 0.65rem; color: var(--text-dim); font-family: var(--font-code); }

    /* Middle Layout Grid */
    .dashboard-main-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.25rem;
    }

    .left-column { display: flex; flex-direction: column; gap: 1.25rem; }
    .right-column { display: flex; flex-direction: column; gap: 1.25rem; }

    .forecast-cards-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }

    .risk-badge {
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      font-size: 0.7rem;
      font-weight: 800;
      text-transform: uppercase;
    }

    .risk-badge.high { background: rgba(239, 68, 68, 0.2); color: var(--rose); border: 1px solid rgba(239, 68, 68, 0.4); }
    .risk-badge.moderate { background: rgba(245, 158, 11, 0.2); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.4); }
    .risk-badge.low { background: rgba(16, 185, 129, 0.2); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.4); }

    .forecast-metrics-list {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
      margin-top: 0.25rem;
    }

    .fm-item { display: flex; flex-direction: column; gap: 0.2rem; }
    .fm-label { font-size: 0.68rem; color: var(--text-muted); display: flex; align-items: center; gap: 0.3rem; }
    .fm-val { font-size: 0.95rem; font-weight: 800; color: #ffffff; font-family: var(--font-code); }

    /* Map Card */
    .map-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }

    .map-tabs-header {
      background: rgba(7, 11, 20, 0.8);
      border-bottom: 1px solid var(--border-color);
      padding: 0.5rem 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .map-tab-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.4rem 0.85rem;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.15s ease;
    }

    .map-tab-btn:hover { color: var(--text-main); background: rgba(255, 255, 255, 0.05); }
    .map-tab-btn.active { background: var(--blue); color: #ffffff; border-color: var(--blue-light); font-weight: 700; }

    @keyframes map-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    .map-viewport-container {
      position: relative;
      width: 100%;
      height: 480px;
      background: #050810;
    }

    #home-leaflet-map { width: 100%; height: 100%; z-index: 1; }
    #leaflet-map { width: 100%; height: 600px; border-radius: 12px; border: 1px solid var(--border-accent); background: #050810; z-index: 1; }

    .map-storm-callout {
      position: absolute;
      top: 1rem;
      right: 1rem;
      background: rgba(9, 18, 36, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-accent);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      z-index: 500;
      font-size: 0.75rem;
      color: #ffffff;
      box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }

    .map-legend-overlay {
      position: absolute;
      top: 1rem;
      left: 1rem;
      background: rgba(9, 18, 36, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-accent);
      border-radius: 8px;
      padding: 0.6rem 0.85rem;
      z-index: 500;
      font-size: 0.7rem;
    }

    .dbz-bar-gradient {
      height: 8px;
      width: 160px;
      border-radius: 4px;
      background: linear-gradient(to right, #00ffff, #00ff00, #ffff00, #ff0000, #ff00ff);
      margin: 0.3rem 0;
    }

    .map-scrubber-bar {
      background: rgba(7, 11, 20, 0.9);
      border-top: 1px solid var(--border-color);
      padding: 0.65rem 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    .scrubber-play-btn {
      background: var(--blue);
      border: none;
      color: #fff;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 0.9rem;
    }

    .scrubber-timeline {
      display: flex;
      align-items: center;
      gap: 1.5rem;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      font-family: var(--font-code);
      flex: 1;
    }

    .scrubber-step.active { color: var(--cyan); border-bottom: 2px solid var(--cyan); }

    /* Active Alert Card */
    .active-alert-card { background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.4); }
    .alert-header { display: flex; align-items: flex-start; gap: 0.75rem; }
    .alert-icon-box { font-size: 1.5rem; color: var(--rose); }
    .alert-details-list { font-size: 0.75rem; color: var(--text-muted); display: flex; flex-direction: column; gap: 0.25rem; margin-top: 0.35rem; }
    .alert-actions-row { display: flex; gap: 0.75rem; margin-top: 0.5rem; }
    .btn-alert-red { background: linear-gradient(135deg, #ef4444, #dc2626); color: #ffffff; border: none; font-weight: 700; padding: 0.45rem 0.9rem; border-radius: 6px; font-size: 0.78rem; cursor: pointer; flex: 1; }
    .btn-alert-outline { background: transparent; color: var(--text-main); border: 1px solid var(--border-accent); font-weight: 600; padding: 0.45rem 0.9rem; border-radius: 6px; font-size: 0.78rem; cursor: pointer; flex: 1; }

    /* Potential Impact */
    .impact-list { display: flex; flex-direction: column; gap: 0.6rem; }
    .impact-item { display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; }
    .impact-name { display: flex; align-items: center; gap: 0.5rem; color: var(--text-main); }

    /* Risk Factors */
    .risk-factors-list { display: flex; flex-direction: column; gap: 0.45rem; font-size: 0.78rem; }
    .rf-item { display: flex; align-items: center; justify-content: space-between; }
    .rf-bar { height: 5px; width: 70px; border-radius: 3px; background: #1e293b; overflow: hidden; }
    .rf-bar-fill { height: 100%; }

    /* Quick Actions */
    .quick-actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
    .qa-btn, .btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 0.6rem 0.75rem;
      color: var(--text-main);
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease;
    }
    .qa-btn:hover, .btn:hover { background: rgba(255, 255, 255, 0.08); border-color: var(--blue-light); color: var(--cyan); }
    .btn-primary { background: linear-gradient(135deg, var(--cyan), var(--blue)); color: #000; border: none; font-weight: 700; }
    .btn-warning { background: linear-gradient(135deg, var(--amber), var(--rose)); color: #fff; border: none; font-weight: 700; }

    /* Controls Panel */
    .controls-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1rem 1.25rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }
    .control-group { display: flex; align-items: center; gap: 0.6rem; }
    .control-label { font-size: 0.75rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; }
    select, input[type="text"], input[type="number"] {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border-accent);
      color: var(--text-main);
      padding: 0.45rem 0.75rem;
      border-radius: 6px;
      font-family: var(--font-main);
      font-size: 0.8rem;
      outline: none;
    }

    .search-wrapper { position: relative; width: 320px; }
    .search-input { width: 100%; padding-left: 2.2rem; }
    .search-icon { position: absolute; left: 0.75rem; top: 50%; transform: translateY(-50%); color: var(--text-muted); }
    .search-dropdown { position: absolute; top: 100%; left: 0; right: 0; background: #0f172a; border: 1px solid var(--border-accent); border-radius: 8px; max-height: 220px; overflow-y: auto; z-index: 1000; display: none; }
    .search-item { padding: 0.5rem 0.75rem; font-size: 0.78rem; cursor: pointer; border-bottom: 1px solid var(--border-color); }
    .search-item:hover { background: rgba(0, 242, 254, 0.1); color: var(--cyan); }

    .dataset-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; }
    .dataset-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem; }
    .metrics-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1rem; }
    .metric-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; display: flex; flex-direction: column; gap: 0.25rem; }
    .metric-val { font-size: 1.4rem; font-weight: 800; font-family: var(--font-code); color: var(--cyan); }
    .metric-label { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; }
    .nowcast-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }

    /* Footer */
    footer.app-footer {
      background: var(--bg-header);
      border-top: 1px solid var(--border-color);
      padding: 0.75rem 1.5rem;
      font-size: 0.72rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Comprehensive Light Theme Overrides */
    body.light-theme {
      --bg-dark: #f8fafc;
      --bg-sidebar: #ffffff;
      --bg-card: #ffffff;
      --bg-card-hover: #f1f5f9;
      --bg-header: #ffffff;
      --border-color: #e2e8f0;
      --border-accent: #cbd5e1;
      
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;

      background-color: #f8fafc !important;
      color: #0f172a !important;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(2, 132, 199, 0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(121, 40, 202, 0.04) 0%, transparent 40%) !important;
    }

    body.light-theme header {
      background: #ffffff !important;
      border-bottom-color: #e2e8f0 !important;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05) !important;
    }

    body.light-theme .brand-title,
    body.light-theme .app-logo-text {
      color: #0f172a !important;
    }

    body.light-theme .app-sidebar {
      background: #ffffff !important;
      border-right-color: #e2e8f0 !important;
    }

    body.light-theme .nav-item {
      color: #475569 !important;
    }

    body.light-theme .nav-item:hover {
      background: #f1f5f9 !important;
      color: #0284c7 !important;
    }

    body.light-theme .nav-item.active {
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(0, 242, 254, 0.08)) !important;
      color: #2563eb !important;
      border-left-color: #2563eb !important;
    }

    body.light-theme .card,
    body.light-theme .horizon-box,
    body.light-theme .status-card,
    body.light-theme .metric-card,
    body.light-theme .impact-card,
    body.light-theme .map-card,
    body.light-theme .explain-card,
    body.light-theme .perf-card,
    body.light-theme .forensic-box,
    body.light-theme .dataset-card,
    body.light-theme .pipeline-step,
    body.light-theme .controls-panel,
    body.light-theme .header-widget {
      background: #ffffff !important;
      border-color: #e2e8f0 !important;
      color: #0f172a !important;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
    }

    body.light-theme .card:hover,
    body.light-theme .horizon-box:hover,
    body.light-theme .dataset-card:hover {
      border-color: #cbd5e1 !important;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08) !important;
    }

    body.light-theme .horizon-time,
    body.light-theme .card-title,
    body.light-theme .fm-val,
    body.light-theme .impact-name,
    body.light-theme .metric-val,
    body.light-theme .explain-item-title,
    body.light-theme .perf-val,
    body.light-theme .qa-btn {
      color: #0f172a !important;
    }

    body.light-theme .horizon-risk,
    body.light-theme .card-subtitle,
    body.light-theme .fm-label,
    body.light-theme .impact-desc,
    body.light-theme .metric-label,
    body.light-theme .explain-item-desc,
    body.light-theme .perf-label,
    body.light-theme .status-time,
    body.light-theme .emblem-sub {
      color: #64748b !important;
    }

    body.light-theme .qa-btn {
      background: #f8fafc !important;
      border-color: #cbd5e1 !important;
    }

    body.light-theme .qa-btn:hover {
      background: #f1f5f9 !important;
      border-color: #0284c7 !important;
      color: #0284c7 !important;
    }

    body.light-theme .tab-btn {
      background: #ffffff !important;
      color: #334155 !important;
      border: 1px solid #cbd5e1 !important;
    }

    body.light-theme .tab-btn:hover {
      background: #f1f5f9 !important;
      color: #0284c7 !important;
    }

    body.light-theme .tab-btn.active {
      background: #2563eb !important;
      color: #ffffff !important;
      border-color: #2563eb !important;
    }

    body.light-theme .btn,
    body.light-theme .btn-sm,
    body.light-theme .btn-outline {
      background: #ffffff !important;
      color: #1e293b !important;
      border: 1px solid #cbd5e1 !important;
    }

    body.light-theme .btn:hover,
    body.light-theme .btn-sm:hover,
    body.light-theme .btn-outline:hover {
      background: #f0f9ff !important;
      color: #0284c7 !important;
      border-color: #0284c7 !important;
    }

    body.light-theme .btn-primary {
      background: linear-gradient(135deg, #0284c7, #2563eb) !important;
      color: #ffffff !important;
      border: none !important;
    }

    body.light-theme table th {
      background: #f1f5f9 !important;
      color: #475569 !important;
      border-bottom: 1px solid #cbd5e1 !important;
    }

    body.light-theme table td {
      border-bottom: 1px solid #e2e8f0 !important;
      color: #1e293b !important;
    }

    body.light-theme pre,
    body.light-theme code,
    body.light-theme input,
    body.light-theme select {
      background: #ffffff !important;
      color: #0f172a !important;
      border: 1px solid #cbd5e1 !important;
    }

    body.light-theme .modal-box {
      background: #ffffff !important;
      border-color: #cbd5e1 !important;
      color: #0f172a !important;
      box-shadow: 0 20px 60px rgba(15, 23, 42, 0.15) !important;
    }

    body.light-theme .leaflet-container {
      background: #e2e8f0 !important;
    }

    body.light-theme .map-legend-overlay,
    body.light-theme .map-storm-callout {
      background: rgba(255, 255, 255, 0.92) !important;
      color: #0f172a !important;
      border-color: #cbd5e1 !important;
      backdrop-filter: blur(8px);
    }

    body.light-theme .app-footer {
      background: #ffffff !important;
      border-top-color: #e2e8f0 !important;
      color: #64748b !important;
    }


    /* Modal Backdrop */
    .modal-backdrop {
      position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(5px);
      z-index: 2000; display: none; align-items: center; justify-content: center;
    }
    .modal-box {
      background: var(--bg-card); border: 1px solid var(--border-accent); border-radius: 14px;
      width: 520px; max-width: 90vw; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem;
      box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    }
  </style>
</head>
<body>

  <!-- Top Main Header -->
  <header>
    <div class="header-left">
      <!-- Ministry Emblem -->
      <div class="emblem-box">
        <div style="font-size: 1.5rem;">🏛️</div>
        <div class="emblem-text">
          <span>Ministry of Earth Sciences</span>
          Government of India • पृथ्वी विज्ञान मंत्रालय
        </div>
      </div>

      <!-- VAJRA-AI Logo -->
      <div class="brand-logo-area">
        <div class="brand-shield">⚡</div>
        <div class="brand-title-group">
          <div class="brand-title">VAJRA-AI</div>
          <div class="brand-subtitle">AI-Powered Thunderstorm & Lightning Decision Support</div>
        </div>
      </div>
    </div>

    <div class="header-center-tagline">
      <span>Towards a Safer & Smarter Tomorrow</span>
    </div>

    <div class="header-right">
      <!-- Live Time & Date -->
      <div class="header-widget">
        <span class="header-widget-icon">🕒</span>
        <span id="header-clock">24 Sep 2026 | 15:42 IST</span>
      </div>

      <!-- Selected Location -->
      <div class="header-widget" style="cursor: pointer;" onclick="promptLocationChange()">
        <span class="header-widget-icon">📍</span>
        <span id="header-location-name">Ajmer, Rajasthan</span>
      </div>

      <!-- Language Switcher -->
      <div class="lang-toggle">
        <span>🌐</span>
        <span>हिन्दी</span> | <span class="active">English</span>
      </div>

      <!-- Verification Tests Button -->
      <button class="btn" style="background: rgba(0, 242, 254, 0.1); color: var(--cyan); border: 1px solid rgba(0, 242, 254, 0.4);" onclick="runVerificationSuite()">
        🧪 Run Verification Tests
      </button>

      <!-- Theme Switcher -->
      <button id="theme-toggle-btn" onclick="toggleTheme()" style="background: rgba(255,255,255,0.08); border: 1px solid var(--border-color); color: var(--text-main); padding: 0.4rem 0.75rem; border-radius: 6px; font-weight: 600; font-size: 0.75rem; cursor: pointer;">
        ☀️ Light Mode
      </button>

      <!-- User Avatar -->
      <div class="user-avatar" title="User Profile">👤</div>
    </div>
  </header>

  <!-- Main Body Layout -->
  <div class="app-body">
    
    <!-- Left Navigation Sidebar -->
    <aside class="sidebar">
      <ul class="nav-list">
        <li class="nav-item active" id="nav-tab-home" onclick="switchTab('tab-home')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🏠</span>
            <span>Home</span>
          </div>
        </li>
        <li class="nav-item" id="nav-tab-india-map" onclick="switchTab('tab-india-map')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🛰️</span>
            <span>Live Map</span>
          </div>
        </li>
        <li class="nav-item" id="nav-tab-dispatch" onclick="switchTab('tab-dispatch')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🚨</span>
            <span>Alerts</span>
          </div>
          <span class="nav-badge">3</span>
        </li>
        <li class="nav-item" id="nav-tab-nowcast" onclick="switchTab('tab-nowcast')">
          <div class="nav-item-left">
            <span class="nav-item-icon">📡</span>
            <span>Forecast</span>
          </div>
        </li>
        <li class="nav-item" id="nav-tab-forensics" onclick="switchTab('tab-forensics')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🔍</span>
            <span>Reports</span>
          </div>
        </li>
        <li class="nav-item" id="nav-tab-telemetry" onclick="switchTab('tab-telemetry')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🧠</span>
            <span>Data & AI</span>
          </div>
        </li>
        <li class="nav-item" id="nav-tab-datacatalog" onclick="switchTab('tab-datacatalog')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🌐</span>
            <span>Data Sources</span>
          </div>
        </li>
        <li class="nav-item" id="nav-tab-pipeline" onclick="switchTab('tab-pipeline')">
          <div class="nav-item-left">
            <span class="nav-item-icon">⚙️</span>
            <span>Settings</span>
          </div>
        </li>
      </ul>

      <div class="sidebar-footer">
        <div class="preparedness-tag">Preparedness Saves Lives</div>
        <div class="tricolor-bar"></div>
      </div>
    </aside>

    <!-- Main Content Container -->
    <main class="content-area">

      <!-- TAB 0: HOME DASHBOARD (IMAGE 1 TARGET DESIGN) -->
      <div id="tab-home" class="tab-content active">
        
        <!-- Top Row Cards -->
        <div class="top-dashboard-row">
          <!-- Multi-Horizon Forecast Card -->
          <div class="card">
            <div class="card-title-header">
              <div class="card-title">⚡ Multi-Horizon Forecast</div>
            </div>
            
            <div class="horizon-grid">
              <div class="horizon-box active" onclick="switchTab('tab-nowcast')">
                <div class="horizon-header">
                  <span class="horizon-icon">📅</span>
                  <div>
                    <div class="horizon-time">24 Hours</div>
                    <div class="horizon-desc">(Day-Ahead)</div>
                  </div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem;">Thunderstorm & Lightning Risk</div>
                <span class="status-badge active-green" style="margin-top: 0.3rem;">● ACTIVE</span>
              </div>

              <div class="horizon-box" onclick="switchTab('tab-nowcast')">
                <div class="horizon-header">
                  <span class="horizon-icon">⏱️</span>
                  <div>
                    <div class="horizon-time">6 Hours</div>
                    <div class="horizon-desc">(Short Range)</div>
                  </div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem;">Storm Potential</div>
                <span class="status-badge active-green" style="margin-top: 0.3rem;">● ACTIVE</span>
              </div>

              <div class="horizon-box" onclick="switchTab('tab-nowcast')">
                <div class="horizon-header">
                  <span class="horizon-icon">⚡</span>
                  <div>
                    <div class="horizon-time">120 Minutes</div>
                    <div class="horizon-desc">(Nowcast)</div>
                  </div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem;">Storm Tracking & Movement</div>
                <span class="status-badge active-green" style="margin-top: 0.3rem;">● ACTIVE</span>
              </div>

              <div class="horizon-box" onclick="switchTab('tab-nowcast')">
                <div class="horizon-header">
                  <span class="horizon-icon">🎯</span>
                  <div>
                    <div class="horizon-time">30 Minutes</div>
                    <div class="horizon-desc">(Real-time)</div>
                  </div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem;">Lightning Risk & New Cell Detection</div>
                <span class="status-badge active-green" style="margin-top: 0.3rem;">● ACTIVE</span>
              </div>
            </div>
          </div>

          <!-- Data Sources Status Card -->
          <div class="card">
            <div class="card-title-header">
              <div class="card-title">🌐 Data Sources Status</div>
              <span id="ds-main-status-tag" style="font-size: 0.7rem; color: var(--emerald); font-weight: 700;">● All Systems Online</span>
            </div>

            <div class="data-sources-grid">
              <div class="data-source-item" style="cursor: pointer;" onclick="switchTab('tab-datacatalog')">
                <span class="ds-icon">📡</span>
                <span class="ds-name">Radar</span>
                <span class="ds-status" id="ds-radar-status">Live</span>
                <span class="ds-time" id="ds-radar-time">42s</span>
              </div>
              <div class="data-source-item" style="cursor: pointer;" onclick="switchTab('tab-datacatalog')">
                <span class="ds-icon">🛰️</span>
                <span class="ds-name">Satellite</span>
                <span class="ds-status" id="ds-sat-status">Live</span>
                <span class="ds-time" id="ds-sat-time">1.8m</span>
              </div>
              <div class="data-source-item" style="cursor: pointer;" onclick="switchTab('tab-datacatalog')">
                <span class="ds-icon">⚡</span>
                <span class="ds-name">Lightning</span>
                <span class="ds-status" id="ds-ltg-status">Live</span>
                <span class="ds-time" id="ds-ltg-time">3m</span>
              </div>
              <div class="data-source-item" style="cursor: pointer;" onclick="switchTab('tab-datacatalog')">
                <span class="ds-icon">🎯</span>
                <span class="ds-name">AWS</span>
                <span class="ds-status" id="ds-aws-status">Live</span>
                <span class="ds-time" id="ds-aws-time">58s</span>
              </div>
              <div class="data-source-item" style="cursor: pointer;" onclick="switchTab('tab-datacatalog')">
                <span class="ds-icon">🌀</span>
                <span class="ds-name">NWP</span>
                <span class="ds-status" id="ds-nwp-status">Updated</span>
                <span class="ds-time" id="ds-nwp-time">35m</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Main Middle Grid -->
        <div class="dashboard-main-grid">
          
          <!-- Left Column -->
          <div class="left-column">
            
            <!-- Forecast Cards Row -->
            <div class="forecast-cards-row">
              <!-- Next Day Forecast -->
              <div class="card">
                <div class="card-title-header">
                  <div class="card-title">🌩️ Next Day Forecast (Tomorrow)</div>
                </div>
                <div style="font-size: 0.75rem; color: var(--text-dim);" id="next-day-date">25 Sep 2026 | Ajmer</div>

                <div style="display: flex; gap: 1.5rem; align-items: center; margin: 0.2rem 0;">
                  <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.8rem;">🌧️</span>
                    <div>
                      <div style="font-size: 0.72rem; color: var(--text-muted);">Thunderstorm Risk</div>
                      <span class="risk-badge high">HIGH</span>
                    </div>
                  </div>

                  <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.8rem;">⚡</span>
                    <div>
                      <div style="font-size: 0.72rem; color: var(--text-muted);">Lightning Risk</div>
                      <span class="risk-badge moderate">MODERATE</span>
                    </div>
                  </div>
                </div>

                <div class="forecast-metrics-list">
                  <div class="fm-item">
                    <span class="fm-label">🕒 Most Likely Window</span>
                    <span class="fm-val" style="font-size: 0.85rem;" id="next-day-window">14:00 – 18:00</span>
                  </div>
                  <div class="fm-item">
                    <span class="fm-label">💧 Rain Probability</span>
                    <span class="fm-val" style="color: var(--blue-light);" id="next-day-rain">72%</span>
                  </div>
                  <div class="fm-item">
                    <span class="fm-label">🎯 Confidence</span>
                    <span class="fm-val" style="color: var(--emerald);" id="next-day-conf">78%</span>
                  </div>
                </div>
              </div>

              <!-- Nowcast (Next 120 Minutes) -->
              <div class="card">
                <div class="card-title-header">
                  <div class="card-title">⚡ Nowcast (Next 120 Minutes)</div>
                </div>
                <div style="font-size: 0.75rem; color: var(--text-dim);">Ajmer District</div>

                <div style="display: flex; gap: 1.5rem; align-items: center; margin: 0.2rem 0;">
                  <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.8rem;">🌩️</span>
                    <div>
                      <div style="font-size: 0.72rem; color: var(--text-muted);">Thunderstorm Probability</div>
                      <div style="font-size: 1.25rem; font-weight: 900; color: var(--blue-light); font-family: var(--font-code);" id="nowcast-prob">72%</div>
                    </div>
                  </div>

                  <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.8rem;">⚡</span>
                    <div>
                      <div style="font-size: 0.72rem; color: var(--text-muted);">Lightning Risk</div>
                      <span class="risk-badge high">HIGH</span>
                    </div>
                  </div>
                </div>

                <div class="forecast-metrics-list">
                  <div class="fm-item">
                    <span class="fm-label">⏱️ Storm Arrival</span>
                    <span class="fm-val" style="font-size: 0.85rem;" id="nowcast-eta">32 – 45 min</span>
                  </div>
                  <div class="fm-item" style="grid-column: span 2;">
                    <span class="fm-label">📊 Expected Intensity</span>
                    <span class="fm-val" style="font-size: 0.85rem; color: var(--rose);" id="nowcast-intensity">Moderate – Severe</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Main Live Situation Map Container -->
            <div class="map-card">
              <!-- Map Controls Header Bar -->
              <div class="map-tabs-header">
                <button class="map-tab-btn active" onclick="switchHomeMapLayer('live')">🛰️ Live Situation</button>
                <button class="map-tab-btn" onclick="switchHomeMapLayer('radar')">🌧️ Radar</button>
                <button class="map-tab-btn" onclick="switchHomeMapLayer('satellite')">🌍 Satellite</button>
                <button class="map-tab-btn" onclick="switchHomeMapLayer('lightning')">⚡ Lightning</button>
                <button class="map-tab-btn" onclick="switchHomeMapLayer('track')">🏹 Storm Track</button>
                <button class="map-tab-btn" onclick="switchHomeMapLayer('forecast')">🔮 Forecast Overlay</button>
              </div>

              <!-- Map Viewport -->
              <div class="map-viewport-container">
                <div id="home-map-loader" style="position: absolute; top:0; left:0; width:100%; height:100%; background:#050810; display:flex; align-items:center; justify-content:center; color:var(--text-muted); font-size:0.85rem; z-index:2; gap:0.5rem;">
                  <span class="spinner" style="display:inline-block; width:18px; height:18px; border:2px solid var(--cyan); border-top-color:transparent; border-radius:50%; animation: map-spin 1s linear infinite;"></span>
                  Loading GIS Satellite & Radar Map...
                </div>
                <div id="home-leaflet-map"></div>

                <!-- Floating Storm Cell Callout Widget -->
                <div class="map-storm-callout">
                  <div style="font-weight: 800; color: var(--cyan); margin-bottom: 0.2rem;">Storm Cell #C-104</div>
                  <div style="color: var(--text-muted); font-size: 0.7rem;">Movement: NE @ 25 km/h</div>
                  <div style="color: var(--rose); font-weight: 700; font-size: 0.72rem; margin-top: 0.2rem;">ETA (Ajmer): 32 min</div>
                </div>

                <!-- dBZ Legend Overlay -->
                <div class="map-legend-overlay">
                  <div style="font-weight: 700; color: var(--text-main); font-size: 0.68rem;">Rainfall (dBZ)</div>
                  <div class="dbz-bar-gradient"></div>
                  <div style="display: flex; justify-content: space-between; color: var(--text-dim); font-size: 0.62rem; font-family: var(--font-code);">
                    <span>10</span><span>20</span><span>30</span><span>40</span><span>50</span><span>60</span>
                  </div>
                </div>
              </div>

              <!-- Map Scrubber Footer -->
              <div class="map-scrubber-bar">
                <button class="scrubber-play-btn" id="home-play-btn" onclick="toggleHomeTimelinePlay()">▶</button>
                <div class="scrubber-timeline">
                  <span class="scrubber-step active" onclick="setHomeTimelineStep(0)">Now 15:42 IST</span>
                  <span class="scrubber-step" onclick="setHomeTimelineStep(1)">+15m</span>
                  <span class="scrubber-step" onclick="setHomeTimelineStep(2)">+30m</span>
                  <span class="scrubber-step" onclick="setHomeTimelineStep(3)">+60m</span>
                  <span class="scrubber-step" onclick="setHomeTimelineStep(4)">+90m</span>
                  <span class="scrubber-step" onclick="setHomeTimelineStep(5)">+120m</span>
                </div>
                <button style="background: rgba(255,255,255,0.08); border: 1px solid var(--border-color); color: #fff; padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.72rem; cursor: pointer;" onclick="toggleMapLayersModal()">🥞 Layers</button>
              </div>
            </div>

          </div>

          <!-- Right Column -->
          <div class="right-column">
            
            <!-- Active Alert Card -->
            <div class="card active-alert-card">
              <div class="alert-header">
                <div class="alert-icon-box">⚠️</div>
                <div>
                  <div style="font-size: 0.7rem; color: var(--rose); font-weight: 800; text-transform: uppercase;">Active Alert</div>
                  <div style="font-size: 1rem; font-weight: 900; color: #ffffff;">Severe Lightning Risk</div>
                </div>
              </div>

              <div class="alert-details-list">
                <div><b>Location:</b> Ajmer District</div>
                <div><b>Expected:</b> 25 – 40 min</div>
                <div><b>Affected Area:</b> ~18 km</div>
                <div><b>Confidence:</b> 82%</div>
              </div>

              <div class="alert-actions-row">
                <button class="btn-alert-red" onclick="openAlertDetailsModal()">View Details</button>
                <button class="btn-alert-outline" id="ack-btn" onclick="acknowledgeAlert()">Acknowledge</button>
              </div>
            </div>

            <!-- Potential Impact Card -->
            <div class="card">
              <div class="card-title-header">
                <div class="card-title">🎯 Potential Impact</div>
              </div>

              <div class="impact-list">
                <div class="impact-item">
                  <div class="impact-name">✈️ Airport (Kishangarh)</div>
                  <span class="risk-badge moderate">Moderate</span>
                </div>
                <div class="impact-item">
                  <div class="impact-name">⚡ Power Substation</div>
                  <span class="risk-badge high">High</span>
                </div>
                <div class="impact-item">
                  <div class="impact-name">🚆 Rail Corridor</div>
                  <span class="risk-badge moderate">Moderate</span>
                </div>
                <div class="impact-item">
                  <div class="impact-name">🚴 Outdoor Areas</div>
                  <span class="risk-badge high">High</span>
                </div>
                <div class="impact-item">
                  <div class="impact-name">🌾 Agriculture</div>
                  <span class="risk-badge low">Low</span>
                </div>
              </div>
            </div>

            <!-- Why is Lightning Risk High? -->
            <div class="card">
              <div class="card-title-header">
                <div class="card-title">⚡ Why is Lightning Risk High?</div>
                <span style="font-size: 0.7rem; color: var(--cyan); cursor: pointer;" onclick="openAlertDetailsModal()">> View Details</span>
              </div>

              <div class="risk-factors-list">
                <div class="rf-item">
                  <span>Radar Reflectivity</span>
                  <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span style="color: var(--rose); font-weight: 700;">High</span>
                    <div class="rf-bar"><div class="rf-bar-fill" style="width: 85%; background: var(--rose);"></div></div>
                  </div>
                </div>

                <div class="rf-item">
                  <span>Cloud Growth</span>
                  <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span style="color: var(--rose); font-weight: 700;">High</span>
                    <div class="rf-bar"><div class="rf-bar-fill" style="width: 80%; background: var(--rose);"></div></div>
                  </div>
                </div>

                <div class="rf-item">
                  <span>Atmospheric Instability (CAPE)</span>
                  <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span style="color: var(--rose); font-weight: 700;">High</span>
                    <div class="rf-bar"><div class="rf-bar-fill" style="width: 90%; background: var(--rose);"></div></div>
                  </div>
                </div>

                <div class="rf-item">
                  <span>Humidity</span>
                  <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span style="color: var(--amber); font-weight: 700;">Moderate</span>
                    <div class="rf-bar"><div class="rf-bar-fill" style="width: 60%; background: var(--amber);"></div></div>
                  </div>
                </div>

                <div class="rf-item">
                  <span>Wind Shear</span>
                  <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span style="color: var(--amber); font-weight: 700;">Moderate</span>
                    <div class="rf-bar"><div class="rf-bar-fill" style="width: 55%; background: var(--amber);"></div></div>
                  </div>
                </div>

                <div class="rf-item">
                  <span>Lightning Activity (Current)</span>
                  <span style="color: var(--rose); font-weight: 700;">Rising</span>
                </div>
              </div>

              <div style="background: rgba(239, 68, 68, 0.1); border-radius: 6px; padding: 0.5rem; font-size: 0.72rem; color: var(--rose); font-weight: 700; text-align: center; margin-top: 0.3rem;">
                🧠 AI Assessment: High Lightning Risk
              </div>
            </div>

            <!-- Model Performance Panel -->
            <div class="card">
              <div class="card-title-header">
                <div class="card-title">🎯 Model Performance</div>
              </div>

              <div style="display: flex; flex-direction: column; gap: 0.6rem; font-size: 0.75rem;">
                <div>
                  <div style="font-weight: 700; color: #fff; margin-bottom: 0.2rem;">Nowcast (0–120 min)</div>
                  <div style="display: flex; justify-content: space-between; color: var(--text-muted); font-family: var(--font-code);">
                    <span>Acc: <b style="color: var(--cyan);">92.4%</b></span>
                    <span>F1: <b style="color: var(--emerald);">89.7%</b></span>
                    <span>FAR: <b style="color: var(--rose);">2.1%</b></span>
                  </div>
                </div>

                <div style="border-top: 1px solid var(--border-color); padding-top: 0.5rem;">
                  <div style="font-weight: 700; color: #fff; margin-bottom: 0.2rem;">Day-Ahead Forecast</div>
                  <div style="display: flex; justify-content: space-between; color: var(--text-muted); font-family: var(--font-code);">
                    <span>Acc: <b style="color: var(--cyan);">88.3%</b></span>
                    <span>F1: <b style="color: var(--emerald);">84.6%</b></span>
                    <span>FAR: <b style="color: var(--rose);">3.7%</b></span>
                  </div>
                </div>
              </div>

              <div style="font-size: 0.7rem; color: var(--cyan); cursor: pointer; margin-top: 0.2rem;" onclick="switchTab('tab-telemetry')">
                View Detailed Metrics →
              </div>
            </div>

            <!-- Forecast & Forensics Panel -->
            <div class="card">
              <div class="card-title-header">
                <div class="card-title">📈 Forecast & Forensics</div>
              </div>

              <div style="display: flex; gap: 0.4rem; font-size: 0.7rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.4rem;">
                <span style="color: var(--cyan); font-weight: 700; border-bottom: 2px solid var(--cyan); padding-bottom: 2px;">24h Forecast</span>
                <span style="color: var(--text-dim); cursor: pointer;" onclick="switchTab('tab-forensics')">Historical</span>
                <span style="color: var(--text-dim); cursor: pointer;" onclick="switchTab('tab-nowcast')">Compare</span>
              </div>

              <div style="display: flex; justify-content: space-between; text-align: center; margin-top: 0.4rem; font-size: 0.7rem;">
                <div>
                  <div style="color: var(--text-dim);">Today</div>
                  <div style="font-size: 1.1rem; margin: 0.1rem 0;">🌧️</div>
                  <div style="font-weight: 700; color: var(--cyan);">72%</div>
                </div>
                <div>
                  <div style="color: var(--text-dim);">Tue</div>
                  <div style="font-size: 1.1rem; margin: 0.1rem 0;">⛈️</div>
                  <div style="font-weight: 700; color: var(--cyan);">78%</div>
                </div>
                <div>
                  <div style="color: var(--text-dim);">Wed</div>
                  <div style="font-size: 1.1rem; margin: 0.1rem 0;">🌧️</div>
                  <div style="font-weight: 700; color: var(--cyan);">56%</div>
                </div>
                <div>
                  <div style="color: var(--text-dim);">Thu</div>
                  <div style="font-size: 1.1rem; margin: 0.1rem 0;">☁️</div>
                  <div style="font-weight: 700; color: var(--amber);">32%</div>
                </div>
                <div>
                  <div style="color: var(--text-dim);">Fri</div>
                  <div style="font-size: 1.1rem; margin: 0.1rem 0;">🌤️</div>
                  <div style="font-weight: 700; color: var(--emerald);">18%</div>
                </div>
              </div>

              <div style="font-size: 0.7rem; color: var(--cyan); cursor: pointer; margin-top: 0.3rem;" onclick="switchTab('tab-nowcast')">
                View Full Forecast →
              </div>
            </div>

            <!-- Quick Actions Card -->
            <div class="card">
              <div class="card-title-header">
                <div class="card-title">⚡ Quick Actions</div>
              </div>

              <div class="quick-actions-grid">
                <button class="qa-btn" onclick="downloadReport()">
                  <span>📥</span> Download Report
                </button>
                <button class="qa-btn" onclick="shareAlert()">
                  <span>🔗</span> Share Alert
                </button>
                <button class="qa-btn" onclick="showNotifications()">
                  <span>🔔</span> Notifications
                </button>
                <button class="qa-btn" onclick="toggleTheme()">
                  <span>🌙</span> Dark Mode
                </button>
              </div>
            </div>

          </div>
        </div>

      </div>

  <script>
    let homeMap = null;
    let homeBasemapLayer = null;
    let homeRadarLayer = null;
    let homeSatelliteLayer = null;
    let homeLightningLayer = null;
    let homeTrackLayer = null;
    let homeForecastLayer = null;
    let activeHomeLayer = 'live';

    function initHomeMap() {
      if (homeMap) {
        setTimeout(() => homeMap.invalidateSize(), 100);
        return;
      }
      const elem = document.getElementById('home-leaflet-map');
      if (!elem) return;

      const loader = document.getElementById('home-map-loader');

      try {
        homeMap = L.map('home-leaflet-map', {
          center: [26.9124, 74.6399],
          zoom: 9,
          zoomControl: true,
          attributionControl: false
        });

        // Dynamic theme-aware basemap selection
        const isLight = document.body.classList.contains('light-theme');
        const lightTileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
        const lightAttr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
        const darkTileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}';
        const darkAttr = 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ';

        homeBasemapLayer = L.tileLayer(isLight ? lightTileUrl : darkTileUrl, {
          maxZoom: 19,
          attribution: isLight ? lightAttr : darkAttr
        });

        homeBasemapLayer.on('tileerror', function() {
          if (homeMap && !homeMap.hasLayer(homeSatelliteLayer)) {
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(homeMap);
          }
        });

        homeBasemapLayer.addTo(homeMap);
        if (loader) loader.style.display = 'none';

        // Satellite layer
        homeSatelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
          maxZoom: 18,
          attribution: 'Esri World Imagery'
        });

        // Layer groups for weather overlays
        homeLightningLayer = L.layerGroup();
        homeTrackLayer = L.layerGroup();
        homeForecastLayer = L.layerGroup();
        homeRadarLayer = L.layerGroup();

        // Populate Lightning Strike Markers
        const strikeCoords = [
          [26.9124, 74.6399, 'CG -45kA (Severe)', '#ef4444'],
          [26.9450, 74.5800, 'IC +12kA', '#f59e0b'],
          [26.8800, 74.7100, 'CG -28kA', '#ef4444'],
          [26.8200, 74.5000, 'IC +8kA', '#00f2fe'],
          [27.0100, 74.6900, 'CG -62kA (Extreme)', '#ff007f']
        ];

        strikeCoords.forEach(([lat, lng, label, color]) => {
          L.circleMarker([lat, lng], {
            radius: 7,
            fillColor: color,
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 0.95
          }).bindPopup(`<b>⚡ Lightning Strike</b><br/>Type: ${label}<br/>Location: ${lat.toFixed(4)}, ${lng.toFixed(4)}`).addTo(homeLightningLayer);
        });

        // Populate Storm Track Vector & Core
        L.polyline([
          [26.65, 74.25],
          [26.78, 74.42],
          [26.9124, 74.6399],
          [27.05, 74.82]
        ], {
          color: '#00f2fe',
          weight: 3,
          dashArray: '8, 6'
        }).addTo(homeTrackLayer);

        L.circleMarker([26.9124, 74.6399], {
          radius: 9,
          fillColor: '#ef4444',
          color: '#ffffff',
          weight: 2,
          fillOpacity: 0.9
        }).bindPopup("<b>🎯 Ajmer Sector HQ</b><br/>Severe Thunderstorm Core<br/>Max dBZ: 58 | Top: 14.2 km").addTo(homeTrackLayer);

        // Populate AI Forecast Convective Contours
        L.circle([26.95, 74.68], {
          radius: 12000,
          color: '#ff007f',
          fillColor: '#ff007f',
          fillOpacity: 0.25,
          weight: 2
        }).bindPopup("<b>🔮 +15 Min AI Forecast Cell</b><br/>Lightning Prob: 88%").addTo(homeForecastLayer);

        L.circle([27.08, 74.85], {
          radius: 18000,
          color: '#f59e0b',
          fillColor: '#f59e0b',
          fillOpacity: 0.18,
          weight: 1.5
        }).bindPopup("<b>🔮 +30 Min AI Forecast Zone</b><br/>Lightning Prob: 45%").addTo(homeForecastLayer);

        // Fetch dynamic radar metadata
        if (typeof apiFetch === 'function') {
          apiFetch('/api/radar/metadata').then(r => r.json()).then(meta => {
            if (meta.tile_template) {
              const radarTileLayer = L.tileLayer(meta.tile_template, { opacity: 0.65 });
              radarTileLayer.addTo(homeRadarLayer);
            }
          }).catch(e => console.warn('Radar metadata error:', e));
        }

        // Add overlays for default 'live' view
        homeLightningLayer.addTo(homeMap);
        homeTrackLayer.addTo(homeMap);
        homeForecastLayer.addTo(homeMap);
        homeRadarLayer.addTo(homeMap);

        setTimeout(() => {
          if (homeMap) homeMap.invalidateSize();
          if (loader) loader.style.display = 'none';
        }, 150);

      } catch (err) {
        console.error('Home map initialization failed:', err);
        if (loader) {
          loader.innerHTML = '<div style="color:#ef4444; padding: 1rem; text-align: center;">⚠️ Map load issue. Retrying...</div>';
        }
      }
    }

    function switchHomeMapLayer(layerName) {
      if (!homeMap) initHomeMap();
      if (!homeMap) return;

      activeHomeLayer = layerName;

      // Highlight active button
      document.querySelectorAll('.map-tab-btn').forEach(btn => {
        const attr = btn.getAttribute('onclick') || '';
        if (attr.includes(`'${layerName}'`)) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });

      // Clear layers
      if (homeMap.hasLayer(homeBasemapLayer)) homeMap.removeLayer(homeBasemapLayer);
      if (homeMap.hasLayer(homeSatelliteLayer)) homeMap.removeLayer(homeSatelliteLayer);
      if (homeMap.hasLayer(homeLightningLayer)) homeMap.removeLayer(homeLightningLayer);
      if (homeMap.hasLayer(homeTrackLayer)) homeMap.removeLayer(homeTrackLayer);
      if (homeMap.hasLayer(homeForecastLayer)) homeMap.removeLayer(homeForecastLayer);
      if (homeMap.hasLayer(homeRadarLayer)) homeMap.removeLayer(homeRadarLayer);

      if (layerName === 'satellite') {
        homeSatelliteLayer.addTo(homeMap);
      } else {
        homeBasemapLayer.addTo(homeMap);
      }

      if (layerName === 'live') {
        homeLightningLayer.addTo(homeMap);
        homeTrackLayer.addTo(homeMap);
        homeForecastLayer.addTo(homeMap);
        homeRadarLayer.addTo(homeMap);
      } else if (layerName === 'radar') {
        homeRadarLayer.addTo(homeMap);
        homeTrackLayer.addTo(homeMap);
      } else if (layerName === 'lightning') {
        homeLightningLayer.addTo(homeMap);
      } else if (layerName === 'track') {
        homeTrackLayer.addTo(homeMap);
      } else if (layerName === 'forecast') {
        homeForecastLayer.addTo(homeMap);
        homeTrackLayer.addTo(homeMap);
      }

      setTimeout(() => homeMap.invalidateSize(), 50);
    }

    function acknowledgeAlert() {
      if (typeof apiFetch === 'function') {
        apiFetch('/api/acknowledge_alert', { method: 'POST' }).then(r => r.json()).then(data => {
          const btn = document.getElementById('ack-btn');
          if (btn) {
            btn.textContent = '✓ Acknowledged';
            btn.style.background = 'var(--emerald)';
            btn.style.color = '#fff';
            btn.style.borderColor = 'var(--emerald)';
          }
          alert('Active Alert acknowledged. Status recorded on backend.');
        }).catch(e => alert('Alert acknowledged locally.'));
      }
    }

    function downloadReport() { window.location.href = '/api/download_report'; }

    function shareAlert() {
      openModal('Share Alert', `
        <div>
          <p style="margin-bottom: 0.5rem;">Share live nowcast advisory link:</p>
          <input type="text" readonly value="${window.location.origin}/index.html?location=Ajmer" style="width: 100%; padding: 0.5rem; background: #050810; color: var(--cyan); border: 1px solid var(--border-accent); border-radius: 6px;">
          <button class="qa-btn" style="margin-top: 0.75rem; background: var(--blue); color: #fff;" onclick="navigator.clipboard.writeText(window.location.origin); alert('Link copied to clipboard!');">📋 Copy Link</button>
        </div>
      `);
    }

    function showNotifications() {
      openModal('Set Notifications', `
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
          <label><input type="checkbox" checked> GEOWEA SMS Cell Broadcast</label>
          <label><input type="checkbox" checked> Email Advisory Reports</label>
          <label><input type="checkbox" checked> SCADA Power Grid Decoupling Trigger</label>
        </div>
      `);
    }

    function openAlertDetailsModal() {
      openModal('Severe Lightning Risk Details - Ajmer Sector', `
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
          <div style="background: rgba(239,68,68,0.15); border: 1px solid var(--rose); padding: 0.75rem; border-radius: 8px; color: #fff;">
            <b>Hazard:</b> Cloud-to-Ground Lightning Surge & Microburst Front<br>
            <b>Location:</b> Ajmer District Sector 1<br>
            <b>Peak Reflectivity:</b> 52.4 dBZ at -10°C isotherm<br>
            <b>ETA (Ajmer):</b> 32 minutes
          </div>
          <p>Convective potential energy (CAPE) is currently at 1,850 J/kg. Disconnect high-voltage transformers and suspend tarmac operations.</p>
        </div>
      `);
    }

    function openModal(title, bodyHtml) {
      document.getElementById('modal-title').textContent = title;
      document.getElementById('modal-body').innerHTML = bodyHtml;
      document.getElementById('modal-container').style.display = 'flex';
    }

    function closeModal() { document.getElementById('modal-container').style.display = 'none'; }

    function promptLocationChange() {
      const loc = prompt("Enter location:", "Ajmer, Rajasthan");
      if (loc) document.getElementById('header-location-name').textContent = loc;
    }

    function setHomeTimelineStep(step) {
      document.querySelectorAll('.scrubber-step').forEach((s, idx) => s.classList.toggle('active', idx === step));
    }

    function toggleHomeTimelinePlay() {
      const btn = document.getElementById('home-play-btn');
      if (isHomePlaying) {
        clearInterval(homeTimelineTimer); isHomePlaying = false; btn.textContent = '▶';
      } else {
        isHomePlaying = true; btn.textContent = '⏸';
        let step = 0;
        homeTimelineTimer = setInterval(() => { step = (step + 1) % 6; setHomeTimelineStep(step); }, 1200);
      }
    }

    function updateMapBasemapsForTheme(isLight) {
      const lightTileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
      const lightAttr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
      const darkTileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}';
      const darkAttr = 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ';

      const targetUrl = isLight ? lightTileUrl : darkTileUrl;
      const targetAttr = isLight ? lightAttr : darkAttr;

      if (homeMap && homeBasemapLayer && activeHomeLayer !== 'satellite') {
        homeMap.removeLayer(homeBasemapLayer);
        homeBasemapLayer = L.tileLayer(targetUrl, { maxZoom: 19, attribution: targetAttr });
        homeBasemapLayer.addTo(homeMap);
      }

      if (typeof map !== 'undefined' && map && typeof currentTileLayer !== 'undefined' && currentTileLayer) {
        map.removeLayer(currentTileLayer);
        currentTileLayer = L.tileLayer(targetUrl, { maxZoom: 19, attribution: targetAttr });
        currentTileLayer.addTo(map);
      }
    }
  </script>
"""

# Patch original_scripts to trigger initHomeMap on page load and tab switch
patched_scripts = original_scripts.replace(
    'window.onload = function() {',
    'window.onload = function() {\n      initHomeMap();'
)

# Patch toggleTheme to update map basemaps dynamically
patched_scripts = patched_scripts.replace(
    "localStorage.setItem('vajra_theme', 'light');",
    "localStorage.setItem('vajra_theme', 'light');\n        updateMapBasemapsForTheme(true);"
)
patched_scripts = patched_scripts.replace(
    "localStorage.setItem('vajra_theme', 'dark');",
    "localStorage.setItem('vajra_theme', 'dark');\n        updateMapBasemapsForTheme(false);"
)

# Replace Carto labels URL with Esri World Dark Gray Reference tile layer
patched_scripts = patched_scripts.replace(
    'https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png',
    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}'
)

switch_tab_patch = """if (target) target.classList.add('active');

      if (tabId === 'tab-home' || tabId === 'home') {
        initHomeMap();
        if (homeMap) setTimeout(() => homeMap.invalidateSize(), 100);
      }"""

patched_scripts = patched_scripts.replace("if (target) target.classList.add('active');", switch_tab_patch)

verif_old_fn = """    async function runVerificationSuite() {
      switchTab('tab-pipeline');
      const consoleOut = document.getElementById('console-output');
      consoleOut.textContent = "Running verification test suite (run_tests.py)...\\nPlease wait...\\n";

      try {
        const res = await fetch('/api/run_tests', { method: 'POST' });
        const data = await res.json();
        
        let txt = `========================================================================\\n`;
        txt += `VERIFICATION SUITE COMPLETED in ${data.total_time}s\\n`;
        txt += `Passed: ${data.passed} | Failed: ${data.failed}\\n`;
        txt += `========================================================================\\n\\n`;

        data.results.forEach(r => {
          txt += `[${r.status.padEnd(10)}] ${r.name.padEnd(42)} ${r.seconds}s\\n`;
          txt += `           why: ${r.detail}\\n\\n`;
        });

        consoleOut.textContent = txt;
      } catch (err) {
        consoleOut.textContent += "\\nVerification suite failed: " + err;
      }
    }"""

verif_new_fn = """    async function runVerificationSuite() {
      const consoleOut = document.getElementById('console-output');
      if (consoleOut) {
        consoleOut.textContent = "Running 25-test verification suite (run_tests.py)...\\nPlease wait...\\n";
      }

      openModal("🧪 System Verification Test Suite", `
        <div style="font-family: var(--font-code); font-size: 0.8rem; color: var(--text-main);">
          <div id="verif-modal-status" style="margin-bottom: 0.75rem; color: var(--cyan); display: flex; align-items: center; gap: 0.5rem;">
            <span class="spinner" style="display:inline-block; width:16px; height:16px; border:2px solid var(--cyan); border-top-color:transparent; border-radius:50%; animation: map-spin 1s linear infinite;"></span>
            Executing 25 Ground-Truth Verification & Causality Proof Tests...
          </div>
          <pre id="verif-modal-log" style="max-height: 380px; overflow-y: auto; background: #050810; padding: 0.85rem; border-radius: 8px; font-size: 0.75rem; line-height: 1.4; color: var(--text-main); border: 1px solid var(--border-color);">Starting run_tests.py execution...</pre>
        </div>
      `);

      const modalLog = document.getElementById('verif-modal-log');
      const modalStatus = document.getElementById('verif-modal-status');

      try {
        const res = await fetch('/api/run_tests', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });

        if (!res.ok) {
          throw new Error(`Server returned HTTP ${res.status} (${res.statusText})`);
        }

        const data = await res.json();

        let txt = `========================================================================\\n`;
        txt += `VERIFICATION SUITE COMPLETED in ${data.total_time || 0}s\\n`;
        txt += `Passed: ${data.passed || 0} | Failed: ${data.failed || 0}\\n`;
        txt += `========================================================================\\n\\n`;

        const testResults = Array.isArray(data.results) ? data.results : [];

        if (testResults.length > 0) {
          testResults.forEach(r => {
            const statusStr = (r.status || 'UNKNOWN').padEnd(12);
            const nameStr = (r.name || 'Unnamed Test').padEnd(38);
            txt += `[${statusStr}] ${nameStr} ${r.seconds || 0}s\\n`;
            txt += `           why: ${r.detail || 'No description'}\\n\\n`;
          });
        } else {
          txt += `No test result items found in response.\\n`;
        }

        if (consoleOut) consoleOut.textContent = txt;
        if (modalLog) modalLog.textContent = txt;

        if (modalStatus) {
          if ((data.failed || 0) === 0) {
            modalStatus.innerHTML = `<span style="color: var(--emerald); font-weight: 700;">✅ All ${data.passed || 0} System Tests Passed (100% Skill & Causality Proofs Validated)</span>`;
          } else {
            modalStatus.innerHTML = `<span style="color: var(--rose); font-weight: 700;">⚠️ ${data.failed} Tests Failed (${data.passed} Passed)</span>`;
          }
        }
      } catch (err) {
        const errTxt = `\\nVerification suite execution failed: ${err.message || err}`;
        if (consoleOut) consoleOut.textContent += errTxt;
        if (modalLog) modalLog.textContent += errTxt;
        if (modalStatus) {
          modalStatus.innerHTML = `<span style="color: var(--rose); font-weight: 700;">❌ Verification Request Error</span>`;
        }
      }
    }"""

patched_scripts = patched_scripts.replace(verif_old_fn, verif_new_fn)

full_html = clean_html + original_tabs_content + """
    </main>
  </div>

  <!-- Footer -->
  <footer class="app-footer">
    <div>
      <b>VAJRA-AI</b> | Ministry of Earth Sciences, Government of India • Proposed solution for MoES Problem Statement SIH26072
    </div>
    <div>Smart Data • Better Decisions • Safer Communities</div>
  </footer>

  <!-- Modal Backdrop for Alerts / Details / Share / Notifications -->
  <div class="modal-backdrop" id="modal-container">
    <div class="modal-box">
      <h3 id="modal-title" style="color: var(--text-main); font-size: 1.1rem; display: flex; align-items: center; justify-content: space-between;">
        <span>Information</span>
        <button onclick="closeModal()" style="background: none; border: none; color: var(--text-muted); font-size: 1.2rem; cursor: pointer;">&times;</button>
      </h3>
      <div id="modal-body" style="color: var(--text-muted); font-size: 0.85rem; line-height: 1.5;"></div>
      <div style="display: flex; justify-content: flex-end; margin-top: 0.5rem;">
        <button class="qa-btn" onclick="closeModal()">Close</button>
      </div>
    </div>
  </div>
""" + patched_scripts

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print("Clean unified index.html built successfully!")
