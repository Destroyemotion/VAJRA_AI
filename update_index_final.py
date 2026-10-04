"""
Python script to generate restored, publication-quality index.html for VAJRA-AI dashboard.
Matches Image 1 target dashboard design, fixes all blank navigation pages, and connects backend APIs.
"""
import os

html_content = r'''<!DOCTYPE html>
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

    .emblem-icon {
      height: 38px;
      width: auto;
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
      gap: 1rem;
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

    /* Main App Layout */
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

    /* Page View Containers */
    .page-view {
      display: none;
      flex-direction: column;
      gap: 1.25rem;
      width: 100%;
    }

    .page-view.active {
      display: flex;
    }

    /* --- HOME DASHBOARD (IMAGE 1 TARGET DESIGN) --- */
    /* Top Row Cards */
    .top-dashboard-row {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.25rem;
    }

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

    /* Multi-Horizon Grid */
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

    .horizon-icon {
      font-size: 1.2rem;
    }

    .horizon-time {
      font-size: 0.85rem;
      font-weight: 800;
      color: #ffffff;
    }

    .horizon-desc {
      font-size: 0.72rem;
      color: var(--text-muted);
    }

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

    /* Data Sources Status Card */
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

    .ds-icon {
      font-size: 1.25rem;
    }

    .ds-name {
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--text-main);
    }

    .ds-status {
      font-size: 0.65rem;
      color: var(--emerald);
      font-weight: 600;
    }

    .ds-time {
      font-size: 0.65rem;
      color: var(--text-dim);
      font-family: var(--font-code);
    }

    /* Middle Layout Grid: Left Column & Right Column */
    .dashboard-main-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.25rem;
    }

    .left-column {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .right-column {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    /* Next Day & Nowcast Row */
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

    .risk-badge.high {
      background: rgba(239, 68, 68, 0.2);
      color: var(--rose);
      border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .risk-badge.moderate {
      background: rgba(245, 158, 11, 0.2);
      color: var(--amber);
      border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .risk-badge.low {
      background: rgba(16, 185, 129, 0.2);
      color: var(--emerald);
      border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .forecast-metrics-list {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
      margin-top: 0.25rem;
    }

    .fm-item {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }

    .fm-label {
      font-size: 0.68rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.3rem;
    }

    .fm-val {
      font-size: 0.95rem;
      font-weight: 800;
      color: #ffffff;
      font-family: var(--font-code);
    }

    /* Live Situation Map Area */
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

    .map-tab-btn:hover {
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.05);
    }

    .map-tab-btn.active {
      background: var(--blue);
      color: #ffffff;
      border-color: var(--blue-light);
      font-weight: 700;
    }

    .map-viewport-container {
      position: relative;
      width: 100%;
      height: 480px;
      background: #050810;
    }

    #home-leaflet-map {
      width: 100%;
      height: 100%;
      z-index: 1;
    }

    /* Floating Overlay Widgets on Map */
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

    /* Map Scrubber Footer */
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

    .scrubber-step.active {
      color: var(--cyan);
      border-bottom: 2px solid var(--cyan);
    }

    /* Active Alert Card */
    .active-alert-card {
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .alert-header {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
    }

    .alert-icon-box {
      font-size: 1.5rem;
      color: var(--rose);
    }

    .alert-details-list {
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      margin-top: 0.35rem;
    }

    .alert-actions-row {
      display: flex;
      gap: 0.75rem;
      margin-top: 0.5rem;
    }

    .btn-alert-red {
      background: linear-gradient(135deg, #ef4444, #dc2626);
      color: #ffffff;
      border: none;
      font-weight: 700;
      padding: 0.45rem 0.9rem;
      border-radius: 6px;
      font-size: 0.78rem;
      cursor: pointer;
      flex: 1;
    }

    .btn-alert-outline {
      background: transparent;
      color: var(--text-main);
      border: 1px solid var(--border-accent);
      font-weight: 600;
      padding: 0.45rem 0.9rem;
      border-radius: 6px;
      font-size: 0.78rem;
      cursor: pointer;
      flex: 1;
    }

    /* Potential Impact List */
    .impact-list {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .impact-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.8rem;
    }

    .impact-name {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text-main);
    }

    /* Why High Risk Panel */
    .risk-factors-list {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      font-size: 0.78rem;
    }

    .rf-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .rf-bar {
      height: 5px;
      width: 70px;
      border-radius: 3px;
      background: #1e293b;
      overflow: hidden;
    }

    .rf-bar-fill {
      height: 100%;
    }

    /* Quick Actions */
    .quick-actions-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.6rem;
    }

    .qa-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 0.6rem 0.75rem;
      color: var(--text-main);
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease;
    }

    .qa-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      border-color: var(--blue-light);
      color: var(--cyan);
    }

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

    /* --- LIGHT THEME COMPREHENSIVE STYLING --- */
    body.light-theme {
      --bg-dark: #f1f5f9;
      --bg-sidebar: #0f172a;
      --bg-card: #ffffff;
      --bg-card-hover: #f8fafc;
      --bg-header: #ffffff;
      --border-color: #cbd5e1;
      --border-accent: #94a3b8;
      
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;
    }

    body.light-theme header {
      background: #ffffff !important;
      border-bottom-color: #cbd5e1 !important;
    }

    body.light-theme .brand-title {
      color: #0f172a !important;
    }

    body.light-theme .card,
    body.light-theme .horizon-box,
    body.light-theme .header-widget {
      background: #ffffff !important;
      border-color: #cbd5e1 !important;
      color: #0f172a !important;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
    }

    body.light-theme .horizon-time,
    body.light-theme .card-title,
    body.light-theme .fm-val,
    body.light-theme .impact-name {
      color: #0f172a !important;
    }

    body.light-theme .qa-btn {
      background: #f8fafc !important;
      color: #0f172a !important;
      border-color: #cbd5e1 !important;
    }

    /* Modal Backdrop */
    .modal-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(5px);
      z-index: 2000;
      display: none;
      align-items: center;
      justify-content: center;
    }

    .modal-box {
      background: var(--bg-card);
      border: 1px solid var(--border-accent);
      border-radius: 14px;
      width: 520px;
      max-width: 90vw;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
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

      <!-- Theme Switcher -->
      <button id="theme-toggle-btn" onclick="toggleTheme()" style="background: rgba(255,255,255,0.08); border: 1px solid var(--border-color); color: var(--text-main); padding: 0.4rem 0.75rem; border-radius: 6px; font-weight: 600; font-size: 0.75rem; cursor: pointer;">
        ☀️ Light Mode
      </button>

      <!-- User Avatar -->
      <div class="user-avatar" title="User Profile">👤</div>
    </div>
  </header>

  <!-- Main Body -->
  <div class="app-body">
    
    <!-- Left Sidebar -->
    <aside class="sidebar">
      <ul class="nav-list">
        <li class="nav-item active" id="nav-home" onclick="navigateTo('home')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🏠</span>
            <span>Home</span>
          </div>
        </li>
        <li class="nav-item" id="nav-live-map" onclick="navigateTo('live-map')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🛰️</span>
            <span>Live Map</span>
          </div>
        </li>
        <li class="nav-item" id="nav-alerts" onclick="navigateTo('alerts')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🚨</span>
            <span>Alerts</span>
          </div>
          <span class="nav-badge">3</span>
        </li>
        <li class="nav-item" id="nav-forecast" onclick="navigateTo('forecast')">
          <div class="nav-item-left">
            <span class="nav-item-icon">📡</span>
            <span>Forecast</span>
          </div>
        </li>
        <li class="nav-item" id="nav-reports" onclick="navigateTo('reports')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🔍</span>
            <span>Reports</span>
          </div>
        </li>
        <li class="nav-item" id="nav-data-ai" onclick="navigateTo('data-ai')">
          <div class="nav-item-left">
            <span class="nav-item-icon">🧠</span>
            <span>Data & AI</span>
          </div>
        </li>
        <li class="nav-item" id="nav-settings" onclick="navigateTo('settings')">
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

    <!-- Main Content Area -->
    <main class="content-area">

      <!-- PAGE VIEW 0: HOME DASHBOARD (IMAGE 1 TARGET DESIGN) -->
      <div id="page-home" class="page-view active">
        
        <!-- Top Row Cards -->
        <div class="top-dashboard-row">
          <!-- Multi-Horizon Forecast Card -->
          <div class="card">
            <div class="card-title-header">
              <div class="card-title">⚡ Multi-Horizon Forecast</div>
            </div>
            
            <div class="horizon-grid">
              <div class="horizon-box active">
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

              <div class="horizon-box">
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

              <div class="horizon-box">
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

              <div class="horizon-box">
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
              <div class="data-source-item">
                <span class="ds-icon">📡</span>
                <span class="ds-name">Radar</span>
                <span class="ds-status" id="ds-radar-status">Live</span>
                <span class="ds-time" id="ds-radar-time">42s</span>
              </div>
              <div class="data-source-item">
                <span class="ds-icon">🛰️</span>
                <span class="ds-name">Satellite</span>
                <span class="ds-status" id="ds-sat-status">Live</span>
                <span class="ds-time" id="ds-sat-time">1.8m</span>
              </div>
              <div class="data-source-item">
                <span class="ds-icon">⚡</span>
                <span class="ds-name">Lightning</span>
                <span class="ds-status" id="ds-ltg-status">Live</span>
                <span class="ds-time" id="ds-ltg-time">3m</span>
              </div>
              <div class="data-source-item">
                <span class="ds-icon">🎯</span>
                <span class="ds-name">AWS</span>
                <span class="ds-status" id="ds-aws-status">Live</span>
                <span class="ds-time" id="ds-aws-time">58s</span>
              </div>
              <div class="data-source-item">
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

              <!-- Map Scrubber Timeline Footer -->
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

              <div style="font-size: 0.7rem; color: var(--cyan); cursor: pointer; margin-top: 0.2rem;" onclick="navigateTo('data-ai')">
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
                <span style="color: var(--text-dim); cursor: pointer;" onclick="navigateTo('reports')">Historical</span>
                <span style="color: var(--text-dim); cursor: pointer;" onclick="navigateTo('forecast')">Compare</span>
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

              <div style="font-size: 0.7rem; color: var(--cyan); cursor: pointer; margin-top: 0.3rem;" onclick="navigateTo('forecast')">
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

      <!-- PAGE VIEW 1: LIVE MAP TAB -->
      <div id="page-live-map" class="page-view">
        <div class="card" style="padding: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div style="display: flex; gap: 0.75rem; align-items: center;">
              <input type="text" id="live-map-search" placeholder="Search Indian City / District..." style="background: rgba(15,23,42,0.8); border: 1px solid var(--border-accent); color: #fff; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.8rem; width: 260px;" oninput="handleSearchInput(event)">
              <button style="background: var(--blue); color: #fff; border: none; padding: 0.4rem 0.8rem; border-radius: 6px; font-weight: 700; font-size: 0.8rem; cursor: pointer;" onclick="locateUserPosition()">📍 My Location</button>
            </div>

            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <span style="font-size: 0.75rem; color: var(--text-muted);">Basemap:</span>
              <select id="live-map-style" style="background: rgba(15,23,42,0.8); color: #fff; border: 1px solid var(--border-accent); padding: 0.4rem; border-radius: 6px; font-size: 0.78rem;" onchange="changeMapStyle()">
                <option value="satellite">🛰️ Real-Time Satellite (Esri)</option>
                <option value="dark">🌑 Dark GIS Canvas</option>
                <option value="terrain">🗺️ Topographic</option>
              </select>
              <button style="background: rgba(245,158,11,0.2); color: var(--amber); border: 1px solid var(--amber); padding: 0.4rem 0.75rem; border-radius: 6px; font-size: 0.78rem; font-weight: 700; cursor: pointer;" onclick="simulateIngestStrike()">⚡ Ingest Strike</button>
            </div>
          </div>

          <!-- Quick Region Chips -->
          <div style="display: flex; gap: 0.35rem; flex-wrap: wrap; margin-top: 0.75rem;">
            <button class="qa-btn" style="padding: 0.25rem 0.5rem; font-size: 0.7rem;" onclick="selectQuickLocation(26.91, 74.64, 'Ajmer')">📍 Ajmer</button>
            <button class="qa-btn" style="padding: 0.25rem 0.5rem; font-size: 0.7rem;" onclick="selectQuickLocation(28.61, 77.20, 'Delhi NCR')">🏛️ Delhi NCR</button>
            <button class="qa-btn" style="padding: 0.25rem 0.5rem; font-size: 0.7rem;" onclick="selectQuickLocation(19.07, 72.87, 'Mumbai')">🌊 Mumbai</button>
            <button class="qa-btn" style="padding: 0.25rem 0.5rem; font-size: 0.7rem;" onclick="selectQuickLocation(12.97, 77.59, 'Bengaluru')">💻 Bengaluru</button>
            <button class="qa-btn" style="padding: 0.25rem 0.5rem; font-size: 0.7rem;" onclick="selectQuickLocation(22.57, 88.36, 'Kolkata')">🌉 Kolkata</button>
            <button class="qa-btn" style="padding: 0.25rem 0.5rem; font-size: 0.7rem;" onclick="selectQuickLocation(13.08, 80.27, 'Chennai')">🏖️ Chennai</button>
            <button class="qa-btn" style="padding: 0.25rem 0.5rem; font-size: 0.7rem;" onclick="selectQuickLocation(18.52, 73.85, 'Pune')">🏔️ Pune</button>
          </div>
        </div>

        <!-- Full GIS Leaflet Map Container -->
        <div class="card" style="padding: 0; overflow: hidden; height: 550px;">
          <div id="leaflet-map" style="width: 100%; height: 100%; z-index: 1;"></div>
        </div>
      </div>

      <!-- PAGE VIEW 2: ALERTS TAB -->
      <div id="page-alerts" class="page-view">
        <div class="card">
          <div class="card-title-header">
            <div class="card-title">🚨 Early Warning & Emergency Dispatch Engine</div>
            <span class="risk-badge high">DEFCON-2 DISPATCH MODE</span>
          </div>
          
          <div class="forecast-metrics-list" style="grid-template-columns: repeat(4, 1fr); margin-top: 0.5rem;">
            <div class="fm-item">
              <span class="fm-label">👥 Monitored Population</span>
              <span class="fm-val" style="color: #fff;">14.2M Civilian</span>
            </div>
            <div class="fm-item">
              <span class="fm-label">🎯 Active Geofences</span>
              <span class="fm-val" style="color: var(--amber);">6 Zones</span>
            </div>
            <div class="fm-item">
              <span class="fm-label">⏱️ Lead-Time Delta</span>
              <span class="fm-val" style="color: var(--cyan);">+38.5m vs NWP</span>
            </div>
            <div class="fm-item">
              <span class="fm-label">🚀 CAP Dispatches</span>
              <span class="fm-val" style="color: var(--emerald);">28 / 60m</span>
            </div>
          </div>
        </div>

        <!-- Dispatch Geofence Cards List -->
        <div class="card">
          <div class="card-title-header">
            <div class="card-title">🎯 Active Geofence Alert Polygons</div>
            <button style="background: var(--rose); color: #fff; border: none; padding: 0.4rem 0.8rem; border-radius: 6px; font-weight: 700; font-size: 0.75rem; cursor: pointer;" onclick="broadcastAlert()">⚡ Broadcast Emergency CAP XML</button>
          </div>

          <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.5rem;">
            <div style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 0.85rem;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: 800; color: #fff; font-size: 0.9rem;">Ajmer District Sector 1 - Civic Zone</div>
                <span class="risk-badge high">CRITICAL</span>
              </div>
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">Expected: 25 - 40 min • Affected Area: ~18 km • Population: 850,000</div>
              <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                <button class="qa-btn" style="background: rgba(239,68,68,0.2); color: #fff;" onclick="openAlertDetailsModal()">View CAP Details</button>
                <button class="qa-btn" onclick="acknowledgeAlert()">Acknowledge</button>
              </div>
            </div>

            <div style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 8px; padding: 0.85rem;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: 800; color: #fff; font-size: 0.9rem;">Kishangarh Airport Tarmac</div>
                <span class="risk-badge moderate">ARMED</span>
              </div>
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">Ground Stop Advisory • 5 nm lightning envelope</div>
            </div>
          </div>
        </div>
      </div>

      <!-- PAGE VIEW 3: FORECAST TAB -->
      <div id="page-forecast" class="page-view">
        <div class="card">
          <div class="card-title-header">
            <div class="card-title">📡 Nowcast & Spatial Probability Model Visualizer</div>
          </div>

          <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
            <select id="model-select-tab" style="background: rgba(15,23,42,0.8); color: #fff; border: 1px solid var(--border-accent); padding: 0.4rem; border-radius: 6px; font-size: 0.78rem;" onchange="loadSample()">
              <option value="advection">Lagrangian Advection (Optical Flow)</option>
              <option value="convlstm">ConvLSTM (Eulerian)</option>
              <option value="charging_rule">Charging-Layer Physics Rule</option>
              <option value="ensemble">Ensemble Model</option>
            </select>
            <button class="qa-btn" onclick="prevSample()">◀ Prev Sample</button>
            <span id="sample-indicator-tab" style="font-family: var(--font-code); font-size: 0.8rem; font-weight: 700;">Sample 1 / 6</span>
            <button class="qa-btn" onclick="nextSample()">Next Sample ▶</button>
          </div>
        </div>

        <!-- 4 Canvases Grid -->
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
          <div class="card" style="align-items: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: var(--cyan); width: 100%;">❄️ Charging-Layer Radar (t=0)</div>
            <canvas id="canvas-radar" width="220" height="220" style="border: 1px solid var(--border-accent); background: #050810; margin-top: 0.5rem;"></canvas>
          </div>

          <div class="card" style="align-items: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: var(--rose); width: 100%;">🎯 Observed Strikes (Target)</div>
            <canvas id="canvas-obs" width="220" height="220" style="border: 1px solid var(--border-accent); background: #050810; margin-top: 0.5rem;"></canvas>
          </div>

          <div class="card" style="align-items: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: var(--violet); width: 100%;">🔮 Probability Forecast</div>
            <canvas id="canvas-prob" width="220" height="220" style="border: 1px solid var(--border-accent); background: #050810; margin-top: 0.5rem;"></canvas>
          </div>

          <div class="card" style="align-items: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: var(--emerald); width: 100%;">🧩 Spatial Contingency Overlay</div>
            <canvas id="canvas-contingency" width="220" height="220" style="border: 1px solid var(--border-accent); background: #050810; margin-top: 0.5rem;"></canvas>
          </div>
        </div>
      </div>

      <!-- PAGE VIEW 4: REPORTS TAB -->
      <div id="page-reports" class="page-view">
        <div class="card">
          <div class="card-title-header">
            <div class="card-title">🔍 Meteorological Forensics & Historical Case Studies</div>
            <button class="qa-btn" onclick="downloadReport()">📥 Download Full Report (.docx)</button>
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">Detailed re-analysis of Derecho squall lines, severe supercells, and cloudburst events.</div>
        </div>

        <div class="card">
          <div class="card-title-header">
            <div class="card-title">📁 Case Archive Records</div>
          </div>
          <div id="reports-case-list" style="display: flex; flex-direction: column; gap: 0.5rem;">
            <!-- Populated dynamically from /api/historical_archive -->
          </div>
        </div>
      </div>

      <!-- PAGE VIEW 5: DATA & AI TAB -->
      <div id="page-data-ai" class="page-view">
        <div class="card">
          <div class="card-title-header">
            <div class="card-title">🧠 AI Model Architecture & System Telemetry</div>
          </div>

          <div class="forecast-metrics-list" style="grid-template-columns: repeat(4, 1fr); margin-top: 0.5rem;">
            <div class="fm-item">
              <span class="fm-label">🖥️ Hardware Node</span>
              <span class="fm-val" style="color: var(--cyan);">DGX-H100-SX5</span>
            </div>
            <div class="fm-item">
              <span class="fm-label">⚡ Inference Latency</span>
              <span class="fm-val" style="color: var(--emerald);">110 ms</span>
            </div>
            <div class="fm-item">
              <span class="fm-label">📊 GPU Util</span>
              <span class="fm-val" style="color: var(--amber);">68.0%</span>
            </div>
            <div class="fm-item">
              <span class="fm-label">💾 VRAM Usage</span>
              <span class="fm-val" style="color: #fff;">54.2 / 80.0 GB</span>
            </div>
          </div>
        </div>

        <!-- Grad-CAM Panel -->
        <div class="card">
          <div class="card-title-header">
            <div class="card-title">🔍 Grad-CAM Explainability Feature Attribution</div>
          </div>

          <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.5rem; font-size: 0.78rem;">
            <div>40 dBZ Echo Tops at -10°C Layer (Mixed-Phase Glaciation): <b>41.2%</b></div>
            <div>LLN Intra-Cloud Flash Rate Gradient Jump: <b>28.8%</b></div>
            <div>Mesocyclone Radial Velocity Shear: <b>19.4%</b></div>
            <div>Satellite Band 13 Cloud-Top Cooling Signature: <b>10.6%</b></div>
          </div>
        </div>
      </div>

      <!-- PAGE VIEW 6: SETTINGS TAB -->
      <div id="page-settings" class="page-view">
        <div class="card">
          <div class="card-title-header">
            <div class="card-title">⚙️ System Verification & Model Pipeline Controls</div>
          </div>

          <div style="display: flex; gap: 0.75rem; margin-top: 0.5rem; flex-wrap: wrap;">
            <button class="qa-btn" style="background: var(--blue); color: #fff;" onclick="runVerificationSuite()">🧪 Run Verification Suite</button>
            <button class="qa-btn" onclick="runMakeSynth()">🎲 Generate Synthetic Dataset</button>
            <button class="qa-btn" onclick="runBaselines()">📈 Evaluate Baselines</button>
            <button class="qa-btn" onclick="runTrain()">⚡ Train ConvLSTM Model</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title-header">
            <div class="card-title">🖥️ Execution Console Output</div>
          </div>
          <pre id="console-output" style="background: #050810; border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; color: var(--cyan); font-family: var(--font-code); font-size: 0.78rem; max-height: 250px; overflow-y: auto;">System operational. Ready for verification commands.</pre>
        </div>
      </div>

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
    <div class="modal-box" id="modal-box-content">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h3 id="modal-title" style="color: #fff; font-size: 1.1rem; font-weight: 800;">Alert Details</h3>
        <button style="background: transparent; border: none; color: var(--text-muted); font-size: 1.2rem; cursor: pointer;" onclick="closeModal()">✖</button>
      </div>
      <div id="modal-body" style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;"></div>
      <div style="display: flex; justify-content: flex-end; margin-top: 0.5rem;">
        <button class="qa-btn" onclick="closeModal()">Close</button>
      </div>
    </div>
  </div>

  <!-- Main JavaScript Logic -->
  <script>
    // State variables
    let homeMap = null;
    let liveMap = null;
    let currentData = null;
    let sampleIdx = 0;
    let totalSamples = 6;
    let currentLead = 2;
    let homeTimelineTimer = null;
    let isHomePlaying = false;

    // Generic API fetch helper
    async function apiFetch(url, options = {}) {
      try {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res;
      } catch (err) {
        console.warn(`API fetch error for ${url}:`, err);
        throw err;
      }
    }

    // Live Clock
    function updateClock() {
      const now = new Date();
      const options = { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' };
      const timeStr = now.toLocaleDateString('en-GB', options).replace(',', ' |') + ' IST';
      const clockElem = document.getElementById('header-clock');
      if (clockElem) clockElem.textContent = timeStr;
    }

    // Page Navigation Router
    function navigateTo(pageId) {
      document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
      const activeNav = document.getElementById(`nav-${pageId}`);
      if (activeNav) activeNav.classList.add('active');

      document.querySelectorAll('.page-view').forEach(el => el.classList.remove('active'));
      const targetPage = document.getElementById(`page-${pageId}`);
      if (targetPage) targetPage.classList.add('active');

      // Trigger Leaflet map invalidateSize
      if (pageId === 'home' && homeMap) {
        setTimeout(() => homeMap.invalidateSize(), 150);
      }
      if (pageId === 'live-map' && liveMap) {
        setTimeout(() => liveMap.invalidateSize(), 150);
      }
      if (pageId === 'reports') {
        fetchHistoricalArchive();
      }
    }

    // Initialize Home Satellite Leaflet Map
    function initHomeMap() {
      if (homeMap) return;
      const homeMapElem = document.getElementById('home-leaflet-map');
      if (!homeMapElem) return;

      homeMap = L.map('home-leaflet-map', {
        center: [26.9124, 74.6399], // Ajmer, Rajasthan
        zoom: 9,
        zoomControl: true
      });

      // Esri World Imagery Satellite Tile Layer
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri'
      }).addTo(homeMap);

      // RainViewer Radar Overlay
      apiFetch('/api/radar/metadata').then(r => r.json()).then(meta => {
        if (meta.tile_template) {
          L.tileLayer(meta.tile_template, { opacity: 0.65 }).addTo(homeMap);
        }
      }).catch(e => console.warn('Radar overlay error:', e));

      // Ajmer Marker
      const ajmerMarker = L.circleMarker([26.9124, 74.6399], {
        radius: 8,
        fillColor: '#ef4444',
        color: '#ffffff',
        weight: 2,
        fillOpacity: 0.9
      }).addTo(homeMap);

      ajmerMarker.bindPopup("<b>Ajmer Sector HQ</b><br/>High Lightning Risk Warning Active").openPopup();

      // Storm Cell Track Vector
      const trackPoints = [[26.70, 74.40], [26.85, 74.55], [26.9124, 74.6399]];
      L.polyline(trackPoints, { color: '#00f2fe', weight: 3, dashArray: '6, 6' }).addTo(homeMap);

      // Strike markers
      L.circleMarker([26.82, 74.50], { radius: 5, fillColor: '#ffff00', color: '#ff0000', weight: 1, fillOpacity: 0.9 }).addTo(homeMap);
      L.circleMarker([26.88, 74.58], { radius: 5, fillColor: '#ffff00', color: '#ff0000', weight: 1, fillOpacity: 0.9 }).addTo(homeMap);
    }

    // Initialize Live Map GIS
    function initLiveMap() {
      if (liveMap) return;
      const liveMapElem = document.getElementById('leaflet-map');
      if (!liveMapElem) return;

      liveMap = L.map('leaflet-map', {
        center: [26.9124, 74.6399],
        zoom: 7
      });

      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}').addTo(liveMap);
    }

    // Data Sources Health Updates
    async function updateDataSourcesHealth() {
      try {
        const res = await apiFetch('/api/sources/health');
        const data = await res.json();
        const sources = data.sources || {};
        
        if (sources.radar) {
          document.getElementById('ds-radar-status').textContent = sources.radar.state || 'Live';
          document.getElementById('ds-radar-time').textContent = `${sources.radar.latency_ms || 42}ms`;
        }
        if (sources.satellites) {
          document.getElementById('ds-sat-status').textContent = sources.satellites.state || 'Live';
          document.getElementById('ds-sat-time').textContent = '1.8m';
        }
        if (sources.ground_lln) {
          document.getElementById('ds-ltg-status').textContent = sources.ground_lln.state || 'Live';
          document.getElementById('ds-ltg-time').textContent = '3m';
        }
      } catch (err) {
        console.warn('Sources health fetch error:', err);
      }
    }

    // Fetch Weather & Forecast Data
    async function fetchHomeData() {
      try {
        const res = await apiFetch('/api/weather?lat=26.9124&lon=74.6399&city=Ajmer');
        const data = await res.json();
        if (data.current) {
          document.getElementById('nowcast-prob').textContent = `${Math.round(data.current.rain_prob || 72)}%`;
        }
      } catch (err) {
        console.warn('Home weather error:', err);
      }
    }

    // Historical Archives for Reports Tab
    async function fetchHistoricalArchive() {
      const container = document.getElementById('reports-case-list');
      if (!container) return;
      try {
        const res = await apiFetch('/api/historical_archive');
        const data = await res.json();
        container.innerHTML = '';
        (data.cases || []).forEach(c => {
          const item = document.createElement('div');
          item.style.cssText = 'background: rgba(15,23,42,0.6); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.85rem; font-size: 0.8rem;';
          item.innerHTML = `
            <div style="display: flex; justify-content: space-between; font-weight: 800; color: #fff;">
              <span>${c.name} (${c.id})</span>
              <span style="color: var(--cyan);">${c.date}</span>
            </div>
            <div style="color: var(--text-muted); margin-top: 0.3rem;">${c.summary}</div>
            <div style="display: flex; gap: 1rem; margin-top: 0.4rem; font-family: var(--font-code); font-size: 0.72rem; color: var(--text-dim);">
              <span>Strikes: ${c.total_strikes}</span>
              <span>Peak Gust: ${c.peak_gust_kts} kts</span>
              <span>CSI Score: ${c.csi_score}</span>
            </div>
          `;
          container.appendChild(item);
        });
      } catch(e) {
        console.warn('Archive fetch error:', e);
      }
    }

    // Actions & Modals
    function acknowledgeAlert() {
      apiFetch('/api/acknowledge_alert', { method: 'POST' }).then(r => r.json()).then(data => {
        const ackBtn = document.getElementById('ack-btn');
        if (ackBtn) {
          ackBtn.textContent = '✓ Acknowledged';
          ackBtn.style.background = 'var(--emerald)';
          ackBtn.style.color = '#ffffff';
          ackBtn.style.borderColor = 'var(--emerald)';
        }
        alert('Active Alert acknowledged. Confirmation recorded on backend.');
      }).catch(err => {
        alert('Alert acknowledged locally.');
      });
    }

    function downloadReport() {
      window.location.href = '/api/download_report';
    }

    function shareAlert() {
      openModal('Share Alert', `
        <div>
          <p>Share this live nowcast advisory for Ajmer District:</p>
          <input type="text" readonly value="${window.location.origin}/index.html?location=Ajmer&alert=severe" style="width: 100%; padding: 0.5rem; margin-top: 0.5rem; background: #050810; color: var(--cyan); border: 1px solid var(--border-accent); border-radius: 6px;">
          <button class="qa-btn" style="margin-top: 0.75rem; background: var(--blue); color: #fff;" onclick="navigator.clipboard.writeText(window.location.origin); alert('Link copied to clipboard!');">📋 Copy Link</button>
        </div>
      `);
    }

    function showNotifications() {
      openModal('Set Notifications', `
        <div>
          <p>Configure automated emergency dispatches for your zone:</p>
          <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.75rem;">
            <label><input type="checkbox" checked> SMS Warning (Civilian GEOWEA)</label>
            <label><input type="checkbox" checked> Email Advisory Report</label>
            <label><input type="checkbox" checked> Push Siren Trigger</label>
          </div>
        </div>
      `);
    }

    function openAlertDetailsModal() {
      openModal('Severe Lightning Risk Details - Ajmer District', `
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
          <div style="background: rgba(239,68,68,0.15); border: 1px solid var(--rose); padding: 0.75rem; border-radius: 8px; color: #fff;">
            <b>Hazard:</b> Cloud-to-Ground Lightning Surge & Microburst Front<br>
            <b>Target Sector:</b> Ajmer District & Surrounding Highways<br>
            <b>Peak Reflectivity:</b> 52.4 dBZ at -10°C isotherm layer<br>
            <b>ETA (Ajmer City):</b> 32 minutes
          </div>
          <p>Atmospheric instability (CAPE) is currently at 1,850 J/kg with active updraft electrification. Outdoor activities should be suspended immediately.</p>
        </div>
      `);
    }

    function openModal(title, contentHtml) {
      document.getElementById('modal-title').textContent = title;
      document.getElementById('modal-body').innerHTML = contentHtml;
      document.getElementById('modal-container').style.display = 'flex';
    }

    function closeModal() {
      document.getElementById('modal-container').style.display = 'none';
    }

    function toggleTheme() {
      const isLight = document.body.classList.toggle('light-theme');
      const btn = document.getElementById('theme-toggle-btn');
      if (btn) btn.textContent = isLight ? '🌙 Dark Mode' : '☀️ Light Mode';
    }

    function promptLocationChange() {
      const loc = prompt("Enter location (e.g. Ajmer, Rajasthan or Mumbai):", "Ajmer, Rajasthan");
      if (loc) {
        document.getElementById('header-location-name').textContent = loc;
      }
    }

    // Timeline Scrubber Controls
    function setHomeTimelineStep(step) {
      document.querySelectorAll('.scrubber-step').forEach((s, idx) => {
        s.classList.toggle('active', idx === step);
      });
    }

    function toggleHomeTimelinePlay() {
      const btn = document.getElementById('home-play-btn');
      if (isHomePlaying) {
        clearInterval(homeTimelineTimer);
        isHomePlaying = false;
        btn.textContent = '▶';
      } else {
        isHomePlaying = true;
        btn.textContent = '⏸';
        let step = 0;
        homeTimelineTimer = setInterval(() => {
          step = (step + 1) % 6;
          setHomeTimelineStep(step);
        }, 1200);
      }
    }

    // Canvases for Forecast tab
    function drawRadar() {
      const c = document.getElementById('canvas-radar');
      if (!c) return;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#050810';
      ctx.fillRect(0, 0, c.width, c.height);

      ctx.fillStyle = 'rgba(0, 242, 254, 0.4)';
      ctx.beginPath(); ctx.arc(110, 110, 60, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = 'rgba(239, 68, 68, 0.7)';
      ctx.beginPath(); ctx.arc(100, 105, 30, 0, Math.PI*2); ctx.fill();
    }

    function drawObserved() {
      const c = document.getElementById('canvas-obs');
      if (!c) return;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#050810'; ctx.fillRect(0, 0, c.width, c.height);
      ctx.fillStyle = '#ffffff';
      for (let i = 0; i < 15; i++) {
        ctx.beginPath();
        ctx.arc(80 + Math.random()*50, 80 + Math.random()*50, 3, 0, Math.PI*2);
        ctx.fill();
      }
    }

    function drawProbability() {
      const c = document.getElementById('canvas-prob');
      if (!c) return;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#050810'; ctx.fillRect(0, 0, c.width, c.height);
      ctx.fillStyle = 'rgba(121, 40, 202, 0.6)';
      ctx.beginPath(); ctx.arc(110, 110, 50, 0, Math.PI*2); ctx.fill();
    }

    // On Load Initializer
    window.onload = function() {
      setInterval(updateClock, 1000);
      updateClock();
      initHomeMap();
      initLiveMap();
      updateDataSourcesHealth();
      fetchHomeData();
      drawRadar(); drawObserved(); drawProbability();
    };
  </script>
</body>
</html>
'''

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Restored index.html generated successfully!")
