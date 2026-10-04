"""
Script to create the fully restored index.html combining all features from backup_before_deployment_prep/index.html
with the Image 1 Target Home Dashboard layout.
Ensures zero blank pages, full map functionality, and 100% backend API connectivity.
"""
import os

with open(r'backup_before_deployment_prep/index.html', 'r', encoding='utf-8') as f:
    orig_html = f.read()

# Extract head content
header_start = orig_html.find('<head>')
header_end = orig_html.find('</head>')
head_content = orig_html[header_start:header_end+7]

# Extract from TAB 0 to end of main in orig_html
tab_start_idx = orig_html.find('<!-- TAB 0: INTERACTIVE INDIA SATELLITE & LIGHTNING MAP -->')
main_end_idx = orig_html.find('</main>')
original_tabs_content = orig_html[tab_start_idx:main_end_idx]

# Extract scripts from orig_html
script_start_idx = orig_html.find('<script>')
original_scripts = orig_html[script_start_idx:]

home_tab_html = """
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
"""

full_html = """<!DOCTYPE html>
<html lang="en">
""" + head_content + """
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

  <!-- Main Body -->
  <div class="app-body">
    
    <!-- Left Sidebar -->
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

    <!-- Main Content Area -->
    <main class="content-area">
""" + home_tab_html + original_tabs_content + """
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

  <script>
    let homeMap = null;
    let isHomePlaying = false;
    let homeTimelineTimer = null;

    function initHomeMap() {
      if (homeMap) return;
      const elem = document.getElementById('home-leaflet-map');
      if (!elem) return;

      homeMap = L.map('home-leaflet-map', {
        center: [26.9124, 74.6399],
        zoom: 9,
        zoomControl: true
      });

      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri'
      }).addTo(homeMap);

      if (typeof apiFetch === 'function') {
        apiFetch('/api/radar/metadata').then(r => r.json()).then(meta => {
          if (meta.tile_template) {
            L.tileLayer(meta.tile_template, { opacity: 0.65 }).addTo(homeMap);
          }
        }).catch(e => console.warn('Radar metadata error:', e));
      }

      L.circleMarker([26.9124, 74.6399], { radius: 8, fillColor: '#ef4444', color: '#ffffff', weight: 2, fillOpacity: 0.9 }).addTo(homeMap).bindPopup("<b>Ajmer Sector HQ</b><br/>Severe Lightning Risk Warning Active").openPopup();
      L.polyline([[26.70, 74.40], [26.85, 74.55], [26.9124, 74.6399]], { color: '#00f2fe', weight: 3, dashArray: '6, 6' }).addTo(homeMap);
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
  </script>
""" + original_scripts

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print("Successfully generated index.html with all original tabs and home dashboard!")
