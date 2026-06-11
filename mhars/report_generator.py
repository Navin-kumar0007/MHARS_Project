from typing import Dict, Any, List
from datetime import datetime

class ReportGenerator:
    """
    Generates a structured HTML diagnostic report of the hardware state.
    """
    
    @staticmethod
    def generate_html_report(state: Any) -> str:
        """
        Builds the HTML report string using current system state.
        Args:
            state: SystemState object from api/main.py
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        machine_name = state.machine_profile["name"]
        
        # Gather metrics safely
        current_temp = getattr(state.env, "temp", 0.0)
        
        # Telemetry history
        history = list(state.telemetry_history)
        if history:
            latest = history[-1]
            # Advanced analytics live in the metadata sub-dict of the telemetry payload.
            meta = latest.get("metadata", {}) or {}
            health_score = meta.get("health_score", 0)
            status_color = "#10b981" if health_score >= 70 else "#f59e0b" if health_score >= 40 else "#ef4444"
            status_text = "Healthy" if health_score >= 70 else "Warning" if health_score >= 40 else "Critical"

            rul = meta.get("rul_minutes", "N/A")
            rul_text = f"{rul} mins" if isinstance(rul, (int, float)) else str(rul)

            fault_type = meta.get("fault_type", "None detected")

            # Sub-scores
            breakdown = meta.get("health_breakdown", {})
            thermal_score = breakdown.get("thermal", "N/A")
            mech_score = breakdown.get("mechanical", "N/A")

        else:
            health_score, status_color, status_text, rul_text, fault_type = 0, "#ef4444", "Unknown", "N/A", "Unknown"
            thermal_score, mech_score = "N/A", "N/A"
            rul = "N/A"
            
        # Recent Alerts
        alerts_html = ""
        recent_alerts = list(state.alert_history)[-5:]
        for a in reversed(recent_alerts):
            alerts_html += f"<li><span class='time'>{a.get('time', '')}</span> {a.get('alert', '')}</li>"
            
        if not alerts_html:
            alerts_html = "<li>No recent alerts.</li>"
            
        # Recommendation Logic
        recs_html = "<ul>"
        if health_score >= 80:
            recs_html += "<li>System operating nominally. Continue standard monitoring.</li>"
        else:
            if isinstance(rul, (int, float)) and rul < 120:
                recs_html += "<li><strong>URGENT:</strong> Remaining useful life is critically low. Schedule maintenance immediately.</li>"
            if isinstance(mech_score, (int, float)) and mech_score < 60:
                recs_html += "<li>Vibration anomalies detected. Inspect bearings and mounting hardware.</li>"
            if fault_type != "Normal Operations" and fault_type != "None detected":
                recs_html += f"<li>Fault signature matches: <strong>{fault_type}</strong>. Investigate specifically for this failure mode.</li>"
        recs_html += "</ul>"
            
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>MHARS Diagnostic Report - {machine_name}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 40px; }}
                h1 {{ color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
                h2 {{ color: #374151; margin-top: 30px; }}
                .header-meta {{ display: flex; justify-content: space-between; color: #6b7280; font-size: 0.9em; margin-bottom: 30px; }}
                .score-card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 30px; }}
                .score-value {{ font-size: 48px; font-weight: bold; color: {status_color}; }}
                .score-label {{ font-size: 14px; text-transform: uppercase; color: #6b7280; letter-spacing: 1px; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .metric-box {{ border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; }}
                .metric-title {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
                .metric-val {{ font-size: 20px; font-weight: 600; margin-top: 5px; }}
                ul.alerts {{ list-style-type: none; padding: 0; }}
                ul.alerts li {{ padding: 10px; border-bottom: 1px solid #f3f4f6; }}
                .time {{ color: #9ca3af; font-size: 0.85em; margin-right: 10px; }}
                .recs {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px 20px; margin-top: 30px; }}
                .recs ul {{ margin: 10px 0 0 0; padding-left: 20px; }}
                .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #9ca3af; }}
                @media print {{
                    body {{ padding: 0; }}
                    .score-card, .metric-box, .recs {{ break-inside: avoid; }}
                }}
            </style>
        </head>
        <body>
            <div class="header-meta">
                <span><strong>Machine:</strong> {machine_name}</span>
                <span><strong>Generated:</strong> {now}</span>
            </div>
            
            <h1>Diagnostic Report</h1>
            
            <div class="score-card">
                <div>
                    <div class="score-label">Overall System Health</div>
                    <div style="font-size: 24px; font-weight: 600; color: #374151;">{status_text}</div>
                </div>
                <div>
                    <div class="score-value">{health_score}</div>
                    <div class="score-label" style="text-align: right;">/ 100</div>
                </div>
            </div>
            
            <div class="grid">
                <div class="metric-box">
                    <div class="metric-title">Current Temperature</div>
                    <div class="metric-val">{current_temp:.1f}°C</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Estimated RUL</div>
                    <div class="metric-val">{rul_text}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Thermal Sub-score</div>
                    <div class="metric-val">{thermal_score}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Mechanical Sub-score</div>
                    <div class="metric-val">{mech_score}</div>
                </div>
            </div>
            
            <h2>Fault Signature Analysis</h2>
            <p>The AI pipeline's multi-modal fusion layer indicates the current operational profile matches: <strong>{fault_type}</strong></p>
            
            <h2>Recent Alerts</h2>
            <ul class="alerts">
                {alerts_html}
            </ul>
            
            <div class="recs">
                <strong>Actionable Recommendations</strong>
                {recs_html}
            </div>
            
            <div class="footer">
                MHARS Digital Twin System v2.0 • Phase 4 Enhanced Edition
            </div>
        </body>
        </html>
        """
        return html
