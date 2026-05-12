import os
import random

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class SystemHealthMonitor:
    """Reads real hardware metrics via psutil (CPU) or generates realistic simulated physics metrics (Motor, Server, Engine)."""
    
    @staticmethod
    def get_cpu_temp_fallback():
        if PSUTIL_AVAILABLE:
            try:
                cpu_pct = psutil.cpu_percent(interval=0.1)
                return round(36.0 + (cpu_pct * 0.48), 1)
            except Exception:
                pass
        try:
            load_1min = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 4
            load_pct = min(100, (load_1min / cpu_count) * 100)
            return round(36.0 + (load_pct * 0.48), 1)
        except Exception:
            return 42.0

    @staticmethod
    def _snapshot_mac_cpu() -> dict:
        """Returns real host Mac metrics (Machine 0)"""
        cpu_pct = 0.0
        cpu_cores = os.cpu_count() or 4
        if PSUTIL_AVAILABLE:
            cpu_pct = psutil.cpu_percent(interval=0.1)
        else:
            try:
                load_1min = os.getloadavg()[0]
                cpu_pct = min(100.0, (load_1min / cpu_cores) * 100)
            except:
                pass
                
        cpu_status = "critical" if cpu_pct > 90 else "warning" if cpu_pct > 75 else "healthy"
        cpu_verdict = "Critical load" if cpu_status == "critical" else "High load" if cpu_status == "warning" else "Normal load"

        mem_pct = 0.0
        if PSUTIL_AVAILABLE:
            mem_pct = psutil.virtual_memory().percent
        mem_status = "critical" if mem_pct > 90 else "warning" if mem_pct > 80 else "healthy"
        mem_verdict = "Critical RAM" if mem_status == "critical" else "High RAM" if mem_status == "warning" else "Sufficient RAM"

        disk_pct = 0.0
        if PSUTIL_AVAILABLE:
            try: disk_pct = psutil.disk_usage('/').percent
            except: pass
        disk_status = "critical" if disk_pct > 95 else "warning" if disk_pct > 85 else "healthy"
        disk_verdict = "Disk almost full" if disk_status == "critical" else "Consider cleanup" if disk_status == "warning" else "Adequate space"

        batt_pct = 100
        batt_plugged = True
        if PSUTIL_AVAILABLE and hasattr(psutil, "sensors_battery"):
            batt = psutil.sensors_battery()
            if batt:
                batt_pct = round(batt.percent)
                batt_plugged = batt.power_plugged
        batt_status = "warning" if (not batt_plugged and batt_pct < 40) else "healthy"
        batt_verdict = "Charging" if batt_plugged else "Discharging"

        net_val = "Idle"
        if PSUTIL_AVAILABLE:
            net = psutil.net_io_counters()
            net_val = f"↑{round(net.bytes_sent/(1024**2))} ↓{round(net.bytes_recv/(1024**2))} MB"

        score = 1.0
        if cpu_pct > 90: score -= 0.3
        elif cpu_pct > 75: score -= 0.1
        if mem_pct > 90: score -= 0.3
        elif mem_pct > 80: score -= 0.1

        score = max(0.0, score)
        overall_status = "critical" if score < 0.4 else "warning" if score < 0.7 else "healthy"

        summary = f"System is {overall_status}. CPU load is {cpu_pct:.1f}%. RAM is at {mem_pct:.1f}%. Disk usage is {disk_pct:.1f}%."

        return {
            "title": "SYSTEM HEALTH (Live macOS)",
            "overall_score": round(score * 100),
            "ai_summary": summary,
            "components": [
                {"icon": "🖥️", "name": "CPU", "val": f"{cpu_pct}%", "verdict": cpu_verdict, "status": cpu_status},
                {"icon": "💾", "name": "RAM", "val": f"{mem_pct}%", "verdict": mem_verdict, "status": mem_status},
                {"icon": "💿", "name": "Disk", "val": f"{disk_pct}%", "verdict": disk_verdict, "status": disk_status},
                {"icon": "🔋", "name": "Battery", "val": f"{batt_pct}%", "verdict": batt_verdict, "status": batt_status},
                {"icon": "🌐", "name": "Network", "val": net_val, "verdict": "Good", "status": "healthy"}
            ]
        }

    @staticmethod
    def _snapshot_motor() -> dict:
        rpm = random.randint(1420, 1460)
        bearing = random.uniform(5.0, 15.0)
        voltage = random.uniform(215.0, 225.0)
        torque = random.uniform(40.0, 60.0)
        
        bearing_status = "warning" if bearing > 12.0 else "healthy"
        torque_status = "warning" if torque > 55.0 else "healthy"
        
        score = 100
        if bearing_status == "warning": score -= 20
        if torque_status == "warning": score -= 15
        
        summary = f"Motor is operating normally. RPM is stable at {rpm}. Bearing wear is {bearing:.1f}%."
        if bearing_status == "warning":
            summary = f"Attention: Bearing wear is elevated ({bearing:.1f}%). Torque is {torque:.1f} Nm. Inspect soon."

        return {
            "title": "MOTOR HEALTH (Simulated)",
            "overall_score": score,
            "ai_summary": summary,
            "components": [
                {"icon": "⚙️", "name": "RPM Speed", "val": str(rpm), "verdict": "Stable", "status": "healthy"},
                {"icon": "🔧", "name": "Bearing Wear", "val": f"{bearing:.1f}%", "verdict": "Check recommended" if bearing_status=="warning" else "Within limits", "status": bearing_status},
                {"icon": "⚡", "name": "Voltage", "val": f"{voltage:.1f}V", "verdict": "Optimal", "status": "healthy"},
                {"icon": "🏋️", "name": "Torque", "val": f"{torque:.1f} Nm", "verdict": "High load" if torque_status=="warning" else "Normal", "status": torque_status},
                {"icon": "🌡️", "name": "Coolant", "val": "Good", "verdict": "Flowing", "status": "healthy"}
            ]
        }

    @staticmethod
    def _snapshot_server() -> dict:
        cpu = random.uniform(40.0, 85.0)
        ram = random.uniform(60.0, 95.0)
        iops = random.randint(12000, 18000)
        
        cpu_status = "warning" if cpu > 80.0 else "healthy"
        ram_status = "critical" if ram > 90.0 else "healthy"
        
        score = 100
        if cpu_status == "warning": score -= 15
        if ram_status == "critical": score -= 30
        
        summary = f"Server rack is stable. CPU array at {cpu:.1f}%. IOPS normal."
        if ram_status == "critical":
            summary = f"Warning: Server RAM is nearly exhausted ({ram:.1f}%). Prepare to spin up additional nodes."

        return {
            "title": "SERVER HEALTH (Simulated)",
            "overall_score": score,
            "ai_summary": summary,
            "components": [
                {"icon": "🖥️", "name": "CPU Array", "val": f"{cpu:.1f}%", "verdict": "High load" if cpu_status=="warning" else "Normal", "status": cpu_status},
                {"icon": "💾", "name": "Active RAM", "val": f"{ram:.1f}%", "verdict": "Critical" if ram_status=="critical" else "Normal", "status": ram_status},
                {"icon": "🌡️", "name": "Rack Temp", "val": f"{random.uniform(22, 28):.1f}°C", "verdict": "Optimal cooling", "status": "healthy"},
                {"icon": "🖧", "name": "Bandwidth", "val": f"{random.uniform(4.0, 8.5):.1f} Gbps", "verdict": "High traffic", "status": "healthy"},
                {"icon": "💿", "name": "Storage", "val": f"{iops} IOPS", "verdict": "Good", "status": "healthy"}
            ]
        }

    @staticmethod
    def _snapshot_engine() -> dict:
        rpm = random.randint(3200, 3800)
        oil_pres = random.uniform(28.0, 45.0)
        fuel = random.uniform(12.0, 15.0)
        
        oil_status = "critical" if oil_pres < 30.0 else "healthy"
        
        score = 100
        if oil_status == "critical": score -= 40
        
        summary = f"Engine operating within nominal parameters. RPM {rpm}, Oil Pressure {oil_pres:.1f} PSI."
        if oil_status == "critical":
            summary = f"CRITICAL: Oil pressure is dropping dangerously low ({oil_pres:.1f} PSI). Risk of engine stall."

        return {
            "title": "ENGINE HEALTH (Simulated)",
            "overall_score": score,
            "ai_summary": summary,
            "components": [
                {"icon": "⚙️", "name": "RPM Speed", "val": str(rpm), "verdict": "Stable", "status": "healthy"},
                {"icon": "🛢️", "name": "Oil Pressure", "val": f"{oil_pres:.1f} PSI", "verdict": "Low pressure" if oil_status=="critical" else "Good", "status": oil_status},
                {"icon": "🔥", "name": "Exhaust", "val": f"{random.uniform(400, 450):.0f}°C", "verdict": "Nominal", "status": "healthy"},
                {"icon": "⛽", "name": "Fuel Flow", "val": f"{fuel:.1f} L/h", "verdict": "Steady", "status": "healthy"},
                {"icon": "📳", "name": "Vibration", "val": "0.02g", "verdict": "Smooth", "status": "healthy"}
            ]
        }

    @staticmethod
    def snapshot(machine_id: int) -> dict:
        """Returns dynamic snapshot based on machine ID."""
        if machine_id == 0:
            return SystemHealthMonitor._snapshot_mac_cpu()
        elif machine_id == 1:
            return SystemHealthMonitor._snapshot_motor()
        elif machine_id == 2:
            return SystemHealthMonitor._snapshot_server()
        else:
            return SystemHealthMonitor._snapshot_engine()
