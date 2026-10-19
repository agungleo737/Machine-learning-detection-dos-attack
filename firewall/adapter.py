from datetime import datetime
import json
import os
import platform
import subprocess


def buat_perintah_blokir(src_ip, sistem=None):
    os_target = (sistem or platform.system()).lower()
    nama_rule = f"ML_DoS_Block_{src_ip}"
    if "windows" in os_target:
        return [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={nama_rule}", "dir=in", "action=block", f"remoteip={src_ip}"
        ]
    return ["iptables", "-A", "INPUT", "-s", str(src_ip), "-j", "DROP"]


def jalankan_aksi_firewall(daftar_ip_blokir, dry_run=True, log_path="firewall_action_log.json"):
    ip_unik = sorted(set(str(ip).strip() for ip in daftar_ip_blokir if not str(ip).startswith("flow_")))
    catatan = []
    for ip in ip_unik:
        perintah = buat_perintah_blokir(ip)
        entri = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "src_ip": ip,
            "mode": "DRY_RUN" if dry_run else "ACTIVE",
            "command": " ".join(perintah),
            "status": "SIMULATED",
        }
        if not dry_run:
            res = subprocess.run(perintah, capture_output=True, text=True, check=False)
            entri["status"] = "APPLIED" if res.returncode == 0 else f"FAILED ({res.stderr.strip()})"
        catatan.append(entri)
    if catatan and log_path:
        riwayat = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    riwayat = json.load(f)
            except Exception:
                riwayat = []
        riwayat.extend(catatan)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(riwayat, f, indent=2)
    return catatan