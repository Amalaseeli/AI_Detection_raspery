import os
import yaml
import pyodbc
from pathlib import Path


class DatabaseConnector:
    def __init__(self):
        self.cfg = self._load_config()

    def _load_config(self):
        cfg = {
            "server": os.getenv("DB_SERVER"),
            "port": os.getenv("DB_PORT", "1433"),
            "database": os.getenv("DB_NAME"),
            "username": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "driver": os.getenv("DB_DRIVER") or "ODBC Driver 18 for SQL Server",
            "encrypt": os.getenv("DB_ENCRYPT", "yes"),
            "trust_server_certificate": os.getenv("DB_TRUST_SERVER_CERT", "yes"),
            "trusted_connection": os.getenv("DB_TRUSTED_CONNECTION"),
        }
        yml = Path(__file__).resolve().parent / "db_cred_sql.yaml"
        if yml.exists():
            try:
                with open(yml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for k in cfg:
                    if not cfg[k] and k in data:
                        val = data.get(k)
                        cfg[k] = str(val) if val is not None else cfg[k]
            except Exception as e:
                print(f"Warning: failed to read {yml}: {e}")
        return cfg

    def create_connection(self):
        try:
            server = self.cfg["server"]
            database = self.cfg["database"]
            driver = self.cfg["driver"]
            if not server or not database or not driver:
                print("DB config missing: server/database/driver.")
                return None

            parts = [
                f"DRIVER={{{driver}}}",
                f"SERVER={server},{self.cfg.get('port','1433')}",
                f"DATABASE={database}",
            ]

            # Windows integrated auth if requested
            if (self.cfg.get("trusted_connection") or "").lower() in ("1", "true", "yes"):
                parts.append("Trusted_Connection=yes")
            else:
                user = self.cfg.get("username")
                pwd = self.cfg.get("password")
                if not user or not pwd:
                    print("DB config missing username/password.")
                    return None
                parts.append(f"UID={user}")
                parts.append(f"PWD={pwd}")

            enc = (self.cfg.get("encrypt") or "yes").lower()
            parts.append(f"Encrypt={'yes' if enc in ('1','true','yes') else 'no'}")
            if (self.cfg.get('trust_server_certificate') or 'yes').lower() in ('1','true','yes'):
                parts.append("TrustServerCertificate=yes")

            parts.append("Connection Timeout=5")

            conn_str = ";".join(parts) + ";"
            return pyodbc.connect(conn_str)
        except Exception as e:
            print(f"DB connection error: {e}")
            return None

