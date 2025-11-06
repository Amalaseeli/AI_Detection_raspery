import os
import yaml
import pyodbc
from pathlib import Path


class DatabaseConnector:
    def __init__(self):
        self.cfg = self._load_config()

    def _load_config(self):
        # Defaults
        cfg = {
            "server": None,
            "port": "1433",
            "database": None,
            "username": None,
            "password": None,
            "driver": "ODBC Driver 18 for SQL Server",
            "encrypt": "yes",
            "trust_server_certificate": "yes",
            "trusted_connection": None,
        }

        # Load YAML and override defaults
        yml = Path(__file__).resolve().parent / "db_cred_sql.yaml"
        if yml.exists():
            try:
                with open(yml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for k in cfg:
                    if k in data and data[k] is not None:
                        v = data[k]
                        cfg[k] = str(v) if not isinstance(v, str) else v
            except Exception as e:
                print(f"Warning: failed to read {yml}: {e}")

        # Environment variables take highest precedence
        env_map = {
            "server": "DB_SERVER",
            "port": "DB_PORT",
            "database": "DB_NAME",
            "username": "DB_USER",
            "password": "DB_PASSWORD",
            "driver": "DB_DRIVER",
            "encrypt": "DB_ENCRYPT",
            "trust_server_certificate": "DB_TRUST_SERVER_CERT",
            "trusted_connection": "DB_TRUSTED_CONNECTION",
        }
        for k, envk in env_map.items():
            val = os.getenv(envk)
            if val is not None and str(val).strip() != "":
                cfg[k] = val

        return cfg

    def create_connection(self):
        try:
            server = self.cfg["server"]
            database = self.cfg["database"]
            driver = self.cfg["driver"]
            if not server or not database or not driver:
                print("DB config missing: server/database/driver.")
                return None

            drv_lower = (driver or "").lower()
            is_freetds = ("freetds" in drv_lower) or ("tdsodbc" in drv_lower)

            if is_freetds:
                parts = [
                    f"DRIVER={{{driver}}}",
                    f"SERVER={server}",
                    f"PORT={self.cfg.get('port','1433')}",
                    f"DATABASE={database}",
                ]

                # FreeTDS typically uses SQL auth
                user = self.cfg.get("username")
                pwd = self.cfg.get("password")
                if not user or not pwd:
                    print("DB config missing username/password for FreeTDS.")
                    return None
                parts.append(f"UID={user}")
                parts.append(f"PWD={pwd}")

                # Modern SQL Server versions work best with TDS 8.0
                parts.append("TDS_Version=8.0")
                parts.append("ClientCharset=UTF-8")
                parts.append("Connection Timeout=5")

                conn_str = ";".join(parts) + ";"
                return pyodbc.connect(conn_str)
            else:
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
            try:
                # Help diagnose by printing installed ODBC drivers
                print(f"Available ODBC drivers: {pyodbc.drivers()}")
            except Exception:
                pass
            return None
