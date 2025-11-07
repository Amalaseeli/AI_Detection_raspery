from db_utils import DatabaseConnector
import pyodbc
import threading
import queue
import time

_db = DatabaseConnector()
_write_q: "queue.Queue[str]" = queue.Queue(maxsize=32)
_stop_evt = threading.Event()

def _ensure_table(cursor):
    cursor.execute("""
IF OBJECT_ID('dbo.AITransaction','U') IS NULL
    CREATE TABLE dbo.AITransaction (AIJsonTxt NVARCHAR(MAX) NOT NULL);
""")
    cursor.execute("""
IF NOT EXISTS (SELECT 1 FROM dbo.AITransaction)
    INSERT INTO dbo.AITransaction (AIJsonTxt) VALUES (N'[]');
""")

def _writer_loop():
    conn = None
    cursor = None
    last_payload = None
    while not _stop_evt.is_set():
        try:
            payload = _write_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if conn is None:
                conn = _db.create_connection()
                if conn is None:
                    time.sleep(1.0)
                    continue
                cursor = conn.cursor()
                _ensure_table(cursor)
                conn.commit()
            # Debounce identical payloads
            if payload == last_payload:
                continue
            cursor.execute("UPDATE TOP (1) dbo.AITransaction SET AIJsonTxt = ?", (payload,))
            if cursor.rowcount == 0:
                cursor.execute("INSERT INTO dbo.AITransaction (AIJsonTxt) VALUES (?)", (payload,))
            conn.commit()
            last_payload = payload
        except pyodbc.Error:
            try:
                if cursor: cursor.close()
            except Exception:
                pass
            try:
                if conn: conn.close()
            except Exception:
                pass
            conn, cursor = None, None
            time.sleep(0.5)
        except Exception:
            time.sleep(0.1)

_thread = threading.Thread(target=_writer_loop, name="DBWriter", daemon=True)
_thread.start()

def save_detected_product(json_txt: str):
    try:
        _write_q.put_nowait(json_txt)
    except queue.Full:
        pass

def clear_database():
        save_detected_product('[]')
