from db_utils import DatabaseConnector
import pyodbc

db = DatabaseConnector()


def save_detected_product(json_txt):
    connection = db.create_connection()
    if connection is None:
        print("Warning: Database connection not established. Skipping save.")
        return
    try:
        cursor = None
        cursor = connection.cursor()

        # Check if table exists
        cursor.execute(
            """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'AITransaction'
        """
        )

        # If the table does not exist
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                """
            CREATE TABLE AITransaction (
                AIJsonTxt NVARCHAR(MAX) NOT NULL
            )
            """
            )
            # Immediately seed with current payload so a row always exists
            cursor.execute("INSERT INTO AITransaction (AIJsonTxt) VALUES (?)", (json_txt,))
            connection.commit()
        else:
            # Ensure at least one row exists; insert if table is empty
            cursor.execute("SELECT COUNT(*) FROM AITransaction")
            row_count = cursor.fetchone()[0]
            if row_count == 0:
                cursor.execute("INSERT INTO AITransaction (AIJsonTxt) VALUES (?)", (json_txt,))
            else:
                # Update a single row to hold the latest payload
                cursor.execute("UPDATE TOP (1) AITransaction SET AIJsonTxt = ?", (json_txt,))
            connection.commit()
    except pyodbc.Error as e:
        print(f"Error: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def fetch_barcode_with_product_name(product_name):
    connection = db.create_connection()
    if connection is None:
        return None
    cursor = connection.cursor()
    try:
        cursor.execute(
            """SELECT Barcode FROM dbo.Item_Barcode WHERE Description = ?""",
            (product_name,),
        )
        barcode = cursor.fetchone()
        if barcode:
            return barcode[0]
        else:
            print("No barcode found with the given product.")
            return None
    except Exception as e:
        print(f"Error fetching product {e}")


def clear_database():
    connection = db.create_connection()
    if connection is None:
        return
    try:
        save_detected_product('[]')
    except pyodbc.Error as e:
        print(f"Error: {e}")
    finally:
        connection.close()

