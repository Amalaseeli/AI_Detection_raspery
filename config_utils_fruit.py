import os
import sys

APP_NAME = "SCSF"

def resource_path(rel_path:str) -> str:
    """
    Get the absolute path to a resource, works for dev and for PyInstaller.
    This is for read-only files packaged with the application.

    When you package a Python app with PyInstaller, 
    your data files (images, templates, etc.) are bundled and extracted to a temporary folder at runtime.
    Their location isn’t the same as during normal development.
    This helper returns the correct absolute path to a resource in both cases.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return(os.path.join(sys._MEIPASS, rel_path))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

def get_data_dir() -> str:
    """
    Returns a writable directory for user data (e.g., roi.yaml).
    For a packaged app, this will be in the user's AppData folder.
    For development, it will be the script's directory.
    """
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            data_dir = os.path.join(os.environ["APPDATA"], APP_NAME)
        else:
            data_dir = os.path.join(os.path.expanduser("~"), ".config", APP_NAME)
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    else:
        return os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = resource_path(os.path.join("models", "fruit_saved_model", "fruit_float32.tflite"))

DATA_DIR = get_data_dir()
ROI_PATH = os.path.join(DATA_DIR, "roi.txt")


product_data = {
    'Apple': {'price': '£1.25', 'barcode': '204001'},
    'Avocado': {'price': '£1.05', 'barcode': '204014'},    
    'Banana': {'price': '£0.89', 'barcode': '204004'},
    'Orange': {'price': '£0.79', 'barcode': '204003'},
    'Passion Fruit': {'price': '75p', 'barcode': '204013'},
    'Tomato': {'price': '£1.20', 'barcode': '204010'},
  

}

classNames = ['Apple', 'Avocado', 'Banana', 'Orange', 'Passion Fruit', 'Tomato']
