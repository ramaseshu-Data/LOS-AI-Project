"""Re-download the hospital length-of-stay dataset into data/ (a copy is already in the repo).

The same 100,000-record dataset is published on Kaggle
(https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft)
and in Microsoft's public GitHub repository, which needs no login:
https://github.com/microsoft/r-server-hospital-length-of-stay

Usage:  python scripts/download_data.py
"""
from pathlib import Path
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/microsoft/"
       "r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv")
DEST = Path(__file__).resolve().parents[1] / "data" / "LengthOfStay.csv"


def main() -> int:
    if DEST.exists():
        print(f"Already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed ({exc}).\nDownload LengthOfStay.csv manually from Kaggle "
              f"and save it as {DEST}")
        return 1
    print(f"Saved to {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
