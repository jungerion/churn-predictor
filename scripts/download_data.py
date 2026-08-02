"""One-off script to fetch the Telco churn dataset and place it where the
pipeline expects it. Not part of the production app — run manually, once."""
import shutil
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("Downloaded to:", path)

src_csv = Path(path) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
dest_csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"
dest_csv.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src_csv, dest_csv)
print("Copied to:", dest_csv)