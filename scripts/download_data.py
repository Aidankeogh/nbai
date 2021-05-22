import zipfile
import urllib3
import shutil
import os

http = urllib3.PoolManager()
url = "https://pbpstats.s3.amazonaws.com/data.zip"
zip_path = "cache/data.zip"
extract_dir = "cache"
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with open(zip_path, 'wb') as out:
    r = http.request('GET', url, preload_content=False)
    shutil.copyfileobj(r, out)
print("Data downloaded, extracting...")
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(extract_dir)
print("Done!")