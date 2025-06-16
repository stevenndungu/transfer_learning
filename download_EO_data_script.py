import os
import time
import math
import pathlib
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple
import tensorflow as tf
from google.cloud import storage
import ee
import geemap
# ee_utils should be in the working folder
os.chdir('/dbfs/mnt/raw/DataScientistSDSData/TLSED_test')
import ee_utils 
# =============== CONFIGURATION ====================
MS_BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
PROJECTION = 'EPSG:3857'
SCALE = 30
EXPORT_TILE_RADIUS = 127
CHUNK_SIZE = 1

# ============= AUTHENTICATION =====================
def init_earth_engine(service_account: str, key_path: str):
    credentials = ee.ServiceAccountCredentials(service_account, key_path)
    ee.Initialize(credentials)
    print('Earth Engine authenticated!')

def get_storage_client(key_path: str):
    return storage.Client.from_service_account_json(key_path)

# ============= EXPORT IMAGES ======================
def export_images(
        df: pd.DataFrame,
        country: str,
        year: int,
        export_folder: str,
        chunk_size: Optional[int],
        ee_utils,
        BUCKET,
        MS_BANDS,
        SCALE,
        EXPORT_TILE_RADIUS,
        EXPORT,
        PROJECTION
        ) -> Dict[Tuple[Any], ee.batch.Task]:
    subset_df = df[(df['country'] == country) & (df['year'] == year)].reset_index(drop=True)
    if chunk_size is None:
        num_chunks = 1
    else:
        num_chunks = int(math.ceil(len(subset_df) / chunk_size))
    tasks = {}
    for i in range(num_chunks):
        chunk_slice = slice(i * chunk_size, (i+1) * chunk_size - 1)
        fc = ee_utils.df_to_fc(subset_df.loc[chunk_slice, :])
        try:
            start_date, end_date = ee_utils.surveyyear_to_range(year, nl=True)
        except ValueError as e:
            #print(f"Skipping {country}, {year}: {e}")
            continue
        roi = fc.geometry()
        imgcol = ee_utils.LandsatSR(roi, start_date=start_date, end_date=end_date).merged
        imgcol = imgcol.map(ee_utils.mask_qaclear).select(MS_BANDS)
        img = imgcol.median()
        img = ee_utils.add_latlon(img)
        img = img.addBands(ee_utils.composite_nl(year))
        fname = f'{country}_{year}_{i:02d}'
        tasks[(export_folder, country, year, i)] = ee_utils.get_array_patches(
            img=img, scale=SCALE,
            ksize=EXPORT_TILE_RADIUS,
            points=fc,
            export=EXPORT,
            prefix=export_folder,
            fname=fname,
            bucket=BUCKET)
    return tasks

# =========== DOWNLOAD BLOBS FROM GCS ==============
def download_blob_if_missing(blob, prefix, dest_dir):
    rel_path = blob.name[len(prefix):]
    local_path = os.path.join(dest_dir, rel_path)
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(local_path):
        blob.download_to_filename(local_path)
        return True
    else:
        #print(f"File exists: {local_path}, skipping.")
        return False

#Uncomment to try running on commandline.
# # ================== MAIN SCRIPT ===================
# def main(args):
#     # Authenticate EE and GCS
#     init_earth_engine(args.service_account, args.key_path)
#     client = get_storage_client(args.key_path)

#     # Import utils AFTER you have correct path
#     import ee_utils  

#     # directories
#     os.makedirs(args.dest_dir, exist_ok=True)

#     # DHS data
#     dhs_df = pd.read_csv(args.dhs_csv_path, float_precision='high', index_col=False).head(1100)
#     dhs_df = dhs_df.tail(100)
#     dhs_surveys = list(dhs_df.groupby(['country', 'year']).groups.keys())

#     # Export images and wait for completion
#     start_time = time.time()
#     for country, year in dhs_surveys:
#         print(f"Starting export for {country}, {year}...")
#         tasks = export_images(
#             df=dhs_df,
#             country=country,
#             year=year,
#             export_folder=args.export_folder,
#             chunk_size=args.chunk_size,
#             ee_utils=ee_utils,
#             BUCKET=args.bucket,
#             MS_BANDS=MS_BANDS,
#             SCALE=SCALE,
#             EXPORT_TILE_RADIUS=EXPORT_TILE_RADIUS,
#             EXPORT=args.export,
#             PROJECTION=PROJECTION
#         )
#         print(f"Waiting on export tasks for {country}, {year}...")
#         ee_utils.wait_on_tasks(tasks, poll_interval=60)
#         print(f"Finished export for {country}, {year}.")
#         time.sleep(100)  # API quota protection

#     elapsed_time = time.time() - start_time
#     print(f"Total time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

#     # Download files from bucket
#     blobs = [blob for blob in client.list_blobs(args.bucket, prefix=args.prefix) if blob.name.endswith(".gz")]
#     print(f"Total files to check: {len(blobs)}")
#     downloaded_count = 0
#     skipped_count = 0
#     for blob in tqdm(blobs, desc="Downloading blobs"):
#         if download_blob_if_missing(blob, args.prefix, args.dest_dir):
#             downloaded_count += 1
#         else:
#             skipped_count += 1
#     print(f"Downloaded: {downloaded_count}, Skipped: {skipped_count}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--service-account', required=True, help='Service account email')
#     parser.add_argument('--key-path', required=True, help='Path to the GCP JSON key')
#     parser.add_argument('--bucket', default='dhs_tfrecords_raw_v2', help='GCS Bucket name')
#     parser.add_argument('--prefix', default='dhs_tfrecords_raw_v2/', help='Prefix inside bucket')
#     parser.add_argument('--dest-dir', default='/dbfs/mnt/raw/DataScientistSDSData/TLSED_test/raw', help='Destination dir')
#     parser.add_argument('--export', default='gcs', help='Export destination (gcs)')
#     parser.add_argument('--export-folder', default='dhs_tfrecords_raw_v2', help='Export folder in bucket')
#     parser.add_argument('--dhs-csv-path', default='/dbfs/mnt/raw/DataScientistSDSData/TLSED/dhs_clusters.csv', help='CSV path')
#     parser.add_argument('--chunk-size', type=int, default=1, help='Chunk size for export')
#     args = parser.parse_args()
#     main(args)


#TODO
# 1. Create requirements.txt file
# 2. run: python your_script.py \
#    --service-account "ee-transfer-learning@rare-deployment-461721-a2.iam.gserviceaccount.com" \
#   --key-path "/dbfs/mnt/raw/DataScientistSDSData/transfer_learning_rare-deployment-461721-a2-0ba3a6ab5a71.json"


######## Test #############
service_account = 'ee-transfer-learning@rare-deployment-461721-a2.iam.gserviceaccount.com'
key_path = '/dbfs/mnt/raw/DataScientistSDSData/transfer_learning_rare-deployment-461721-a2-0ba3a6ab5a71.json'
dest_dir = '/dbfs/mnt/raw/DataScientistSDSData/TLSED_test/raw'
export = 'gcs'
export_folder = 'dhs_tfrecords_raw_v2'
dhs_csv_path = '/dbfs/mnt/raw/DataScientistSDSData/TLSED/dhs_clusters.csv'
chunk_size = 1
bucket = 'dhs_tfrecords_raw_v2'
prefix = 'dhs_tfrecords_raw_v2/'
## Authenticate EE and GCS
init_earth_engine(service_account, key_path)
client = get_storage_client(key_path)

# directories
os.makedirs(dest_dir, exist_ok=True)

# DHS data
dhs_df = pd.read_csv(dhs_csv_path, float_precision='high', index_col=False).head(11200)
dhs_df = dhs_df.tail(10000)
dhs_surveys = list(dhs_df.groupby(['country', 'year']).groups.keys())

# Export images and wait for completion
start_time = time.time()
for country, year in dhs_surveys:
    print(f"Starting export for {country}, {year}...")
    tasks = export_images(
        df=dhs_df,
        country=country,
        year=year,
        export_folder=export_folder,
        chunk_size=chunk_size,
        ee_utils=ee_utils,
        BUCKET=bucket,
        MS_BANDS=MS_BANDS,
        SCALE=SCALE,
        EXPORT_TILE_RADIUS=EXPORT_TILE_RADIUS,
        EXPORT=export,
        PROJECTION=PROJECTION
    )
    print(f"Waiting on export tasks for {country}, {year}...")
    ee_utils.wait_on_tasks(tasks, poll_interval=60)
    print(f"Finished export for {country}, {year}.")
    time.sleep(100)  # API quota protection

elapsed_time = time.time() - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# Download files from bucket
blobs = [blob for blob in client.list_blobs(bucket, prefix=prefix) if blob.name.endswith(".gz")]
print(f"Total files to check: {len(blobs)}")
downloaded_count = 0
skipped_count = 0
for blob in tqdm(blobs, desc="Downloading blobs"):
    if download_blob_if_missing(blob, prefix, dest_dir):
        downloaded_count += 1
    else:
        skipped_count += 1
print(f"Downloaded: {downloaded_count}, Skipped: {skipped_count}")