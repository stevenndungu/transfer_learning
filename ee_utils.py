from typing import Any, Mapping, Optional, Tuple, Union

import ee
import pandas as pd
import time
from tqdm.auto import tqdm


Numeric = Union[int, float]


# --- DataFrame to EE FeatureCollection ---
def df_to_fc(df: pd.DataFrame, lat_colname: str = 'lat',
             lon_colname: str = 'lon') -> ee.FeatureCollection:
    '''
    Args
    - csv_path: str, path to CSV file that includes at least two columns for
        latitude and longitude coordinates
    - lat_colname: str, name of latitude column
    - lon_colname: str, name of longitude column

    Returns: ee.FeatureCollection, contains one feature per row in the CSV file
    '''
    # convert values to Python native types
    # see https://stackoverflow.com/a/47424340
    df = df.astype('object')

    ee_features = []
    for i in range(len(df)):
        props = df.iloc[i].to_dict()

        # oddly EE wants (lon, lat) instead of (lat, lon)
        _geometry = ee.Geometry.Point([
            props[lon_colname],
            props[lat_colname],
        ])
        ee_feat = ee.Feature(_geometry, props)
        ee_features.append(ee_feat)

    return ee.FeatureCollection(ee_features)

# --- Date Ranges for Survey Years ---
def surveyyear_to_range(survey_year: int, nl: bool = False) -> Tuple[str, str]:
    '''Returns the start and end dates for filtering satellite images for a
    survey beginning in the specified year.

    Calibrated DMSP Nighttime Lights only exist for 3 relevant date ranges,
    which Google Earth Engine filters by their start date. For more info, see
    https://www.ngdc.noaa.gov/eog/dmsp/download_radcal.html.

        DMSP range               | we use for these surveys
        -------------------------|-------------------------
        2010-01-11 to 2011-07-31 | 2006 to 2011
        2010-01-11 to 2010-12-09 | 2006 to 2011
        2005-11-28 to 2006-12-24 | 2003 to 2005

    Args
    - survey_year: int, year that survey was started
    - nl: bool, whether to use special range for night lights

    Returns
    - start_date: str, represents start date for filtering satellite images
    - end_date: str, represents end date for filtering satellite images
    '''
    if 2003 <= survey_year and survey_year <= 2005:
        start_date = '2003-1-1'
        end_date = '2005-12-31'
    elif 2006 <= survey_year and survey_year <= 2008:
        start_date = '2006-1-1'
        end_date = '2008-12-31'
        if nl:
            end_date = '2010-12-31'  # artificially extend end date for DMSP
    elif 2009 <= survey_year and survey_year <= 2011:
        start_date = '2009-1-1'
        end_date = '2011-12-31'
    elif 2012 <= survey_year and survey_year <= 2014:
        start_date = '2012-1-1'
        end_date = '2014-12-31'
    elif 2015 <= survey_year and survey_year <= 2017:
        start_date = '2015-1-1'
        end_date = '2017-12-31'
    else:
        raise ValueError(f'Invalid survey_year: {survey_year}. '
                         'Must be between 2009 and 2017 (inclusive)')
    return start_date, end_date

# --- QA MASK HANDLING (no major changes needed for C2) ---
def decode_qamask(img: ee.Image) -> ee.Image:
    '''
    Args
    - img: ee.Image, Landsat 5/7/8 image containing 'QA_PIXEL' band

    Returns
    - masks: ee.Image, contains 5 bands of masks

    Pixel QA Bit Flags (universal across Landsat 5/7/8)
    Bit  Attribute
    0    Fill
    1    Clear
    2    Water
    3    Cloud Shadow
    4    Snow
    5    Cloud
    '''
    qa = img.select('QA_PIXEL') 
    clear = qa.bitwiseAnd(2).neq(0).updateMask(qa.bitwiseAnd(2).neq(0)).rename(['pxqa_clear'])
    water = qa.bitwiseAnd(4).neq(0).updateMask(qa.bitwiseAnd(4).neq(0)).rename(['pxqa_water'])
    cloud_shadow = qa.bitwiseAnd(8).eq(0).updateMask(qa.bitwiseAnd(8).eq(0)).rename(['pxqa_cloudshadow'])
    snow = qa.bitwiseAnd(16).eq(0).updateMask(qa.bitwiseAnd(16).eq(0)).rename(['pxqa_snow'])
    cloud = qa.bitwiseAnd(32).eq(0).updateMask(qa.bitwiseAnd(32).eq(0)).rename(['pxqa_cloud'])

    #Previous code was 
    #water = qa.bitwiseAnd(4).neq(0)  # 0 = not water, 1 = water
    #water = water.updateMask(water).rename(['pxqa_water'])

    #cloud_shadow = qa.bitwiseAnd(8).eq(0)  # 0 = shadow, 1 = not shadow
    #cloud_shadow = cloud_shadow.updateMask(cloud_shadow).rename(['pxqa_cloudshadow'])

    #snow = qa.bitwiseAnd(16).eq(0)  # 0 = snow, 1 = not snow
    #snow = snow.updateMask(snow).rename(['pxqa_snow'])

    #cloud = qa.bitwiseAnd(32).eq(0)  # 0 = cloud, 1 = not cloud
    #cloud = cloud.updateMask(cloud).rename(['pxqa_cloud'])
    
    return ee.Image.cat([clear, water, cloud_shadow, snow, cloud])


def mask_qaclear(img: ee.Image) -> ee.Image:
    '''
    Args
    - img: ee.Image

    Returns
    - img: ee.Image, input image with cloud-shadow, snow, cloud, and unclear
        pixels masked out
    '''
    qam = decode_qamask(img)
    return (img
            .updateMask(qam.select('pxqa_cloudshadow'))
            .updateMask(qam.select('pxqa_snow'))
            .updateMask(qam.select('pxqa_cloud')))

# --- Add Lat/Lon Bands ---
def add_latlon(img: ee.Image) -> ee.Image:
    '''Creates a new ee.Image with 2 added bands of longitude and latitude
    coordinates named 'LON' and 'LAT', respectively
    '''
    latlon = ee.Image.pixelLonLat().select(
        opt_selectors=['longitude', 'latitude'],
        opt_names=['LON', 'LAT'])
    return img.addBands(latlon)


# --- Nightlights Composite ---
def composite_nl(year: int) -> ee.Image:
    '''Creates a median-composite nightlights (NL) image.

    Args
    - year: int, start year of survey

    Returns: ee.Image, contains a single band named 'NIGHTLIGHTS'
    '''
    if year <= 2011:
        img_col = ee.ImageCollection('NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4')
    else:
        img_col = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')

    start_date, end_date = surveyyear_to_range(year, nl=True)
    return img_col.filterDate(start_date, end_date).median().select([0], ['NIGHTLIGHTS'])

# --- TFRecord Exporter ---
def tfexporter(collection: ee.FeatureCollection, export: str, prefix: str,
               fname: str, selectors: Optional[ee.List] = None,
               dropselectors: Optional[ee.List] = None,
               bucket: Optional[str] = None) -> ee.batch.Task:
    '''Creates and starts a task to export a ee.FeatureCollection to a TFRecord
    file in Google Drive or Google Cloud Storage (GCS).

    GCS:   gs://bucket/prefix/fname.tfrecord
    Drive: prefix/fname.tfrecord

    Args
    - collection: ee.FeatureCollection
    - export: str, 'drive' for Drive, 'gcs' for GCS
    - prefix: str, folder name in Drive or GCS to export to, no trailing '/'
    - fname: str, filename
    - selectors: None or ee.List of str, names of properties to include in
        output, set to None to include all properties
    - dropselectors: None or ee.List of str, names of properties to exclude
    - bucket: None or str, name of GCS bucket, only used if export=='gcs'

    Returns
    - task: ee.batch.Task
    '''
    if dropselectors is not None:
        if selectors is None:
            selectors = collection.first().propertyNames()

        selectors = selectors.removeAll(dropselectors)

    if export == 'gcs':
        task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=fname,
            bucket=bucket,
            fileNamePrefix=f'{prefix}/{fname}',
            fileFormat='TFRecord',
            selectors=selectors)

    elif export == 'drive':
        task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=fname,
            folder=prefix,
            fileNamePrefix=fname,
            fileFormat='TFRecord',
            selectors=selectors)

    else:
        raise ValueError(f'export "{export}" is not one of ["gcs", "drive"]')

    task.start()
    return task

# --- Array Patch Sampling for Deep Learning ---
def sample_patch(point: ee.Feature, patches_array: ee.Image,
                 scale: Numeric) -> ee.Feature:
    '''Extracts an image patch at a specific point.

    Args
    - point: ee.Feature
    - patches_array: ee.Image, Array Image
    - scale: int or float, scale in meters of the projection to sample in

    Returns: ee.Feature, 1 property per band from the input image
    '''
    arrays_samples = patches_array.sample(
        region=point.geometry(),
        scale=scale,
        projection='EPSG:3857',
        factor=None,
        numPixels=None,
        dropNulls=False,
        tileScale=12)
    return arrays_samples.first().copyProperties(point)


def get_array_patches(
        img: ee.Image, scale: Numeric, ksize: Numeric,
        points: ee.FeatureCollection, export: str,
        prefix: str, fname: str,
        selectors: Optional[ee.List] = None,
        dropselectors: Optional[ee.List] = None, bucket: Optional[str] = None
        ) -> ee.batch.Task:
    '''Creates and starts a task to export square image patches in TFRecord
    format to Google Drive or Google Cloud Storage (GCS). The image patches are
    sampled from the given ee.Image at specific coordinates.

    Args
    - img: ee.Image, image covering the entire region of interest
    - scale: int or float, scale in meters of the projection to sample in
    - ksize: int or float, radius of square image patch
    - points: ee.FeatureCollection, coordinates from which to sample patches
    - export: str, 'drive' for Google Drive, 'gcs' for GCS
    - prefix: str, folder name in Drive or GCS to export to, no trailing '/'
    - fname: str, filename for export
    - selectors: None or ee.List, names of properties to include in output,
        set to None to include all properties
    - dropselectors: None or ee.List, names of properties to exclude
    - bucket: None or str, name of GCS bucket, only used if export=='gcs'

    Returns: ee.batch.Task
    '''
    kern = ee.Kernel.square(radius=ksize, units='pixels')
    patches_array = img.neighborhoodToArray(kern)

    # ee.Image.sampleRegions() does not cut it for larger collections,
    # using mapped sample instead
    samples = points.map(lambda pt: sample_patch(pt, patches_array, scale))

    # export to a TFRecord file which can be loaded directly in TensorFlow
    return tfexporter(collection=samples, export=export, prefix=prefix,
                      fname=fname, selectors=selectors,
                      dropselectors=dropselectors, bucket=bucket)

# --- Progress Bar for Export Tasks ---
def wait_on_tasks(tasks: Mapping[Any, ee.batch.Task],
                  show_probar: bool = True,
                  poll_interval: int = 20,
                  ) -> None:
    '''Displays a progress bar of task progress.

    Args
    - tasks: dict, maps task ID to a ee.batch.Task
    - show_progbar: bool, whether to display progress bar
    - poll_interval: int, # of seconds between each refresh
    '''
    remaining_tasks = list(tasks.keys())
    done_states = {ee.batch.Task.State.COMPLETED,
                   ee.batch.Task.State.FAILED,
                   ee.batch.Task.State.CANCEL_REQUESTED,
                   ee.batch.Task.State.CANCELLED}

    progbar = tqdm(total=len(remaining_tasks))
    while len(remaining_tasks) > 0:
        new_remaining_tasks = []
        for taskID in remaining_tasks:
            status = tasks[taskID].status()
            state = status['state']

            if state in done_states:
                progbar.update(1)

                if state == ee.batch.Task.State.FAILED:
                    state = (state, status['error_message'])
                elapsed_ms = status['update_timestamp_ms'] - status['creation_timestamp_ms']
                elapsed_min = int((elapsed_ms / 1000) / 60)
                progbar.write(f'Task {taskID} finished in {elapsed_min} min with state: {state}')
            else:
                new_remaining_tasks.append(taskID)
        remaining_tasks = new_remaining_tasks
        time.sleep(poll_interval)
    progbar.close()

# --- Landsat Collection 2, Level 2 Handling ---
class LandsatSR:
    def __init__(self, filterpoly: ee.Geometry, start_date: str, end_date: str) -> None:
        '''
        Args
        - filterpoly: ee.Geometry
        - start_date: str, string representation of start date
        - end_date: str, string representation of end date
        '''
        self.filterpoly = filterpoly
        self.start_date = start_date
        self.end_date = end_date
        
        #UPD by SN: Replace C01 with C02 and SR with L2 in image and collection IDs. For example, Landsat 8 OLI Tier 1 (Landsat 9 OLI-2 was not included in Collection 1):
        # Collection 1 was LANDSAT/LC08/C01/T1_SR updated to LANDSAT/LC08/C02/T1_L2 in collection 2
        # refer: https://developers.google.com/earth-engine/landsat_c1_to_c2
        # Updated to Collection 2, Level 2 datasets (C01 now deprecated)
        self.l8 = self.init_coll('LANDSAT/LC08/C02/T1_L2').map(self.rename_l8).map(self.rescale_l8)
        self.l7 = self.init_coll('LANDSAT/LE07/C02/T1_L2').map(self.rename_l57).map(self.rescale_l57)
        self.l5 = self.init_coll('LANDSAT/LT05/C02/T1_L2').map(self.rename_l57).map(self.rescale_l57)

        self.merged = self.l5.merge(self.l7).merge(self.l8).sort('system:time_start')

    def init_coll(self, name: str) -> ee.ImageCollection:
        '''
        Creates a ee.ImageCollection containing images of desired points
        between the desired start and end dates.

        Args
        - name: str, name of collection

        Returns: ee.ImageCollection
        '''
        return (ee.ImageCollection(name)
                .filterBounds(self.filterpoly)
                .filterDate(self.start_date, self.end_date))

    @staticmethod
    def rename_l8(img: ee.Image) -> ee.Image:
        # Bands for Landsat 8/9 C02 L2 (server side)           
        bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7','ST_B10', 'SR_QA_AEROSOL', 'QA_PIXEL', 'QA_RADSAT'
        ]
        newnames = ['AEROS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2','TEMP1',  'SR_QA_AEROSOL', 'QA_PIXEL', 'QA_RADSAT'
        ]

      
        return img.select(bands, newnames)

    @staticmethod
    def rescale_l8(img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image, Landsat 8 image, with bands already renamed
            by rename_l8()

        Returns
        - img: ee.Image, with bands rescaled
        '''
        # Surface reflectance and thermal scaling for Landsat 5/7 C02 L2
        opt = img.select(['AEROS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
        #opt = img.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'SWIR3'])
        therm = img.select(['TEMP1'])
        masks = img.select(['SR_QA_AEROSOL', 'QA_PIXEL', 'QA_RADSAT'])

        opt = opt.multiply(0.0000275).add(-0.2) # Collection 1 used a 0.0001 scale factor. Collection 2 uses a 2.75e-05 scale factor and -0.2 offset.
        therm = therm.multiply(0.00341802).add(149.0)  # C2 L2 thermal scaling

        scaled = ee.Image.cat([opt, therm, masks]).copyProperties(img)
        # system properties are not copied
        scaled = scaled.set('system:time_start', img.get('system:time_start'))
        return scaled

    @staticmethod
    def rename_l57(img: ee.Image) -> ee.Image:
        # Bands for Landsat 5/7 C02 L2
        bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'SR_ATMOS_OPACITY', 'SR_CLOUD_QA', 'QA_PIXEL', 'QA_RADSAT']

        newnames = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2','TEMP1','SR_ATMOS_OPACITY', 'SR_CLOUD_QA', 'QA_PIXEL', 'QA_RADSAT' ]


        return img.select(bands, newnames)

    @staticmethod
    def rescale_l57(img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image, Landsat 5/7 image, with bands already renamed
            by rename_157()

        Returns
        - img: ee.Image, with bands rescaled
        '''
        opt = img.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
        atmos = img.select(['SR_ATMOS_OPACITY'])
        therm = img.select(['TEMP1'])
        masks = img.select(['SR_CLOUD_QA', 'QA_PIXEL', 'QA_RADSAT'])

        opt = opt.multiply(0.0000275).add(-0.2) # Collection 1 used a 0.0001 scale factor. Collection 2 uses a 2.75e-05 scale factor and -0.2 offset.
        atmos = atmos.multiply(0.001)
        therm = therm.multiply(0.00341802).add(149.0)  # C2 L2 thermal scaling

        scaled = ee.Image.cat([opt, therm, masks, atmos]).copyProperties(img)
        # system properties are not copied
        scaled = scaled.set('system:time_start', img.get('system:time_start'))
        return scaled
