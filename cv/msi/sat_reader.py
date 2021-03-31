import os
import glob

import os.path as op
import numpy as np

__all__=["SatReader", "BigEarthReader", "SenENVIReader", "split_bands"]

class SatReader():
    """ Sentinel-2 Satellite Reader Base Class.

        Use gdal or rasterio to load each bands of a L1A level Sentinel-2
        HSI. Support bands status check and read.

    Args:
        root: The root directory of multiple sentinel-2 folders.
    """
    def __init__(self, root):
        self.root = root

        # Checks the existence of required python packages
        self.gdal_existed = self.rasterio_existed = self.georasters_existed = False
        try:
            from osgeo import gdal
            self.gdal_existed = True
            print('INFO: GDAL package will be used to read GeoTIFF files')
        except ImportError:
            try:
                import rasterio
                self.rasterio_existed = True
                print('INFO: rasterio package will be used to read GeoTIFF files')
            except ImportError:
                print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')

    # Reset the root directory
    def reset_root(self, root):
        self.root = root

    # Check the status of each bands
    def check_bands(self, patch_name):
        self._read(patch_name, ret=False)

    # Read all bands and return
    def read_file(self, patch_name):
        return self._read(patch_name, ret=True)
        

class BigEarthReader(SatReader):
    """ The reader for BigEarthNet dataset.
    """
    def __init__(self, root):
        super(BigEarthReader, self).__init__(root)

        # Spectral band names to read related GeoTIFF files
        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                           'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    def _read(self, patch_name, ret=False):
        # Reads spectral bands of all patches whose folder names are populated before
        bands = []
        for band_name in self.band_names:
            # First finds related GeoTIFF path and reads values as an array
            band_path = os.path.join(
                self.root, patch_name, patch_name + '_' + band_name + '.tif')
            if self.gdal_existed:
                band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = raster_band.ReadAsArray()
            elif self.rasterio_existed:
                band_ds = rasterio.open(band_path)
                band_data = band_ds.read(1)
            if ret:
                bands.append(band_data)
            else:
                # band_data keeps the values of band band_name for the patch patch_name
                print('INFO: band', band_name, 'of patch', patch_name,
                        'is ready with size', band_data.shape)
        if ret:
            return bands

class SenENVIReader(SatReader):
    """ The reader for BigEarthNet dataset.
    """
    def __init__(self, root):
        super(SenENVIReader, self).__init__(root)

        # Spectral band names to read related GeoTIFF files
        self.band_names = ['B1', 'B2', 'B3', 'B4', 'B5',
              'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    def _read(self, patch_name, ret=False):
        # Reads spectral bands of all patches whose folder names are populated before
        bands = []
        if patch_name.endswith('.data'):
            subroot = op.join(self.root, patch_name)
        else:
            subroot = glob.glob(op.join(self.root, patch_name, '*.data'))[0]

        for band_name in self.band_names:
            # First finds related GeoTIFF path and reads values as an array
            band_path = op.join(subroot, (band_name+'.img'))
            if self.gdal_existed:
                band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = raster_band.ReadAsArray()
            elif self.rasterio_existed:
                band_ds = rasterio.open(band_path)
                band_data = band_ds.read(1)
            if ret:
                bands.append(band_data)
            else:
                # band_data keeps the values of band band_name for the patch patch_name
                print('INFO: band', band_name, 'of patch', patch_name,
                        'is ready with size', band_data.shape)
        if ret:
            return bands

# Split sentinel-2 HSI bands into 10m, 20m, and 60m resolution groups
def split_bands(bands):
    d10 = np.array([bands[i] for i in (1,2,3,7)])
    d20 = np.array([bands[i] for i in (4,5,6,8,10,11)])
    d60 = np.array([bands[i] for i in (0,9)])
    return d10.transpose(1,2,0), d20.transpose(1,2,0), d60.transpose(1,2,0)