"""
    Geography related functions. Mainly the interaction between raster and shapefile.
"""

import numpy as np
import json
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import Affine, MemoryFile
from contextlib import contextmanager
from copy import deepcopy

def bbox(shp):
    """ Compute the bounding box of a certain shapefile.
    """
    piece = np.array([i.bounds for i in shp['geometry']])
    minx = piece[:, 0].min()
    miny = piece[:, 1].min()
    maxx = piece[:, 2].max()
    maxy = piece[:, 3].max()
    return minx, miny, maxx, maxy


def edge_length(shp):
    """ Compute the x and y edge length for a ceratin shapefile.
    """
    minx, miny, maxx, maxy = bbox(shp)
    return round(maxx-minx, 3), round(maxy-miny, 3)


def shape2latlong(shp):
    """ Turn the shapefile unit from meters/other units to lat/long.
    """
    return shp.to_crs(epsg=4326)


def bbox_latlong(shp):
    """ Compute the latitude-longitude bounding box of a certain shapefile.
    """
    shp = shape2latlong(shp)
    return bbox(shp)


def bbox_polygon(shp):
    """ Return the rectangular Polygon bounding box of a certain shapefile.
    """
    minx, miny, maxx, maxy = bbox(shp)
    return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])


def merge_polygon(shp):
    """ Merge a shapefile to one single polygon.
    """
    return shp.dissolve(by='Id').iloc[0].geometry


def polygon2geojson(polygon):
    """ Turn a polygon to a geojson format string.
        This is used for rasterio mask operation.
    """
    if type(polygon) == Polygon:
        polygon = gpd.GeoSeries(polygon)
    return [json.loads(polygon.to_json())['features'][0]['geometry']]


def sen2rgb(img, scale=30):
    """ Turn the 12 channel float32 format sentinel-2 images to a RGB uint8 image. 
    """
    return (img[(3, 2, 1), ]/256*scale).astype(np.uint8)


def cropbyshp(raster, shp, boundary_clip=True, ret_transform=False):
    """ Crop a raster using a shapefile.
    """
    # Reproject the shapefile to the same crs of raster.
    shp = shp.to_crs({"init": str(raster.crs)})
    # Compute the rectangular Polygon bounding box of a certain shapefile.
    if boundary_clip:
        bbpoly = bbox_polygon(shp)
    else:
        bbpoly = shp
    # Execute the mask operation.
    out_img, out_transform = mask(dataset=raster, shapes=polygon2geojson(
        bbpoly), crop=True, all_touched=True)
    if ret_transform:
        return out_img, out_transform
    else:
        return out_img


def write_raster(raster, path):
    """ Write a created raster object to file.
    """
    with rio.open(
        path,
        'w',
        **raster.meta
    ) as dst:
        dst.write(raster.read())


def sen_reproject(src, dst_crs, out_path):
    """ Reproject a raster to a new CRS coordinate, and save it in out_path.
    Args:
        src: Input raster.
        dst_crs: Target CRS. String.
        out_path: The path of the output file.
    """
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(out_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.cubic)


def mask_A_by_B(A, B):
    """ Generate a mask from B, and applied it to A.
        All 0 values are excluded.
    """
    mask = B.sum(axis=0) > 1e-3
    masked_A = mask*A
    return masked_A


def adjust_A_by_B(A, B):
    """ Adjust image A's each band by the corresponding B's band.

        The output is A with each band have the same mean and std
        value of the corresponding B's band.

        A/B: Shape: [C, H, W]
    """
    temp = deepcopy(A)
    for i in range(A.shape[0]):
        ap = A[i].flatten()
        ap = ap[ap > 1e-3]
        bp = B[i].flatten()
        bp = bp[bp > 1e-3]
        temp[i] = (A[i]-ap.mean())/ap.std()
        temp[i] = temp[i]*bp.std()+bp.mean()
    temp = mask_A_by_B(temp, A)
    return temp

# use context manager so DatasetReader and MemoryFile get cleaned up automatically


@contextmanager
def resample_raster(raster, scale=2):
    """ Resample the raster without changing the geo transform coverage.

    Example:
        with rasterio.open(dat) as src:
        with resample_raster(src, 3.5) as resampled:
            print('Orig dims: {}, New dims: {}'.format(src.shape, resampled.shape))
            print(repr(resampled))

    From:
        https://gis.stackexchange.com/questions/329945/should-resampling-downsampling-a-raster-using-rasterio-cause-the-coordinates-t
        https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array/329439#329439
    """
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = raster.height * scale
    width = raster.width * scale

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff',
                   height=height, width=width)

    data = raster.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
        out_shape=(raster.count, height, width),
        resampling=Resampling.cubic,
    )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return
