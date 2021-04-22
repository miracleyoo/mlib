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
from rasterio.plot import show

def bbox(shp):
    """ Compute the bounding box of a certain shapefile.
    """
    piece = np.array([i.bounds for i in shp['geometry']])
    minx = piece[:,0].min()
    miny = piece[:,1].min()
    maxx = piece[:,2].max()
    maxy = piece[:,3].max()
    return minx, miny, maxx, maxy

def edge_length(shp):
    """ Compute the x and y edge length for a ceratin shapefile.
    """
    minx, miny, maxx, maxy = bbox(shp)
    return round(maxx-minx,3), round(maxy-miny,3)

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
    return Polygon([(minx, miny), (minx, maxy), (maxx,maxy), (maxx, miny)])

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
    return (img[(3,2,1),]/256*scale).astype(np.uint8)

def cropbyshp(raster, shp, boundary_clip=True):
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
    out_img, out_transform = mask(dataset=raster, shapes=polygon2geojson(bbpoly), crop=True, all_touched=True)
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
    mask = B.sum(axis=0)>1e-3
    masked_A = mask*A
    return masked_A

def adjust_A_by_B(A, B):
    """ Adjust image A's each band by the corresponding B's band.

        The output is A with each band have the same mean and std
        value of the corresponding B's band.

        A/B: Shape: [C, H, W]
    """
    for i in range(A.shape[0]):
        ap = A[i].flatten()
        ap = ap[ap!=0]
        bp = B[i].flatten()
        bp = bp[bp!=0]
        A[i] = (A[i]-ap.mean())/ap.std()
        A[i] = A[i]*bp.std()+bp.mean()
    return A