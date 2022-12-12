# -*- coding: utf-8 -*-

"""
***************************************************************************
    CrossingVectorRasterStatScriptQgis.py
    -------------------------------------
    Date                 : February 2019
    Copyright            : (C) 2019 by Gilles Fouvet (CEREMA)
    Email                : gilles dot fouvet at cerma dot fr
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

__author__ = 'Gilles Fouvet'
__date__ = 'February 2019'
__copyright__ = '(C) 2019, Gilles Fouvet'


from qgis.PyQt.QtGui import QIcon

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing,
                       QgsApplication,
                       QgsMessageLog,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterString,
                       QgsProcessingParameterVectorDestination)
                   

# Import des bibliothèques python pour l'application
import sys,os,glob,argparse,platform,shutil,time
import numpy as np
import json, math, shapely
import unicodedata
import processing
from shapely.geometry import shape, box, MultiPolygon
from collections import Counter
from osgeo import gdal, osr, ogr
from osgeo.gdalconst import GA_ReadOnly

class CrossingVectorRasterStatProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    All Processing algorithms should extend the QgsProcessingAlgorithm class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_VECTOR = 'INPUT_VECTOR'
    OUTPUT_VECTOR = 'OUTPUT_VECTOR'
    BOOL_ALL_COUNT = 'BOOL_ALL_COUNT'
    BOOL_COLUMNS_STR = 'BOOL_COLUMNS_STR'
    BOOL_COLUMNS_REAL = 'BOOL_COLUMNS_REAL'
    DICO_CLASS_LABEL = 'DICO_CLASS_LABEL'
    
    DICO_TXT_DEFAULT = "" #"11110:build, 11120:Road, 12200:Water, 13000:bared_soil, 20000:Vegetation"
    
    scriptsProcessingPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0] + os.sep + 'processing' + os.sep + 'scripts'

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CrossingVectorRasterStatProcessingAlgorithm()

    def icon(self):
        return QIcon(os.path.join(self.scriptsProcessingPath, 'images', 'logoCerema.png'))
    
    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'CrossingVectorRasterStatScript'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('CrossingVectorRasterStat')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('Cerema scripts')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'ceremascriptsid'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        text_info = "This algorithm create statistics from a crossing vector file and raster file\n"\
        + "Input Dico Class to Label example = 11110:build, 11120:Road, 12200:Water, 13000:bared_soil, 20000:Vegetation"
        return self.tr(text_info)

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input raster features source. It can have any kind of geometry.
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Input layer raster'),
                None,
                False
            )
        )

        # We add the input vector features source. It can have any kind of geometry.
        """
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR,
                self.tr('Input layer vector'),
                [QgsProcessing.TypeVectorPolygon ],
                None,
                False
            )
        )
        """
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_VECTOR,
                self.tr('Input layer vector'),
                [QgsProcessing.TypeVectorPolygon ],
                None,
                False
            )
        )
        
        
        # We add the input boolean parameters.
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.BOOL_ALL_COUNT, 
                self.tr("Enable Stats All Count"),
                QVariant(True),
                False
            )                
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.BOOL_COLUMNS_STR, 
                self.tr("Enable Stats Columns String"),
                QVariant(False),
                False
            )               
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.BOOL_COLUMNS_REAL, 
                self.tr("Enable Stats Columns Real"),
                QVariant(False),
                False
            )          
        )
        
        # We add the input dico parameter.
        self.addParameter(
            QgsProcessingParameterString(
                self.DICO_CLASS_LABEL, 
                self.tr("Input Dico Class to Label"),
                QVariant(self.DICO_TXT_DEFAULT),
                False,
                True
            )          
        )
        
        # We add the output vector features. It can have any kind of geometry.
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VECTOR,
                self.tr('Output layer vector'),
                type = QgsProcessing.TypeVectorAnyGeometry
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        # Retrieve the feature source. The 'dest_id' variable is used
        # to uniquely identify the feature, and must be included in the
        # dictionary returned by the processAlgorithm function.
        
        source_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        source_vector = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)
        output_path_vector = self.parameterAsOutputLayer(parameters, self.OUTPUT_VECTOR, context)
        
        #feedback.pushInfo(self.tr("Warning! test message... "))
        #feedback.setProgressText(self.tr("Info! test progress ... "))
        #feedback.reportError (self.tr("Error! test... "))
        
        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated

        if source_raster is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT_RASTER))    
        #source_path_raster = source_raster.dataProvider().dataSourceUri() 
        source_path_raster = source_raster.source()        
            
        if source_vector is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT_VECTOR))
        #source_path_vector = source_vector.dataProvider().dataSourceUri().split('|')[0] 
        source_path_vector = source_vector.source()       

        if output_path_vector is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT_VECTOR))
            
        enable_stats_all_count = self.parameterAsBool(parameters, self.BOOL_ALL_COUNT, context)
        enable_stats_columns_str = self.parameterAsBool(parameters, self.BOOL_COLUMNS_STR, context)
        enable_stats_columns_real = self.parameterAsBool(parameters, self.BOOL_COLUMNS_REAL, context)
        class_label_string = self.parameterAsString(parameters, self.DICO_CLASS_LABEL, context)
    
        col_to_delete_list = [] 
        col_to_add_list = []
        class_label_dico = {}
        
        if class_label_string != "":
            class_label_list = class_label_string.split(',')
            for class_label_str in class_label_list :
                classId = int(cleanSpaceText(cleanSpaceText(class_label_str).split(':')[0]))
                labelStr = removeAccents(cleanSpaceText(cleanSpaceText(class_label_str).split(':')[1]).replace("\'", "").replace('\"', ""))
                class_label_dico[classId] = labelStr
        
        if enable_stats_columns_real:
            class_label_dico = {}
			
        # Si le dossier de sortie n'existe pas, on le crée
        repertory_output = os.path.dirname(output_path_vector)
        if not os.path.isdir(repertory_output):
            os.makedirs(repertory_output)

        # Call application
        statisticsVectorRaster(self, feedback, source_path_raster, source_path_vector, output_path_vector, enable_stats_all_count, enable_stats_columns_str, enable_stats_columns_real, col_to_delete_list, col_to_add_list, class_label_dico, clean_small_polygons=False, overwrite=True)

        # Return the results of the algorithm. In this case our only result is
        # the feature which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        results = {}
        results[self.OUTPUT_VECTOR] = output_path_vector
        return results
        
#########################################################################
# LIBRARY raster_stats                                                  #
#########################################################################

# Correspondance entre python2 "basestring" et python3 "str"
try:
  basestring
except NameError:
  basestring = str

class RasterStatsError(Exception):
    pass

class OGRError(Exception):
    pass

#########################################################################    
def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]

    x1 = int(math.floor((bbox[0] - originX) / pixel_width))
    x2 = int(math.ceil((bbox[2] - originX) / pixel_width))

    y1 = int(math.floor((bbox[3] - originY) / pixel_height))
    y2 = int(math.ceil((bbox[1] - originY) / pixel_height))

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)

#########################################################################
def raster_extent_as_bounds(gt, size):
    east1 = gt[0]
    east2 = gt[0] + (gt[1] * size[0])
    west1 = gt[3] + (gt[5] * size[1])
    west2 = gt[3]
    return (east1, west1, east2, west2)

#########################################################################    
def feature_to_geojson(feature):
    # This duplicates the feature.ExportToJson ogr method
    # but is safe across gdal versions since it was fixed only in 1.8+

    geom = feature.GetGeometryRef()
    if geom is not None:
        geom_json_string = geom.ExportToJson()
        geom_json_object = json.loads(geom_json_string)
    else:
        geom_json_object = None

    output = {'type':'Feature',
               'geometry': geom_json_object,
               'properties': {}
              }

    fid = feature.GetFID()
    if fid:
        output['id'] = fid

    for key in feature.keys():
        output['properties'][key] = feature.GetField(key)

    return output

#########################################################################
def shapely_to_ogr_type(shapely_type):
    from osgeo import ogr
    if shapely_type == "Polygon":
        return ogr.wkbPolygon
    elif shapely_type == "LineString":
        return ogr.wkbLineString
    elif shapely_type == "MultiPolygon":
        return ogr.wkbMultiPolygon
    elif shapely_type == "MultiLineString":
        return ogr.wkbLineString
    raise TypeError("shapely type %s not supported" % shapely_type)

#########################################################################
def parse_geo(thing):
    # Given a python object, try to get a geo-json like mapping from it

    from shapely import wkt, wkb

    # object implementing geo_interface
    try:
        geo = thing.__geo_interface__
        return geo
    except AttributeError:
        pass

    # wkt
    try:
        shape = wkt.loads(thing)
        return shape.__geo_interface__
    except Exception:
        pass

    # geojson-like python mapping
    try:
        assert thing['type'] in ["Feature", "Point", "LineString", "Polygon",
                                "MultiPoint", "MultiLineString", "MultiPolygon"]
        return thing
    except (AssertionError, TypeError):
        pass

    # geojson string
    try:
        maybe_geo = json.loads(thing)
        assert maybe_geo['type'] in ["Feature", "Point", "LineString", "Polygon",
                       "MultiPoint", "MultiLineString", "MultiPolygon"]
        return maybe_geo
    except (ValueError, AssertionError):
        pass

    # wkb
    try:
        shape = wkb.loads(thing)
        return shape.__geo_interface__
    except Execption:
        pass

    raise RasterStatsError("Can't parse %s as a geo-like object" % thing)

#########################################################################
def get_ogr_ds(vds):
    from osgeo import ogr
    if not isinstance(vds, basestring):
        raise OGRError("OGR cannot open %r: not a string" % vds)

    ds = ogr.Open(vds)
    if not ds:
        raise OGRError("OGR cannot open %r" % vds)

    return ds

#########################################################################
def ogr_srs(vector, layer_num):
    ds = get_ogr_ds(vector)
    layer = ds.GetLayer(layer_num)
    return layer.GetSpatialRef()

#########################################################################
def ogr_records(vector, layer_num=0):
    ds = get_ogr_ds(vector)
    layer = ds.GetLayer(layer_num)
    for i in range(layer.GetFeatureCount()):
        try:
            feature = layer.GetFeature(i)
        except RuntimeError:
            continue
        else:
            yield feature_to_geojson(feature)
    return        

#########################################################################
def geo_records(vectors):
    for vector in vectors:
        yield parse_geo(vector)
    return    

#########################################################################
def get_features(vectors, layer_num=0):
    from osgeo import osr
    spatial_ref = osr.SpatialReference()
    if isinstance(vectors, basestring):
        try:
        # either an OGR layer ...
            get_ogr_ds(vectors)
            features_iter = ogr_records(vectors, layer_num)
            spatial_ref = ogr_srs(vectors, layer_num)
            strategy = "ogr"
        except OGRError:
        # ... or a single string to be parsed as wkt/wkb/json
            feat = parse_geo(vectors)
            features_iter = [feat]
            strategy = "single_geo"
    elif hasattr(vectors, '__geo_interface__'):
        # ... or an single object
        feat = parse_geo(vectors)
        features_iter = [feat]
        strategy = "single_geo"
    elif isinstance(vectors, dict):
        # ... or an python mapping
        feat = parse_geo(vectors)
        features_iter = [feat]
        strategy = "single_geo"
    else:
        # ... or an iterable of objects
        features_iter = geo_records(vectors)
        strategy = "iter_geo"

    return features_iter, strategy, spatial_ref

#########################################################################
if ogr.GetUseExceptions() != 1:
    ogr.UseExceptions()


DEFAULT_STATS = ['count', 'min', 'max', 'mean']
VALID_STATS = DEFAULT_STATS + \
    ['sum', 'std', 'median', 'all', 'majority', 'minority', 'unique', 'range']

#########################################################################
def raster_stats(self, feedback, vectors, raster, layer_num=0, band_num=1, nodata_value=None,
                 global_src_extent=False, categorical=False, stats=None,
                 copy_properties=False):

    if not stats:
        if not categorical:
            stats = DEFAULT_STATS
        else:
            stats = []
    else:
        if isinstance(stats, str):
            if stats in ['*', 'ALL']:
                stats = VALID_STATS
            else:
                stats = stats.split()
    for x in stats:
        if x not in VALID_STATS:
            raise RasterStatsError("Stat `%s` not valid;" \
                " must be one of \n %r" % (x, VALID_STATS))

    run_count = False
    if categorical or 'majority' in stats or 'minority' in stats or 'unique' in stats or 'all' in stats :
        # run the counter once, only if needed
        run_count = True

    rds = gdal.Open(raster, GA_ReadOnly)
    if not rds:
        raise RasterStatsError("Cannot open %r as GDAL raster" % raster)
    rb = rds.GetRasterBand(band_num)
    rgt = rds.GetGeoTransform()
    rsize = (rds.RasterXSize, rds.RasterYSize)
    rbounds = raster_extent_as_bounds(rgt, rsize)

    if nodata_value is not None:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)
    else:
        nodata_value = rb.GetNoDataValue()

    features_iter, strategy, spatial_ref = get_features(vectors, layer_num)

    if global_src_extent:
        # create an in-memory numpy array of the source raster data
        # covering the whole extent of the vector layer
        if strategy != "ogr":
            raise RasterStatsError("global_src_extent requires OGR vector")

        # find extent of ALL features
        ds = ogr.Open(vectors)
        layer = ds.GetLayer(layer_num)
        ex = layer.GetExtent()
        # transform from OGR extent to xmin, ymin, xmax, ymax
        layer_extent = (ex[0], ex[2], ex[1], ex[3])

        global_src_offset = bbox_to_pixel_offsets(rgt, layer_extent)
        global_src_array = rb.ReadAsArray(*global_src_offset)

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    results = []

    for i, feat in enumerate(features_iter):
        if feat['type'] == "Feature":
            try :
                geom = shape(feat['geometry'])
            except :
                next
        else:  # it's just a geometry
            geom = shape(feat)

        # Point and MultiPoint don't play well with GDALRasterize
        # convert them into box polygons the size of a raster cell
        buff = rgt[1] / 2.0
        if geom.type == "MultiPoint":
            geom = MultiPolygon([box(*(pt.buffer(buff).bounds))
                                for pt in geom.geoms])
        elif geom.type == 'Point':
            geom = box(*(geom.buffer(buff).bounds))

        ogr_geom_type = shapely_to_ogr_type(geom.type)

        # "Clip" the geometry bounds to the overall raster bounding box
        # This should avoid any rasterIO errors for partially overlapping polys
        geom_bounds = list(geom.bounds)
        if geom_bounds[0] < rbounds[0]:
            geom_bounds[0] = rbounds[0]
        if geom_bounds[1] < rbounds[1]:
            geom_bounds[1] = rbounds[1]
        if geom_bounds[2] > rbounds[2]:
            geom_bounds[2] = rbounds[2]
        if geom_bounds[3] > rbounds[3]:
            geom_bounds[3] = rbounds[3]

        # calculate new geotransform of the feature subset
        src_offset = bbox_to_pixel_offsets(rgt, geom_bounds)

        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5]
        )

        if src_offset[2] < 0 or src_offset[3] < 0:
            # we're off the raster completely, no overlap at all, so there's no need to even bother trying to calculate
            feature_stats = dict([(s, None) for s in stats])
        else:
            if not global_src_extent:
                # use feature's source extent and read directly from source
                # fastest option when you have fast disks and well-indexed raster
                # advantage: each feature uses the smallest raster chunk
                # disadvantage: lots of disk reads on the source raster
                src_array = rb.ReadAsArray(*src_offset)
                if src_array is None:
                    src_offset = (src_offset[0],src_offset[1],src_offset[2],src_offset[3] - 1)
                    src_array = rb.ReadAsArray(*src_offset)

            else:
                # derive array from global source extent array
                # useful *only* when disk IO or raster format inefficiencies are your limiting factor
                # advantage: reads raster data in one pass before loop
                # disadvantage: large vector extents combined with big rasters need lot of memory
                xa = src_offset[0] - global_src_offset[0]
                ya = src_offset[1] - global_src_offset[1]
                xb = xa + src_offset[2]
                yb = ya + src_offset[3]
                src_array = global_src_array[ya:yb, xa:xb]

            # Create a temporary vector layer in memory
            mem_ds = mem_drv.CreateDataSource('out')
            mem_layer = mem_ds.CreateLayer('out', spatial_ref, ogr_geom_type)
            ogr_feature = ogr.Feature(feature_def=mem_layer.GetLayerDefn())
            ogr_geom = ogr.CreateGeometryFromWkt(geom.wkt)
            ogr_feature.SetGeometryDirectly(ogr_geom)
            mem_layer.CreateFeature(ogr_feature)

            # Rasterize it
            rvds = driver.Create('rvds', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
            rvds.SetGeoTransform(new_gt)

            gdal.RasterizeLayer(rvds, [1], mem_layer, None, None, burn_values=[1])
            rv_array = rvds.ReadAsArray()
            # Mask the source data array with our current feature
            # we take the logical_not to flip 0<->1 to get the correct mask effect
            # we also mask out nodata values explicitly
            # ATTENTION : probleme possible si src_array == None.

            test_ok = True
            if src_array is None:
                feedback.pushInfo(self.tr("Warning! src_array = "+ str(src_array) + ", nodata_value = " + str(nodata_value)))
                test_ok = False
            else :
                masked = np.ma.MaskedArray(
                    src_array,
                    mask=np.logical_or(
                        src_array is nodata_value,
                        np.logical_not(rv_array)
                    )
                )

            if run_count:
                if test_ok :
                    pixel_count = Counter(masked.compressed())
                else :
                    pixel_count = 0
            if categorical:
                feature_stats = dict(pixel_count)
            else:
                feature_stats = {}

            if 'min' in stats:
                if test_ok :
                    feature_stats['min'] = float(masked.min())
                else :
                    feature_stats['min'] = 0.0
            if 'max' in stats:
                if test_ok :
                    feature_stats['max'] = float(masked.max())
                else :
                    feature_stats['max'] = 0.0
            if 'mean' in stats:
                if test_ok :
                    feature_stats['mean'] = float(masked.mean())
                else :
                    feature_stats['mean'] = 0.0
            if 'count' in stats:
                if test_ok :
                    feature_stats['count'] = int(masked.count())
                else :
                    feature_stats['count'] = 0
            # optional
            if 'sum' in stats:
                if test_ok :
                    feature_stats['sum'] = float(masked.sum())
                else :
                    feature_stats['sum'] = 0.0
            if 'std' in stats:
                if test_ok :
                    feature_stats['std'] = float(masked.std())
                else :
                    feature_stats['std'] = 0.0
            if 'median' in stats:
                if test_ok :
                    feature_stats['median'] = float(np.median(masked.compressed()))
                else :
                    feature_stats['median'] = 0.0

            # Add option 'all' GFT 17/03/2014
            if 'all' in stats:
                try:
                    feature_stats['all'] = pixel_count.most_common()
                except IndexError:
                    feature_stats['all'] = None

            if 'majority' in stats:
                try:
                    feature_stats['majority'] = pixel_count.most_common(1)[0][0]
                except IndexError:
                    feature_stats['majority'] = None

            if 'minority' in stats:
                try:
                    feature_stats['minority'] = pixel_count.most_common()[-1][0]
                except IndexError:
                    feature_stats['minority'] = None

            if 'unique' in stats:
                if test_ok :
                    feature_stats['unique'] = len(pixel_count.keys())
                else :
                    feature_stats['unique'] = 0

            if 'range' in stats:
                try:
                    rmin = feature_stats['min']
                except KeyError:
                    if test_ok :
                        rmin = float(masked.min())
                    else :
                        rmin = 0.0
                try:
                    rmax = feature_stats['max']
                except KeyError:
                    if test_ok :
                        rmax = float(masked.max())
                    else :
                        rmax = 0.0
                feature_stats['range'] = rmax - rmin

        try:
            # Use the provided feature id as __fid__
            feature_stats['__fid__'] = feat['id']
        except KeyError:
            # use the enumerator
            feature_stats['__fid__'] = i

        if 'properties' in feat and copy_properties:
            for key, val in feat['properties'].items():
                feature_stats[key] = val

        results.append(feature_stats)

    return results

#########################################################################    
def stats_to_csv(stats):
    from cStringIO import StringIO
    import csv

    csv_fh = StringIO()

    keys = set()
    for stat in stats:
        for key in stat.keys():
            keys.add(key)

    fieldnames = sorted(list(keys))

    csvwriter = csv.DictWriter(csv_fh, delimiter=',', fieldnames=fieldnames)
    csvwriter.writerow(dict((fn,fn) for fn in fieldnames))
    for row in stats:
        csvwriter.writerow(row)
    contents = csv_fh.getvalue()
    csv_fh.close()
    return contents

#########################################################################
# FONCTION cleanBeginSpaceText()                                        #
#########################################################################
#   Role : Fonction qui nettoye les espaces en debut uniquement
#   Entrees :
#       text_input : le text en entree pouvant contenir des espaces en debut et/ou en fin de chaineExemple : text = "  Exemple de Texte"
#   Sortie : :
#       return text_output : Le texte netoye des espaces de debut. Exemple : text = "Exemple de Texte"
def cleanBeginSpaceText(text_input):
    text_output = ""
    begin = False
    for char in str(text_input) :
        if char != ' ' or begin :
            begin = True
            text_output = text_output + char

    return text_output
    
#########################################################################
# FONCTION cleanSpaceText()                                             #
#########################################################################
#   Role : Fonction qui nettoye les espaces en debut et fin de chaine
#   Entrees :
#       text_input : le text en entree pouvant contenir des espaces en debut et/ou en fin de chaineExemple : text = "  Exemple de Texte "
#   Sortie : :
#       return text_output : Le texte netoye des espaces de debut et fin. Exemple : text = "Exemple de Texte"
def cleanSpaceText(text_input):
    text_temp = cleanBeginSpaceText(text_input[::-1])
    text_output = cleanBeginSpaceText(text_temp[::-1])

    return text_output
    
#########################################################################
# FONCTION removeAccents()                                              #
#########################################################################   
def removeAccents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return str("".join([c for c in nfkd_form if not unicodedata.combining(c)]))
    
#########################################################################
# FONCTION removeVectorFile()                                           #
#########################################################################
#   Role : Suppression de fichier vecteur
#   Paramètres :
#      del_vector_name : Nom du fichier vecteur à supprimer
#      format_vector : format du fichier vecteur (par defaut : format shapefile)
def removeVectorFile(del_vector_name, format_vector='ESRI Shapefile'):
    # Récupération du driver
    driver = ogr.GetDriverByName(format_vector)
    # Supression du vecteur
    if os.path.exists(del_vector_name):
        driver.DeleteDataSource(del_vector_name)
    return

#########################################################################
# FONCTION copyVectorFile()                                             #
#########################################################################
#   Role : Copy de fichier vecteur
#   Paramètres :
#      input_vector_name : Nom du fichier vecteur à copier
#      output_vector_name : Nom du fichier vecteur de sortie recopié
#      format_vector : format du fichier vecteur (par defaut : format shapefile)
def copyVectorFile(self, feedback, input_vector_name, output_vector_name, format_vector='ESRI Shapefile'):
    # Récupération du driver
    driver = ogr.GetDriverByName(format_vector)
    # Copie du vecteur
    if os.path.exists(input_vector_name):
        data_source = driver.Open(input_vector_name, 0)  # en lecture
        if os.path.exists(output_vector_name):
            driver.DeleteDataSource(output_vector_name)
        driver.CopyDataSource(data_source,output_vector_name)
        data_source.Destroy()
    else :
        feedback.pushInfo(self.tr("Warning! Impossible to copy file : " + input_vector_name + ". file not found!"))
    return

#########################################################################
# FONCTION renameVectorFile()                                           #
#########################################################################
#   Role : Renomage de fichier vecteur
#   Paramètres :
#      input_vector_name : Nom du fichier vecteur à copier
#      output_vector_name : Nom du fichier vecteur de sortie renomé
def renameVectorFile(self, feedback, input_vector_name, output_vector_name):
    # Renomage du fichier
    if os.path.exists(input_vector_name):
        base_name_input = os.path.splitext(input_vector_name)[0]
        base_name_output = os.path.splitext(output_vector_name)[0]
        os.rename(base_name_input + ".dbf", base_name_output + ".dbf")
        os.rename(base_name_input + ".prj", base_name_output + ".prj")
        os.rename(base_name_input + ".shx", base_name_output + ".shx")
        os.rename(base_name_input + ".shp", base_name_output + ".shp")
    else :
        feedback.pushInfo(self.tr("Warning! Impossible to rename file : " + input_vector_name + ". file not found!"))
    return

#########################################################################
# FONCTION getPixelSizeImage()                                          #
#########################################################################
#   Rôle : Cette fonction permet de retourner la taille d'un pixel de l'image
#   Paramètres :
#       image_raster : fichier image d'entrée
def getPixelSizeImage(image_raster):
    pixel_size = 0.0
    dataset = gdal.Open(image_raster, GA_ReadOnly)
    if dataset is not None:
        geotransform = dataset.GetGeoTransform()
        pixel_width = geotransform[1]  # w-e pixel resolution
        pixel_height = geotransform[5] # n-s pixel resolution
        pixel_size = pixel_width * pixel_height
    dataset = None

    return abs(pixel_size)

#########################################################################
# FONCTION getEmpriseImage()                                            #
#########################################################################
#   Rôle : Cette fonction permet de retourner les coordonnées xmin,xmax,ymin,ymax de l'emprise de l'image
#   Paramètres :
#       image_raster : fichier image d'entrée
#   Paramétres de retour :
#       xmin, xmax, ymin, ymax
def getEmpriseImage(image_raster):
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    if not os.path.isfile(image_raster) :
        raise NameError( "getEmpriseImage() : File " + image_raster + " does not exist")

    dataset = gdal.Open(image_raster, GA_ReadOnly)
    if dataset is not None:
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        geotransform = dataset.GetGeoTransform()
        pixel_width = geotransform[1]  # w-e pixel resolution
        pixel_height = geotransform[5] # n-s pixel resolution
        xmin = geotransform[0]     # top left x
        ymax = geotransform[3]     # top left y
        xmax = xmin + (cols * pixel_width)
        ymin = ymax + (rows * pixel_height)

    dataset = None

    return xmin, xmax, ymin, ymax

#########################################################################
# FONCTION identifyPixelValues ()                                       #
#########################################################################
# Rôle : Cette fonction identifie les valeurs présentes dans une image
# Paramètres :
#     image_raster : fichier image d'entrée
# Paramétres de retour :
#     liste des valeurs des pixels de l'image
#
# Exemple d'utilisation: image_values_list = identifyPixelValues(image_raster)
def identifyPixelValues(image_raster):
    image_open = gdal.Open(image_raster)                   # Ouverture de l'image
    image = image_open.GetRasterBand(1)                   # Extraction de la premiere bande
    image_array = image.ReadAsArray()                     # Transformation de la bande en tableau numpy
    image_values_nparray = np.unique(image_array)         # Extraction des valeurs uniques dans un tableau numpy
    image_values_list = image_values_nparray.tolist()     # Transformation du tableau en liste
    image_open = None

    return (image_values_list)

###########################################################################################################################################
# FONCTION getEmpriseFile()                                                                                                               #
###########################################################################################################################################
# http://geoinformaticstutorial.blogspot.fr/2012/09/reading-raster-data-with-python-and-gdal.html
#   Role : Cette fonction permet de retourner les coordonnées xmin,xmax,ymin,ymax de l'emprise d'un fichier shape
#   Parametres :
#       vector_input : nom du fichier vecteur d'entrée
#       format_shape : format du fichier shape
#   Paramétres de retour :
#       xmin, xmax, ymin, ymax
def getEmpriseFile(self, feedback, empr_file, format_shape='ESRI Shapefile'):
    # Recuperation du  driver pour le format shape
    driver = ogr.GetDriverByName(format_shape)

    # Ouverture du fichier d'emprise
    data_source = driver.Open(empr_file, 0)
    if data_source is None:
        feedback.reportError(self.tr("Error! Impossible to open hold file : " + empr_file))
        raise QgsProcessingException(self.tr("Error! Impossible to open hold file : " + empr_file))
        #sys.exit(1) #exit with an error code

    # Recuperation des couches de donnees
    layer = data_source.GetLayer(0)
    num_features = layer.GetFeatureCount()
    extent = layer.GetExtent()

    # Fermeture du fichier d'emprise
    data_source.Destroy()

    xmin = extent[0]
    xmax = extent[1]
    ymin = extent[2]
    ymax = extent[3]

    return xmin,xmax,ymin,ymax

#########################################################################
# FONCTION cleanMiniAreaPolygons()                                      #
#########################################################################
#   Role : Fonction qui supprime les polygones de surfaces minimales d'un shapefile
#   Paramètres :
#       vector_input : nom du fichier vecteur d'entrée
#       vector_output : nom du fichier vecteur de sortie
#       min_size_area : valeur de la taille minimale de surface des polygones à nettoyer
#       col : nom du champs (colonne) à regarder
#       format_shape : format d'entrée et de sortie des fichiers vecteurs. Par default : 'ESRI Shapefile'
#  Exemple d'utilisation: cleanMiniAreaPolygons("vectorInput.shape","vectorOutput.shape",0.45,'ESRI Shapefile')
def cleanMiniAreaPolygons(self, feedback, vector_input, vector_output, min_size_area, col='id', format_shape='ESRI Shapefile'):
    # Initialisation du dictionaire de surface par classe
    size_area_by_class_dico = {}

    # Recuperation du driver pour le format shape fichier entrée
    driver_input = ogr.GetDriverByName(format_shape)

    # Ouverture du fichier shape en lecture
    data_source_input = driver_input.Open(vector_input, 0) # 0 means read-only. 1 means writeable.
    if data_source_input is None:
        feedback.pushInfo(self.tr("Warning! Format file unknown, unable to open the file : " + vector_input))
        return size_area_by_class_dico

    # Récupération des cractéristiques du fichier en entrée (type de géométrie, projection...)
    layer_input = data_source_input.GetLayer()
    feature_input = layer_input.GetNextFeature()

    # Cas ou le fichier est vide on recopie juste le fichier d'entrée
    if feature_input is None:
        feedback.pushInfo(self.tr("Warning! the input file is empty so no treats it is just copy in output : " + str(vector_input)))
        # Copy vector_output
        driver_input.CopyDataSource(data_source_input,vector_output)
        data_source_input.Destroy()
        return size_area_by_class_dico

    # Lecture des infos de la couche
    name_layer_input = layer_input.GetName()
    out_srs = layer_input.GetSpatialRef ()
    geom_type_input = layer_input.GetGeomType()

    # Recuperation du driver pour le format shape fichier de sortie
    driver_output = ogr.GetDriverByName(format_shape)

    # Si le fichier destination existe deja on l ecrase
    if os.path.exists(vector_output) :
        feedback.pushInfo(self.tr("Warning! The shape file already exists it will be crushed : " + vector_output))
        driver_output.DeleteDataSource(vector_output)

    # Creation du fichier shape de sortie en écriture
    data_source_output = driver_output.CreateDataSource(vector_output)
    layer_output = data_source_output.CreateLayer(name_layer_input, out_srs, geom_type=geom_type_input)

    # Ajouter les champs du fichier d'entrée au fichier de sortie
    defn_layer_input = layer_input.GetLayerDefn()
    for i in range(0, defn_layer_input.GetFieldCount()):
        field_defn = defn_layer_input.GetFieldDefn(i)
        layer_output.CreateField(field_defn)

    # Recupère le output Layer's Feature Definition
    defn_layer_output = layer_output.GetLayerDefn()

    # Comptage du nombre de polygones sources
    num_features = layer_input.GetFeatureCount()

    # Réinitialiser la lecture des géométries
    layer_input.ResetReading()

    # Pour chaque polygone
    # Add features to the ouput Layer
    for i in range(0, layer_input.GetFeatureCount()):

         feature_input = layer_input.GetFeature(i)  # Get the input Feature
         geometry = feature_input.GetGeometryRef() # Calculating the actual area

         # Si la geometry est non nulle
         if not geometry is None :
            idfeat = feature_input.GetFID()
            polygonArea = geometry.GetArea()

            # Si la surface est superieur a min_size_area, on copie le polygone
            if polygonArea >= min_size_area :

                # Add new feature to output Layer
                layer_output.CreateFeature(feature_input)

                # Mettre a jour information du dictionaire surface total par classe
                if col != "":
                    class_label = int(feature_input.GetFieldAsString(col))

                    if class_label not in size_area_by_class_dico :
                        size_area_by_class_dico[class_label] = polygonArea
                    else :
                        size_area_by_class_dico[class_label] += polygonArea

    # Comptage du nombre de polygones destinations
    num_features = layer_output.GetFeatureCount()

    # Fermeture des fichiers shape
    layer_output.SyncToDisk()
    data_source_input.Destroy()
    data_source_output.Destroy()
    
    return size_area_by_class_dico

###########################################################################################################################################
# DEFINITION DE LA FONCTION statisticsVectorRaster                                                                                        #
###########################################################################################################################################
# ROLE:
#     Fonction qui calcule pour chaque polygone d'un fichier vecteur (shape) les statistiques associées de l'intersection avec une image raster (tif)
#
# ENTREES DE LA FONCTION :
#    feedback : pour log qgis
#    image_input : Fichier image raster de la classification information pour le calcul des statistiques
#    vector_input : Fichier vecteur d'entrée defini les zones de polygones pour le calcul des statistiques
#    vector_output : Fichier vecteur de sortie
#    enable_stats_all_count : Active le calcul statistique 'all','count' sur les pixels de l'image raster
#    enable_stats_columns_str : Active le calcul statistique 'majority','minority' sur les pixels de l'image raster
#    enable_stats_columns_real : Active le calcul statistique 'min', 'max', 'mean' , 'median','sum', 'std', 'unique', 'range' sur les pixels de l'image raster.
#    col_to_delete_list : liste des colonnes a suprimer
#    col_to_add_list : liste des colonnes à ajouter
#         NB: ce parametre n a de sens que sur une image rvb ou un MNT par exemple
#    class_label_dico : dictionaire affectation de label aux classes de classification
#    overwrite : supprime ou non les fichiers existants ayant le meme nom
#
# SORTIES DE LA FONCTION :
#    Eléments modifiés le fichier shape d'entrée
#
def statisticsVectorRaster(self, feedback, image_input, vector_input, vector_output, enable_stats_all_count, enable_stats_columns_str, enable_stats_columns_real, col_to_delete_list, col_to_add_list, class_label_dico, clean_small_polygons=False, overwrite=True) :
 
    # INITIALISATION

    # Constantes
    PREFIX_AREA_COLUMN = "S_"
    FORMAT_VECTOR = "ESRI Shapefile"

    # creation du fichier vecteur de sortie
    if vector_output == "":
        vector_output = vector_input # Précisé uniquement pour l'affichage
    else :
        # copy vector_output
        copyVectorFile(self, feedback, vector_input, vector_output, FORMAT_VECTOR)

    # Vérifications
    image_xmin, image_xmax, image_ymin, image_ymax = getEmpriseImage(image_input)
    vector_xmin, vector_xmax, vector_ymin, vector_ymax = getEmpriseFile(self, feedback, vector_input)

    if round(vector_xmin,4) < round(image_xmin,4) or round(vector_xmax,4) > round(image_xmax,4) or round(vector_ymin,4) < round(image_ymin,4) or round(vector_ymax,4) > round(image_ymax,4) :
        raise NameError("Error! The extend of the vector file (%s) is greater than the image file (%s)" %(vector_input,image_input))

    pixel_size = getPixelSizeImage(image_input)

    # Suppression des très petits polygones qui introduisent des valeurs NaN

    if clean_small_polygons:
        min_size_area = pixel_size * 2
        vector_temp = os.path.splitext(vector_output)[0] + "_temp.shp"

        cleanMiniAreaPolygons(self, feedback, vector_output, vector_temp, min_size_area, '', FORMAT_VECTOR)
        removeVectorFile(vector_output, FORMAT_VECTOR)
        renameVectorFile(self, feedback, vector_temp, vector_output)

    # Récuperation du driver pour le format shape
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Ouverture du fichier shape en lecture-écriture
    data_source = driver.Open(vector_output, 1) # 0 means read-only - 1 means writeable.
    if data_source is None:
        feedback.reportError(self.tr("Error! Impossible to open vector file : " + vector_output))
        raise QgsProcessingException(self.tr("Error! Impossible to open vector file : " + vector_output))
        #sys.exit(1) # exit with an error code

    # Récupération du vecteur
    layer = data_source.GetLayer(0)         # Recuperation de la couche (une couche contient les polygones)
    layer_definition = layer.GetLayerDefn() # GetLayerDefn => returns the field names of the user defined (created) fields

    # ETAPE 0/3 : CREATION AUTOMATIQUE DU DICO DE VALEUR SI IL N'EXISTE PAS
    if enable_stats_all_count and class_label_dico == {}:
        image_values_list = identifyPixelValues(image_input)
        # Pour toutes les valeurs
        for id_value in image_values_list :
            class_label_dico[id_value] = str(id_value)
        # Suppression de la valeur no date à 0
        if 0 in class_label_dico :
            del class_label_dico[0]

    # ETAPE 1/3 : CREATION DES COLONNES DANS LE FICHIER SHAPE

    # En entrée :
    # col_to_add_list = [UniqueID, majority/DateMaj/SrcMaj, minority, min, max, mean, median, sum, std, unique, range, all, count, all_S, count_S] - all traduisant le class_label_dico en autant de colonnes
    # Sous_listes de col_to_add_list à identifier pour des facilités de manipulations ultérieures:
    # col_to_add_inter01_list = [majority/DateMaj/SrcMaj, minority, min, max, mean, median, sum, std, unique, range]
    # col_to_add_inter02_list = [majority, minority, min, max, mean, median, sum, std, unique, range, all, count, all_S, count_S]
    # Construction des listes intermédiaires
    col_to_add_inter01_list = []

    # valeurs à injecter dans des colonnes - Format String
    if enable_stats_columns_str :
        stats_columns_str_list = ['majority','minority']
        for e in stats_columns_str_list :
            col_to_add_list.append(e)

    # valeurs à injecter dans des colonnes - Format Nbr
    if enable_stats_columns_real :
        stats_columns_real_list = ['min', 'max', 'mean' , 'median','sum', 'std', 'unique', 'range']
        for e in stats_columns_real_list :
            col_to_add_list.append(e)

    # valeurs à injecter dans des colonnes - Format Nbr
    if enable_stats_all_count :
        stats_all_count_list = ['all','count']
        for e in stats_all_count_list :
            col_to_add_list.append(e)

    # valeurs à injecter dans des colonnes - si class_label_dico est non vide
    if class_label_dico != {}:
        stats_all_count_list = ['all','count']
        for e in stats_all_count_list :
            if not e in col_to_add_list :
                col_to_add_list.append(e)

    # Ajout colonne par colonne
    if "majority" in col_to_add_list:
        col_to_add_inter01_list.append("majority")
    if "DateMaj" in col_to_add_list:
        col_to_add_inter01_list.append("DateMaj")
    if "SrcMaj" in col_to_add_list:
        col_to_add_inter01_list.append("SrcMaj")
    if "minority" in col_to_add_list:
        col_to_add_inter01_list.append("minority")
    if "min" in col_to_add_list:
        col_to_add_inter01_list.append("min")
    if "max" in col_to_add_list:
        col_to_add_inter01_list.append("max")
    if "mean" in col_to_add_list:
        col_to_add_inter01_list.append("mean")
    if "median" in col_to_add_list:
        col_to_add_inter01_list.append("median")
    if "sum" in col_to_add_list:
        col_to_add_inter01_list.append("sum")
    if "std" in col_to_add_list:
        col_to_add_inter01_list.append("std")
    if "unique" in col_to_add_list:
        col_to_add_inter01_list.append("unique")
    if "range" in col_to_add_list:
        col_to_add_inter01_list.append("range")

    # Copy de col_to_add_inter01_list dans col_to_add_inter02_list
    col_to_add_inter02_list = list(col_to_add_inter01_list)

    if "all" in col_to_add_list:
        col_to_add_inter02_list.append("all")
    if "count" in col_to_add_list:
        col_to_add_inter02_list.append("count")
    if "all_S" in col_to_add_list:
        col_to_add_inter02_list.append("all_S")
    if "count_S" in col_to_add_list:
        col_to_add_inter02_list.append("count_S")
    if "DateMaj" in col_to_add_inter02_list:
        col_to_add_inter02_list.remove("DateMaj")
        col_to_add_inter02_list.insert(0,"majority")
    if "SrcMaj" in col_to_add_inter02_list:
        col_to_add_inter02_list.remove("SrcMaj")
        col_to_add_inter02_list.insert(0,"majority")

    # valeurs à injecter dans des colonnes - Format Nbr
    if enable_stats_all_count :
        stats_all_count_list = ['all_S', 'count_S']
        for e in stats_all_count_list :
            col_to_add_list.append(e)

    # Creation de la colonne de l'identifiant unique
    if ("UniqueID" in col_to_add_list) or ("uniqueID" in col_to_add_list) or ("ID" in col_to_add_list):
        field_defn = ogr.FieldDefn("ID", ogr.OFTInteger)    # Création du nom du champ dans l'objet stat_classif_field_defn
        layer.CreateField(field_defn)

    # Creation des colonnes de col_to_add_inter01_list ([majority/DateMaj/SrcMaj, minority, min, max, mean, median, sum, std, unique, range])
    for col in col_to_add_list:
        if layer_definition.GetFieldIndex(col) == -1 :                          # Vérification de l'existence de la colonne col (retour = -1 : elle n'existe pas)
            if col == 'majority' or col == 'DateMaj' or col == 'SrcMaj' or col == 'minority':  # Identification de toutes les colonnes remplies en string
                stat_classif_field_defn = ogr.FieldDefn(col, ogr.OFTString)     # Création du champ (string) dans l'objet stat_classif_field_defn
                layer.CreateField(stat_classif_field_defn)
            elif col == 'mean' or col == 'median' or col == 'sum' or col == 'std' or col == 'unique' or col == 'range' or col == 'max' or col == 'min':
                stat_classif_field_defn = ogr.FieldDefn(col, ogr.OFTReal)       # Création du champ (real) dans l'objet stat_classif_field_defn
                # Définition de la largeur du champ
                stat_classif_field_defn.SetWidth(20)
                # Définition de la précision du champ valeur flottante
                stat_classif_field_defn.SetPrecision(2)
                layer.CreateField(stat_classif_field_defn)

    # Creation des colonnes reliées au dictionnaire
    if ('all' in col_to_add_list) or ('count' in col_to_add_list) or ('all_S' in col_to_add_list) or ('count_S' in col_to_add_list):
        for col in class_label_dico:

            # Gestion du nom de la colonne correspondant à la classe
            name_col = class_label_dico[col]
            if len(name_col) > 10:
                name_col = name_col[:10]
                feedback.pushInfo(self.tr("Warning! Column name too long. It will be truncated to 10 characters in case of use : " + name_col))

            # Gestion du nom de la colonne correspondant à la surface de la classe
            name_col_area =  PREFIX_AREA_COLUMN + name_col
            if len(name_col_area) > 10:
                name_col_area = name_col_area[:10]
                feedback.pushInfo(self.tr("Warning! Column name too long. It will be truncated to 10 characters in case of use : " + name_col_area))

            # Ajout des colonnes de % de répartition des éléments du raster
            if ('all' in col_to_add_list) or ('count' in col_to_add_list):
                if layer_definition.GetFieldIndex(name_col) == -1 :                     # Vérification de l'existence de la colonne name_col (retour = -1 : elle n'existe pas)
                    stat_classif_field_defn = ogr.FieldDefn(name_col, ogr.OFTReal)      # Création du champ (real) dans l'objet stat_classif_field_defn
                    # Définition de la largeur du champ
                    stat_classif_field_defn.SetWidth(20)
                    # Définition de la précision du champ valeur flottante
                    stat_classif_field_defn.SetPrecision(2)
                    layer.CreateField(stat_classif_field_defn)                          # Ajout du champ

            # Ajout des colonnes de surface des éléments du raster
            if ('all_S' in col_to_add_list) or ('count_S' in col_to_add_list):
                if layer_definition.GetFieldIndex(name_col_area) == -1 :                # Vérification de l'existence de la colonne name_col_area (retour = -1 : elle n'existe pas)
                    stat_classif_field_defn = ogr.FieldDefn(name_col_area, ogr.OFTReal) # Création du nom du champ dans l'objet stat_classif_field_defn
                    # Définition de la largeur du champ
                    stat_classif_field_defn.SetWidth(20)
                    # Définition de la précision du champ valeur flottante
                    stat_classif_field_defn.SetPrecision(2)
                    layer.CreateField(stat_classif_field_defn)                          # Ajout du champ

    # ETAPE 2/3 : REMPLISSAGE DES COLONNES DU VECTEUR

    # Calcul des statistiques col_to_add_inter02_list = [majority, minority, min, max, mean, median, sum, std, unique, range, all, count, all_S, count_S] de croisement images_raster / vecteur
    # Utilisation de la librairie rasterstat
    stats_info_list = raster_stats(self, feedback, vector_output, image_input, stats=col_to_add_inter02_list)

    # Decompte du nombre de polygones
    num_features = layer.GetFeatureCount()
    polygone_count = 0

    for polygone_stats in stats_info_list : # Pour chaque polygone représenté dans stats_info_list - et il y a autant de polygone que dans le fichier vecteur

        polygone_count = polygone_count + 1

        # Extraction de feature
        feature = layer.GetFeature(polygone_stats['__fid__'])

        # Remplissage de l'identifiant unique
        if ("UniqueID" in col_to_add_list) or ("uniqueID" in col_to_add_list) or ("ID" in col_to_add_list):
            feature.SetField('ID', int(stats_info_list.index(polygone_stats)))

        # Initialisation à 0 des colonnes contenant le % de répartition de la classe - Verifier ce qu'il se passe si le nom dépasse 10 caracteres
        if ('all' in col_to_add_list) or ('count' in col_to_add_list):
            for element in class_label_dico:
                name_col = class_label_dico[element]
                if len(name_col) > 10:
                    name_col = name_col[:10]
                feature.SetField(name_col,0)

        # Initialisation à 0 des colonnes contenant la surface correspondant à la classe - Verifier ce qu'il se passe si le nom dépasse 10 caracteres
        if ('all_S' in col_to_add_list) or ('count_S' in col_to_add_list):
            for element in class_label_dico:
                name_col = class_label_dico[element]
                name_col_area =  PREFIX_AREA_COLUMN + name_col
                if len(name_col_area) > 10:
                    name_col_area = name_col_area[:10]
                feature.SetField(name_col_area,0)

        # Remplissage des colonnes contenant le % de répartition et la surface des classes
        if ('all' in col_to_add_list) or ('count' in col_to_add_list) or ('all_S' in col_to_add_list) or ('count_S' in col_to_add_list):
            majority_all = polygone_stats['all']          # 'all' est une liste des couples : (Valeur_du_pixel_sur_le_raster, Nbr_pixel_ayant_cette_valeur) pour le polygone observe.
                                                          # Ex : [(0,183),(803,45),(801,4)] : dans le polygone, il y a 183 pixels de valeur 0, 45 pixels de valeur 803 et 4 pixels de valeur 801
            # Deux valeurs de pixel peuvent faire référence à une même colonne. Par exemple : les pixels à 201, 202, 203 peuvent correspondre à la BD Topo
            # Regroupement des éléments de majority_all allant dans la même colonne au regard de class_label_dico
            count_for_idx_couple = 0            # Comptage du nombre de modifications (suppression de couple) de majority_all pour adapter la valeur de l'index lors de son parcours

            for idx_couple in range(1,len(majority_all)) :  # Inutile d'appliquer le traitement au premier élément (idx_couple == 0)

                idx_couple = idx_couple - count_for_idx_couple    # Prise en compte dans le parcours de majority_all des couples supprimés
                couple = majority_all[idx_couple]          # Ex : couple = (803,45)

                if (couple is None) or (couple == "") :    # en cas de bug de rasterstats (erreur geometrique du polygone par exemple)
                    feedback.pushInfo(self.tr("Warning! Problem detected in polygon management %s" %(polygone_count)))
                    pass
                else :
                    for idx_verif in range(idx_couple):
                        # Vérification au regard des éléments présents en amont dans majority_all
                        # Cas où le nom correspondant au label a déjà été rencontré dans majority_all
                        # Vérification que les pixels de l'image sont réferncés dans le dico
                        if couple[0] in class_label_dico:

                            if class_label_dico[couple[0]] == class_label_dico[majority_all[idx_verif][0]]:
                                majority_all[idx_verif] = (majority_all[idx_verif][0] , majority_all[idx_verif][1] + couple[1])  # Ajout du nombre de pixels correspondant dans le couple précédent
                                majority_all.remove(couple)                                                                      # Supression du couple présentant le "doublon"
                                count_for_idx_couple = count_for_idx_couple + 1                                                  # Mise à jour du décompte de modifications
                                break
                        else:
                           raise NameError( "statisticsVectorRaster() : The image file (%s) contain pixel value '%d' not identified into class_label_dico" %(image_input, couple[0]))

            # Intégration des valeurs de majority all dans les colonnes
            for couple_value_count in majority_all :                             # Parcours de majority_all. Ex : couple_value_count = (803,45)
                if (couple_value_count is None) or (couple_value_count == "") :  # en cas de bug de rasterstats (erreur geometrique du polygone par exemple)
                    feedback.pushInfo(self.tr("Warning! Problem detected in polygon management %s" %(polygone_count)))
                    pass
                else :
                    nb_pixel_total = polygone_stats['count']       # Nbr de pixels du polygone
                    pixel_value = couple_value_count[0]            # Valeur du pixel
                    value_count = couple_value_count[1]            # Nbr de pixels ayant cette valeur
                    name_col = class_label_dico[pixel_value]       # Transformation de la valeur du pixel en "signification" au regard du dictionnaire. Ex : BD Topo ou 2011
                    name_col_area =  PREFIX_AREA_COLUMN + name_col # Identification du nom de la colonne en surfaces

                    if len(name_col) > 10:
                        name_col = name_col[:10]
                    if len(name_col_area) > 10:
                        name_col_area = name_col_area[:10]

                    value_area = pixel_size * value_count                                    # Calcul de la surface du polygone correspondant à la valeur du pixel
                    if nb_pixel_total != None and nb_pixel_total != 0:
                        percentage = (float(value_count)/float(nb_pixel_total)) * 100  # Conversion de la surface en pourcentages, arondi au pourcent
                    else :
                        feedback.pushInfo(self.tr("Waring! Problem in identifying the number of pixels in the polygon %s : the percentage of%s is set to 0" %(polygone_count,name_col)))
                        percentage = 0.0

                    if ('all' in col_to_add_list) or ('count' in col_to_add_list):
                        feature.SetField(name_col, percentage)      # Injection du pourcentage dans la colonne correpondante
                    if ('all_S' in col_to_add_list) or ('count_S' in col_to_add_list):
                        feature.SetField(name_col_area, value_area) # Injection de la surface dans la colonne correpondante
        else :
            pass

        # Remplissage des colonnes statistiques demandées ( col_to_add_inter01_list = [majority/DateMaj/SrcMaj, minority, min, max, mean, median, sum, std, unique, range] )
        for stats in col_to_add_inter01_list :

            if stats == 'DateMaj' or  stats == 'SrcMaj' :                # Cas particulier de 'DateMaj' et 'SrcMaj' : le nom de la colonne est DateMaj ou SrcMaj, mais la statistique utilisée est identifiée par majority
                name_col = stats                                         # Nom de la colonne. Ex : 'DateMaj'
                value_statis = polygone_stats['majority']                # Valeur majoritaire. Ex : '203'
                if value_statis == None:
                    value_statis_class = 'nan'
                else :
                    value_statis_class = class_label_dico[value_statis]  # Transformation de la valeur au regard du dictionnaire. Ex : '2011'
                feature.SetField(name_col, value_statis_class)           # Ajout dans la colonne

            elif (stats is None) or (stats == "") or (polygone_stats[stats] is None) or (polygone_stats[stats]) == "" or (polygone_stats[stats]) == 'nan' :
                # En cas de bug de rasterstats (erreur geometrique du polygone par exemple)
                pass

            else :
                name_col = stats                                         # Nom de la colonne. Ex : 'majority', 'max'
                value_statis = polygone_stats[stats]                     # Valeur à associer à la colonne, par exemple '2011'

                if (name_col == 'majority' or name_col == 'minority') and class_label_dico != [] : # Cas où la colonne fait référence à une valeur du dictionnaire
                    value_statis_class = class_label_dico[value_statis]
                else:
                    value_statis_class = value_statis

                feature.SetField(name_col, value_statis_class)

        layer.SetFeature(feature)
        feature.Destroy()

    # ETAPE 3/3 : SUPRESSION DES COLONNES NON SOUHAITEES

    if col_to_delete_list != []:

        for col_to_delete in col_to_delete_list :
            if layer_definition.GetFieldIndex(col_to_delete) != -1 :                   # Vérification de l'existence de la colonne col (retour = -1 : elle n'existe pas)
                layer.DeleteField(layer_definition.GetFieldIndex(col_to_delete))       # Suppression de la colonne

    # Fermeture du fichier shape
    layer.SyncToDisk()
    layer = None
    data_source.Destroy()

    return
