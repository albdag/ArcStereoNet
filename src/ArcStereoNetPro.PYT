#- - - - - - - - - - - - - I N I T I A L I Z A T I O N - - - - - - - - - - - - #

#----------------------------Import Internal Libraries-------------------------#

import arcpy, io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import numpy as np
from os import path, remove, startfile



#----------------------------Import  mplstereonet------------------------------#

try:
    from mplstereonet import *
    MPLSTEREONET_AVAILABLE = True
except ImportError:
    MPLSTEREONET_AVAILABLE = False

#-----------------------------Add output function------------------------------#

arcpy.env.addOutputsToMap = False                                               # manage the addition of output image into T.O.C.



#------------------------------Global  Variables-------------------------------#

contour_methods = {'Kamb (no smoothing)':'kamb',                                # Contour methods dictionary
                   'Kamb (linear smoothing)':'linear_kamb',
                   'Kamb (inverse square smoothing)':'square_kamb',
                   'Kamb (exponential smoothing)':'exponential_kamb',
                   'Schmidt (a.k.a. 1%)':'schmidt'}

colorMaps = ['Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds', 'bwr']   # Contour color maps

nets_dict = {'Schmidt (Equal Area)':'equal_area_stereonet',                     # Net types dictionary
             'Wulff (Equal Angle)':'equal_angle_stereonet'}

cardinals = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')                        # Supported Cardinal Points
card_hybrid = ('N', u'45\u00B0', u'90\u00B0', u'135\u00B0', u'180\u00B0',
               u'225\u00B0', u'270\u00B0', u'315\u00B0')

# https://xkcd.com/color/rgb/ for more colors
colors_dict = {'Red':'r', 'Green':'g', 'Blue':'b', 'Cyan':'c', 'Magenta':'m',   # Colors dictionary
               'Yellow':'y', 'Black':'k', 'Orange':'#ff8c00', 'Grey':'#808080',
               'Salmon':'#e9967a', 'Olive':'#808000', 'Violet':'#440d80',
               'Gold':'#daa520', 'Water green':'#00ff7f', 'Peach':'#ffb07c',
               'Light yellow':'#f0e68c', 'Greenish Blue':'#0b8b87',
               'Puke Brown':'#947706', 'Indigo':'#380282', 'Dust':'#b2996e',
               'Very Light Blue':'#d5ffff', 'Muddy Green':'#657432',
               'Dirty Pink':'#ca7b80'}

markers_dict = {'Point':'.', 'Square':'s', 'Triangle':'^', 'Hexagon':'H',       # Markers dictionary
                'Thin diamond':'d', 'Star':'*', 'Circle':'o', 'Diamond':'D',
                'Pentagon':'p'}
                # add 'Plus':'P' and 'X':'X' with updated version of matplotlib

lines_dict = {'Solid':'-', 'Dashed':'--', 'Dotted':':', 'Dashdot':'-.'}         # Lines dictionary



#- - - - - - - - - - - - U S E F U L  F U N C T I O N S - - - - - - - - - - - -#

#---------------------------Generate Temp File Path----------------------------#

def temp_filepath(file_name: str) -> str:
    """Generate a temporary file path in the ArcGIS scratch folder.

    Args:
        file_name (str): Name of the temporary file, including extension.

    Returns:
        str: Full temporary file path.
    """
    scratch_folder = arcpy.env.scratchFolder
    temp_path = path.join(scratch_folder, file_name)
    return temp_path



#-------------------------------Auto-Fill Fields-------------------------------#

def AutoFill_fields(param, field_filter, default_name):

    all_defaults = (None, 'azimuth', 'dip_angle', 'method', 'type')
    current_value = param.valueAsText
    try:
        current_value = current_value.lower()
    except AttributeError:
        pass

    if not param.altered or current_value in all_defaults:
        for name in field_filter:
            if name.lower() == default_name.lower():
                param.value = name
                break
        else:
            param.value = None



#-----------------------------Auto-Fill ValueTable-----------------------------#

def AutoFill_vT(param, cols, default_values, checkList=True):

    vT = param.values

    for col, val in zip(cols, default_values):
        if checkList:
            editCol = vT[-1][col] not in param.filters[col].list
        else:
            editCol = vT[-1][col] in (0, u'')

        if editCol:
            vT[-1][col] = val

    param.values = vT



#------------------------------Parameter Changed-------------------------------#

def paramChanged(param):

    if param.altered and not param.hasBeenValidated:
        return True
    else:
        return False



#-------------------Distinguish Planar from Linear features--------------------#

def filterData(dataset, type):

    ''' dataset-> [[azimuth, dip angle, method, type], ...]'''

    strikes, dips, trends, plunges = [], [], [], []

    for d in dataset:
        if d[3] == type:

            if d[2] == 'RHR':
                strikes.append(d[0])
                dips.append(d[1])
            else:
                trends.append(d[0])
                plunges.append(d[1])

    return strikes, dips, trends, plunges



#---------------------------Convert Unicode to String--------------------------#

def uni2str(listObject):

    try:
        for i in listObject:
            if type(i) == unicode:
                listObject[listObject.index(i)] = str(i)
    except NameError: # deal with python 3 error for unicode data type
        pass
    
    return listObject



#- - - - - - - - - - - - M. E. A. D.  A L G O R I T H M - - - - - - - - - - - -#

def cos_r(values):
    return np.cos(np.radians(values))


def sin_r(values):
    return np.sin(np.radians(values))


def linearize(values):
    return [v if abs(v-min(values)) <= 180 else v-360 for v in values]

def isObtuse(values):
    values_fix = linearize(values)
    if max(values_fix) - min(values_fix) > 180:
        return True
    else:
        return False


def avg_angle(values, weights=None):

    if weights == None:
        weights = np.ones(len(values))
    elif sum(weights) == 0:
        return np.NaN, np.NaN

    x = (cos_r(values)*weights).sum() / sum(weights)
    y = (sin_r(values)*weights).sum() / sum(weights)

    avg_deg = int(round(np.degrees(np.arctan2(y, x)) % 360, 0))
    R_mean = np.sqrt(x**2 + y**2)

    return avg_deg, R_mean


def median_angles(values):

    values_fix = linearize(values)
    res = np.median(values_fix) % 360

    return res


def group_angles(angles, med_str, med_dip, tol_str, tol_dip):
    grouped, excluded = [], []
    gamma = True

    while gamma:
        gamma = False

        for a in angles:
            S, D = a

            if (
            abs(sin_r(S) - sin_r(med_str)) <= 2*tol_str and
            abs(cos_r(S) - cos_r(med_str)) <= 2*tol_str and
            abs(sin_r(D) - sin_r(med_dip)) <= tol_dip
            ):

                grouped.append(a)
                angles.remove(a)
                med_str = median_angles([g[0] for g in grouped])
                med_dip = np.median([g[1] for g in grouped])
                gamma = True


    for g in grouped:                                                           # post-grouping cleaning of data.
            S, D = g

            if (
            abs(sin_r(S) - sin_r(med_str)) > 2*tol_str or
            abs(cos_r(S) - cos_r(med_str)) > 2*tol_str or
            abs(sin_r(D) - sin_r(med_dip)) > tol_dip
            ):
                excluded.append(g)
                grouped.remove(g)

    excluded.extend(angles)
    return grouped, excluded


def angles_clustering(strikes, dips, measurement='poles', tol_str=0.5, tol_dip=0.3, n_clusts=None):

    if measurement == 'lines':
        strikes, dips = dips, strikes

    max_Sfreq = float(max([strikes.count(s) for s in strikes]))
    norm_Sfreq = {s: round(strikes.count(s)/max_Sfreq, 3) for s in strikes}
    max_Dfreq = float(max([dips.count(d) for d in dips]))
    norm_Dfreq = {d: round(dips.count(d)/max_Dfreq, 3) for d in dips}

    angles = [(s, d) for s,d in zip(strikes, dips)]

    families = []
    SDfreq_tresh = 2.000

    while SDfreq_tresh > 0.000:
        for a in angles:
            if norm_Sfreq[a[0]] + norm_Dfreq[a[1]] == SDfreq_tresh:
                grouped, excluded = group_angles(angles, a[0], a[1], tol_str, tol_dip)
                break

        try:
            angles = excluded

            if len(grouped) > 0:
                families.append([(g[0], g[1]) for g in grouped])
                grouped = []
            else:
                raise UnboundLocalError

        except UnboundLocalError:
            SDfreq_tresh = round(SDfreq_tresh - 0.001, 3)



    spurious = angles

    if n_clusts == None:
        for fam in families[:]:
            if len(fam) < len(max(families, key=len))/3.0:                      # 1/3 of main family ("dead" if, may be useful for future updates)
                spurious.extend(fam)
                families.remove(fam)

    else:                                                                       # Number of clusts selected by user
        while True:
            try:
                spurious.extend(sorted(families, key=len, reverse=True)[n_clusts:])
                families = sorted(families, key=len, reverse=True)[:n_clusts]
                if min([len(f) > 1 for f in families]) == True:
                    break
                else:
                    raise IndexError
            except IndexError:
                n_clusts -= 1
            except ValueError:  # Raises when families is an empty list
                return []



    return families                                                             # "return families, spurious" to also obtain spurious values list



#- - - - - - A R C S T E R E O N E T    I N I T I A L I S A T I O N - - - - - -#

class Toolbox(object):

    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""

        self.label = 'ArcStereoNet'
        self.alias = 'ArcStereoNet'

        # List of tool classes associated with this toolbox
        self.tools = [RoseDiagrams, StereoPlots, GraphToHyperlink]


class _ToolBase(object):

    def __init__(self) -> None:

        """Base class for all tools."""

        self.label = 'Tool label'
        self.description = 'Tool description.'
        self.canRunInBackground = False


    def checkExternalDependencies(self, messages) -> bool:

        """Check for the availabilty of all required external Python libraries.
        Add dedicated error message if dependencies are missing.

        Args:
            messages: Geoprocessing messages object.

        Returns:
            bool: Whether all external dependencies are available.
        """

        if MPLSTEREONET_AVAILABLE:
            return True
        else:
            link = "https://github.com/albdag/ArcStereoNet/wiki/1.-Installation#installation-for-arcgis-pro"
            error_msg = "Mplstereonet dependency is missing from your " \
            "current Python environment. For more info, check the " \
            f"installation guide at {link}"
            messages.addErrorMessage(error_msg)
            return False


#- - - - - - - - - - R  O  S  E     D  I  A  G  R  A  M  S - - - - - - - - - - #

class RoseDiagrams(_ToolBase):

    def __init__(self):

        """Define the tool (tool name is the name of the class)."""

        super().__init__()
        self.label = 'Rose Diagrams'
        self.description = 'Use this tool to plot rose diagrams'



#- - - - - - - - - - - - - - - - - PARAMETERS - - - - - - - - - - - - - - - - -#

    def getParameterInfo(self):

        """Define parameter definitions"""



#---------------------------Input feature (param0)-----------------------------#

        param0 = arcpy.Parameter(displayName = 'Input Feature',
                                 name = 'in_feature',
                                 datatype = 'GPFeatureLayer',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param0.filter.list = ['Point', 'Polyline', 'Polygon']



#---------------------------Azimuth Field (param1)-----------------------------#

        param1 = arcpy.Parameter(displayName = 'Azimuth Field',
                                 name = 'azm_field',
                                 datatype = 'Field',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param1.filter.list = ['Short', 'Long', 'Float', 'Single', 'Double']
        param1.parameterDependencies = [param0.name]



#-----------------------------Type Field (param2)------------------------------#

        param2 = arcpy.Parameter(displayName = 'Type Field',
                                 name = 'type_field',
                                 datatype = 'Field',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param2.filter.list = ['Text']
        param2.parameterDependencies = [param0.name]



#------------------------Data to be plotted (param3)---------------------------#

        param3 = arcpy.Parameter(displayName = 'Data to be plotted',
                                 name = 'plotting_data',
                                 datatype = 'GPValueTable',
                                 parameterType = 'Required',
                                 direction = 'Input',
                                 multiValue = True)

        param3.columns = [['GPString', 'Data Type'],
                          ['GPString', 'Color'],
                          ['GPString', 'Show M.E.A.D. Mean Vector(s)'],
                          ['GPLong', 'Number of Clusters'],
                          ['GPLong', 'M.E.A.D. Azimuth tolerance(%)']]

        param3.filters[0].type = 'ValueList'
        param3.filters[1].type = 'ValueList'
        param3.filters[2].type = 'ValueList'
        param3.filters[4].type = 'Range'

        param3.filters[0].list = ['NONE']
        param3.filters[1].list = ['Random'] + sorted(colors_dict.keys())
        param3.filters[2].list = ['NO', 'YES']
        param3.filters[4].list = [0, 100]



#---------------------Store Image Output checkbox (param4)---------------------#

        param4 = arcpy.Parameter(displayName = 'Store Image Output',
			                     name = 'store_img',
			                     datatype = 'GPBoolean',
			                     parameterType = 'Optional',
			                     direction = 'Input')

        param4.value = True



#-----------------------------Output image (param5)----------------------------#

        param5 = arcpy.Parameter(displayName = 'Output Image',
                                 name = 'output_img',
                                 datatype = 'DEFile',
                                 parameterType = 'Required',
                                 direction = 'Output')

        param5.filter.list = ['png', 'eps', 'pdf', 'pgf', 'ps', 'raw', 'rgba',
                              'svg', 'svgz']



#--------------------------------Image DPI (param6)----------------------------#

        param6 = arcpy.Parameter(displayName = 'Image Resolution (DPI)',
                                 name = 'img_DPI',
                                 datatype = 'GPLong',
                                 parameterType = 'Optional',
                                 direction = 'Input')

        param6.value = 200
        param6.filter.type = 'Range'
        param6.filter.list = [0, 600]



#----------------------------Title Label (param7)------------------------------#

        param7 = arcpy.Parameter(displayName = 'Title Label',
                                 name = 'title_label',
                                 datatype = 'GPString',
                                 parameterType = 'Optional',
                                 direction = 'Input')

        param7.category = 'Plot Customisation'



#-------------------------Show Grid checkbox (param8)--------------------------#

        param8 = arcpy.Parameter(displayName = 'Show Grid',
			         name = 'show_grid',
			         datatype = 'GPBoolean',
			         parameterType = 'Optional',
			         direction = 'Input')

        param8.category = 'Plot Customisation'

        param8.value = True



#------------------------Show Legend checkbox (param9)-------------------------#

        param9 = arcpy.Parameter(displayName = 'Show Legend',
			         name = 'show_legend',
                                 datatype = 'GPBoolean',
			         parameterType = 'Optional',
			         direction = 'Input')

        param9.category = 'Plot Customisation'

        param9.value = True



#----------------Show Samples Number label checkbox (param10)------------------#

        param10 = arcpy.Parameter(displayName = 'Show Samples Number label',
			          name = 'show_nSamples',
		                  datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param10.category = 'Plot Customisation'

        param10.value = True



#-------------------Mirrored behaviour checkbox (param11)----------------------#

        param11 = arcpy.Parameter(displayName = 'Mirrored behaviour',
			          name = 'mirrored',
			          datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param11.category = 'Plotting Options'

        param11.value = False



#----------------------Weighted Plot checkbox (param12)------------------------#

        param12 = arcpy.Parameter(displayName = 'Weighted Rose Diagram',
			          name = 'weighted',
			          datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param12.category = 'Plotting Options'

        param12.value = False



#---------------------------Weight Field (param13)-----------------------------#

        param13 = arcpy.Parameter(displayName = 'Weight Field',
                                  name = 'wgt_field',
                                  datatype = 'Field',
                                  parameterType = 'Required',
                                  direction = 'Input',
                                  enabled = False)

        param13.category = 'Plotting Options'

        param13.filter.list = ['Short', 'Long', 'Float', 'Single', 'Double']
        param13.parameterDependencies = [param0.name]



#---------------------Write Log File checkbox (param14)------------------------#

        param14 = arcpy.Parameter(displayName = 'Write Log File',
			          name = 'write_log',
			          datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param14.category = 'Plotting Options'

        param14.value = False



#--------------------------Tick Marks Spacing (param15)------------------------#

        param15 = arcpy.Parameter(displayName = 'Tick Marks Spacing (degrees)',
                                  name = 'tickspacing',
                                  datatype = 'GPLong',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param15.category = 'Plot Customisation'

        param15.value = 10
        param15.filter.type = 'Range'
        param15.filter.list = [1, 360]



#------------------Radius Labels Angular Position (param16)--------------------#

        param16 = arcpy.Parameter(displayName = 'Radius labels angular position (degrees)',
                                  name = 'rGrids_angle',
                                  datatype = 'GPLong',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param16.category = 'Plot Customisation'

        param16.value = 0
        param16.filter.type = 'Range'
        param16.filter.list = [0, 360]



#-----------------------------Grid Space (param17)-----------------------------#

        param17 = arcpy.Parameter(displayName = 'Set Grid Space',
                                  name = 'gridSpace',
                                  datatype = 'GPLong',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param17.category = 'Plot Customisation'

        param17.value = 8



#-------------------------------Parameters LIST--------------------------------#

        parameters = [param0, param1, param2, param3, param4, param5, param6,
                      param7, param8, param9, param10, param11, param12,
                      param13, param14, param15, param16, param17]

        return parameters



#- - - - - - - - - - - O P T I O N A L  F U N C T I O N S - - - - - - - - - - -#


#- - - - - - - - - - - - - - - UPDATE  PARAMETERS - - - - - - - - - - - - - - -#

    def updateParameters(self, parameters):

        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""


#-------------Updating parameters[1, 2] according to parameters[0]-------------#

        if paramChanged(parameters[0]):                                         # parameters[0] = Input Feature
                                                                                # parameters[1] = Azimuth Field
            try:                                                                # parameters[2] = Type Field
                fields = arcpy.Describe(parameters[0].value).fields
                string_fields = [f.name for f in fields if f.type == 'String']
                num_types = ('SmallInteger', 'Integer', 'Single', 'Double')
                num_fields = [f.name for f in fields if f.type in num_types]

                AutoFill_fields(parameters[1], num_fields, 'Azimuth')
                AutoFill_fields(parameters[2], string_fields, 'Type')

            except (AttributeError, IOError):
                for p in parameters[1:3]:
                    p.value = None



#-------------Updating parameters[3] according to parameters[0, 2]-------------#
                                                                                # parameters[0] = Input Feature
        if parameters[2].altered:                                               # parameters[2] = Type Field
                                                                                # parameters[3] = Data to be plotted
            try:
                cursor = arcpy.da.SearchCursor(parameters[0].valueAsText,
                                               parameters[2].valueAsText)
                entries_p3 = set([str(c[0]) for c in cursor if str(c[0]).strip() != ''])
                parameters[3].filters[0].list = sorted(list(entries_p3))

            except (RuntimeError, TypeError):
                parameters[3].filters[0].list = ['NONE']

        else:
            parameters[3].filters[0].list = ['NONE']


        if parameters[3].altered and parameters[3].values != None:
            AutoFill_vT(parameters[3], (1, 2), ('Random', 'NO'))
            AutoFill_vT(parameters[3], (3, 4), (1, 100), checkList=False)
        else:
            pass



#--------------Updating parameters[5] according to parameters[4]---------------#
                                                                                # parameters[4] = Store Image Output checkbox
        if parameters[4].altered:                                               # parameters[5] = Output Image

            if parameters[4].value == False:
                parameters[5].value = temp_filepath('RoseDiagram_temp.png')
                parameters[5].enabled = False

            else:
                parameters[5].enabled = True



#------------Updating parameters[13] according to parameters[1, 12]------------#
                                                                                # parameters[1] = Azimuth Value
        if parameters[1].altered or paramChanged(parameters[12]):               # parameters[12] = Weighted Plot checkbox
                                                                                # parameters[13] = Weight Field
            if parameters[12].value:
                parameters[13].enabled = True

            else:
                parameters[13].value = parameters[1].valueAsText
                parameters[13].enabled = False



        return



#--------------------------------Other Functions-------------------------------#

    def updateMessages(self, parameters):

        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""

#---Updating parameters[11] warn message according to parameters[0, 1, 2, 3]---#
                                                                                # parameters[0] = Input Feature
        if (paramChanged(parameters[1]) or                                      # parameters[1] = Azimuth Field
            paramChanged(parameters[3]) or                                      # parameters[2] = Type Field
            paramChanged(parameters[11])):                                      # parameters[3] = Data to be plotted
                                                                                # parameters[11] = Mirrored behaviour checkbox
            if parameters[11].value:

                try:
                    dataRange = {}
                    cursor =  arcpy.da.SearchCursor(parameters[0].valueAsText,
                                                    (parameters[1].valueAsText,
                                                     parameters[2].valueAsText))

                    for c in cursor:
                        if str(c[1]) in list(zip(*parameters[3].values))[0]:
                            if str(c[1]) not in dataRange:
                                dataRange[str(c[1])] = [c[0]]
                            else:
                                dataRange[str(c[1])].append(c[0])

                    warnList = [k for k, v in dataRange.items() if isObtuse(v)]
                    if len(warnList) != 0:
                        msg = 'The following data types seem to cover a range greater than 180 degrees: {}'.format(str(warnList))
                        parameters[11].setWarningMessage(msg)
                    else:
                        parameters[11].clearMessage()

                except (RuntimeError, TypeError):
                    pass

            else:
                parameters[11].clearMessage()



#----Updating parameters[13] warn message according to parameters[0, 12, 13]---#
                                                                                #parameters[0] = Input Feature
        if parameters[12].value:                                                #parameters[12] = Weighted Plot checkbox
                                                                                #parameters[13] = Weight Field
            try:
                wgts = arcpy.da.SearchCursor(parameters[0].valueAsText,
                                             parameters[13].valueAsText)
                if min(wgts)[0] < 0:
                    msg = 'The selected field contains negative values. Weights should be >= 0.'
                    parameters[13].setWarningMessage(msg)
                else:
                    parameters[13].clearMessage()

            except (RuntimeError, TypeError):
                pass

        else:
            parameters[13].clearMessage()



        return



    def isLicensed(self):

        """Set whether tool is licensed to execute."""

        return True



#- - - - - - - - - - - - - T O O L  E X E C U T I O N - - - - - - - - - - - - -#

    def execute(self, parameters, messages):

        """The source code of the tool."""
        
        ok = self.checkExternalDependencies(messages)
        if not ok:
            return


#---------------------------Parameters assignation-----------------------------#

        IN_FEATURE = parameters[0]
        AZM_FIELD = parameters[1]
        TYPE_FIELD = parameters[2]
        PLOTTING_DATA = parameters[3]
        STORE_IMG = parameters[4]
        OUTPUT_IMG = parameters[5]
        IMG_DPI = parameters[6]
        TITLE_LABEL = parameters[7]
        SHOW_GRID = parameters[8]
        SHOW_LEGEND = parameters[9]
        SHOW_NSAMPLES = parameters[10]
        MIRRORED = parameters[11]
        WEIGHTED = parameters[12]
        WGT_FIELD = parameters[13]
        WRITE_LOG = parameters[14]
        TICKSPACING = parameters[15]
        RGRIDS_ANGLE = parameters[16]
        GRIDSPACE = parameters[17]



#-----------------------------Setting Figure Title-----------------------------#

        title = IN_FEATURE.valueAsText
        if TITLE_LABEL.value:
            title = TITLE_LABEL.valueAsText



#-------------------------------Importing Data---------------------------------#

        fields_params = [AZM_FIELD, TYPE_FIELD]
        if WEIGHTED.value:
            fields_params.append(WGT_FIELD)
        fields = tuple(fp.valueAsText for fp in fields_params)                  # Supported field names, later referred as: [0],[1],([2])
        source = IN_FEATURE.valueAsText
        data_array = arcpy.da.FeatureClassToNumPyArray(source, fields,
                                                       null_value='')
        data = data_array.tolist()

        if STORE_IMG.value:
            coords = arcpy.da.FeatureClassToNumPyArray(source, 'Shape')         # Extracting centroid coordinates
            centroid = np.nanmean(coords.tolist(), axis=0)[0]



#-------------------------Pre-Elaboration of Data------------------------------#

#>>> Initialize projected data types set & statistics dictionaries for log file

        proj_types = set()

        if WRITE_LOG.value:
            stats_log = {}



#-----------------------Initialization of MPLStereonet-------------------------#

        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(111, projection = 'polar')

        ax.set_title(title, y = 1.1, fontsize = 18)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        gridFormat = np.arange(0, 360, TICKSPACING.value)
        ax.set_thetagrids(gridFormat, gridFormat)



#----------------------------Plotting of Data----------------------------------#

        barLengths = []

        for row in PLOTTING_DATA.values:
            row = uni2str(row)                                                  # Converting from unicode to string
            _type, _color, _meanVect, _nClusts, _azmTol = row

            if _color == 'Random':                                              # Select a random color if not specified by the user
                random_index = np.random.randint(len(colors_dict))
                _color = list(colors_dict.keys())[random_index]

            proj_types.add(_type)                                               # Add data type to projected data types set


            strikes = [s[0] % 360 for s in data if s[1] == _type]
            if WEIGHTED.value:
                weights = [w[2] for w in data if w[1] == _type]

            strike_freq, bin_edges = np.histogram(strikes,
                                                  np.arange(-5, 366, 10),
                                                  weights = weights if WEIGHTED.value else None)

            strike_freq[0] += strike_freq[-1]

            if MIRRORED.value:
                half = np.sum(np.split(strike_freq[:-1], 2), 0)
                dataShow = np.concatenate([half, half])
            else:
                dataShow = strike_freq[:-1]

            barLengths.append(max(dataShow))

            ax.bar(np.radians(np.arange(0, 360, 10)), bottom = 0.0,
                   height = dataShow, width = np.radians(10),
                   align = 'center',
                   color = colors_dict[_color],
                   edgecolor = 'w' if _color == 'Black' else 'k',
                   label = _type)


            if _meanVect == 'YES':                                              # Plot mean vector if required

                wgt = weights if WEIGHTED.value else [1]*len(strikes)
                _azmTol = _azmTol / 100.

                mead_clust = angles_clustering(strikes, wgt,
                                               tol_str=_azmTol,
                                               tol_dip=1,                       # Treat weights as dips data and set the dip tolerance to 100%
                                               n_clusts=_nClusts)               # so that the output couples will be (strike-weight) and the
                                                                                # weight info for each strike value is preserved and used inside avg_angle function
                for mc in mead_clust:
                    a, w = zip(*mc)

                    avg_deg, R_mean = avg_angle(a, w)

                    quiver_ec = 'black' if _color == 'Red' else 'red'
                    n_quivs = 2 if MIRRORED.value else 1
                    sign = 1
                    for q in range(n_quivs):
                        ax.quiver(0, 0,
                                  sign * sin_r(avg_deg), sign * cos_r(avg_deg),
                                  facecolor = colors_dict[_color],
                                  edgecolor = quiver_ec,
                                  linewidth = 1.25,
                                  scale = 2. / R_mean,
                                  width = 0.01,
                                  zorder = 2)
                        sign *= -1


                    if WRITE_LOG.value:
                        if n_quivs == 2:
                            avg_deg = '{} ({})'.format(avg_deg, (avg_deg+180)%360)
                        if _type not in stats_log:
                            stats_log[_type] = (avg_deg, R_mean)
                        else:
                            stats_log[_type] += (avg_deg, R_mean)




#----------------------------Adjusting radius labels---------------------------#

        radii = np.linspace(10**-10, max(barLengths), GRIDSPACE.value)          # in some mpl versions radii must be strictly positive values, therefore 10**-10 instead of 0
        format_type = '%.2f'
        if radii[1] <= 10**-3 or max(radii) >= 10**6:                           # radii[0] is always 0, then we check radii[1]
            format_type = '%.2e'

        ax.set_rgrids(radii = radii,
                      fmt = format_type,
                      angle = RGRIDS_ANGLE.value,
                      weight = 'semibold',
                      size = 10,
                      color = 'white',
                      path_effects = [mpe.withStroke(linewidth=2, foreground='k')])


#-------Adding Grid, Weighted on, Legend and n.of Data Label to the plot-------#

        ax.grid(SHOW_GRID.value)


        if WEIGHTED.value:
            ax.text(0.0, -0.1,
                    u'Weighted on:\n{}'.format(WGT_FIELD.valueAsText),
                    fontsize = 8,
                    transform = ax.transAxes)


        if SHOW_LEGEND.value == True:
            from collections import OrderedDict
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))

            if len(by_label) > 0:
                ax.legend(by_label.values(),
                          by_label.keys(),
                          loc = 'upper left',
                          bbox_to_anchor = (1.0, 0.5, 0.5, 0.6),
                          ncol = 1,
                          fontsize = 8,
                          numpoints = 1)


        if SHOW_NSAMPLES.value == True:
            txt4lbl = ''
            for pt in sorted(proj_types):
                pt_num = sum([D.count(pt) for D in data])
                txt4lbl += pt + ' = ' + str(pt_num) + '\n'

            ax.text(1.0, -0.1,
                    txt4lbl,
                    fontsize = 8,
                    transform = ax.transAxes)



#-------------------------------Write Log File---------------------------------#

        if WRITE_LOG.value:
            logPath = path.splitext(OUTPUT_IMG.valueAsText)[0] + '_LOG.txt'

            with io.open(logPath, 'w') as LOG:
                LOG.write(title.upper() + '\n\n')

                if SHOW_NSAMPLES.value == False:
                    txt4lbl = ''
                    for pt in sorted(proj_types):
                        pt_num = sum([D.count(pt) for D in data])
                        txt4lbl += pt + ' = ' + str(pt_num) + '\n'
                LOG.write(txt4lbl + u'\n')

                if WEIGHTED.value:
                    LOG.write(u'Weighted on {}\n\n'.format(WGT_FIELD.valueAsText))

                if len(stats_log) > 0:
                    LOG.write(u'STATISTICS\n')
                    for key, val in stats_log.items():
                        avgs = '\t'.join(['{} deg'.format(i) for i in val[0::2]])
                        R_means = '\t'.join([str(j) for j in val[1::2]])
                        avg_desc = '{} mean vector(s) direction ->\t{}'.format(key, avgs)
                        R_mean_desc = '{} mean resultant(s) length ->\t{}'.format(key, R_means)
                        LOG.write(avg_desc + u'\n' + R_mean_desc + '\n\n')

                LOG.write(u'\nLog file automatically compiled by ArcStereoNet')
                LOG.close()



#----------------------Save the figure and End the tool------------------------#

        plt.savefig(OUTPUT_IMG.valueAsText,
                    dpi = IMG_DPI.value,
                    bbox_inches = 'tight')
        plt.close('all')

        if STORE_IMG.value == False:
            startfile(OUTPUT_IMG.valueAsText)
            if WRITE_LOG.value:
                startfile(logPath)

        elif OUTPUT_IMG.valueAsText.endswith('.png'):
            arcpy.BuildPyramids_management(OUTPUT_IMG.valueAsText)

            xypath = path.splitext(OUTPUT_IMG.valueAsText)[0] + '_xy.txt'       # Saving centroid coordinates
            clon, clat = centroid
            descr = arcpy.Describe(IN_FEATURE.valueAsText)
            with io.open(xypath, 'w') as xyfile:
                xyfile.write(u'{}\n'.format(clon))
                xyfile.write(u'{}\n'.format(clat))
                xyfile.write(u'{}\n'.format(path.join(descr.path, descr.name)))



        return




################################################################################
################################################################################
################################################################################


#- - - - - - - - - S   T   E   R   E   O   P   L   O   T   S - - - - - - - - - #

class StereoPlots(_ToolBase):

    def __init__(self):

        """Define the tool (tool name is the name of the class)."""

        super().__init__()
        self.label = 'Stereoplots'
        self.description = 'Add description.'



#- - - - - - - - - - - - - - - - - PARAMETERS - - - - - - - - - - - - - - - - -#

    def getParameterInfo(self):

        """Define parameter definitions"""



#---------------------------Input feature (param0)-----------------------------#

        param0 = arcpy.Parameter(displayName = 'Input Feature',
                                 name = 'in_feature',
                                 datatype = 'GPFeatureLayer',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param0.filter.list = ['Point', 'Polyline', 'Polygon']



#---------------------------Azimuth Field (param1)-----------------------------#

        param1 = arcpy.Parameter(displayName = 'Azimuth Field',
                                 name = 'azm_field',
                                 datatype = 'Field',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param1.filter.list = ['Short', 'Long', 'Float', 'Single', 'Double']
        param1.parameterDependencies = [param0.name]



#--------------------------Dip Angle Field (param2)----------------------------#

        param2 = arcpy.Parameter(displayName = 'Dip Angle Field',
                                 name = 'dip_field',
                                 datatype = 'Field',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param2.filter.list = ['Short', 'Long', 'Float', 'Single', 'Double']
        param2.parameterDependencies = [param0.name]



#----------------------------Method Field (param3)-----------------------------#

        param3 = arcpy.Parameter(displayName = 'Method Field',
                                 name = 'met_field',
                                 datatype = 'Field',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param3.filter.list = ['Text']
        param3.parameterDependencies = [param0.name]



#-----------------------------Type Field (param4)------------------------------#

        param4 = arcpy.Parameter(displayName = 'Type Field',
                                 name = 'type_field',
                                 datatype = 'Field',
                                 parameterType = 'Required',
                                 direction = 'Input')

        param4.filter.list = ['Text']
        param4.parameterDependencies = [param0.name]



#---------------------------Plotting Data (param5)-----------------------------#

        param5 = arcpy.Parameter(displayName = 'Plot Cyclographic Traces, Poles and Vectors',
                                 name = 'plotData',
                                 datatype = 'GPValueTable',
                                 parameterType = 'Optional',
                                 direction = 'Input')

        param5.columns = [['GPString', 'Data Type'],
                          ['GPString', 'Color'],
                          ['GPString', 'Pole/Vector Symbol'],
                          ['GPDouble', 'Pole/Vector Size'],
                          ['GPString', 'Cyclographic Trace Style'],
                          ['GPDouble', 'Line Width']]

        param5.filters[0].type = 'ValueList'
        param5.filters[1].type = 'ValueList'
        param5.filters[2].type = 'ValueList'
        param5.filters[4].type = 'ValueList'

        param5.filters[0].list = ['NONE']
        param5.filters[1].list = ['Random'] + sorted(colors_dict.keys())
        param5.filters[2].list = ['NONE'] + sorted(markers_dict.keys())
        param5.filters[4].list = ['NONE'] + sorted(lines_dict.keys())



#---------------------Store Image Output checkbox (param6)---------------------#

        param6 = arcpy.Parameter(displayName = 'Store Image Output',
			         name = 'store_img',
			         datatype = 'GPBoolean',
			         parameterType = 'Optional',
			         direction = 'Input')

        param6.value = True



#-----------------------------Output image (param7)----------------------------#

        param7 = arcpy.Parameter(displayName = 'Output Image',
                                 name = 'output_img',
                                 datatype = 'DEFile',
                                 parameterType = 'Required',
                                 direction = 'Output')

        param7.filter.list = ['png', 'eps', 'pdf', 'pgf', 'ps',
                              'raw', 'rgba', 'svg', 'svgz']



#-----------------------------Image DPI (param8)-------------------------------#

        param8 = arcpy.Parameter(displayName = 'Image Resolution (DPI)',
                                 name = 'image_DPI',
                                 datatype = 'GPLong',
                                 parameterType = 'Optional',
                                 direction = 'Input')

        param8.value = 200
        param8.filter.type = 'Range'
        param8.filter.list = [0, 600]



#-----------------------------Net Type (param9)--------------------------------#

        param9 = arcpy.Parameter(displayName = 'Net Type',
        			 name = 'net_type',
                                 datatype = 'GPString',
                                 parameterType = 'Optional',
			         direction = 'Input')

        param9.category = 'Plotting Options'

        param9.value = 'Schmidt (Equal Area)'
        param9.filter.type = 'ValueList'
        param9.filter.list = list(nets_dict.keys())



#---------------------Write Log File checkbox (param10)------------------------#

        param10 = arcpy.Parameter(displayName = 'Write Log File',
			          name = 'write_log',
			          datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param10.category = 'Plotting Options'

        param10.value = False



#---------------------------Title Label (param11)------------------------------#

        param11 = arcpy.Parameter(displayName = 'Title Label',
                                  name = 'title_label',
                                  datatype = 'GPString',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param11.category = 'Plot Customisation'



#------------------------Tick Marks Number (param12)---------------------------#

        param12 = arcpy.Parameter(displayName = 'Set Number of Tick Marks',
                                  name = 'n_ticks',
                                  datatype = 'GPLong',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param12.category = 'Plot Customisation'

        param12.filter.type = 'ValueList'
        param12.filter.list = [0, 1, 4, 8]
        param12.value = 8



#--------------------------Tick Marks Type (param13)---------------------------#

        param13 = arcpy.Parameter(displayName = 'Tick Marks Type',
                                  name = 'ticks_type',
                                  datatype = 'GPString',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param13.category = 'Plot Customisation'

        param13.filter.type = 'ValueList'
        param13.filter.list = ['Angular Values', 'Cardinal Points', 'Hybrid']
        param13.value = 'Angular Values'



#-------------------------Show Grid checkbox (param14)-------------------------#

        param14 = arcpy.Parameter(displayName = 'Show Grid',
			          name = 'show_grid',
			          datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param14.category = 'Plot Customisation'

        param14.value = True



#-----------------------Show Legend checkbox (param15)-------------------------#

        param15 = arcpy.Parameter(displayName = 'Show Legend',
			          name = 'show_legend',
                                  datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param15.category = 'Plot Customisation'

        param15.value = True



#-----------------Show Samples Number label checkbox (param16)-----------------#

        param16 = arcpy.Parameter(displayName = 'Show Samples Number label',
                                  name = 'show_nSamples',
			          datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param16.category = 'Plot Customisation'

        param16.value = True



#---------------------------Apply Contour (param17)----------------------------#

        param17 = arcpy.Parameter(displayName = 'Apply Contour',
                                  name = 'contour',
                                  datatype = 'GPValueTable',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param17.columns = [['GPString', 'Data Type'],
                           ['GPString', 'Method'],
                           ['GPDouble', u'Standard deviation (\u03C3)'],
                           ['GPString', 'Style'],
                           ['GPString', 'Color'],
                           ['GPLong', 'Transparency (%)']]

        param17.category = 'Contour & Statistics'

        param17.filters[0].type = 'ValueList'
        param17.filters[1].type = 'ValueList'
        param17.filters[3].type = 'ValueList'
        param17.filters[4].type = 'ValueList'
        param17.filters[5].type = 'Range'

        param17.filters[0].list = ['NONE']
        param17.filters[1].list = sorted(contour_methods.keys())
        param17.filters[3].list = ['Filled', 'Unfilled']
        param17.filters[4].list = ['Random'] + colorMaps
        param17.filters[5].list = [0, 100]



#---------------------------Show Colorbar (param18)----------------------------#

        param18 = arcpy.Parameter(displayName = 'Show Contour Colorbar',
			          name = 'show_colorbar',
			          datatype = 'GPBoolean',
			          parameterType = 'Optional',
			          direction = 'Input')

        param18.category = 'Contour & Statistics'

        param18.value = True



#------------------------Extract mean vectors (param19)------------------------#

        param19 = arcpy.Parameter(displayName = 'Extract Mean Vector(s)',
                                  name = 'extract_means',
                                  datatype = 'GPValueTable',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param19.columns = [['GPString', 'Data Type'],
                           ['GPString', 'Algorithm'],
                           ['GPLong', 'Number of Clusters'],
                           ['GPLong', 'M.E.A.D. Azimuth tolerance (%)'],
                           ['GPLong', 'M.E.A.D. Inclination tolerance (%)'],
                           ['GPLong', 'Fisher confidence (%)'],
                           ['GPString', 'Color'],
                           ['GPString', 'Pole/Vector Symbol'],
                           ['GPDouble', 'Pole/Vector Size'],
                           ['GPString', 'Cyclographic Trace Style'],
                           ['GPDouble', 'Line Width']]

        param19.category = 'Contour & Statistics'

        param19.filters[0].type = 'ValueList'
        param19.filters[1].type = 'ValueList'
        param19.filters[3].type = 'Range'
        param19.filters[4].type = 'Range'
        param19.filters[5].type = 'Range'
        param19.filters[6].type = 'ValueList'
        param19.filters[7].type = 'ValueList'
        param19.filters[9].type = 'ValueList'

        param19.filters[0].list = ['NONE']
        param19.filters[1].list = ['M.E.A.D.', 'M.E.A.D. + Fisher', 'K-means',
                                   'Bingham']
        param19.filters[3].list = [0, 100]
        param19.filters[4].list = [0, 100]
        param19.filters[5].list = [0, 99]
        param19.filters[6].list = ['Random'] + sorted(colors_dict.keys())
        param19.filters[7].list = ['NONE'] + sorted(markers_dict.keys())
        param19.filters[9].list = ['NONE'] + sorted(lines_dict.keys())



#----------------------Track M.E.A.D. behaviour (param20)----------------------#

        param20 = arcpy.Parameter(displayName = 'Track M.E.A.D. behaviour',
                                  name = 'track_MEAD',
                                  datatype = 'GPBoolean',
                                  parameterType = 'Optional',
                                  direction = 'Input')

        param20.category = 'Contour & Statistics'

        param20.value = False



#-------------------------------Parameters LIST--------------------------------#

        parameters = [param0, param1, param2, param3, param4, param5, param6,
                      param7, param8, param9, param10, param11, param12,
                      param13, param14, param15, param16, param17, param18,
                      param19, param20]



        return parameters


#- - - - - - - - - - - O P T I O N A L  F U N C T I O N S - - - - - - - - - - -#


#- - - - - - - - - - - - - - - UPDATE  PARAMETERS - - - - - - - - - - - - - - -#

    def updateParameters(self, parameters):

        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""



#----------Updating parameters[1, 2, 3, 4] according to parameters[0]----------#
                                                                                # parameters[0] = Input Feature
        if parameters[0].value:                                                 # parameters[1] = Azimuth Field
            try:                                                                # parameters[2] = Dip Angle Field
                fields = arcpy.Describe(parameters[0].value).fields             # parameters[3] = Method Field
                string_fields = [f.name for f in fields if f.type == 'String']  # parameters[4] = Type Field
                num_types = ('SmallInteger', 'Integer', 'Single', 'Double')
                num_fields = [f.name for f in fields if f.type in num_types]

                AutoFill_fields(parameters[1], num_fields, 'Azimuth')
                AutoFill_fields(parameters[2], num_fields, 'Dip_Angle')
                AutoFill_fields(parameters[3], string_fields, 'Method')
                AutoFill_fields(parameters[4], string_fields, 'Type')

            except IOError:
                for p in parameters[1:5]:
                    p.value = None

        else:
            for p in parameters[1:5]:
                p.value = None
                if p.datatype != 'Field':
                    p.filter.list = []



#---------Updating parameters[5, 17, 19] according to parameters[0, 4]---------#
                                                                                # parameters[0] = Input Feature
        if parameters[4].altered:                                               # parameters[4] = Type Field
                                                                                # parameters[5] = Data to be plotted
            try:                                                                # parameters[17] = Apply Contour
                cursor = arcpy.da.SearchCursor(parameters[0].valueAsText,       # parameters[19] = Extract Mean Vectors
                                               parameters[4].valueAsText)
                feat = set([str(c[0]) for c in cursor if str(c[0]).strip() != ''])
                for p in (5, 17, 19):
                    parameters[p].filters[0].list = sorted(list(feat))

            except (RuntimeError, TypeError):
                for p in (5, 17, 19):
                    parameters[p].filters[0].list = ['NONE']

        else:
            for p in (5, 17, 19):
                parameters[p].filters[0].list = ['NONE']


        if parameters[5].altered and parameters[5].values != None:
            AutoFill_vT(parameters[5], (1, 2, 4), ('Random', 'Point', 'NONE'))
            AutoFill_vT(parameters[5], (3, 5), (5.0, 0.5), checkList=False)
        else:
            pass

        if parameters[17].altered and parameters[17].values != None:
            defaultTexts = ('Kamb (exponential smoothing)', 'Filled', 'Random')
            AutoFill_vT(parameters[17], (1, 3, 4), defaultTexts)
            AutoFill_vT(parameters[17], (2, 5), (2.0, 0), checkList=False)
        else:
            pass

        if parameters[19].altered and parameters[19].values != None:
            defaultTexts = ('M.E.A.D. + Fisher', 'Random', 'Square', 'NONE')
            AutoFill_vT(parameters[19], (1, 6, 7, 9), defaultTexts)
            AutoFill_vT(parameters[19], (2,3,4,5,8,10), (1,50,30,95,5.0,0.5),
                        checkList=False)



#--------------Updating parameters[7] according to parameters[6]---------------#
                                                                                # parameters[6] = Store Image Output checkbox
        if parameters[6].altered:                                               # parameters[7] = Output Image

            if parameters[6].value == False:
                parameters[7].value = temp_filepath('Stereoplot_temp.png')
                parameters[7].enabled = False

            else:
                parameters[7].enabled = True

        else:
            pass



#--------------Updating parameters[18] according to parameters[17]-------------#
                                                                                # parameters[17] = Apply Contour
        if parameters[17].values != None:                                       # parameters[18] = Show Colorbar
            parameters[18].enabled = True

        else:
            parameters[18].value = False
            parameters[18].enabled = False



#--------------Updating parameters[20] according to parameters[19]-------------#
                                                                                # parameters[19] = Extract Mean Vectors
        if parameters[19].values != None:                                       # parameters[20] = Track M.E.A.D. behaviour
            parameters[20].enabled = True

        else:
            parameters[20].value = False
            parameters[20].enabled = False



        return



#- - - - - - - - - - - - - - - UPDATE  MESSAGES - - - - - - - - - - - - - - - -#

    def updateMessages(self, parameters):

        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""

#---Updating parameters[9] warning messages according to parameters[17, 19]---#
                                                                                # parameters[9] = Net Type
        if ((parameters[17].altered and parameters[17].values != None) or       # parameters[17] = Apply Contour
            (parameters[19].altered and parameters[19].values != None)):        # parameters[19] = Extract mean vectors

            if parameters[9].valueAsText == 'Wulff (Equal Angle)':
                warn9 = 'For better statistical results a Schmidt net type is recommended.'
                parameters[9].setWarningMessage(warn9)
            else:
                parameters[9].clearMessage()

        else:
            pass



#-----Updating parameters[17] warning messages according to parameters[17]-----#
                                                                                # parameters[17] = Apply Contour
        if parameters[17].altered and parameters[17].values != None:

            if len(parameters[17].values) > 1:
                warn17 = 'Apply contour on more than 1 feature may cause issues.'
                parameters[17].setWarningMessage(warn17)
            else:
                parameters[17].clearMessage()

        else:
            pass



        return



#- - - - - - - - - - - - - - - - - SET  LICENSE - - - - - - - - - - - - - - - -#

    def isLicensed(self):

        """Set whether tool is licensed to execute."""

        return True



#- - - - - - - - - - - - - T O O L  E X E C U T I O N - - - - - - - - - - - - -#

    def execute(self, parameters, messages):

        """The source code of the tool."""

        ok = self.checkExternalDependencies(messages)
        if not ok:
            return

#---------------------------Parameters assignation-----------------------------#

        IN_FEATURE = parameters[0]
        AZM_FIELD = parameters[1]
        DIP_FIELD = parameters[2]
        MET_FIELD = parameters[3]
        TYPE_FIELD = parameters[4]
        PLOT_DATA = parameters[5]
        STORE_IMG = parameters[6]
        OUTPUT_IMG = parameters[7]
        IMG_DPI = parameters[8]
        NET_TYPE = parameters[9]
        WRITE_LOG = parameters[10]
        TITLE_LABEL = parameters[11]
        N_TICKS = parameters[12]
        TICKS_TYPE = parameters[13]
        SHOW_GRID = parameters[14]
        SHOW_LEGEND = parameters[15]
        SHOW_NSAMPLES = parameters[16]
        CONTOUR = parameters[17]
        SHOW_COLORBAR = parameters[18]
        EXTRACT_MEANS = parameters[19]
        TRACK_MEAD = parameters[20]



#-------------------------------Importing Data---------------------------------#

        fields_params = (AZM_FIELD, DIP_FIELD, MET_FIELD, TYPE_FIELD)
        fields = tuple(fp.valueAsText for fp in fields_params)                  # Supported field names, later referred as: [0],[1],[2],[3]
        source = IN_FEATURE.valueAsText
        data_array = arcpy.da.FeatureClassToNumPyArray(source, fields,
                                                       null_value='')
        data = data_array.tolist()

        if STORE_IMG.value:
            coords = arcpy.da.FeatureClassToNumPyArray(source, 'Shape')         # Extracting centroid coordinates
            centroid = np.nanmean(coords.tolist(), axis=0)[0]


#-------------------------Pre-Elaboration of Data------------------------------#

#>>> Transforming DD data in RHR data <<<

        fixed_data = []
        for x in data:
            if x[2] == 'DD':
                fixed_strike = (x[0] - 90) % 360
                fixed_method = 'RHR'

            else:
                fixed_strike = x[0]
                fixed_method = x[2]

            fixed_data.append((fixed_strike, x[1], fixed_method, x[3]))


#>>> Initialize projected data types set & statistics variables for log file

        proj_types = set()

        if WRITE_LOG.value:
            contour_stats = []
            stats_log = {}
            fisher_stats = {}


#>>> Initialize fisher cones dictionary to show cones in legend

        from matplotlib.lines import Line2D
        fisher_cones = {}



#-----------------------Initialization of MPLStereonet-------------------------#

        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(111, projection = nets_dict[NET_TYPE.valueAsText])

        if TITLE_LABEL.value:
            title = TITLE_LABEL.valueAsText
        else:
            title = IN_FEATURE.valueAsText
        ax.set_title(title, y = 1.1, fontsize = 18)



#-----------------------Plotting of selected data------------------------------#

#>>> PLOTTING DATA Value Table <<<

        if PLOT_DATA.values != None:
            for row in PLOT_DATA.values:
                row = uni2str(row)                                              # Converting from unicode to string
                _type, _color, _marker, _mSize, _line, _lWidth = row

                if _color == 'Random':                                          # Select a random color if not specified by the user
                    random_index = np.random.randint(len(colors_dict))
                    _color = list(colors_dict.keys())[random_index]

                proj_types.add(_type)                                           # Add data type to projected data types set

                strikes, dips, trends, plunges = filterData(fixed_data, _type)  # Splitting RHR data from TP data


                if _marker != 'NONE':                                           # Plotting Poles or Vectors

                    if len(strikes) > len(trends):
                        ax.pole(strikes,
                                dips,
                                c = colors_dict[_color],
                                marker = markers_dict[_marker],
                                mec = 'w' if _color == 'Black' else 'k',
                                markersize = _mSize,
                                label = _type + ' (Poles)')

                    else:
                        ax.line(plunges,
                                trends,
                                c = colors_dict[_color],
                                marker = markers_dict[_marker],
                                mec = 'w' if _color == 'Black' else 'k',
                                markersize = _mSize,
                                label = _type)


                if _line != 'NONE':                                             # Plotting Planes
                        fground = 'white' if _color == 'Black' else 'black'
                        ax.plane(strikes,
                                 dips,
                                 c = colors_dict[_color],
                                 ls = lines_dict[_line],
                                 linewidth = _lWidth,
                                 path_effects = [mpe.withStroke(linewidth=_lWidth*2,
                                                                foreground=fground)],
                                 label = _type)



#>>> CONTOUR Value Table <<<

        if CONTOUR.values != None:
            for row in CONTOUR.values:
                row = uni2str(row)                                              # Converting from unicode to string
                _type, _method, _sigma, _style, _color, _alpha = row

                if _color == 'Random':                                          # Select a random color map if not specified by the user
                    random_index = np.random.randint(len(colorMaps))
                    _color = colorMaps[random_index]

                _alpha = 1 - (_alpha / 100.)                                    # Converting transparency to opacity (matplotlib "alpha" parameter)

                proj_types.add(_type)                                           # Add data type to projected data types set

                strikes, dips, trends, plunges = filterData(fixed_data, _type)  # Splitting RHR data from TP data

                if len(strikes) > len(trends):
                    x1, x2, apply_on = strikes, dips, 'poles'
                else:
                    x1, x2, apply_on = plunges, trends, 'lines'


                if _style == 'Filled':                                          # Applying Filled Density Contour Function on poles or vectors

                    fdcf = ax.density_contourf(x1,
                                               x2,
                                               method = contour_methods[_method],
                                               measurement = apply_on,
                                               sigma = _sigma,
                                               cmap = _color,
                                               alpha = _alpha)

                    mask = ax.density_contour(x1,
                                              x2,
                                              method = contour_methods[_method],
                                              measurement = apply_on,
                                              sigma = _sigma,
                                              colors = 'black',
                                              alpha = _alpha)


                else:                                                           # Applying Unfilled Density Contour Function on poles or vectors

                    udcf = ax.density_contour(x1,
                                              x2,
                                              method = contour_methods[_method],
                                              measurement = apply_on,
                                              sigma = _sigma,
                                              cmap = _color,
                                              alpha = _alpha)


                if WRITE_LOG.value:
                    contour_stats = [_type, _method, _sigma]



#>>> EXTRACT MEAN VECTORS Value Table <<<

        if EXTRACT_MEANS.values != None:
            for row in EXTRACT_MEANS.values:
                row = uni2str(row)                                              # Converting from unicode to string
                _type, _algorithm, _nClusts, _azmTol, _dipTol, _conf = row[:6]
                _color, _marker, _mSize, _line, _lWidth = row[6:]


                if _color == 'Random':                                          # Select a random color if not specified by the user
                    random_index = np.random.randint(len(colors_dict))
                    _color = list(colors_dict.keys())[random_index]

                proj_types.add(_type)                                           # Add data type to projected data types set

                strikes, dips, trends, plunges = filterData(fixed_data, _type)  # Splitting RHR data from TP data

                if len(strikes) > len(trends):
                    x1, x2, apply_on = strikes, dips, 'poles'
                else:
                    x1, x2, apply_on = plunges, trends, 'lines'


                if _algorithm == 'K-means':                                     # Extract K-means mean vector(s)

                    while True:
                        try:
                            centers = kmeans(x1, x2, num=_nClusts,
                                             measurement=apply_on)
                            break
                        except np.linalg.LinAlgError:
                            _nClusts -= 1

                    if apply_on == 'poles':
                        c1, c2 = geographic2pole(*zip(*centers))
                    else:
                        c1, c2 = geographic2plunge_bearing(*zip(*centers))


                elif _algorithm == 'Bingham':                                   # Extract Bingham Statistics

                    c1, c2 = zip(fit_girdle(x1, x2, measurement = apply_on))



                else:                                                           # Extract M.E.A.D. clusters

                    _azmTol, _dipTol = _azmTol / 100., _dipTol / 100.

                    mead_clust = angles_clustering(x1, x2, apply_on,
                                                   _azmTol, _dipTol,
                                                   _nClusts)

                    azm, inc = [], []
                    for mc in mead_clust:
                        fam = list(zip(*mc))
                        azm.append(fam[0])
                        inc.append(fam[1])

                        if TRACK_MEAD.value:                                    # Track M.E.A.D. clustering process behaviour if required
                            num_mark = '${}$'.format(mead_clust.index(mc) + 1)
                            if apply_on == 'poles':
                                ax.pole(fam[0],
                                        fam[1],
                                        c = 'k',
                                        marker = num_mark,
                                        mec = colors_dict[_color],
                                        markersize = _mSize + _mSize/2.)

                            else:
                                ax.line(fam[1],
                                        fam[0],
                                        c = 'k',
                                        marker = num_mark,
                                        mec = colors_dict[_color],
                                        markersize = _mSize + _mSize/2.)


                    c1, c2 = [], []
                    for i in range(len(mead_clust)):

                        if 'Fisher' in _algorithm:                              # Extract Fisher mean vector(s) and stats

                            if apply_on == 'poles':
                                vectorPB, stats = find_fisher_stats(azm[i], inc[i],
                                                                    measurement = 'poles',
                                                                    conf = _conf)
                                vectorSD = plunge_bearing2pole(*vectorPB)
                                c1.append(vectorSD[0][0])
                                c2.append(vectorSD[1][0])

                            else:
                                vectorPB, stats = find_fisher_stats(inc[i], azm[i],
                                                                    measurement = 'lines',
                                                                    conf = _conf)
                                c1.append(vectorPB[0])
                                c2.append(vectorPB[1])

                            ax.cone(vectorPB[0],
                                    vectorPB[1],
                                    stats[1],
                                    facecolor = "none",
                                    edgecolor = colors_dict[_color],
                                    linestyle = '-',
                                    linewidth = _lWidth)

                            cone_lbl = _type + ' mean confidence cone'
                            fisher_cones[cone_lbl] = Line2D([0], [0],
                                                            ls = 'None',
                                                            mec = colors_dict[_color],
                                                            mfc = 'None',
                                                            marker = "o",
                                                            ms = 10,
                                                            mew = _lWidth)

                            if WRITE_LOG.value:
                                if _type not in fisher_stats:
                                    fisher_stats[_type] = stats
                                else:
                                    fisher_stats[_type] += stats


                        else:                                                   # Extract M.E.A.D. mean vector(s)

                            c_azm, _ = avg_angle(azm[i])
                            c_inc = sum(inc[i]) / len(inc[i])
                            c1.append(c_azm if apply_on == 'poles' else c_inc)
                            c2.append(c_inc if apply_on == 'poles' else c_azm)



                label_root = '{} [{}] mean'.format(_type, _algorithm)           # Project Statistics Data and collect mean data and stats for log file if required
                if _algorithm == 'Bingham':
                    label_root = '{} [{}] best fit'.format(_type, _algorithm)

                for n in range(len(c1)):

                    if WRITE_LOG.value:
                        if apply_on == 'poles' or _algorithm == 'Bingham':
                            mean = (c1[n], c2[n])
                        else:
                            mean = (c2[n], c1[n])

                        if (_type, _algorithm) not in stats_log:
                            stats_log[(_type, _algorithm)] = mean
                        else:
                            stats_log[(_type, _algorithm)] += mean


                    if _marker != 'NONE':

                        if apply_on == 'poles' or _algorithm == 'Bingham':
                            ax.pole(c1[n],
                                    c2[n],
                                    c = colors_dict[_color],
                                    marker = markers_dict[_marker],
                                    mec = ('w' if _color == 'Black' else 'k'),
                                    markersize = _mSize,
                                    label = label_root + ' pole')

                        else:
                            ax.line(c1[n],
                                    c2[n],
                                    c = colors_dict[_color],
                                    marker = markers_dict[_marker],
                                    mec = ('w' if _color == 'Black' else 'k'),
                                    markersize = _mSize,
                                    label = label_root + ' vector')



                    if _line != 'NONE' and (apply_on == 'poles' or _algorithm == 'Bingham'):

                        fground = ('white' if _color == 'Black' else 'black')
                        ax.plane(c1[n],
                                 c2[n],
                                 c = colors_dict[_color],
                                 ls = lines_dict[_line],
                                 linewidth = _lWidth,
                                 path_effects = [mpe.withStroke(linewidth=_lWidth*2,
                                                                foreground=fground)],
                                 label = label_root + ' plane')



#----Adding Grid, Legend, n.of Data label, Colorbar & Tick Marks to the plot---#

        ax.grid(SHOW_GRID.value)


        if SHOW_LEGEND.value == True:
            from collections import OrderedDict
            handles, labels = ax.get_legend_handles_labels()
            for lbl, hnd in fisher_cones.items():
                handles.append(hnd)
                labels.append(lbl)
            by_label = OrderedDict(zip(labels, handles))

            if len(by_label) > 0:
                ax.legend(by_label.values(),
                          by_label.keys(),
                          loc = 'upper left',
                          bbox_to_anchor = (1.0, 0.5, 0.5, 0.6),
                          ncol = 1,
                          fontsize = 8,
                          numpoints = 1)


        if SHOW_NSAMPLES.value == True:
            txt4lbl = ''
            for pt in sorted(proj_types):
                pt_num = sum([FD.count(pt) for FD in fixed_data])
                txt4lbl += pt + ' = ' + str(pt_num) + '\n'
            ax.text(1.0, 0.0,
                    txt4lbl,
                    fontsize = 8,
                    transform = ax.transAxes)


        if SHOW_COLORBAR.value == True:
            cbaxes = fig.add_axes([-0.2, 0.1, 0.03, 0.8])                       # left, bottom, width, height
            try:
                plt.colorbar(fdcf, cax = cbaxes)
            except NameError:
                plt.colorbar(udcf, cax = cbaxes)
            contourInfo = u'Contour applied on:\n{}\n\nMethod: {}\nStd. Dev.: {}'.format(*CONTOUR.values[0][:3])
            ax.text(-0.29, 0.0,
                    contourInfo,
                    fontsize = 8,
                    transform = ax.transAxes)


        ax.set_azimuth_ticks(np.linspace(0, 360, N_TICKS.value, False))

        if TICKS_TYPE.valueAsText != 'Angular Values':
            if TICKS_TYPE.valueAsText == 'Cardinal Points':
                card = cardinals
            else:
                card = card_hybrid
            try:
                ax.set_azimuth_ticklabels(card[::8//N_TICKS.value])
            except ZeroDivisionError:
                pass



#-------------------------------Write Log File---------------------------------#

        if WRITE_LOG.value:
            logPath = path.splitext(OUTPUT_IMG.valueAsText)[0] + '_LOG.txt'

            with io.open(logPath, 'w') as LOG:
                LOG.write(title.upper() + '\n\n')

                if SHOW_NSAMPLES.value == False:
                    txt4lbl = ''
                    for pt in sorted(proj_types):
                        pt_num = sum([FD.count(pt) for FD in fixed_data])
                        txt4lbl += pt + ' = ' + str(pt_num) + '\n'
                LOG.write(txt4lbl + u'\n')

                if len(contour_stats) > 0:
                    LOG.write(u'CONTOUR INFO\n')
                    LOG.write(u'Applied on ->\t{}\n'.format(contour_stats[0]))
                    LOG.write(u'Method ->\t{}\n'.format(contour_stats[1]))
                    LOG.write(u'St.Dev. ->\t{}\n'.format(contour_stats[2]))
                    LOG.write(u'\n')

                if len(stats_log) > 0:
                    LOG.write(u'STATISTICS\n')

                    for key, val in sorted(stats_log.items()):
                        m_azm = [u'{0:03d}'.format(int(round(i,0))) for i in val[0::2]]
                        m_inc = [u'{0:02d}'.format(int(round(j,0))) for j in val[1::2]]
                        m_tot = [i + u'/' + j for i, j in zip(m_azm, m_inc)]

                        if key[1] == 'Bingham':
                            desc = '[Bingham best fit plane]'
                        else:
                            desc = '[{} mean(s)]'.format(key[1])

                        typeinfo = '{} {} ->\t'.format(key[0], desc)
                        LOG.write(typeinfo + '\t'.join(m_tot) + u'\n')

                        if key[1] == 'M.E.A.D. + Fisher':
                            r_values = ['{0:0.3f}'.format(r) for r in fisher_stats[key[0]][0::3]]
                            r_title = ' - R value (length of the mean vector) ->\t\t\t'
                            angles = ['{0:0.2f}'.format(a) for a in fisher_stats[key[0]][1::3]]
                            a_title = ' - Fisher angle (confidence radius in degrees) ->\t\t'
                            k_values = ['{0:0.2f}'.format(k) for k in fisher_stats[key[0]][2::3]]
                            k_title = ' - K value (dispersion factor) ->\t\t\t\t'
                            LOG.write(u'<Fisher Statistics>:\n')
                            LOG.write(r_title + '\t'.join(r_values) + u'\n')
                            LOG.write(a_title + '\t'.join(angles) + u'\n')
                            LOG.write(k_title + '\t'.join(k_values) + u'\n')
                        LOG.write(u'\n')

                final_note = u'\nNote that mean values are expressed as follows:\n\
 - strike/dip (planar features)\n\
 - trend/plunge (linear features)\n'
                LOG.write(final_note)
                LOG.write(u'\nLog file automatically compiled by ArcStereoNet')
                LOG.close()



#----------------------Save the figure and End the tool------------------------#

        plt.savefig(OUTPUT_IMG.valueAsText,
                    dpi = IMG_DPI.value,
                    bbox_inches = 'tight')
        plt.close('all')

        if STORE_IMG.value == False:
            startfile(OUTPUT_IMG.valueAsText)
            if WRITE_LOG.value:
                startfile(logPath)

        elif OUTPUT_IMG.valueAsText.endswith('.png'):
            arcpy.BuildPyramids_management(OUTPUT_IMG.valueAsText)

            xypath = path.splitext(OUTPUT_IMG.valueAsText)[0] + '_xy.txt'       # Saving centroid coordinates
            clon, clat = centroid
            descr = arcpy.Describe(IN_FEATURE.valueAsText)
            with io.open(xypath, 'w') as xyfile:
                xyfile.write(u'{}\n'.format(clon))
                xyfile.write(u'{}\n'.format(clat))
                xyfile.write(u'{}\n'.format(path.join(descr.path, descr.name)))



        return



#- - - - - - - G  R  A  P  H    T  O    H  Y  P  E  R  L  I  N  K - - - - - - -#

class GraphToHyperlink(_ToolBase):

    def __init__(self):

        """Define the tool (tool name is the name of the class)."""

        super().__init__()
        self.label = 'Graph To Hyperlink'
        self.description = 'Use this tool to link plots with corresponding data.'



#- - - - - - - - - - - - - - - - - PARAMETERS - - - - - - - - - - - - - - - - -#

    def getParameterInfo(self):

        """Define parameter definitions"""



#-------------------------Input plot images (param0)---------------------------#

        param0 = arcpy.Parameter(displayName = 'Input Plot Images (Raster)',
                                 name = 'in_plots',
                                 datatype = 'DERasterDataset',
                                 parameterType = 'Required',
                                 direction = 'Input',
                                 multiValue = True)



#------------------------Output feature class (param1)-------------------------#

        param1 = arcpy.Parameter(displayName = 'Output Feature Class',
                                 name = 'out_shp',
                                 datatype = 'DeShapefile',
                                 parameterType = 'Required',
                                 direction = 'Output')



#-------------------------------Parameters LIST--------------------------------#

        parameters = [param0, param1]

        return parameters



#- - - - - - - - - - - O P T I O N A L  F U N C T I O N S - - - - - - - - - - -#

#- - - - - - - - - - - - - - - UPDATE  PARAMETERS - - - - - - - - - - - - - - -#

    def updateParameters(self, parameters):

        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        return



#--------------------------------Other Functions-------------------------------#

    def updateMessages(self, parameters):

        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""



#-------Updating parameters[0] error messages according to parameters[0]-------#
                                                                                # parameters[0] = Input plot images
        if parameters[0].altered and parameters[0].value != None:
            for img_path in parameters[0].values:
                xy_path = path.splitext(str(img_path))[0] + '_xy.txt'
                if not path.exists(xy_path):
                    err = u'Unable to find coordinates file for {}.'.format(img_path)
                    parameters[0].setErrorMessage(err)
                    break



        return



    def isLicensed(self):

        """Set whether tool is licensed to execute."""

        return True



#- - - - - - - - - - - - - T O O L  E X E C U T I O N - - - - - - - - - - - - -#

    def execute(self, parameters, messages):

        """The source code of the tool."""


#---------------------------Parameters assignation-----------------------------#

        IN_PLOT = parameters[0]
        OUT_SHP = parameters[1]



#------------------Creation of the new output feature class--------------------#

        rows = []
        sr = []                                                                 # sr = Spatial Reference

        for ID, plot_img in enumerate(IN_PLOT.values):

            xy_file = path.splitext(str(plot_img))[0] + '_xy.txt'
            with io.open(xy_file, 'r') as temp_xy:
                lon, lat, source_path = temp_xy.readlines()                
                lon, lat = float(lon), float(lat)
                sr.append(arcpy.Describe(source_path).spatialReference)
                rows.append(((lon, lat), ID, str(plot_img)))

        out_path, out_name = path.split(OUT_SHP.valueAsText)
        # Set SR to None if multiple SR are found (and send warning)
        if len({s.exporttostring() for s in sr}) == 1:
            spat_ref = sr[0]
        else:
            spat_ref = None
            messages.addWarningMessage('Found conflicting SR: set to Unknown!')
        arcpy.CreateFeatureclass_management(out_path, out_name, "POINT",
                                            spatial_reference=spat_ref)
        arcpy.AddField_management(OUT_SHP.valueAsText, "Plot", "TEXT")
        
        fields = ["SHAPE@XY", "Id", "Plot"]
        with arcpy.da.InsertCursor(OUT_SHP.valueAsText, fields) as curs:
            for row in rows:
                curs.insertRow(row)



        return




