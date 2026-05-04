#- - - - - - - - - - - - - I N I T I A L I Z A T I O N - - - - - - - - - - - - #

#--------------------------Import Internal Libraries--------------------------#
import arcpy
import importlib.util
import io
import re
import shlex
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
# matplotlib.use('TkAgg') use for interactive plots with plt.show()
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Iterable, Literal, TypedDict
from os import path, startfile
from datetime import datetime

#-----------------------------Environment Settings----------------------------#
# Prevent adding output image to Contents pane
arcpy.env.addOutputsToMap = False                                          

#------------------------------Global Constants-------------------------------#
# Contour methods dictionary
CONTOUR_METHODS = {
    'Kamb (no smoothing)': 'kamb',                               
    'Kamb (linear smoothing)': 'linear_kamb',
    'Kamb (inverse square smoothing)': 'square_kamb',
    'Kamb (exponential smoothing)': 'exponential_kamb',
    'Schmidt (a.k.a. 1%)': 'schmidt'
}

# Contour color maps dictionary
CONTOUR_COLORMAPS = {
    "Viridis (Purple-Green-Yellow)": "viridis",
    "Plasma (Blue-Red-Yellow)": "plasma",
    "Inferno (DarkPurple-Red-Yellow)": "inferno",
    "Magma (DarkPurple-DarkPink-PaleYellow)": "magma",
    "Cividis (Blue-Yellow)": "cividis",
    "Turbo (Blue-Yellow-Red)": "turbo",
    "Greys": "Greys",
    "Purples": "Purples",
    "Blues": "Blues",
    "Greens": "Greens",
    "Oranges": "Oranges",
    "Reds": "Reds",
    "Yellow-Orange-Brown": "YlOrBr",
    "Yellow-Orange-Red": "YlOrRd",
    "Orange-Red": "OrRd",
    "Purple-Red": "PuRd",
    "Red-Purple": "RdPu",
    "Blue-Purple": "BuPu",
    "Purple-Blue": "PuBu",
    "Green-Blue": "GnBu",
    "Blue-Green": "BuGn",
    "Purple-Blue-Green": "PuBuGn",
    "Yellow-Green-Blue": "YlGnBu",
    "Yellow-Green": "YlGn",
}
# Compass labels dictionary
COMPASS_LABELS = {
    'Cardinals': ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'),
    'Degrees': ('0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'),
    'Hybrid': ('N', '45°', '90°', '135°', '180°', '225°', '270°', '315°')
}

# Colors dictionary, inspired from https://xkcd.com/color/rgb/
COLORS = {
    'Red': 'r',
    'Green': 'g',
    'Blue': 'b',
    'Cyan': 'c',
    'Magenta': 'm',   
    'Yellow': 'y',
    'Black': 'k',
    'Orange': '#ff8c00',
    'Grey': '#808080',
    'Salmon': '#e9967a',
    'Olive': '#808000',
    'Violet': '#440d80',
    'Gold' : '#daa520',
    'Water green': '#00ff7f',
    'Peach': '#ffb07c',
    'Light yellow': '#f0e68c',
    'Greenish Blue': '#0b8b87',
    'Puke Brown': '#947706',
    'Indigo': '#380282',
    'Dust': '#b2996e',
    'Very Light Blue': '#d5ffff',
    'Muddy Green': '#657432',
    'Dirty Pink': '#ca7b80'
}

# Markers dictionary
MARKERS = {
    'Point (•)': '.',
    'Square (◻)': 's',
    'Triangle (△)': '^',
    'Hexagon (⬡)': 'H',       
    'Thin diamond (◊)': 'd',
    'Star (☆)': '*',
    'Circle (○)': 'o',
    'Diamond (◇)': 'D',
    'Pentagon (⬠)': 'p',
    'Plus (+)': 'P',
    'Cross (✕)': 'X'
}
                
# Lines dictionary
LINES = {
    'Solid (—)': '-',
    'Dashed (---)': '--',
    'Dotted (· · ·)': ':',
    'Dashdot (-·-)': '-.'
}

# ArcGIS numerical field types and field filters
NUM_FIELD_TYPES = ['BigInteger', 'SmallInteger', 'Integer', 'Single', 'Double']
NUM_FIELD_FILTERS = ['Short', 'Long', 'Single', 'Double']

# Available output image file formats
IMG_FORMATS = ['png', 'eps', 'pdf', 'pgf', 'ps', 'raw', 'rgba', 'svg', 'svgz']

#- - - - - - - - - - - - U S E F U L  F U N C T I O N S - - - - - - - - - - - -#

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



#- - - - - - - - - - - - M. E. A. D.  A L G O R I T H M - - - - - - - - - - - -#

def cos_r(values):
    return np.cos(np.radians(values))


def sin_r(values):
    return np.sin(np.radians(values))


def linearize(values):
    return [v if abs(v-min(values)) <= 180 else v-360 for v in values]


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


def angles_clustering(
        strikes: list,
        dips: list,
        measurement: Literal['lines', 'poles'] = 'poles',
        tol_str: float = 0.5,
        tol_dip: float = 0.3,
        n_clusts: int | None = None
    ):

    # In the future modify function to work with numpy arrays?.

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


class MeanVectorStats(TypedDict):
    # Algorithm name
    algorithm: Literal["MEAD", "MEAD+Fisher", "K-means", "Bingham"]
    # Mean azimuth and dip values per cluster id
    vectors: dict[str, tuple[float, float]]
    # Mean resultant lengths per cluster id
    r_means: dict[str, float]
    # Fisher statistics (R value, confidence radius, kappa) per cluster id
    fisher_stats: dict[str, tuple[float, float, float]]
    # MEAD azimuth and dip tolerance values
    tol: tuple[float, float] | None
    # Fisher confidence
    conf: float | None

class StatItem(TypedDict):
    # Number of samples
    n: int
    # Kind of oriented data
    kind: Literal["lines", "planes"]
    # Contour data = list of (method, st_dev)
    contour: list[tuple[str, float]]
    # Clustering data = list of MeanVectorStats
    clustering: list[MeanVectorStats]



class StatsLogger():

    def __init__(
        self,
        stats_dict: dict[str, StatItem],
        kind: Literal["RoseDiagrams", "StereoPlots"]
    ) -> None:

        """Helper class for generating log files from statistics.

        Args:
            stats_dict (dict[str, StatItem]): Dictionary containing statistics.
            kind (Literal["RoseDiagrams", "StereoPlots"]): Requesting tool.
        """
    
        self.stats = stats_dict
        self.kind = kind


    @staticmethod
    def paragraph(text: str) -> str:

        """Wrap 'text' in a paragraph.

        Args:
            text (str): Text to wrap in paragraph.

        Returns:
            str: Text as paragraph.
        """

        return f'{text}\n\n'
    

    @staticmethod
    def head(text: str) -> str:

        """Wrap 'text' in a stylish frame ideal for a section's heading.

        Args:
            text (str): Text to wrap in frame.

        Returns:
            heading (str): Text as heading.
        """

        hline = '-' * (len(text) + 4)
        heading = f'{hline}\n| {text} |\n{hline}'
        return heading


    def custom(self, text_list: Iterable[str], parag: bool = True) -> str:

        """Create new log text from a list of custom texts.

        Args:
            text_list (Iterable[str]): List of texts to merge.
            parag (bool, optional): Wrap text in paragraph. Defaults to True.

        Returns:
            str: Custom log text.
        """

        t = '\n'.join(text_list)
        return self.paragraph(t) if parag else t
    

    def toolName(self, parag: bool = True) -> str:

        """Generate formatted log text with tool name.

        Args:
            parag (bool, optional): Wrap text in paragraph. Defaults to True.

        Returns:
            str: Log text with tool name.
        """

        t = f'Tool: {self.kind}'
        return self.paragraph(t) if parag else t
    

    def dateTime(self, parag: bool = True) -> str:

        """Generate formatted log text with datetime info.

        Args:
            parag (bool, optional): Wrap text in paragraph. Defaults to True.

        Returns:
            str: Log text with datetime.
        """

        t = f'Datetime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        return self.paragraph(t) if parag else t
            

    def samples(self, parag: bool = True) -> str:

        """Generate formatted log text with number of plotted samples per data
        type.

        Args:
            parag (bool, optional): Wrap text in paragraph. Defaults to True.

        Returns:
            str: Log text with number of samples.
        """

        t = '\n'.join(f'{k}: {v["n"]}' for k, v in sorted(self.stats.items()))
        return self.paragraph(t) if parag else t
    

    def contour(self, parag: bool = True) -> str:

        """Generate formatted log text with contour statistics per data type.

        Args:
            parag (bool, optional): Wrap text in paragraph. Defaults to True.

        Returns:
            str: Log text with contour statistics.
        """

        tlist = [self.head('CONTOUR STATISTICS')]
        contour_stats_available = False
        for type_, stats in sorted(self.stats.items()):

            for cont in stats["contour"]:
                contour_stats_available = True

                # Type
                tlist.append(type_)

                # Method 
                tlist.append(f'- Contour method: {cont[0]}')

                # Standard deviation
                tlist.append(f'- Std. deviation: {cont[1]}')

                # Force extra spacing
                tlist.append('')

        if not contour_stats_available:
            return ''
        
        t = '\n'.join(tlist[:-1]) # exclude last extra spacing element
        return self.paragraph(t) if parag else t
        

    def clustering(self, mirrored: bool = False, parag: bool = True) -> str:

        """Generate formatted log text with clustering and mean vector
        statistics per data type.

        Args:
            mirrored (bool, optional): Whether Mirrored option in RoseDiagrams
                tool is active. This parameter is ignored if 'self.kind' is not
                "RoseDiagrams". Defaults to False.
            parag (bool, optional): Wrap text in paragraph. Defaults to True.

        Returns:
            str: Log text with clustering and mean vector statistics.
        """

        tlist = [self.head('MEAN VECTORS STATISTICS')]
        clust_stast_available = False
        for type_, stats in sorted(self.stats.items()): 
        
            for mvs in stats["clustering"]:
                clust_stast_available = True

                # Type
                tlist.append(type_) 
                
                # Algorithm
                algm = mvs["algorithm"]
                desc = 'best fit plane' if algm == 'Bingham' else 'clustering'
                tlist.append(f'- Statistic: {algm} {desc}')

                # MEAD tolerance (only for MEAD and MEAD+Fisher algorithms)
                if (tol := mvs["tol"]) is not None:
                    atol, itol = tol
                    atol = f'{atol}% (Azimuth)'
                    itol = '' if np.isnan(itol) else f'{itol}% (Inclination)'
                    tlist.append(f'- MEAD tolerance: {atol} {itol}'.rstrip())

                # Fisher confidence (only for MEAD+Fisher algorithm)
                if (conf := mvs["conf"]) is not None:
                    tlist.append(f'- Fisher confidence: {conf}%')

                # Mean vectors
                azms, incl = zip(*mvs["vectors"].values())
                azms = np.rint(azms).astype(int)
                incl = np.rint(incl).astype(int)
                
                if self.kind == "StereoPlots":
                    vect = '\t'.join(f'{a}/{i}' for a, i in zip(azms, incl))
                elif mirrored:
                    vect = '\t'.join(f'{a} ({(a + 180) % 360})°' for a in azms)
                else:
                    vect = '\t'.join(f'{a}°' for a in azms)
                tlist.append(f'- Estimated mean vector(s) data: {vect}')

                # R means (only for MEAD and MEAD+Fisher)
                r_means = '\t'.join(
                    (str(round(v, 3)) for v in mvs["r_means"].values())
                )
                if len(r_means):
                    tlist.append(f'- MEAD mean resultant length(s): {r_means}')

                # Fisher stats (only for MEAD+Fisher)
                if len(fisher_stats := mvs["fisher_stats"]):
                    r_val, angle, kappa = zip(*fisher_stats.values())
                    s1 = '  - Magnitude of mean vector(s): ' + '\t'.join(
                        (str(round(r, 3)) for r in r_val)
                    )
                    s2 = '  - Confidence radius (degrees): ' + '\t'.join(
                        (str(round(a, 3)) for a in angle)
                    )
                    s3 = '  - Vector(s) dispersion factor: ' + '\t'.join(
                        (str(round(k, 1)) for k in kappa)
                    )
                    fstats = '\n'.join((s1, s2, s3))
                    tlist.append(f'- Fisher statistics:\n{fstats}')
                
                # Force extra spacing
                tlist.append('')

        if not clust_stast_available:
            return ''
        
        t = '\n'.join(tlist[:-1]) # exclude last extra spacing element
        return self.paragraph(t) if parag else t
    

    def extraInfo(self, parag: bool = True) -> str:

        """Generate formatted log text with extra info, such as tool name, tool
        execution datetime and mean vector notations.

        Args:
            parag (bool, optional): Wrap text in paragraph. Defaults to True.

        Returns:
            str: Log text with extra info.
        """

        tlist = [
            self.head('EXTRA INFO'),
            self.toolName(parag=False),
            self.dateTime(parag=False),
            'Log file automatically compiled by ArcStereoNet'
        ]
        # Add mean vectors notation only for StereoPlots tool
        if self.kind == "StereoPlots":
            notation = (
                'Mean vectors are expressed as strike/dip for planar data '\
                'and trend/plunge for linear data.'
            )
            tlist.insert(-1, notation)

        t = '\n'.join(tlist)
        return self.paragraph(t) if parag else t



class Toolbox(object):

    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""

        self.label = 'ArcStereoNet'
        self.alias = 'ArcStereoNet'

        # List of tool classes associated with this toolbox
        self.tools = [RoseDiagrams, StereoPlots, GraphToHyperlink]



class _ToolBase(object):

    # Field hints for auto filling input Fields parameters (must be lowercase)
    AZIMUTH_HINTS = (
        'azimuth',
        'orientation',
        'orientations',
        'strike',
        'strikes',
        'strike_angle',
        'strike_angles',
        'strike angle',
        'strike angles',
        'trend',
        'trends'
    )
    DIP_HINTS = (
        'dip',
        'dips',
        'dip_angle',
        'dip_angles',
        'dip angle',
        'dip angles',
        'inclination',
        'inclinations',
        'plunge',
        'plunges'
    )
    METHOD_HINTS = (
        'method',
        'methods',
        'rule',
        'rules'
    )
    TYPE_HINTS = (
        'category',
        'categories',
        'kind',
        'kinds',
        'label',
        'labels',
        'type',
        'types'
    )
    WEIGHT_HINTS = (
        'weight',
        'weights'
    )

    def __init__(self) -> None:

        """Base class for all tools."""

        self.label = 'Tool label'
        self.description = 'Tool description.'
        self.canRunInBackground = False


    def isModuleAvailable(self, name: str) -> bool:
        """Check if a module is available in the current Python environment.

        Args:
            name (str): Module's name.

        Returns:
            bool: Whether the module is available.
        """

        return importlib.util.find_spec(name) is not None


    def _getRawValueTable(self, parameter: arcpy.Parameter) -> list[list]:

        """Return raw values extracted from a ValueTable parameter. Raw values
        differ from normal values obtained with 'parameter.values' because are
        extracted from 'parameter.valueAsText', then parsed and recompiled as
        classic ValueTable data (list of lists). This allows preserving special
        value characters (like '#' indicating empty values), not accessible via
        'parameter.values'.

        Raises:
            TypeError: Datatype of 'parameter' must be "Value Table".

        Returns:
            list[list]: Raw ValueTable values.
        """

        # Enforce requirement of a ValueTable parameter
        if (dt := parameter.datatype) != 'Value Table':
            raise TypeError(f"Parameter must be of type Value Table, not {dt}")
        
        # Rows: split on ';' character, excluding ';' inside string values
        # Columns: split on ' ' character, excluding ' ' inside string values
        rows = re.split(r";(?=(?:[^']*'[^']*')*[^']*$)", parameter.valueAsText)
        raw_table = [shlex.split(r.strip()) for r in rows]
        
        return raw_table


    def parameterChanged(self, parameter: arcpy.Parameter) -> bool:

        """Check if a parameter has been altered and not yet validated.

        Args:
            parameter (arcpy.Parameter): Parameter to check.

        Returns:
            bool: Whether the parameter has changed.
        """

        return parameter.altered and not parameter.hasBeenValidated
    
    
    def getFieldParameterAlias(
        self,
        field_parameter: arcpy.Parameter,
        feature_parameter: arcpy.Parameter
    ) -> str:
        
        """Get alias name, if it exists, of Field parameter 'field_parameter' 
        that belongs to Feature Layer parameter 'feature_parameter'. Fallback
        on field name if no alias is found.

        Args:
            field_parameter (arcpy.Parameter): Field parameter to check.
            feature_parameter (arcpy.Parameter): Parent FeatureLayer parameter.

        Raises:
            TypeError: Datatype of 'field_parameter' must be "Field".
            TypeError: Datatype of 'feature_parameter' must be "Feature Layer".

        Returns:
            str: Field alias name, if available. Field name otherwise.
        """

        # Enforce requirement of a Field parameter
        if (dt := field_parameter.datatype) != 'Field':
            raise TypeError(f"Parameter must be of type Field, not {dt}")
        
        # Enforce requirement of a Feature Layer parameter
        if (dt := feature_parameter.datatype) != 'Feature Layer':
            raise TypeError(f"Parameter must be of type Feature Layer, not {dt}")
        
        # Return field alias if it exists. Fallback on field name
        feature_desc = arcpy.Describe(feature_parameter.value)
        field_list: list[arcpy.Field] = feature_desc.fields
        for f in field_list:
            if f.name == field_parameter.valueAsText:
                alias = f.aliasName
                return alias if alias else f.name
        else:
            return field_parameter.valueAsText
    
    
    def inferFieldParameter(
        self,
        parameter: arcpy.Parameter,
        field_list: Iterable[arcpy.Field],
        field_hints: Iterable[str]
    ) -> None:
        
        """Auto fill an empty Field 'parameter' with a field from 'field_list'
        that matches any of the 'field_hints'.

        Args:
            parameter (arcpy.Parameter): Field parameter to be filled.
            field_list (Iterable[arcpy.Field]): Fields available for filling.
            field_hints (Iterable[str]): Field hints to be matched for filling.

        Raises:
            TypeError: Datatype of 'parameter' must be "Field".
        """
        # Enforce requirement of a Field parameter
        if (dt := parameter.datatype) != 'Field':
            raise TypeError(f"Parameter must be of type Field, not {dt}")

        # Extract a dictionary of names and aliases from available fields
        fields_dict = {f.name: f.aliasName for f in field_list}

        # Do not alter a parameter with a valid name or alias
        if (parameter.valueAsText in fields_dict.keys() or 
            parameter.valueAsText in fields_dict.values()):
            return

        # Try to fill parameter with a field whose name or alias matches a hint
        for name, alias in fields_dict.items():
            if name.lower() in field_hints or alias.lower() in field_hints:
                parameter.value = name # make sure to always assign field name
                break
        else:
            parameter.value = None


    def resolveValueTableParameter(
        self,
        parameter: arcpy.Parameter,
        defaults: tuple[int | float | str, ...],
        skip_first_col: bool = True
    ) -> None:
        
        """Auto fill empty or invalid entries in Value Table 'parameter' with 
        values from 'defaults'.

        Args:
            parameter (arcpy.Parameter): ValueTable parameter to be filled.
            defaults (tuple[int | float | str, ...]): Default table values.
            skip_first_col (bool, optional): Whether to skip first column of
             the table. Defaults to True.

        Raises:
            TypeError: Datatype of 'parameter' must be "Value Table".
            ValueError: Lengths of 'defaults' and table columns cannot differ. 
        """

        # Enforce requirement of a ValueTable parameter
        if (dt := parameter.datatype) != 'Value Table':
            raise TypeError(f"Parameter must be of type Value Table, not {dt}")

        # Do nothing if ValueTable parameter is invalid
        if (table := parameter.values) is None:
            return
        
        # Raise error if lengths of 'defaults' and table columns are different
        columns = range(skip_first_col, len(table[0]))
        if (c := len(columns)) != (d := len(defaults)):
            err = f"Expected {c} items in 'defaults', but {d} were found."
            raise ValueError(err)

        # Replace invalid or empty values of table with default ones
        raw_table = self._getRawValueTable(parameter)
        for row in range(len(table)):
            for col, default in zip(columns, defaults):
                value, raw_value = table[row][col], raw_table[row][col]
                filt = parameter.filters[col]

                # Fix empty values
                if raw_value == '#' and value != default:
                    table[row][col] = default
                # Fix numeric values out of range
                elif filt.type == 'Range': 
                    vmin, vmax = filt.list
                    table[row][col] = max(vmin, min(value, vmax))
                # Fix values not in list
                elif filt.type == 'ValueList': 
                    table[row][col] = value if value in filt.list else default

        parameter.values = table


    def whereClause(
        self,
        fc: arcpy.Parameter,
        field: arcpy.Parameter,
        operator: str,
        value: Iterable[Any] | Any
    ) -> str:
        
        """Generate a where clause, useful when iterating data using cursors.

        Args:
            fc (arcpy.Parameter): Feature class to be processed by cursor.
            field (arcpy.Parameter): Field used in where clause.
            operator (str): SQL operator used in where clause.
            value (Iterable[Any] | Any): Target value(s) used in where clause.

        Returns:
            str: SQL-formatted where clause.
        """

        fieldname = arcpy.AddFieldDelimiters(fc.valueAsText, field.valueAsText)
        if operator in ('IN', 'NOT IN'):
            value = f'''({", ".join(f"'{x}'" for x in value)})'''
        return f'{fieldname} {operator} {value}'



class RoseDiagrams(_ToolBase):

    VALUETABLE_DEFAULTS = {'Plot_Data': ('Random', 'NO', 1, 100)}

    def __init__(self):

        """Define the tool (tool name is the name of the class)."""

        super().__init__()
        self.label = 'Rose Diagrams'
        self.description = 'Use this tool to plot rose diagrams'


    def hasObtuseSpread(self, angles: ArrayLike) -> bool:

        """Check if the provided array of angles spreads more than 180 degrees.

        Args:
            angles (ArrayLike): Angles in range [0, 360] degrees.

        Returns:
            bool: Whether the angles spread more than 180 degrees.
        """

        array = np.unwrap(np.asarray(angles), period=360)
        return array.max() - array.min() > 180


    def getParameterInfo(self) -> list[arcpy.Parameter]:

        """Define parameter definitions.

        Returns:
            list[arcpy.Parameter]: List of tool's parameters.
        """

        # INPUT FEATURE (param0) ----------------------------------------------
        param0 = arcpy.Parameter(
            name='input_feature',
            displayName='Input Feature',
            direction='Input',
            datatype='GPFeatureLayer',
            parameterType='Required'
        )
        param0.filter.list = ['Point', 'Polyline', 'Polygon']                  # type: ignore[reportOptionalMemberAccess]

        # AZIMUTH FIELD (param1) ----------------------------------------------
        param1 = arcpy.Parameter(
            name='azimuth_field',
            displayName='Azimuth Field',
            direction='Input',
            datatype='Field',
            parameterType='Required'
        )
        param1.parameterDependencies = [param0.name]                           # type: ignore[reportAttributeAccessIssue]
        param1.filter.list = NUM_FIELD_FILTERS                                 # type: ignore[reportOptionalMemberAccess]

        # TYPE FIELD (param2) -------------------------------------------------
        param2 = arcpy.Parameter(
            name='type_field',
            displayName='Type Field',
            direction='Input',
            datatype='Field',
            parameterType='Required'
        )
        param2.parameterDependencies = [param0.name]                           # type: ignore[reportAttributeAccessIssue]
        param2.filter.list = ['Text']                                          # type: ignore[reportOptionalMemberAccess]

        # PLOT DATA (param3) --------------------------------------------------
        param3 = arcpy.Parameter(
            name='plot_data',
            displayName='Plot Data',
            direction='Input',
            datatype='GPValueTable',
            parameterType='Optional',
            multiValue=True
        )
        param3.columns = [
            ['GPString', 'Data Type'],
            ['GPString', 'Color'],
            ['GPString', 'Show MEAD Mean Vector(s)'],
            ['GPLong', 'Number of Clusters'],
            ['GPLong', 'MEAD Azimuth tolerance(%)']
        ]
        param3.filters[0].type = 'ValueList'
        param3.filters[1].type = 'ValueList'
        param3.filters[2].type = 'ValueList'
        param3.filters[3].type = 'Range'
        param3.filters[4].type = 'Range'
        param3.filters[0].list = []
        param3.filters[1].list = ['Random'] + sorted(COLORS.keys())
        param3.filters[2].list = ['NO', 'YES']
        param3.filters[3].list = [1, 100]
        param3.filters[4].list = [0, 100]

        # SAVE IMAGE (param4) -------------------------------------------------
        param4 = arcpy.Parameter(
            name='save_image',
            displayName='Save Image',
            direction='Input',
			datatype='GPBoolean',
			parameterType='Optional',
		)
        param4.value = True

        # IMAGE OUTPUT (param5) -----------------------------------------------
        param5 = arcpy.Parameter(
            name='image_output',
            displayName='Image Output',
            direction='Output',          
            datatype='DEFile',
            parameterType='Optional',
        )
        param5.filter.list = IMG_FORMATS                                       # type: ignore[reportOptionalMemberAccess]

        # IMAGE DPI (param6) --------------------------------------------------
        param6 = arcpy.Parameter(
            name='image_dpi',
            displayName='Image Resolution (DPI)',
            direction='Input',
            datatype='GPLong',
            parameterType='Optional',
        )
        param6.controlCLSID = '{C8C46E43-3D27-4485-9B38-A49F3AC588D9}'
        param6.filter.type = 'Range'                                           # type: ignore[reportOptionalMemberAccess]
        param6.filter.list = [0, 600]                                          # type: ignore[reportOptionalMemberAccess]
        param6.value = 200

        # TITLE (param7) ------------------------------------------------------
        param7 = arcpy.Parameter(
            name='title',
            displayName='Plot Title',
            direction='Input',
            datatype='GPString',
            parameterType='Optional',
            category='Plot Customization'
        )

        # SHOW GRID (param8)  -------------------------------------------------
        param8 = arcpy.Parameter(
            name='show_grid',
            displayName='Grid',
            direction='Input',
			datatype='GPBoolean',
			parameterType='Optional',
            category='Plot Customization'
		)
        param8.value = True

        # SHOW LEGEND (param9) ------------------------------------------------
        param9 = arcpy.Parameter(
            name='show_legend',
            displayName='Legend',
            direction='Input',
            datatype='GPBoolean',
			parameterType='Optional',
            category='Plot Customization'
		)
        param9.value = True

        # SHOW NUMBER OF SAMPLES (param10) ------------------------------------
        param10 = arcpy.Parameter(
            name='show_num_samples',
            displayName='Number of Samples',
            direction='Input',
		    datatype = 'GPBoolean',
			parameterType = 'Optional',
            category='Plot Customization'
		)
        param10.value = True

        # MIRRORED PLOT (param11) ---------------------------------------------
        param11 = arcpy.Parameter(
            name='mirrored',
            displayName='Mirrored',
            direction='Input',
			datatype='GPBoolean',
			parameterType='Optional',
            category='Advanced Options'
		)
        param11.value = False

        # WEIGHTED PLOT (param12) ---------------------------------------------
        param12 = arcpy.Parameter(
            name='weighted',
            displayName='Weighted',
            direction='Input',
			datatype='GPBoolean',
			parameterType='Optional',
            category='Advanced Options'
		)
        param12.value = False

        # WEIGHT FIELD (param13) ----------------------------------------------
        param13 = arcpy.Parameter(
            name='weight_field',
            displayName='Weight Field',
            direction='Input',
            datatype='Field',
            parameterType='Optional', 
            enabled=False,
            category='Advanced Options'
        )
        param13.parameterDependencies = [param0.name]                          # type: ignore[reportAttributeAccessIssue]
        param13.filter.list = NUM_FIELD_FILTERS                                # type: ignore[reportOptionalMemberAccess]

        # WRITE LOG (param14) -------------------------------------------------
        param14 = arcpy.Parameter(
            name='write_log',
            displayName='Log File',
            direction='Input',
		    datatype='GPBoolean',
			parameterType='Optional',
            category='Advanced Options'
		)
        param14.value = False

        # TICK SPACING (param15) ----------------------------------------------
        param15 = arcpy.Parameter(
            name='tick_spacing',
            displayName='Tick Spacing',
            direction='Input', 
            datatype='GPLong',
            parameterType='Optional',
            category='Plot Customization'
        )
        param15.filter.type = 'Range'                                          # type: ignore[reportOptionalMemberAccess]
        param15.filter.list = [1, 360]                                         # type: ignore[reportOptionalMemberAccess]
        param15.value = 10

        # RADIAL LABELS ANGLE (param16) ---------------------------------------
        param16 = arcpy.Parameter(
            name='radial_labels_angle',
            displayName='Labels Angle',
            direction='Input',
            datatype='GPLong',
            parameterType='Optional',
            category='Plot Customization'
        )
        param16.filter.type = 'Range'                                          # type: ignore[reportOptionalMemberAccess]
        param16.filter.list = [0, 360]                                         # type: ignore[reportOptionalMemberAccess]
        param16.value = 0

        # GRID SPACE (param17) ------------------------------------------------
        param17 = arcpy.Parameter(
            name='grid_space',
            displayName='Set Grid Space',
            direction='Input',
            datatype='GPLong',
            parameterType='Optional',
            category='Plot Customization'
        )
        param17.value = 8

        # Parameters list
        parameters = [
            param0, param1, param2, param3, param4, param5, param6, param7,
            param8, param9, param10, param11, param12, param13, param14, 
            param15, param16, param17
        ]
        return parameters


    def updateParameters(self, parameters: list[arcpy.Parameter]) -> None:

        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter has
        been changed.
        
        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
        """

        # Parameters accessed by this method
        Input_Feature = parameters[0] # Feature Class
        Azimuth_Field = parameters[1] # Field
        Type_Field = parameters[2] # Field
        Plot_Data = parameters[3] # ValueTable
        Save_Image = parameters[4] # GPBoolean
        Image_Output = parameters[5] # DEFile
        Weighted = parameters[12] # GPBoolean
        Weight_Field = parameters[13] # Field

        # Attempt to fill Azimuth, Type and Weight fields parameters with 
        # matching fields from the newly selected Input Feature. Clear them if
        # the Input Feature parameter is empty
        if Input_Feature.value is None:
            Azimuth_Field.value = None
            Type_Field.value = None
            Weight_Field.value = None

        elif self.parameterChanged(Input_Feature):
            describe = arcpy.Describe(Input_Feature.value)
            fields: list[arcpy.Field] = describe.fields
            numf = [f for f in fields if f.type in NUM_FIELD_TYPES]
            strf = [f for f in fields if f.type == 'String']
            self.inferFieldParameter(Azimuth_Field, numf, self.AZIMUTH_HINTS)
            self.inferFieldParameter(Type_Field, strf, self.TYPE_HINTS)
            self.inferFieldParameter(Weight_Field, numf, self.WEIGHT_HINTS)

        # Update data types value list of Plot Data ValueTable parameter when
        # Type Field parameter changes. Clear value list if Type Field is empty
        if Type_Field.value is None:
            Plot_Data.filters[0].list = []

        elif self.parameterChanged(Type_Field):
            with arcpy.da.SearchCursor(
                Input_Feature.value, Type_Field.valueAsText) as curs:
                data_types = sorted(set((row[0] for row in curs)))
                Plot_Data.filters[0].list = data_types

        # Attempt to fill empty or invalid entries of Plot Data ValueTable 
        # parameter when it changes (entry added, modified, moved or removed)
        if self.parameterChanged(Plot_Data):
            defaults = self.VALUETABLE_DEFAULTS['Plot_Data']
            self.resolveValueTableParameter(Plot_Data, defaults)
                                                                                
        # Enable Image Output parameter when Save Image option is checked
        if self.parameterChanged(Save_Image):
            Image_Output.enabled = Save_Image.value

        # Enable Weight Field parameter when Weighted option is checked
        if self.parameterChanged(Weighted):
            Weight_Field.enabled = Weighted.value

        return


    def updateMessages(self, parameters: list[arcpy.Parameter]) -> None:

        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation.

        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
        """

        # Parameters accessed by this method
        Input_Feature = parameters[0] # Feature Class
        Azimuth_Field = parameters[1] # Field
        Type_Field = parameters[2] # Field
        Plot_Data = parameters[3] # ValueTable
        Save_Image = parameters[4] # GPBoolean
        Image_Output = parameters[5] # DEFile
        Mirrored = parameters[11] # GPBoolean
        Weighted = parameters[12] # GPBoolean
        Weight_Field = parameters[13] # Field

        # Image Output parameter is required if Save Image option is checked                                                            
        if Save_Image.value and Image_Output.value is None:
            Image_Output.setIDMessage('ERROR', 530)

        # Set warning to Mirrored option if azimuth data of types added to Plot
        # Data ValueTable spreads more than 180 degrees
        if (
            Mirrored.value and
            all(p.value is not None for p in parameters[0:4])
        ):
            data = {d[0]: [] for d in Plot_Data.values}
            fields = (Azimuth_Field.valueAsText, Type_Field.valueAsText)
            w = self.whereClause(Input_Feature, Type_Field, 'IN', data.keys())
            with arcpy.da.SearchCursor(Input_Feature.value, fields, w) as curs:
                for azimuth, type_lbl in curs:
                    data[type_lbl].append(azimuth % 360)
            warns = [k for k, v in data.items() if self.hasObtuseSpread(v)]
            if len(warns):
                msg = f'Azimuth data of {warns} spreads more than 180 degrees.'
                Mirrored.setWarningMessage(msg)

        # Weight Field parameter is required if Weighted option is checked                                                            
        if Weighted.value and Weight_Field.value is None:
            Weight_Field.setIDMessage('ERROR', 530)
        
        # Set warning if Weight Field parameter contains negative values 
        if (
            self.parameterChanged(Weight_Field) and
            Input_Feature.value is not None
        ):
            with arcpy.da.SearchCursor(
                Input_Feature.value, Weight_Field.valueAsText) as curs:
                if min(curs)[0] < 0:
                    msg = 'The selected data contains negative values.'
                    Weight_Field.setWarningMessage(msg)

        return


    def isLicensed(self):

        """Set whether tool is licensed to execute."""

        return True


    def execute(self, parameters: list[arcpy.Parameter], messages) -> None:

        """The source code of the tool.

        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
            messages: Tool's messages.
        """

        # Intialize statistics dictionary
        statistics: dict[str, StatItem] = {}

        # Get parameters
        Input_Feature = parameters[0]
        Azimuth_Field = parameters[1]
        Type_Field = parameters[2]
        Plot_Data = parameters[3]
        Save_Image = parameters[4]
        Image_Output = parameters[5]
        Image_Dpi = parameters[6]
        Title = parameters[7]
        Show_Grid = parameters[8]
        Show_Legend = parameters[9]
        Show_Num_Samples = parameters[10]
        Mirrored = parameters[11]
        Weighted = parameters[12]
        Weight_Field = parameters[13]
        Write_Log = parameters[14]
        Tick_Spacing = parameters[15]
        Radial_Labels_Angle = parameters[16]
        Grid_Space = parameters[17]

        # PLOT INITIALIZATION -------------------------------------------------
        # Create figure with polar ax
        fig = Figure(figsize=(6, 6), dpi=Image_Dpi.value)
        ax = PolarAxes(fig, 111)
        fig.add_subplot(ax)

        # Set title
        title = Title.valueAsText if Title.value else Input_Feature.valueAsText
        ax.set_title(title, y=1.1, fontsize=18)

        # Set grid
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        grid_format = np.arange(0, 360, Tick_Spacing.value)
        ax.set_thetagrids(grid_format, grid_format)
        ax.grid(alpha=0.4 if Show_Grid.value else 0)

        # DATA PLOTTING -------------------------------------------------------
        # Parse oriented data
        fields = [Azimuth_Field.valueAsText, Type_Field.valueAsText]
        if Weighted.value:
            fields.append(Weight_Field.valueAsText)

        data = arcpy.da.FeatureClassToNumPyArray(
            in_table=Input_Feature.valueAsText,
            field_names=fields,
            null_value=''
        )

        # Plot data
        max_bar_length = 0
        plot_data = Plot_Data.values if Plot_Data.values is not None else []
        for (_type, _color, _mean_vect, _nclusts, _azm_tol) in plot_data:
           
            # Get azimuth and weight values from data filtered by _type
            flt = data[data[Type_Field.valueAsText] == _type]
            azimuth = flt[Azimuth_Field.valueAsText] % 360
            weights = flt[Weight_Field.valueAsText] if Weighted.value else None

            # Define label for _type to ensure tracking all plotted instances
            type_label = _type
            suffix = 0
            while type_label in statistics.keys():
                suffix += 1
                type_label = f'{_type}({suffix})' 

            # Compute histogram
            bins = np.arange(-5, 366, 10)
            hist, _ = np.histogram(azimuth, bins, weights=weights)

            # Wrap around circular histogram data
            hist[0] += hist[-1]
            hist = hist[:-1]

            # Format histogram data for mirrored diagram, if required
            if Mirrored.value:
                half = np.sum(np.split(hist, 2), axis=0)
                hist = np.concatenate([half, half])

            # Update maximum bar lenght
            max_bar_length = max(max_bar_length, hist.max())

            # Add histogram data to plot
            if _color == 'Random':                                              
                _color = np.random.choice(list(COLORS.keys()))
            ax.bar(
                x=np.radians(np.arange(0, 360, 10)),
                height=hist,
                width=np.radians(10),
                bottom=0.0,
                align='center',
                color=COLORS.get(_color),
                edgecolor='w' if _color == 'Black' else 'k', # TODO make outline lightgray when color is black
                label=type_label
            )

            # Compute mean vector(s) if required
            mean_vector_stats: list[MeanVectorStats] = []
            if _mean_vect == 'YES':

                # Initialize MEAD statistics
                mead_stats: MeanVectorStats = {
                    "algorithm": "MEAD",
                    "vectors": {},
                    "r_means": {},
                    "fisher_stats": {},
                    "tol": (_azm_tol, np.nan),
                    "conf": None
                }
                
                # Treat weights as dips data and set the dip tolerance to 100%
                # so that the output couples will be (azimuth-weight) and the
                # weight info for each azimuth value is preserved and used in 
                # avg_angle function
                mead_clusters = angles_clustering(
                    azimuth.tolist(),
                    weights.tolist() if weights is not None else [1] * len(azimuth),
                    tol_str=_azm_tol / 100,
                    tol_dip=1,                       
                    n_clusts=_nclusts
                )               

                # Add arrows to plot indicating mean vector(s) for each cluster                                        
                for i, cluster in enumerate(mead_clusters, start=1):
                    azm, wgt = zip(*cluster)
                    avg_deg, r_mean = avg_angle(azm, wgt)

                    sign = 1
                    for _ in range(1 + Mirrored.value):
                        ax.quiver(
                            0,
                            0,
                            sign * sin_r(avg_deg),
                            sign * cos_r(avg_deg),
                            scale=2 / r_mean,
                            width=0.01,
                            facecolor=COLORS.get(_color),
                            edgecolor='black' if _color == 'Red' else 'red',
                            linewidth=1.25
                        )
                        sign = -1

                    # Update MEAD statistics for i-th cluster 
                    mead_stats["vectors"][i] = (avg_deg, np.nan)
                    mead_stats["r_means"][i] = r_mean

                # Add MEAD statistics to mean vector statistics
                mean_vector_stats.append(mead_stats)

            # Populate overall statistics for _type data
            statistics[type_lbl] = {
                "n": len(azimuth),
                "kind": "lines",
                "contour": [],
                "clustering": mean_vector_stats
            }

        # PLOT FINALIZATION ---------------------------------------------------
        # Set radial gridlines and labels
        radii = np.linspace(0, max_bar_length, Grid_Space.value)
        fmt = '%.2e' if radii[1] <= 10**-3 or radii[-1] >= 10**6 else '%.2f'
        ax.set_rgrids(
            radii=radii,
            angle=Radial_Labels_Angle.value,
            fmt=fmt,
            weight='semibold',
            size=10,
            color='white',
            path_effects=[mpe.withStroke(linewidth=2, foreground='k')]
        )

        # Set optional legend
        if Show_Legend.value:
            hands, lbls = ax.get_legend_handles_labels()
            by_lbl = {l: h for l, h in sorted(dict(zip(lbls, hands)).items())}
            if len(by_lbl):
                ax.legend(
                    handles=by_lbl.values(),
                    labels=by_lbl.keys(),
                    loc='upper left',
                    bbox_to_anchor=(1.0, 0.9),
                    ncols=1,
                    fontsize=8,
                    numpoints=1,
                    bbox_transform=fig.transFigure,
                    borderaxespad=0.0,
                )

        # Set optional "Weighted on" label
        weight_alias = self.getFieldParameterAlias(Weight_Field, Input_Feature) 
        if Weighted.value:
            ax.text(
                x=0.0,
                y=0.0,
                s=f'Weighted on:\n{weight_alias}',
                fontsize=8,
                transform=fig.transFigure,
            )

        # Set optional "Number of Samples" label
        n_samp = StatsLogger(statistics, "RoseDiagrams").samples(parag=False)
        if Show_Num_Samples.value:
            ax.text(
                x=1.0,
                y=0.0,
                s=n_samp,
                fontsize=8,
                transform=fig.transFigure
            )

        # OUTPUT IMAGE AND LOG ------------------------------------------------
        # Set output filepaths
        if Save_Image.value:
            image_outpath = Image_Output.valueAsText
        else:
            image_outpath = temp_filepath('RoseDiagram_temp.png')

        root = path.splitext(image_outpath)[0]
        log_outpath = f'{root}_LOG.txt'
        xy_path = f'{root}_xy.txt'

        # Save figure
        fig.savefig(image_outpath, dpi='figure', bbox_inches='tight')

        # Write log if required
        if Write_Log.value:
            logger = StatsLogger(statistics, "RoseDiagrams")
            with open(log_outpath, 'wt', encoding='utf-8') as log:
                # Title and number of samples
                log.write(logger.paragraph(title.upper()))
                log.write(logger.paragraph(n_samp))
                # Plot options
                opt = [logger.head('OPTIONS'), f'Mirrored: {Mirrored.value}']
                if Weighted.value:
                    opt.append((f'Weighted on: {weight_alias}'))
                log.write(logger.custom(opt))
                # Clustering statistics
                log.write(logger.clustering(mirrored=Mirrored.value))
                # Extra info (tool name, datetime)
                log.write(logger.extraInfo(parag=False))

        # Save centroid coordinates if image is not temporary     
        if Save_Image.value:
            coords = arcpy.da.FeatureClassToNumPyArray(
                in_table=Input_Feature.valueAsText,
                field_names='Shape'
            )        
            clon, clat = np.nanmean(coords, axis=0)[0]
            descr = arcpy.Describe(Input_Feature.valueAsText)
            with open(xy_path, 'w') as xyfile:
                xyfile.write(f'{clon}\n')
                xyfile.write(f'{clat}\n')
                xyfile.write(f'{path.join(descr.path, descr.name)}\n')

        # Auto-show image (and optionally log file) if image is temporary
        else: # add a dedicated boolean parameter to enable auto-show?
            startfile(image_outpath)
            if Write_Log.value:
                startfile(log_outpath)

        return



class StereoPlots(_ToolBase):

    VALUETABLE_DEFAULTS = {
        'Plot_Data': (
            'Random', 'Point (•)', 5.0, 'NONE', 0.5
        ),
        'Contour_Data': (
            'Kamb (exponential smoothing)', 2.0, 'Filled', 'Random', 0
        ),
        'Mean_Vector_Data': (
            'MEAD+Fisher', 1, 50, 30, 95, 'Random', 'Square (◻)', 5.0, 'NONE', 0.5
        )
    }

    def __init__(self):

        """Define the tool (tool name is the name of the class)."""

        super().__init__()
        self.label = 'Stereoplots'
        self.description = 'Add description.'


    def getParameterInfo(self) -> list[arcpy.Parameter]:

        """Define parameter definitions.

        Returns:
            list[arcpy.Parameter]: List of tool's parameters.
        """

        #INPUT FEATURE (param0) -----------------------------------------------
        param0 = arcpy.Parameter(
            name='input_feature',
            displayName='Input Feature',
            direction='Input',
            datatype='GPFeatureLayer',
            parameterType='Required'
        )
        param0.filter.list = ['Point', 'Polyline', 'Polygon']                  # type: ignore[reportOptionalMemberAccess]

        # AZIMUTH FIELD (param1) ----------------------------------------------
        param1 = arcpy.Parameter(
            name='azimuth_field',
            displayName='Azimuth Field',
            direction='Input',
            datatype='Field',
            parameterType='Required'
        )
        param1.parameterDependencies = [param0.name]                           # type: ignore[reportAttributeAccessIssue]
        param1.filter.list = NUM_FIELD_FILTERS                                 # type: ignore[reportOptionalMemberAccess]
        
        # DIP FIELD (param2) --------------------------------------------------
        param2 = arcpy.Parameter(
            name='dip_field',
            displayName='Dip Field',
            direction='Input',
            datatype='Field',
            parameterType='Required'
        )
        param2.parameterDependencies = [param0.name]                           # type: ignore[reportAttributeAccessIssue]
        param2.filter.list = NUM_FIELD_FILTERS                                 # type: ignore[reportOptionalMemberAccess]
        
        # METHOD FIELD (param3) -----------------------------------------------
        param3 = arcpy.Parameter(
            name='method_field',
            displayName='Method Field',
            direction='Input',
            datatype='Field',
            parameterType='Required'
        )
        param3.parameterDependencies = [param0.name]                           # type: ignore[reportAttributeAccessIssue]
        param3.filter.list = ['Text']                                          # type: ignore[reportOptionalMemberAccess]
        
        # TYPE FIELD (param2) -------------------------------------------------
        param4 = arcpy.Parameter(
            name='type_field',
            displayName='Type Field',
            direction='Input',
            datatype='Field',
            parameterType='Required'
        )
        param4.parameterDependencies = [param0.name]                           # type: ignore[reportAttributeAccessIssue]
        param4.filter.list = ['Text']                                          # type: ignore[reportOptionalMemberAccess]

        # PLOT DATA (param5) --------------------------------------------------
        param5 = arcpy.Parameter(
            name='plot_data',
            displayName='Plot Data',
            direction='Input', 
            datatype='GPValueTable',
            parameterType='Optional',
            multiValue=True
        )
        param5.columns = [
            ['GPString', 'Data Type'],
            ['GPString', 'Color'],
            ['GPString', 'Pole/Vector Symbol'],
            ['GPDouble', 'Pole/Vector Size'],
            ['GPString', 'Cyclographic Trace Style'],
            ['GPDouble', 'Line Width']
        ]
        param5.filters[0].type = 'ValueList'
        param5.filters[1].type = 'ValueList'
        param5.filters[2].type = 'ValueList'
        param5.filters[3].type = 'Range'
        param5.filters[4].type = 'ValueList'
        param5.filters[5].type = 'Range'
        param5.filters[0].list = []
        param5.filters[1].list = ['Random'] + sorted(COLORS.keys())
        param5.filters[2].list = ['NONE'] + sorted(MARKERS.keys())
        param5.filters[3].list = [0.1, 100.0]
        param5.filters[4].list = ['NONE'] + sorted(LINES.keys())
        param5.filters[5].list = [0.1, 10.0]

        # SAVE IMAGE (param6) -------------------------------------------------
        param6 = arcpy.Parameter(
            name='save_image',
            displayName='Save Image',
            direction='Input',
			datatype='GPBoolean',
			parameterType='Optional'
		)
        param6.value = True

        # IMAGE OUTPUT (param7) -----------------------------------------------
        param7 = arcpy.Parameter(
            name='image_output',
            displayName='Image Output',
            direction='Output',
            datatype='DEFile',
            parameterType='Optional'
        )
        param7.filter.list = IMG_FORMATS                                       # type: ignore[reportOptionalMemberAccess]

        # IMAGE DPI (param8) --------------------------------------------------
        param8 = arcpy.Parameter(
            name='image_dpi',
            displayName='Image Resolution (DPI)',
            direction='Input',
            datatype='GPLong',
            parameterType='Optional'
        )
        param8.controlCLSID = '{C8C46E43-3D27-4485-9B38-A49F3AC588D9}'
        param8.filter.type = 'Range'                                           # type: ignore[reportOptionalMemberAccess]
        param8.filter.list = [0, 600]                                          # type: ignore[reportOptionalMemberAccess]                                       
        param8.value = 200

        # NET TYPE (param9) ---------------------------------------------------
        param9 = arcpy.Parameter(
            name='net_type',
            displayName='Net Type',
            direction='Input',
            datatype='GPString',
            parameterType='Optional',
            category='Advanced Options'
		)
        param9.filter.type = 'ValueList'                                       # type: ignore[reportOptionalMemberAccess]
        param9.filter.list = ['Schmidt (Equal Area)', 'Wulff (Equal Angle)']   # type: ignore[reportOptionalMemberAccess]
        param9.value = 'Schmidt (Equal Area)'

        # WRITE LOG (param10) -------------------------------------------------
        param10 = arcpy.Parameter(
            name='write_log',
            displayName='Log File',
            direction='Input',
			datatype='GPBoolean',
			parameterType='Optional',
            category='Advanced Options'
		)
        param10.value = False

        # TITLE (param11) -----------------------------------------------------
        param11 = arcpy.Parameter(
            name='title',
            displayName='Plot Title',
            direction='Input',
            datatype='GPString',
            parameterType='Optional',
            category='Plot Customization'
        )

        # NUMBER OF TICK MARKS (param12) --------------------------------------
        param12 = arcpy.Parameter(
            name='num_ticks',
            displayName='Number of Tick Marks',
            direction='Input',
            datatype='GPLong',
            parameterType='Optional',
            category='Plot Customization'
        )
        param12.filter.type = 'ValueList'                                      # type: ignore[reportOptionalMemberAccess]
        param12.filter.list = [0, 1, 4, 8]                                     # type: ignore[reportOptionalMemberAccess]
        param12.value = 8

        # TICK MARKS FORMAT (param13) -----------------------------------------
        param13 = arcpy.Parameter(
            name='ticks_format',
            displayName='Tick Marks Format',
            direction='Input',
            datatype='GPString',
            parameterType='Optional',
            category='Plot Customization'
        )
        param13.filter.type = 'ValueList'                                      # type: ignore[reportOptionalMemberAccess]
        param13.filter.list = list(COMPASS_LABELS.keys())                      # type: ignore[reportOptionalMemberAccess]
        param13.value = 'Degrees'

        # SHOW GRID (param14) -------------------------------------------------
        param14 = arcpy.Parameter(
            name='show_grid',
            displayName='Grid',
            direction='Input',
		    datatype='GPBoolean',
			parameterType='Optional',
            category='Plot Customization'
		)
        param14.value = True

        # SHOW LEGEND (param15) -----------------------------------------------
        param15 = arcpy.Parameter(
            name='show_legend',
            displayName='Legend',
            direction='Input',
            datatype='GPBoolean',
			parameterType='Optional',
            category='Plot Customization'
		)
        param15.value = True

        # SHOW NUMBER OF LABELS (param16) -------------------------------------
        param16 = arcpy.Parameter(
            name='show_num_samples',
            displayName='Number of Samples',
            direction='Input',
			datatype='GPBoolean',
			parameterType='Optional',
            category='Plot Customization'
		)
        param16.value = True

        # CONTOUR DATA (param17) ----------------------------------------------
        param17 = arcpy.Parameter(
            name='contour',
            displayName='Contour',
            direction='Input',
            datatype='GPValueTable',
            parameterType='Optional',
            category='Contour & Statistics'
        )
        param17.columns = [
            ['GPString', 'Data Type'],
            ['GPString', 'Method'],
            ['GPDouble', 'Standard deviation (σ)'],
            ['GPString', 'Style'],
            ['GPString', 'Color'],
            ['GPLong', 'Transparency (%)']
        ]
        param17.filters[0].type = 'ValueList'
        param17.filters[1].type = 'ValueList'
        param17.filters[2].type = 'Range'
        param17.filters[3].type = 'ValueList'
        param17.filters[4].type = 'ValueList'
        param17.filters[5].type = 'Range'
        param17.filters[0].list = []
        param17.filters[1].list = sorted(CONTOUR_METHODS.keys())
        param17.filters[2].list = [0.0, 100.0]
        param17.filters[3].list = ['Filled', 'Unfilled']
        param17.filters[4].list = ['Random'] + list(CONTOUR_COLORMAPS.keys())
        param17.filters[5].list = [0, 100]

        # SHOW COLORBAR (param18) ---------------------------------------------
        param18 = arcpy.Parameter(
            name='show_colorbar',
            displayName='Contour Colorbar',
            direction='Input',
		    datatype='GPBoolean',
			parameterType='Optional',
            category='Contour & Statistics'
		)
        param18.value = True

        # MEAN VECTOR DATA (param19) ------------------------------------------
        param19 = arcpy.Parameter(
            name='mean_vector',
            displayName='Mean Vector(s)',
            direction='Input',
            datatype='GPValueTable',
            parameterType='Optional',
            category='Contour & Statistics'
        )
        param19.columns = [
            ['GPString', 'Data Type'],
            ['GPString', 'Algorithm'],
            ['GPLong', 'Number of Clusters'],
            ['GPLong', 'MEAD Azimuth tolerance (%)'],
            ['GPLong', 'MEAD Inclination tolerance (%)'],
            ['GPLong', 'Fisher confidence (%)'],
            ['GPString', 'Color'],
            ['GPString', 'Pole/Vector Symbol'],
            ['GPDouble', 'Pole/Vector Size'],
            ['GPString', 'Cyclographic Trace Style'],
            ['GPDouble', 'Line Width']
        ]
        param19.filters[0].type = 'ValueList'
        param19.filters[1].type = 'ValueList'
        param19.filters[2].type = 'Range'
        param19.filters[3].type = 'Range'
        param19.filters[4].type = 'Range'
        param19.filters[5].type = 'Range'
        param19.filters[6].type = 'ValueList'
        param19.filters[7].type = 'ValueList'
        param19.filters[8].type = 'Range'
        param19.filters[9].type = 'ValueList'
        param19.filters[10].type = 'Range'
        param19.filters[0].list = []
        param19.filters[1].list = ['MEAD', 'MEAD+Fisher', 'K-means', 'Bingham']
        param19.filters[2].list = [1, 100]
        param19.filters[3].list = [0, 100]
        param19.filters[4].list = [0, 100]
        param19.filters[5].list = [0, 99]
        param19.filters[6].list = ['Random'] + sorted(COLORS.keys())
        param19.filters[7].list = ['NONE'] + sorted(MARKERS.keys())
        param19.filters[8].list = [0.1, 100.0]
        param19.filters[9].list = ['NONE'] + sorted(LINES.keys())
        param19.filters[10].list = [0.1, 10.0]

        # TRACK MEAD (param20) ------------------------------------------------
        param20 = arcpy.Parameter(
            name='track_mead',
            displayName='Track MEAD Clustering',
            direction='Input',
            datatype='GPBoolean',
            parameterType='Optional',
            category='Contour & Statistics'
        )
        param20.value = False

        # Parameters list
        parameters = [
            param0, param1, param2, param3, param4, param5, param6, param7,
            param8, param9, param10, param11, param12, param13, param14,
            param15, param16, param17, param18, param19, param20
        ]
        return parameters
    

    def updateParameters(self, parameters: list[arcpy.Parameter]) -> None:

        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter has
        been changed.
        
        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
        """

        # Parameters accessed by this method
        Input_Feature = parameters[0] # Feature Class
        Azimuth_Field = parameters[1] # Field
        Dip_Field = parameters[2] # Field
        Method_Field = parameters[3] # Field
        Type_Field = parameters[4] # Field
        Plot_Data = parameters[5] # ValueTable
        Save_Image = parameters[6] # GPBoolean
        Image_Output = parameters[7] # DEFile
        Contour_Data = parameters[17] # ValueTable
        Show_Colorbar = parameters[18] # GPBoolean
        Mean_Vector_Data = parameters[19] # ValueTable
        Track_Mead = parameters[20] # GPBoolean

        # Attempt to fill Azimuth, Dip, Method and Type fields parameters with
        # matching fields from the newly selected Input Feature. Clear them if
        # the Input Feature parameter is empty
        if Input_Feature.value is None:
            Azimuth_Field.value = None
            Dip_Field.value = None
            Method_Field.value = None
            Type_Field.value = None

        elif self.parameterChanged(Input_Feature):
            describe = arcpy.Describe(Input_Feature.value)
            fields: list[arcpy.Field] = describe.fields
            numf = [f for f in fields if f.type in NUM_FIELD_TYPES]
            strf = [f for f in fields if f.type == 'String']
            self.inferFieldParameter(Azimuth_Field, numf, self.AZIMUTH_HINTS)
            self.inferFieldParameter(Dip_Field, numf, self.DIP_HINTS)
            self.inferFieldParameter(Method_Field, strf, self.METHOD_HINTS)
            self.inferFieldParameter(Type_Field, strf, self.TYPE_HINTS)

        # Update data types value lists of Plot Data, Contour Data and Mean
        # Vector Data ValueTable parameters when Type Field parameter changes.
        # Clear value lists if Type Field is empty
        if Type_Field.value is None:
            Plot_Data.filters[0].list = []
            Contour_Data.filters[0].list = []
            Mean_Vector_Data.filters[0].list = []

        elif self.parameterChanged(Type_Field):
            with arcpy.da.SearchCursor(
                Input_Feature.value, Type_Field.valueAsText) as curs:
                data_types = sorted(set((row[0] for row in curs)))
                Plot_Data.filters[0].list = data_types
                Contour_Data.filters[0].list = data_types
                Mean_Vector_Data.filters[0].list = data_types

        # Attempt to fill empty or invalid entries of Plot Data, Contour Data
        # and Mean Vector Data ValueTable parameters when they change (entry 
        # added, modified, moved or removed)    
        if self.parameterChanged(Plot_Data):
            defaults = self.VALUETABLE_DEFAULTS['Plot_Data']
            self.resolveValueTableParameter(Plot_Data, defaults)

        if self.parameterChanged(Contour_Data):
            defaults = self.VALUETABLE_DEFAULTS['Contour_Data']
            self.resolveValueTableParameter(Contour_Data, defaults)

        if self.parameterChanged(Mean_Vector_Data):
            defaults = self.VALUETABLE_DEFAULTS['Mean_Vector_Data']
            self.resolveValueTableParameter(Mean_Vector_Data, defaults)

        # Enable Output Image parameter when Save Image option is checked
        if self.parameterChanged(Save_Image):
            Image_Output.enabled = Save_Image.value

        # Disable and uncheck Show Colorbar option when Contour Data is empty
        Show_Colorbar.enabled = Contour_Data.values is not None
        if not Show_Colorbar.enabled:
            Show_Colorbar.value = False

        # Disable and uncheck Track MEAD option when Mean Vector Data is empty
        Track_Mead.enabled = Mean_Vector_Data.values is not None
        if not Track_Mead.enabled:
            Track_Mead.value = False

        return


    def updateMessages(self, parameters: list[arcpy.Parameter]) -> None:

        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation.
        
        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
        """

        # Parameters accessed by this method
        Save_Image = parameters[6] # GPBoolean
        Image_Output = parameters[7] # DEFile
        Net_Type = parameters[9] # GPString
        Contour_Data = parameters[17] # ValueTable
        Mean_Vector_Data = parameters[19] # ValueTable

        # Output Image parameter is required if Save Image option is checked                                                            
        if Save_Image.value and Image_Output.value is None:
            Image_Output.setIDMessage('ERROR', 530)

        # Set warning to Net Type parameter when "Wulff" net is selected with
        # active statistical analysis (Contour or Mean Vector Extraction)
        if (
            Net_Type.valueAsText == 'Wulff (Equal Angle)' and
            any(p.value is not None for p in (Contour_Data, Mean_Vector_Data))
        ):
            msg = 'Schmidt net type is recommended for better statistics.'
            Net_Type.setWarningMessage(msg)

        # Set warning to Contour Data ValueTable when more than one feature is
        # selected
        if (con_data := Contour_Data.values) is not None and len(con_data) > 1:
            msg = 'Applying contour on multiple features may cause issues.'
            Contour_Data.setWarningMessage(msg)

        return


    def isLicensed(self):

        """Set whether tool is licensed to execute."""

        return True


    def execute(self, parameters: list[arcpy.Parameter], messages) -> None:

        """The source code of the tool.

        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
            messages: Tool's messages.
        """

        # Check for mplstereonet availability
        if self.isModuleAvailable('mplstereonet'):
            import mplstereonet as msn
        else:
            wiki_url = "https://github.com/albdag/ArcStereoNet/wiki"
            installation_page = "1.-Installation#installation-for-arcgis-pro"
            link = "/".join((wiki_url, installation_page))
            msg = "You need 'mplstereonet' dependency to run this tool. For "\
                  f"more info, check the installation guide at {link}"
            messages.addErrorMessage(msg)
            return

        # Get parameters
        Input_Feature = parameters[0]
        Azimuth_Field = parameters[1]
        Dip_Field = parameters[2]
        Method_Field = parameters[3]
        Type_Field = parameters[4]
        Plot_Data = parameters[5]
        Save_Image = parameters[6]
        Image_Output = parameters[7]
        Image_Dpi = parameters[8]
        Net_Type = parameters[9]
        Write_Log = parameters[10]
        Title = parameters[11]
        Num_Ticks = parameters[12]
        Ticks_Format = parameters[13]
        Show_Grid = parameters[14]
        Show_Legend = parameters[15]
        Show_Num_Samples = parameters[16]
        Contour_Data = parameters[17]
        Show_Colorbar = parameters[18]
        Mean_Vector_Data = parameters[19]
        Track_Mead = parameters[20]


#-------------------------------Importing Data---------------------------------#

        fields_params = (Azimuth_Field, Dip_Field, Method_Field, Type_Field)
        fields = tuple(fp.valueAsText for fp in fields_params)                  # Supported field names, later referred as: [0],[1],[2],[3]
        source = Input_Feature.valueAsText
        data_array = arcpy.da.FeatureClassToNumPyArray(source, fields,
                                                       null_value='')
        data = data_array.tolist()

        if Save_Image.value:
            coords = arcpy.da.FeatureClassToNumPyArray(source, 'Shape')         # Extracting centroid coordinates
            centroid = np.nanmean(coords.tolist(), axis=0)[0]


#-------------------------Pre-Elaboration of Data------------------------------#

        # Determine image output filepath
        if Save_Image.value:
            image_outpath = Image_Output.valueAsText
        else:
            image_outpath = temp_filepath('Stereoplot_temp.png')

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

        if Write_Log.value:
            contour_stats = []
            stats_log = {}
            fisher_stats = {}


#>>> Initialize fisher cones dictionary to show cones in legend

        from matplotlib.lines import Line2D
        fisher_cones = {}



#-----------------------Initialization of MPLStereonet-------------------------#
        # Initialize plot
        fig = Figure(figsize=(6, 6), dpi=Image_Dpi.value)
        if Net_Type.valueAsText == 'Wulff (Equal Angle)':
            ax = msn.stereonet_axes.EqualAngleAxes(fig, 111)
        else:
            ax = msn.stereonet_axes.EqualAreaAxes(fig, 111)
        fig.add_subplot(ax)

        # Set title
        title = Title.valueAsText if Title.value else Input_Feature.valueAsText
        ax.set_title(title, y=1.1, fontsize=18)

        # Set grid and ticks
        ax.grid(Show_Grid.value)
        ax.set_azimuth_ticks(np.linspace(0, 360, N_TICKS.value, False))
        if (n := Num_Ticks.value) > 0:
            ticklabels = COMPASS_LABELS[Ticks_Format.valueAsText]
            ax.set_azimuth_ticklabels(ticklabels[::8//n])

#-----------------------Plotting of selected data------------------------------#

#>>> PLOTTING DATA Value Table <<<

        if Plot_Data.values != None:
            for row in Plot_Data.values:
                _type, _color, _marker, _mSize, _line, _lWidth = row

                if _color == 'Random':                                          # Select a random color if not specified by the user
                    random_index = np.random.randint(len(COLORS))
                    _color = list(COLORS.keys())[random_index]

                proj_types.add(_type)                                           # Add data type to projected data types set

                strikes, dips, trends, plunges = filterData(fixed_data, _type)  # Splitting RHR data from TP data


                if _marker != 'NONE':                                           # Plotting Poles or Vectors

                    if len(strikes) > len(trends):
                        ax.pole(strikes,
                                dips,
                                c = COLORS[_color],
                                marker = MARKERS[_marker],
                                mec = 'w' if _color == 'Black' else 'k',
                                markersize = _mSize,
                                label = _type + ' (Poles)')

                    else:
                        ax.line(plunges,
                                trends,
                                c = COLORS[_color],
                                marker = MARKERS[_marker],
                                mec = 'w' if _color == 'Black' else 'k',
                                markersize = _mSize,
                                label = _type)


                if _line != 'NONE':                                             # Plotting Planes
                        fground = 'white' if _color == 'Black' else 'black'
                        ax.plane(strikes,
                                 dips,
                                 c = COLORS[_color],
                                 ls = LINES[_line],
                                 linewidth = _lWidth,
                                 path_effects = [mpe.withStroke(linewidth=_lWidth*2,
                                                                foreground=fground)],
                                 label = _type)



#>>> CONTOUR Value Table <<<

        if Contour_Data.values != None:
            for row in Contour_Data.values:
                _type, _method, _sigma, _style, _color, _alpha = row

                if _color == 'Random':                                          # Select a random color map if not specified by the user
                    random_index = np.random.randint(len(CONTOUR_COLORMAPS))
                    _color = list(CONTOUR_COLORMAPS.keys())[random_index]

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
                                               method = CONTOUR_METHODS[_method],
                                               measurement = apply_on,
                                               sigma = _sigma,
                                               cmap = CONTOUR_COLORMAPS[_color],
                                               alpha = _alpha)

                    mask = ax.density_contour(x1,
                                              x2,
                                              method = CONTOUR_METHODS[_method],
                                              measurement = apply_on,
                                              sigma = _sigma,
                                              colors = 'black',
                                              alpha = _alpha)


                else:                                                           # Applying Unfilled Density Contour Function on poles or vectors

                    udcf = ax.density_contour(x1,
                                              x2,
                                              method = CONTOUR_METHODS[_method],
                                              measurement = apply_on,
                                              sigma = _sigma,
                                              cmap = CONTOUR_COLORMAPS[_color],
                                              alpha = _alpha)


                if Write_Log.value:
                    contour_stats = [_type, _method, _sigma]



#>>> EXTRACT MEAN VECTORS Value Table <<<

        if Mean_Vector_Data.values != None:
            for row in Mean_Vector_Data.values:
                _type, _algorithm, _nClusts, _azmTol, _dipTol, _conf = row[:6]
                _color, _marker, _mSize, _line, _lWidth = row[6:]


                if _color == 'Random':                                          # Select a random color if not specified by the user
                    random_index = np.random.randint(len(COLORS))
                    _color = list(COLORS.keys())[random_index]

                proj_types.add(_type)                                           # Add data type to projected data types set

                strikes, dips, trends, plunges = filterData(fixed_data, _type)  # Splitting RHR data from TP data

                if len(strikes) > len(trends):
                    x1, x2, apply_on = strikes, dips, 'poles'
                else:
                    x1, x2, apply_on = plunges, trends, 'lines'


                if _algorithm == 'K-means':                                     # Extract K-means mean vector(s)

                    while True:
                        try:
                            centers = msn.kmeans(x1, x2, num=_nClusts,
                                             measurement=apply_on)
                            break
                        except np.linalg.LinAlgError:
                            _nClusts -= 1

                    if apply_on == 'poles':
                        c1, c2 = msn.geographic2pole(*zip(*centers))
                    else:
                        c1, c2 = msn.geographic2plunge_bearing(*zip(*centers))


                elif _algorithm == 'Bingham':                                   # Extract Bingham Statistics

                    c1, c2 = zip(msn.fit_girdle(x1, x2, measurement = apply_on))



                else:                                                           # Extract MEAD clusters

                    _azmTol, _dipTol = _azmTol / 100., _dipTol / 100.

                    mead_clust = angles_clustering(x1, x2, apply_on,
                                                   _azmTol, _dipTol,
                                                   _nClusts)

                    azm, inc = [], []
                    for mc in mead_clust:
                        fam = list(zip(*mc))
                        azm.append(fam[0])
                        inc.append(fam[1])

                        if Track_Mead.value:                                    # Track MEAD clustering process behaviour if required
                            num_mark = '${}$'.format(mead_clust.index(mc) + 1)
                            if apply_on == 'poles':
                                ax.pole(fam[0],
                                        fam[1],
                                        c = 'k',
                                        marker = num_mark,
                                        mec = COLORS[_color],
                                        markersize = _mSize + _mSize/2.)

                            else:
                                ax.line(fam[1],
                                        fam[0],
                                        c = 'k',
                                        marker = num_mark,
                                        mec = COLORS[_color],
                                        markersize = _mSize + _mSize/2.)


                    c1, c2 = [], []
                    for i in range(len(mead_clust)):

                        if 'Fisher' in _algorithm:                              # Extract Fisher mean vector(s) and stats

                            if apply_on == 'poles':
                                vectorPB, stats = msn.find_fisher_stats(azm[i], inc[i],
                                                                    measurement = 'poles',
                                                                    conf = _conf)
                                vectorSD = msn.plunge_bearing2pole(*vectorPB)
                                c1.append(vectorSD[0][0])
                                c2.append(vectorSD[1][0])

                            else:
                                vectorPB, stats = msn.find_fisher_stats(inc[i], azm[i],
                                                                    measurement = 'lines',
                                                                    conf = _conf)
                                c1.append(vectorPB[0])
                                c2.append(vectorPB[1])

                            ax.cone(vectorPB[0],
                                    vectorPB[1],
                                    stats[1],
                                    facecolor = "none",
                                    edgecolor = COLORS[_color],
                                    linestyle = '-',
                                    linewidth = _lWidth)

                            cone_lbl = _type + ' mean confidence cone'
                            fisher_cones[cone_lbl] = Line2D([0], [0],
                                                            ls = 'None',
                                                            mec = COLORS[_color],
                                                            mfc = 'None',
                                                            marker = "o",
                                                            ms = 10,
                                                            mew = _lWidth)

                            if Write_Log.value:
                                if _type not in fisher_stats:
                                    fisher_stats[_type] = stats
                                else:
                                    fisher_stats[_type] += stats


                        else:                                                   # Extract MEAD mean vector(s)

                            c_azm, _ = avg_angle(azm[i])
                            c_inc = sum(inc[i]) / len(inc[i])
                            c1.append(c_azm if apply_on == 'poles' else c_inc)
                            c2.append(c_inc if apply_on == 'poles' else c_azm)



                label_root = '{} [{}] mean'.format(_type, _algorithm)           # Project Statistics Data and collect mean data and stats for log file if required
                if _algorithm == 'Bingham':
                    label_root = '{} [{}] best fit'.format(_type, _algorithm)

                for n in range(len(c1)):

                    if Write_Log.value:
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
                                    c = COLORS[_color],
                                    marker = MARKERS[_marker],
                                    mec = ('w' if _color == 'Black' else 'k'),
                                    markersize = _mSize,
                                    label = label_root + ' pole')

                        else:
                            ax.line(c1[n],
                                    c2[n],
                                    c = COLORS[_color],
                                    marker = MARKERS[_marker],
                                    mec = ('w' if _color == 'Black' else 'k'),
                                    markersize = _mSize,
                                    label = label_root + ' vector')



                    if _line != 'NONE' and (apply_on == 'poles' or _algorithm == 'Bingham'):

                        fground = ('white' if _color == 'Black' else 'black')
                        ax.plane(c1[n],
                                 c2[n],
                                 c = COLORS[_color],
                                 ls = LINES[_line],
                                 linewidth = _lWidth,
                                 path_effects = [mpe.withStroke(linewidth=_lWidth*2,
                                                                foreground=fground)],
                                 label = label_root + ' plane')



#----Adding Legend, n.of Data label & Colorbar to the plot---#

        if Show_Legend.value == True:
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


        if Show_Num_Samples.value == True:
            txt4lbl = ''
            for pt in sorted(proj_types):
                pt_num = sum([FD.count(pt) for FD in fixed_data])
                txt4lbl += pt + ' = ' + str(pt_num) + '\n'
            ax.text(1.0, 0.0,
                    txt4lbl,
                    fontsize = 8,
                    transform = ax.transAxes)


        if Show_Colorbar.value == True:
            cbax = fig.add_axes((-0.2, 0.1, 0.03, 0.8))                       # left, bottom, width, height
            try:
                fig.colorbar(fdcf, cax=cbax)
            except NameError:
                fig.colorbar(udcf, cax=cbax)
            contourInfo = u'Contour applied on:\n{}\n\nMethod: {}\nStd. Dev.: {}'.format(*CONTOUR.values[0][:3])
            ax.text(-0.29, 0.0,
                    contourInfo,
                    fontsize = 8,
                    transform = ax.transAxes)


#-------------------------------Write Log File---------------------------------#

        if Write_Log.value:
            logPath = path.splitext(image_outpath)[0] + '_LOG.txt'

            with io.open(logPath, 'w') as LOG:
                LOG.write(title.upper() + '\n\n')

                if Show_Num_Samples.value == False:
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

                        if key[1] == 'MEAD+Fisher':
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

        fig.savefig(image_outpath, dpi='figure', bbox_inches='tight')
        plt.close('all')

        if Save_Image.value == False:
            startfile(image_outpath)
            if Write_Log.value:
                startfile(logPath)

        elif image_outpath.endswith('.png'):
            xypath = path.splitext(image_outpath)[0] + '_xy.txt'       # Saving centroid coordinates
            clon, clat = centroid
            descr = arcpy.Describe(Input_Feature.valueAsText)
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


    def getParameterInfo(self) -> list[arcpy.Parameter]:

        """Define parameter definitions.

        Returns:
            list[arcpy.Parameter]: List of tool's parameters.
        """

        # INPUT PLOTS (param0) ------------------------------------------------
        param0 = arcpy.Parameter(
            name='input_plots',
            displayName='Input Plots',
            direction='Input',
            datatype='DERasterDataset',
            parameterType='Required',
            multiValue=True
        )

        # OUTPUT FEATURE (param1) ---------------------------------------------
        param1 = arcpy.Parameter(
            name='output_feature',
            displayName='Output Feature',
            direction='Output',
            datatype='DEFeatureClass',
            parameterType='Required'
        )

        # Parameters list
        parameters = [param0, param1]
        return parameters


    def updateParameters(self, parameters: list[arcpy.Parameter]) -> None:

        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed.
        
        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
        """

        return
    

    def updateMessages(self, parameters: list[arcpy.Parameter]) -> None:

        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation.
        
        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
        """

        # Parameters accessed by this method
        Input_Plots = parameters[0] # Raster (multivalue)

        # Send error message if the sidecar xy file is missing from any plot.
        if self.parameterChanged(Input_Plots) and Input_Plots.value is not None:
            for img_path in Input_Plots.values:
                xy_path = path.splitext(str(img_path))[0] + '_xy.txt'
                if not path.exists(xy_path):
                    err = f'Unable to find coordinates file for {img_path}.'
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
        OUT_FC = parameters[1]



#------------------Creation of the new output feature class--------------------#

        rows = []
        sr = []                                                                 # sr = Spatial Reference

        for plot_img in IN_PLOT.values:

            xy_file = path.splitext(str(plot_img))[0] + '_xy.txt'
            with io.open(xy_file, 'r') as temp_xy:
                lon, lat, source_path = temp_xy.readlines()                
                lon, lat = float(lon), float(lat)
                sr.append(arcpy.Describe(source_path).spatialReference)
                rows.append(((lon, lat), str(plot_img)))

        out_path, out_name = path.split(OUT_FC.valueAsText)
        # Set SR to None if multiple SR are found (and send warning)
        if len({s.exporttostring() for s in sr}) == 1:
            spat_ref = sr[0]
        else:
            spat_ref = None
            messages.addWarningMessage('Found conflicting SR: set to Unknown!')
        arcpy.CreateFeatureclass_management(out_path, out_name, "POINT",
                                            spatial_reference=spat_ref)
        arcpy.AddField_management(OUT_FC.valueAsText, "Plot", "TEXT")
        
        fields = ["SHAPE@XY", "Plot"]
        with arcpy.da.InsertCursor(OUT_FC.valueAsText, fields) as curs:
            for row in rows:
                curs.insertRow(row)



        return




