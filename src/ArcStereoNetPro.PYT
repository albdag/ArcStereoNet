"""
ArcStereoNet Pro is a Python toolbox for ArcGIS Pro. Its main features include:
 - Generate stereographic projections and rose diagrams within ArcGIS Pro.
 - Analyze data with contouring, clustering and mean vector statistics.
 - Link plots with their corresponding geographic position in map.

Author: @albdag
Dependencies: arcpy, numpy, matplotlib, mplstereonet
"""

import arcpy
import importlib.util
import io
import re
import shlex
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections.polar import PolarAxes
# matplotlib.use('TkAgg') use for interactive plots with plt.show()
import matplotlib.patheffects as mpe
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Iterable, Literal, TypedDict
from os import path, startfile
from datetime import datetime

#-----------------------------Environment Settings----------------------------#
# Prevent adding output image to Contents pane
arcpy.env.addOutputsToMap = False                                          

#------------------------------Global Constants-------------------------------#
# Mean vector analysis algorithms
MeanVectorAlgorithm = Literal["MEAD", "MEAD+Fisher", "K-means", "Bingham"]
MEAN_VECTOR_ALGORITHMS = ['MEAD', 'MEAD+Fisher', 'K-means', 'Bingham']

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

            
def mead(*args, vectors: bool = True, **kwargs: dict[str, Any]):
    # ATM only use it for StereoPlots, in future extend to RoseDiagrams
    # if len(args) == 1:
    #     args = (args[0], None)

    # Parse args
    args = (a.tolist() for a in args)
    
    # Parse kwargs
    # mead_kwargs = { # future improvement
    #     'wgts': self.kwargs.get('wgts', None),
    #     'measurement': self.kwargs.get('measurement', 'poles'),
    #     'n_clusts': self.kwargs.get('n_clusts', None),
    #     'azm_tol': self.kwargs.get('azm_tol', 0.5),
    #     'dip_tol': self.kwargs.get('inc_tol', 0.3)
    # }
    mead_kwargs = {
        'measurement': kwargs.get('measurement', 'poles'),
        'tol_str': kwargs.get('azm_tol', 0.5),
        'tol_dip': kwargs.get('inc_tol', 0.3),
        'n_clusts': kwargs.get('n_clusts', None),
    }

    clusters = angles_clustering(*args, **mead_kwargs)
    # clusters, _ = angles_clustering_new(a1, a2, **mead_kwargs) # future
    mean_vect = []
    r_means = []
    for c in clusters:
        azm, inc = zip(*c)
        avg_azm, r_mean = avg_angle(azm)
        # avg_inc = np.nan if a2 is None else np.mean(inc)
        avg_inc = np.mean(inc)
        mean_vect.append((avg_azm, avg_inc))
        r_means.append(r_mean)

    if vectors:
        return clusters, mean_vect, r_means
    else:
        return clusters




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
    algorithm: MeanVectorAlgorithm
    # Mean azimuth and inclination values per cluster id
    vectors: dict[int, tuple[float, float]]
    # Mean resultant lengths per cluster id
    r_means: dict[int, float]
    # Fisher statistics (R value, confidence radius, kappa) per cluster id
    fisher_stats: dict[int, tuple[float, float, float]]
    # MEAD azimuth and inclination tolerance values
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
        'azimuths',
        'bearing',
        'bearings',
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
    INCLINATION_HINTS = (
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
    

    def tempFilepath(self, file_name: str) -> str:
        """Generate a temporary file path in the ArcGIS scratch folder.

        Args:
            file_name (str): Name of the temporary file, including extension.

        Returns:
            str: Full temporary file path.
        """

        scratch_folder = arcpy.env.scratchFolder
        temp_path = path.join(scratch_folder, file_name)
        return temp_path


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

    
    def getUniqueLabel(
        self,
        label: str,
        existent: Iterable[str] | None = None
    ) -> str:
        
        """
        Generate a unique label by adding a progressive numerical suffix to
        'label' if it already exists in 'existent'. If 'existent' is None, the
        original 'label' is returned unchanged.

        Args:
            label (str): Original label.
            existent (Iterable[str] | None, optional): List of existent labels.
                Defaults to None.

        Returns:
            str: Unique label
        """

        # Return original label if there are no existing labels to check
        if existent is None:
            return label
        
        # Add progressive suffix to label
        suffix = 0
        unique_label = label
        while unique_label in existent:
            suffix += 1
            unique_label = f'{label}({suffix})' 
        return unique_label


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
            filtered = data[data[fields[1]] == _type]
            azimuth = filtered[fields[0]] % 360
            weights = filtered[fields[2]] if Weighted.value else None

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

            # Randomize color if required
            if _color == 'Random':
                _color = np.random.choice(list(COLORS.keys()))

            # Ensure tracking all plotted instances of _type
            type_lbl = self.getUniqueLabel(_type, statistics.keys())

            # Add histogram data to plot
            ax.bar(
                x=np.radians(np.arange(0, 360, 10)),
                height=hist,
                width=np.radians(10),
                bottom=0.0,
                align='center',
                color=COLORS.get(_color),
                edgecolor='w' if _color == 'Black' else 'k', # TODO make outline lightgray when color is black
                label=type_lbl,
                linewidth=0.5
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
                            edgecolor='w' if _color == 'Black' else 'k',
                            linewidth=1
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
            hdl, lbl = ax.get_legend_handles_labels()
            by_lbl = dict(sorted(zip(lbl, hdl)))
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
            image_outpath = self.tempFilepath('RoseDiagram_temp.png')

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
            clon, clat = np.nanmean(coords['Shape'], axis=0)
            descr = arcpy.Describe(Input_Feature.valueAsText)
            sr: arcpy.SpatialReference = descr.spatialReference
            with open(xy_path, 'w') as xyfile:
                xyfile.write(f'{clon}\n')
                xyfile.write(f'{clat}\n')
                xyfile.write(f'{sr.factoryCode}\n')

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

    def __init__(self) -> None:

        """Define the tool (tool name is the name of the class)."""

        super().__init__()
        self.label = 'Stereoplots'
        self.description = 'Add description.'


    def splitLineAndPlaneData(self, in_data: NDArray) -> list[NDArray]:
        
        """Split linear data (trends / plunges) from planar data (strikes /
        dips) that populate the same input 'in_data' structured array.

        Args:
            in_data (NDArray): Input structured array containing oriented data.
            It must exactly contain 4 fields (tuple of 4 elements per row),
            including the following information (in order): azimuth (float),
            inclination (float), sampling method (str), data type (str).

        Raises:
            TypeError: 'in_data' must be a structured array with four fields.

        Returns:
            list[NDArray]: Output split planar and linear data, divided in
            strikes, dips, trends and plunges arrays.
        """
        
        # Enforce requirement of a strucured array with exactly four fields
        if (fields := in_data.dtype.names) is None or len(fields) != 4:
            raise TypeError('A structured array with 4 fields is required')
        
        # Split linear from planar data
        azimuth_field, inclination_field, method_field, _ = fields
        planar_data = in_data[in_data[method_field] == 'RHR']
        linear_data = in_data[in_data[method_field] == 'TP']
        
        # Extract strikes / dips from planes and trends / plunges from lines
        strikes = planar_data[azimuth_field] % 360
        dips = planar_data[inclination_field]
        trends = linear_data[azimuth_field] % 360
        plunges = linear_data[inclination_field]

        return [strikes, dips, trends, plunges]
    

    def getMajorityDataKind(
        self,
        strike: ArrayLike,
        trend: ArrayLike
    ) -> tuple[Literal["planes", "lines"], bool]:
        
        """Return the majority data kind amongst planar and linear features, by
        analyzing the amount of 'strikes' and 'trends' records. Tipically, one
        of the two should contain 0 records; if this is not the case, this 
        function will send a warning.

        Args:
            strike (ArrayLike): Strike data.
            trend (ArrayLike): Trend data.

        Returns:
            majority, warning (tuple[Literal["planes", "lines"], bool]): The 
            majority data kind; 'warning' is True if minority data kind is > 0.
        """
        
        # Determine majority data kind (planar or linear)
        strike, trend = np.asarray(strike), np.asarray(trend)
        n_planes, n_lines = len(strike), len(trend)
        majority = 'planes' if n_planes > n_lines else 'lines'

        # Report if both linear and planar data are found for the same type
        warning = min(n_planes, n_lines) > 0

        return majority, warning
    

    def hideFirstContourLevel(self, contour: QuadContourSet) -> None:

        """Makes the first (lowest) filled region of a 'contour' QuadContourSet
        fully transparent while preserving automatic levels.

        Args:
            contour (QuadContourSet): Filled contour.

        Raises:
            TypeError: 'contour' must be a filled QuadContourSet.

        """

        # Enforce requirement of a filled contour
        if not contour.filled:
            raise TypeError('Required a filled QuadContourSet.')
        
        # Generate colors from original colormap, but set first one transparent 
        levels = list(contour.levels)
        colors = np.zeros((len(levels)-1, 4))
        colors[0] = [1, 1, 1, 0]
        colors[1:] = contour.get_cmap()(np.linspace(0, 1, len(levels)-2))

        # Assign new colormap and normalization to contour
        contour.set_cmap(ListedColormap(colors))
        contour.set_norm(BoundaryNorm(levels, len(colors)))
        
        # Fix transparency of first level being ignored by PathCollection
        contour.collections[0].set_alpha(0)


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
        
        # INCLINATION FIELD (param2) --------------------------------------------------
        param2 = arcpy.Parameter(
            name='inclination_field',
            displayName='Inclination Field',
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
        param19.filters[1].list = MEAN_VECTOR_ALGORITHMS
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
        Inclination_Field = parameters[2] # Field
        Method_Field = parameters[3] # Field
        Type_Field = parameters[4] # Field
        Plot_Data = parameters[5] # ValueTable
        Save_Image = parameters[6] # GPBoolean
        Image_Output = parameters[7] # DEFile
        Contour_Data = parameters[17] # ValueTable
        Show_Colorbar = parameters[18] # GPBoolean
        Mean_Vector_Data = parameters[19] # ValueTable
        Track_Mead = parameters[20] # GPBoolean

        # Attempt to fill Azimuth, Inclination, Method and Type fields params
        # with matching fields from the newly selected Input Feature. Clear
        # them if the Input Feature parameter is empty
        if Input_Feature.value is None:
            Azimuth_Field.value = None
            Inclination_Field.value = None
            Method_Field.value = None
            Type_Field.value = None

        elif self.parameterChanged(Input_Feature):
            describe = arcpy.Describe(Input_Feature.value)
            fields: list[arcpy.Field] = describe.fields
            numf = [f for f in fields if f.type in NUM_FIELD_TYPES]
            strf = [f for f in fields if f.type == 'String']
            self.inferFieldParameter(Azimuth_Field, numf, self.AZIMUTH_HINTS)
            self.inferFieldParameter(Inclination_Field, numf, self.INCLINATION_HINTS)
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

        return


    def isLicensed(self) -> bool:

        """Set whether tool is licensed to execute.

        Returns:
            bool: Whether tool is licensed.
        """

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
        
        # Initialize statistics dictionary
        statistics: dict[str, StatItem] = {}

        # Get parameters
        Input_Feature = parameters[0]
        Azimuth_Field = parameters[1]
        Inclination_Field = parameters[2]
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

        # PLOT INITIALIZATION -------------------------------------------------
        # Create figure with stereonet ax
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
        ax.grid(alpha=0.4 if Show_Grid.value else 0)
        ax.set_azimuth_ticks(np.linspace(0, 360, Num_Ticks.value, False))
        if (n := Num_Ticks.value) > 0:
            ticklabels = COMPASS_LABELS[Ticks_Format.valueAsText]
            ax.set_azimuth_ticklabels(ticklabels[::8//n])

        # DATA PLOTTING -------------------------------------------------------
        # Parse oriented data
        fields = [
            f.valueAsText for f in (
                Azimuth_Field, Inclination_Field, Method_Field, Type_Field
            )
        ]

        data = arcpy.da.FeatureClassToNumPyArray(
            in_table=Input_Feature.valueAsText,
            field_names=fields,
            null_value=''
        )

        # Convert Dip-Direction (DD) data to Right-Hand Rule (RHR)
        dd_mask = data[fields[2]] == 'DD'
        data[fields[0]][dd_mask] -= 90 # module 360 is computed later
        data[fields[2]][dd_mask] = 'RHR'

        # Plot data
        plot_data = pd if (pd := Plot_Data.values) is not None else []
        for _type, _color, _marker, _msize, _line, _lwidth in plot_data:

            # Determine majority data kind (planar or linear)
            filtered = data[data[fields[3]] == _type]
            strike, dip, trend, plunge = self.splitLineAndPlaneData(filtered)
            kind, warn = self.getMajorityDataKind(strike, trend)

            # Report if both linear and planar data are found for the same type
            if warn:
                msg = f'Found both planar and linear data for type "{_type}".'
                messages.addWarningMessage(f'Plot data: {msg}')

            # Ensure tracking all plotted instances of _type
            type_lbl = self.getUniqueLabel(_type, statistics.keys()) 

            # Populate general statistics for _type data
            statistics[type_lbl] = {
                "n": len(strike if kind == 'planes' else trend),
                "kind": kind,
                "contour": [],
                "clustering": []
            }

            # Randomize color if required
            if _color == 'Random':
                _color = np.random.choice(list(COLORS.keys()))

            # Add markers (poles to planes or line vectors) to plot
            if _marker != 'None':
                kwargs = {
                    'color': COLORS.get(_color),
                    'marker': MARKERS.get(_marker),
                    'markeredgecolor': 'w' if _color == 'Black' else 'k', # TODO make outline lightgray when color is black
                    'markersize': _msize
                }
                if kind == 'planes':
                    ax.pole(strike, dip, label=f'{type_lbl} (Poles)', **kwargs)
                else:
                    ax.line(plunge, trend, label=type_lbl, **kwargs)
            
            # Add cyclographic traces to plot
            if _line != 'NONE':
                ax.plane(
                    strike=strike,
                    dip=dip,
                    color=COLORS.get(_color),
                    label=type_lbl,
                    linestyle=LINES.get(_line),
                    linewidth=_lwidth,
                    path_effects=[
                        mpe.withStroke(
                            linewidth=_lwidth * 2,
                            foreground='w' if _color == 'Black' else 'k'
                        )
                    ]
                )

        # Plot contour data
        contour_data = cd if (cd := Contour_Data.values) is not None else []
        n_contours = 0
        for _type, _method, _sigma, _style, _cmap, _alpha in contour_data:
            n_contours += 1

            # Determine majority data kind (planar or linear)
            filtered = data[data[fields[3]] == _type]
            strike, dip, trend, plunge = self.splitLineAndPlaneData(filtered)
            kind, warn = self.getMajorityDataKind(strike, trend)

            # Report if both linear and planar data are found for the same type
            if warn:
                msg = f'Found both planar and linear data for type "{_type}".'
                messages.addWarningMessage(f'Contour data: {msg}')

            # Add contour statistics for _type data
            contour_stats = (_method, _sigma)
            if _type in statistics.keys():
                statistics[_type]['contour'].append(contour_stats)
            else:
                statistics[_type] = {
                    "n": len(strike if kind == 'planes' else trend),
                    "kind": kind,
                    "contour": [(_method, _sigma)],
                    "clustering": []
                }

            # Randomize colormap if required
            if _cmap == 'Random':
                _cmap = np.random.choice(list(CONTOUR_COLORMAPS.keys()))
            cmap = CONTOUR_COLORMAPS.get(_cmap)

            # Define contour args and kwargs
            args = (strike, dip) if kind == 'planes' else (plunge, trend)
            kwargs = {
                'measurement': 'poles' if kind == 'planes' else 'lines',
                'method': CONTOUR_METHODS.get(_method),
                'sigma': _sigma,
                'alpha': 1 - (_alpha / 100),
                'linewidths': 1,
                'zorder': 1 * n_contours
            }

            # Add filled or unfilled countour to plot
            if _style == 'Filled':
                contour = ax.density_contourf(*args, cmap=cmap, **kwargs)
                self.hideFirstContourLevel(contour)
                ax.density_contour(*args, colors='gray', **kwargs) # outline
                    
            else:
                contour = ax.density_contour(*args, cmap=cmap, **kwargs)

            # Add colorbar to plot if required
            if Show_Colorbar.value:
                x, y = -0.65,  1.2 - 0.2 * n_contours
                cbax = ax.inset_axes((x, y, 0.5, 0.02))
                cbax.tick_params(labelsize=7)
                fig.colorbar(
                    mappable=contour,
                    cax=cbax,
                    orientation='horizontal',
                    ticklocation='top'
                )
                info = f'Contour on: {_type}\nMethod: {_method}\nσ: {_sigma}'
                ax.text(x, y - 0.1, info, fontsize=8, transform=ax.transAxes)

        # Plot mean vector data
        cones_proxy: dict[str, Line2D] = {} # only used for MEAD+Fisher
        mv_data = mvd if (mvd := Mean_Vector_Data.values) is not None else []
        for (_type, _algorithm, _nclusts, _azm_tol, _inc_tol, _fisher_conf,
             _color, _marker, _msize, _line, _lwidth) in mv_data:
            
            # Determine majority data kind (planar or linear)
            filtered = data[data[fields[3]] == _type]
            strike, dip, trend, plunge = self.splitLineAndPlaneData(filtered)
            kind, warn = self.getMajorityDataKind(strike, trend)

            # Report if both linear and planar data are found for the same type
            if warn:
                msg = f'Found both planar and linear data for type "{_type}".'
                messages.addWarningMessage(f'Mean vector data: {msg}')

            # Initialize mean vector statistics
            mean_vector_stats: MeanVectorStats = {
                "algorithm": _algorithm,
                "vectors": {},
                "r_means": {},
                "fisher_stats": {},
                "tol": None,
                "conf": None
            }

            # Define type of data measurement
            if kind == 'planes':
                args, measurement = (strike, dip), 'poles'
            else:
                args, measurement = (plunge, trend), 'lines'
            
            # Find mean vector(s) and update mean vector statistics
            match _algorithm:
                case 'Bingham':
                    vect = msn.fit_girdle(*args, measurement=measurement)
                    mean_vector_stats["vectors"] = {0: vect}

                case 'K-means':
                    kwargs = {'num': _nclusts, 'measurement': measurement}
                    while True:
                        try: # TEST bidirectional=False
                            centers = np.asarray(msn.kmeans(*args, **kwargs))
                            break
                        except np.linalg.LinAlgError:
                            kwargs["num"] -= 1

                    if measurement == 'poles':
                        c1, c2 = msn.geographic2pole(*centers.T)
                        vect = [(c1[i], c2[i]) for i in range(len(centers))]
                    else:
                        c1, c2 = msn.geographic2plunge_bearing(*centers.T)
                        vect = [(c2[i], c1[i]) for i in range(len(centers))]
                    
                    mean_vector_stats["vectors"] = dict(enumerate(vect))

                case 'MEAD':
                    kwargs = {
                        'measurement': measurement,
                        'n_clusts': _nclusts,
                        'azm_tol': _azm_tol / 100,
                        'inc_tol': _inc_tol / 100
                    }
                    mead_clusters, vect, _ = mead(*args, **kwargs)

                    mean_vector_stats["vectors"] = dict(enumerate(vect))
                    mean_vector_stats["tol"] = (_azm_tol, _inc_tol)

                case 'MEAD+Fisher':
                    mead_kwargs = {
                        'measurement': measurement,
                        'n_clusts': _nclusts,
                        'azm_tol': _azm_tol / 100,
                        'inc_tol': _inc_tol / 100
                    }
                    fisher_kwargs = {
                        'measurement': measurement,
                        'conf': _fisher_conf
                    }
                    
                    vect = {}
                    fisher_stats: dict[int, tuple[float, float, float]] = {}
                    mead_clusters = mead(*args, vectors=False, **mead_kwargs)
                    for i, cluster in enumerate(mead_clusters):
                        azm, inc = zip(*cluster)
                        if measurement == 'poles':
                            pb_vect, stats = msn.find_fisher_stats(
                                azm, inc, **fisher_kwargs
                            )
                            sd_vect = msn.plunge_bearing2pole(*pb_vect)
                            vect[i] = (*sd_vect[0], *sd_vect[1])
                        else:
                            pb_vect, stats = msn.find_fisher_stats(
                                inc, azm, **fisher_kwargs
                            )
                            vect[i] = tuple(reversed(pb_vect))

                        if stats is None:
                            stats = (np.nan, np.nan, np.nan)
                        r_value, angle, kappa = (
                            np.nan if s is None else float(s) for s in stats
                        )
                        fisher_stats[i] = (r_value, angle, kappa)

                    mean_vector_stats["vectors"] = vect
                    mean_vector_stats["fisher_stats"] = fisher_stats
                    mean_vector_stats["tol"] = (_azm_tol, _inc_tol)
                    mean_vector_stats["conf"] = _fisher_conf

                case _: # technically impossible
                    msg = f'Algorithm "{_algorithm}" not recognized.'
                    messages.addErrorMessage(msg)
                    return
                
            # Add clustering statistics to _type data
            if _type in statistics.keys():
                statistics[_type]['clustering'].append(mean_vector_stats)
            else:
                statistics[_type] = {
                    "n": len(strike if kind == 'planes' else trend),
                    "kind": kind,
                    "contour": [],
                    "clustering": [mean_vector_stats]
                }

            # Randomize color if required
            if _color == 'Random':
                _color = np.random.choice(list(COLORS.keys()))

            # Add track MEAD clustering to plot if required
            if Track_Mead.value and 'MEAD' in _algorithm:
                kwargs = {
                    'color': 'k',
                    'markeredgecolor': COLORS.get(_color),
                    'markersize': _msize * 1.5
                }
                for i, cluster in enumerate(mead_clusters, start=1):           # type: ignore[reportPossiblyUnboundVariable]
                    azm, inc = zip(*cluster)
                    if measurement == 'poles':
                        ax.pole(azm, inc, marker=f'${i}$', **kwargs)
                    else:
                        ax.line(inc, azm, marker=f'${i}$', **kwargs)

            # Add poles, line vectors and cyclographic traces to plot
            avg_azm, avg_inc = zip(*mean_vector_stats["vectors"].values())
            suffix = 'best fit' if _algorithm == 'Bingham' else 'mean'
            label_root = f'{_type}\n[{_algorithm}] {suffix}'
            
            if _marker != 'NONE':
                kwargs = {
                    'color': COLORS.get(_color),
                    'marker': MARKERS.get(_marker),
                    'markeredgecolor': 'w' if _color == 'Black' else 'k', #TODO use lightgray instead of w
                    'markersize': _msize
                }
                if measurement == 'poles' or _algorithm == 'Bingham': 
                    lbl = f'{label_root} pole(s)'
                    ax.pole(avg_azm, avg_inc, label=lbl, **kwargs)
                else:
                    lbl = f'{label_root} vector(s)'
                    ax.line(avg_inc, avg_azm, label=lbl, **kwargs)

            if (
                _line != 'NONE' and (
                    measurement == 'poles' or _algorithm == 'Bingham'
                )
            ):
                ax.plane(
                    strike=avg_azm,
                    dip=avg_inc,
                    color=COLORS.get(_color),
                    label=f'{label_root} plane(s)',
                    linestyle=LINES.get(_line),
                    linewidth=_lwidth,
                    path_effects=[
                        mpe.withStroke(
                            linewidth=_lwidth * 2,
                            foreground='w' if _color == 'Black' else 'k'
                        )
                    ]
                )

            # Add fisher cones to plot
            if _algorithm == 'MEAD+Fisher':
                angles = [v[1] for v in mean_vector_stats["fisher_stats"].values()]
                if measurement == 'poles':
                    plng, bear = msn.pole2plunge_bearing(avg_azm, avg_inc)
                else:
                    plng, bear = avg_inc, avg_azm
                
                ax.cone(
                    plunge=plng,
                    bearing=bear,
                    angle=angles,
                    facecolor='none',
                    edgecolor=COLORS.get(_color),
                    linestyle='-',
                    linewidth=_lwidth
                )
                # Fix fisher cones not showing up in legend
                cones_label = f'{_type}\n[MEAD+Fisher] confidence cone'
                cones_proxy[cones_label] = Line2D(
                    xdata=(0, ), 
                    ydata=(0, ),
                    linestyle='None',
                    marker="o",
                    markeredgecolor=COLORS.get(_color),
                    markeredgewidth=_lwidth,
                    markerfacecolor='None',
                    markersize=10
                )

        # PLOT FINALIZATION --------------------------------------------------- 
        # Set optional legend
        if Show_Legend.value:
            hdl, lbl = ax.get_legend_handles_labels()
            hdl.extend(cones_proxy.values())
            lbl.extend(cones_proxy.keys())
            by_lbl = dict(sorted(zip(lbl, hdl), key=lambda x: x[0]))
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

        # Set optional "Number of Samples" label
        n_samp = StatsLogger(statistics, "StereoPlots").samples(parag=False)
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
            image_outpath = self.tempFilepath('StereoPlot_temp.png')

        root = path.splitext(image_outpath)[0]
        log_outpath = f'{root}_LOG.txt'
        xy_path = f'{root}_xy.txt'

        # Save figure
        fig.savefig(image_outpath, dpi='figure', bbox_inches='tight')

        # Write log if required
        if Write_Log.value:
            logger = StatsLogger(statistics, "StereoPlots")
            with open(log_outpath, 'wt', encoding='utf-8') as log:
                # Title and number of samples
                log.write(logger.paragraph(title.upper()))
                log.write(logger.paragraph(n_samp))
                # Contour statistics
                log.write(logger.contour())
                # Clustering statistics
                log.write(logger.clustering())
                # Extra info (tool name, datetime)
                log.write(logger.extraInfo(parag=False))

        # Save centroid coordinates if image is not temporary     
        if Save_Image.value:
            coords = arcpy.da.FeatureClassToNumPyArray(
                in_table=Input_Feature.valueAsText,
                field_names='Shape'
            )
            clon, clat = np.nanmean(coords['Shape'], axis=0)
            descr = arcpy.Describe(Input_Feature.valueAsText)
            sr: arcpy.SpatialReference = descr.spatialReference
            with open(xy_path, 'w') as xyfile:
                xyfile.write(f'{clon}\n')
                xyfile.write(f'{clat}\n')
                xyfile.write(f'{sr.factoryCode}\n')

        # Auto-show image (and optionally log file) if image is temporary
        else: # add a dedicated boolean parameter to enable auto-show?
            startfile(image_outpath)
            if Write_Log.value:
                startfile(log_outpath)

        return



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
                    Input_Plots.setErrorMessage(err)
                    break

        return


    def isLicensed(self):

        """Set whether tool is licensed to execute.

        Returns:
            bool: Whether tool is licensed.
        """

        return True


    def execute(self, parameters: list[arcpy.Parameter], messages) -> None:

        """The source code of the tool.

        Args:
            parameters (list[arcpy.Parameter]): Tool's parameters.
            messages: Tool's messages.
        """

        # Get Parameters
        Input_Plots = parameters[0]
        Output_Feature = parameters[1]

        # Initialize output feature class rows and input spatial references
        # 'fc_rows' scheme: [( (lon, lat), plot image filepath ), ...]
        # 'sr_wkid' scheme: {WKID, ...}
        fc_rows: list[tuple[tuple[float, float], str]] = []
        sr_wkid: set[int] = set()

        # Populate 'fc_rows' and 'sr' with data from input plots
        for plot in Input_Plots.values:
            xy_file = f'{path.splitext(str(plot))[0]}_xy.txt'
            with io.open(xy_file, 'r') as temp_xy:
                lon, lat, wkid = temp_xy.readlines()                
                fc_rows.append(((float(lon), float(lat)), str(plot)))
                sr_wkid.add(int(wkid))

        # Set WKID to None if multiple WKIDs are found, and send warning
        wkid = None if len(sr_wkid) > 1 else sr_wkid.pop()
        if wkid is None:
            messages.addWarningMessage('Found conflicting SR: set to Unknown!')
        
        # Create and populate output feature class
        arcpy.CreateFeatureclass_management(
            *path.split(Output_Feature.valueAsText),
            geometry_type='POINT',
            spatial_reference=arcpy.SpatialReference(wkid)
        )
        arcpy.AddField_management(Output_Feature.valueAsText, "Plot", "TEXT")
        fields = ["SHAPE@XY", "Plot"]
        with arcpy.da.InsertCursor(Output_Feature.valueAsText, fields) as curs:
            for row in fc_rows:
                curs.insertRow(row)

        return