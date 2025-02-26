# ArcStereoNet
ArcStereoNet is a Python toolbox for generating high quality *rose diagrams* and *stereographic projections* within **ArcGIS®**. With ArcStereoNet you can quickly analyse your oriented data, while taking advantage of the powerful geospatial tools provided by ArcGIS®. The main benefit of using this toolbox is being able to <ins>analyse your data within the same software environment</ins>, minding the gap between spatial orientation of data and its georeferenced location.

It has been developed mainly for providing support to structural geology and mineralogy analysis. Oriented data can be acquired on field (e.g., beddings, fault lines etc.) but also extracted from thin section high-resolution images (e.g., mineral preferred orientations). However, ASN is potentially suitable to work with any type of oriented data (e.g., wind directions).

<img src="https://github.com/user-attachments/assets/2e78d961-8f7d-4d7d-a5bd-7cc167850c8b" height="256">
<img src="https://github.com/user-attachments/assets/d16a1501-f3ee-4ade-847c-6193435fd504" height="256">


This repository includes:
 * An updated version of the [original *ArcStereoNet*](https://doi.org/10.3390/ijgi10020050) (compatible with ArcGIS Desktop - ArcMap), initially published in the International Journal of Geo-Information (MDPI) under the CC BY license ([https://www.mdpi.com/openaccess](https://www.mdpi.com/openaccess)).
 * *ArcStereoNet Pro*, a new version of the toolbox compatible with ArcGIS Pro.


# Citing This Work
If you use this software in your research, please cite the original paper:

**"ArcStereoNet: A New ArcGIS® Toolbox for Projection and Analysis of Meso- and Micro-Structural Data"**, by
Ortolano, G., D’Agostino, A., Pagano, M., Visalli, R., Zucali, M., Fazio, E., Alsop, I. and Cirrincione, R.
ISPRS International Journal of Geo-Information, 2021, 10(2), 50.
DOI: [10.3390/ijgi10020050](https://doi.org/10.3390/ijgi10020050).

Export citation file: [BibTeX](Cite/ijgi-v10-i02_20250225.bib) | [EndNote](Cite/ijgi-v10-i02_20250225.enw) | [RIS](Cite/ijgi-v10-i02_20250225.ris)


# Installation for ArcGIS Desktop (ArcMap) users
 1. Download *ArcStereoNet* toolbox.
 2. Copy the toolbox file "ArcStereoNet.pyt" inside ArcGIS® toolboxes folder. By default, the full path is:\
    `"C:/Program Files (x86)/ArcGis/Desktop10.x/ArcToolbox/Toolboxes"`.
 3. Open the ArcToolbox window inside ArcMap®, right-click and then select "Add Toolbox" option to import ArcStereoNet.
 4. The installation will automatically start; this requires an internet connection. A pop-up window will inform the user whether the components have been successfully installed or not.

>[!Note]
> - ArcGIS Desktop versions 10.4+ are fully compatible.
> - Versions 10.3 and 10.3.1 require a manual installation of *`pip`* [https://pip.pypa.io/en/stable/](https://pip.pypa.io/en/stable/) in the dedicated ArcGIS Python instance directory.
> - Versions < 10.3 are incompatible.


# Installation for ArcGIS Pro

Unfortunately, due to the internal changes of Python Environments in ArcGIS Pro, the *ArcStereoNet Pro* version cannot simply be dragged and dropped for it to be installed. The installation process requires a few steps, which are listed below.

>[!Note]
>With these steps you will create a new Python environment within ArcGis Pro, where the toolbox dependencies will be installed. This is in order to avoid possible conflicts with other Python modules that could be installed in your default environment. However, if you are familiar with Python environments in ArcGis Pro, you can also try installing the dependencies within your preferred environment.

>[!Warning]
>You may require admin privileges to install ArcStereoNet Pro on your machine.

 1. Download *ArcStereoNet Pro* toolbox.
 2. Follow the [official ESRI guide](https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/clone-an-environment.htm) to clone the default Python environment inside ArcGIS Pro.
 3. Close ArcGIS Pro.
 4. Open the **ArcGIS Pro Python Command Prompt** by navigating to Start Menu > All Programs > ArcGIS > Python Command Prompt.
 5. Activate your cloned environment by typing `proswap arcgispro-py3-clone` or `proswap your_custom_name` if you have renamed the cloned environment.
 6. Install dependencies on the environment by typing `pip install mplstereonet --no-deps`
 7. Open ArcGIS Pro and add the "ArcStereoNetPro.pyt" file directly inside a new Map project folder, by right-clicking on Toolboxes inside the Catalog panel.

>[!Warning]
>Whenever you want to use ArcStereoNet Pro, you need to make sure that the Python environment where its dependencies are installed is the active environment. 


# User Guide
ArcStereoNet includes: 
 * **Stereoplots** tool - for stereographic projections and oriented data analysis.
 * **Rose Diagrams** tool - for rose histograms creation.
 * **Graph To Hyperlink** tool - to link plots with the corresponding geographic position in map.

The *Stereoplot* tool is useful to produce lower hemisphere stereographic projections, showing cyclographic traces and poles for the selected planar measurements and points for the linear ones. Several contouring and statistical functions can also be applied to oriented data. 
The *Rose Diagrams* tool can be used to create weighted and unweighted rose diagrams and to extract mean vectors directions. 
Finally, the *Graph To Hyperlink* tool allows to link realised graphs to the georeferenced position of plotted data.

This quick guide shows you how to use them by providing information about their required and optional parameters.

## 1. Preparing Your Data
ArcStereoNet's tools require as input the following information:
 - Azimuth angle (both *Stereoplots* tool and *Rose Diagrams* tool)
 - Dip angle (only *Stereoplots* tool)
 - Sampling method (only *Stereoplots tool*)
 - Feature type (both *Stereoplots* tool and *Rose Diagrams* tool)

These must be stored in different fields within the attribute table of your **input shapefile**. The tools can automatically fill the required inputs if such fields are renamed as follows (not case sensitive):
 * *Azimuth* – here azimuthal values (i.e., direction, dip direction or trend) must be stored as numeric values.
 * *Dip Angle* or *Dip_Angle* – here inclination values (i.e., dip or plunge) must be stored as numeric values.
 * *Method* – here data format must be specified as text values, choosing from "RHR", "DD" and "TP" (must be written in uppercases), indicating, respectively, the following conventional sampling methods: *Right Hand Rule*, *Dip Direction/Dip* and *Trend-Plunge*.
 * *Type* – here the feature type should be specified as text values (e.g. "Main Foliation", "Fault", "Bedding", etc.). This information is not mandatory, though highly recommended. It is functional for the plot legend and guides the algorithm to a correct grouping and graphical representation of the different types of data.

You may also populate with such information a tabular data file and then import it as a new vector layer in ArcGIS; each column will be parsed as a different field. If the fields are not renamed as suggested above, it is still possible to select manually the corresponding ones within the tools interface. Extra fields can be added to the attribute table according to your requirements and preferences.
> [!TIP]
> If you have an active selection of some features in your input data, the tools will only process those!

## 2. Stereoplots

The following screenshots shows the *Stereoplots* tool layout. A detailed description for each parameter is provided below. Green dots indicate required parameters. 
> [!NOTE]
> While this screenshot is taken from *ArcStereoNet* basic version, the same parameters are available for *ArcStereoNet Pro* version.

![image](https://github.com/user-attachments/assets/4e8e35c5-4ae6-4e39-b65d-eb26cf0dd22f)

<ins>**(A)** Oriented dataset input parameter.</ins>\
A shapefile is required: point, line and polygon feature types are supported.

<ins>**(B)** *Azimuth*, *Dip Angle*, *Method* and *Type* fields input parameters.</ins>\
If you have formatted the shapefile fields names as recommended ([Preparing Your Data](1-preparing-your-data)), these parameters will be automatically filled, otherwise they can be selected manually through the drop-down menus.

<ins>**(C)** Plotting data Value Table.</ins>\
The feature types can be added through the drop-down menu. For each feature type, the user can specify the plotting colour, pole or vector symbol and size, cyclographic trace style and width (Value Table columns).
> [!TIP]
> If a linear datatype is added, the corresponding cyclographic trace style and width entries will be ignored by the tool. A linear datatype is automatically identified by checking the *Method* field.

<ins>**(D)** Output image settings.</ins>\
By unchecking the *Store Image Output* checkbox, the user can prompt the tool to save a temporary output image file and automatically open it after the tool execution. Otherwise, the output image file path can be specified in the *Output Image* parameter box. The image DPI value can be set through the parameter below.

<ins>**(E)** Contour & Statistics sub-menu (collapsed).</ins>\
See [Contour & Statistics](2.1contour-&-statistics) for details.

<ins>**(F)** Plot customisation sub-menu (expanded).</ins>\
The following parameters can be set (from top to bottom): the title label that will appear in the plot (if not specified, it will be the name of the shapefile); the number of Tick Marks surrounding the plot and their type (i.e., angular values, cardinal points or a mix of them); whether to enable or disable the appearance of the stereonet grid, the legend and the number of plotted data.

<ins>**(G)** Plotting options sub-menu (expanded).</ins>\
The stereonet type can be chosen between equal-area (**Schmidt net**) and equal-angle (**Wulff net**). The *Write Log File* option can be checked to prompt the tool to compile a text file storing statistical information concerning the plotted data and its statistics. If the *Store Image Output* option **(D)** is not enabled, a temporary log file will be created and automatically opened together with the image. Otherwise, the log file will be saved in the same output directory selected for the image.


### 2.1 Contour & Statistics
The following screenshot shows the expanded *Contour & Statistics* sub-menu of Stereoplots tool.

![image](https://github.com/user-attachments/assets/db4cacd8-2899-4de0-aa5d-035deb7f57af)

<ins>**(A)** *Apply Contour* Value Table.</ins>\
The feature types can be added through the drop-down menu. You can choose the preferred contour density function under the "Method" column, as well as other parameters such as standard deviation, contour style ("filled" or "unfilled"), color and transparency through the corresponding columns.

> [!Tip]
> A warning message will pop-up if two or more contour functions are applied contemporaneously. However, it is still possible to prompt for multiple contour functions if you really want to. An "unfilled" contour style is recommended.

<ins>**(B)** Show a contour color.</ins>\
This setting will be enabled after the *Apply Contour* Value Table has been populated. If two or more contours are applied at the same time, the colour bar will be referred to the last one.

<ins>**(C)** *Extract Mean Vectors* Value Table.</ins>\
The feature types can be added through the drop-down menu. Under the "Algorithm" column the preferred algorithm can be chosen. The choices are: **MEAD**, **MEAD + Fisher**, **K-means**, **Bingham**.
> [!Tip]
> Multiple feature types can be analysed contemporaneously, and the same feature type can be analysed with different algorithms as well.

  - The **MEAD** algorithm can be used for a fully user-controlled clustering extraction, thanks to the azimuth and inclination tolerances parameters.
  - The **MEAD + Fisher** algorithm adds to the previous one the extraction of Fisher statistics (mean vector magnitude, confidence radius and dispersion factor – Fisher et al., 1993[^1]) that will be stored into the log file (if prompted by the user). A confidence cone surrounding the pole to extracted mean plane will also be plotted.
  - The K-means algorithm employs a k-means approach (MacQueen, 1967[^2]) modified for spherical data.
  - The Bingham algorithm allows to extract the best fit plane of a "girdle-like" distribution pattern (Bingham, 1974[^3]) of poles to planes or lines.

[^1]: Bingham, C. (1974). An antipodally symmetric distribution on the sphere. The Annals of Statistics, 1201-1225.
[^2]: Fisher, N. I., Lewis, T., & Embleton, B. J. (1993). Statistical analysis of spherical data. Cambridge university press
[^3]: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability (Vol. 1, No. 14, pp. 281-297).

The algorithm-control parameters can be set through the following columns (from left to right): "Number of Clusters", to specify the number of families to split data into; "MEAD Azimuth tolerance" and "MEAD Inclination tolerance", to constrain the algorithm to define more or less dispersed clusters; "Fisher confidence", that influences the Fisher confidence radius computed by MEAD + Fisher algorithm. Each of these control parameters affects only some of the algorithms as summarised in the following table.
  
![image](https://github.com/user-attachments/assets/869dd319-9a65-44dc-935a-5cc837974301)

If a not affecting parameter is specified, it will simply be skipped by the tool. Some columns within the Value Table gather graphic-related parameters such as plotting colour, pole or vector symbol and size, cyclographic trace style and width. For further information on the algorithms, please refer to the original scientific paper: [**"ArcStereoNet: A New ArcGIS® Toolbox for Projection and Analysis of Meso- and Micro-Structural Data"**](https://doi.org/10.3390/ijgi10020050) 

<ins>**(D)** Track M.E.A.D. clustering behaviour.</ins>\
This option will be enabled after the *Extract Mean Vectors* Value Table has been populated and will only apply on clusters extracted through the MEAD. clustering process (**MEAD** and **MEAD + Fisher** algorithms). If checked, the tool will plot the clustered data with different symbols for each extracted cluster. The users can take advantage of this option to identify the *MEAD Tolerance parameters* that best suit their preferences.
>[!Tip]
>It is likely that you will test the algorithms several times, modifying the control parameters repeatedly, until finding the ones that better suit your preferences and needs. During this phase (testing phase), it is highly recommend using the ArcGIS *Results* window to re-launch any previously used tool. In such a way you can quickly modify only the desired parameter without having to set all the others each time. We also recommend deselecting the *Store Image Output* option during the testing phase, in order to rapidly view the graphic result and save memory space on the disk.


## 3. Rose Diagrams
The following screenshot shows the *Rose Diagrams* tool layout. A detailed description for each parameter is provided below. Green dots indicate required parameters.
> [!NOTE]
> While this screenshot is taken from *ArcStereoNet* basic version, the same parameters are available for *ArcStereoNet Pro* version.

![image](https://github.com/user-attachments/assets/1f252494-316e-4417-a9b3-4271e888a56a)

<ins>**(A)** Oriented dataset input parameter.</ins>\
A shapefile is required: point, line and polygon feature types are supported.

<ins>**(B)** *Azimuth* and *Type* fields input parameters.</ins>\
If the user has formatted the shapefile fields names as recommended ([Preparing Your Data](1-preparing-your-data)), these parameters will be automatically filled, otherwise they can be selected manually through the drop-down menus.

<ins>**(C)** Plotting data Value Table.</ins>\
The feature types can be added through the drop-down menu. For each feature, the user can specify the bar colour and whether to show the mean vectors or not, with a determined number of clusters and azimuth tolerance (Value Table columns).

<ins>**(D)** Output image settings.</ins>\
By unchecking the *Store Image Output* checkbox, the user can prompt the tool to save a temporary output image file and automatically open it after the tool execution. Otherwise, the output image file path can be specified in the *Output Image* parameter box. The image DPI value can be set through the parameter below.

<ins>**(E)** Plot customisation sub-menu (expanded).</ins>\
The following parameters can be set (from top to bottom): the title label that will appear in the final plot (if not specified, it will be the name of the shapefile); whether to enable or disable the appearance of the polar grid, the legend, and the number of plotted data; the spacing (in degrees) of the tick marks surrounding the diagram; the orientation of the radial scale (in degrees from N); the grid spacing (i.e., spacing of the radial scale).

<ins>**(F)** Plotting options sub-menu (expanded).</ins>\
If the *Mirrored behaviour* option is checked a specular rose diagram will be generated. A warning message will popup if the azimuthal dataset spreads for more than 180°. If the *Weighted Rose Diagram* option is selected, the below *Weight Field* parameter enables and must be filled with a field of the input shapefile storing weight data. A warning message will appear if one or more weights are negative values. The *Write Log File* option can be checked to prompt the tool to compile a text file storing statistical information concerning the plotted data. If the *Store Image Output* option **(D)** is not enabled, a temporary log file will be created and automatically opened together with the image. Otherwise, the log file will be saved in the same output directory selected for the image.


## 4. Graph To Hyperlink
The following screenshot shows the *Graph To Hyperlink* tool layout. All parameters are here required.
> [!NOTE]
> While this screenshot is taken from *ArcStereoNet* basic version, the same parameters are available for *ArcStereoNet Pro* version.

![image](https://github.com/user-attachments/assets/9e43549a-e380-45a5-96c6-1a3e54f8e1e8)

 * The *Input Plot Images* parameter requires one or more raster images to be provided. The files can be selected through the drop-down menu. Such images are meant to be the plots resulting from the application of the *Stereoplots* and/or the *Rose Diagrams* tools.
>[!WARNING]
>Each time a new stereoplot or rose diagram is saved as a PNG image, a text file (named as “imageName_xy.txt”) is automatically compiled and stored in the same output directory. Such file contains the spatial information referred to the corresponding plot. Make sure to not delete or move these files, as they are required by the *Graph To Hyperlink* tool to link each image to the correct georeferenced position. If the user selects an image whose _xy.txt file is missing, an error message will popup.

 * The output file path can be chosen through the *Output Feature Class* parameter. A new point shapefile, gathering all the links to the selected images and their corresponding latitude and longitude values, will be created.
The spatial information attached to each plot is computed as the mean latitude and the mean longitude (centroid) of the processed data. The adopted reference system will be the same as the dataset from which the plots were extracted.

Once the *Graph To Hyperlink* tool has been executed and the new resulting shapefile imported within ArcGIS, you can enable an hyperlink or HTML popup view. This can be done by from the properties of the new shapefile. The following screenshot shows an example of HTML popups displaying georeferenced stereoplots on map.

![image](https://github.com/user-attachments/assets/66966d8b-d98b-4fa1-ad7c-5d13ed2f319a)
