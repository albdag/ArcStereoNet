# ArcStereoNet

<p align="center">
  <img width="512" src="https://github.com/user-attachments/assets/284911d5-39ef-4434-a46d-d7f7311f6461">
</p>

ArcStereoNet is a Python toolbox for generating high quality *rose diagrams* and *stereographic projections* within **ArcGIS®**. With ArcStereoNet you can quickly analyse your oriented data, while taking advantage of the powerful geospatial tools provided by ArcGIS®. The main benefit of using this toolbox is being able to <ins>analyse your data within the same software environment</ins>, minding the gap between spatial orientation of data and its georeferenced location.

It has been developed mainly for providing support to structural geology and mineralogy analysis. Oriented data can be acquired on field (e.g., beddings, fault lines etc.) but also extracted from thin section high-resolution images (e.g., mineral preferred orientations). However, ArcStereoNet is potentially suitable to work with any type of oriented data (e.g., wind directions).

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


# User Guide
ArcStereoNet includes: 
 * **Stereoplots** tool - for stereographic projections and oriented data analysis.
 * **Rose Diagrams** tool - for rose histograms creation.
 * **Graph To Hyperlink** tool - to link plots with the corresponding geographic position in map.

The *Stereoplot* tool is useful to produce lower hemisphere stereographic projections, showing cyclographic traces and poles for the selected planar measurements and points for the linear ones. Several contouring and statistical functions can also be applied to oriented data. 
The *Rose Diagrams* tool can be used to create weighted and unweighted rose diagrams and to extract mean vectors directions. 
Finally, the *Graph To Hyperlink* tool allows to link realised graphs to the georeferenced position of plotted data.

This [quick guide](https://github.com/albdag/ArcStereoNet/wiki) shows you how to use them by providing information about their required and optional parameters.
