# ArcStereoNet

<p align="center">
  <img width="512" src="https://github.com/user-attachments/assets/284911d5-39ef-4434-a46d-d7f7311f6461">
</p>

ArcStereoNet is an **ArcGIS® Python toolbox** that integrates the [*mplstereonet*](https://github.com/joferkington/mplstereonet) library within the ArcGIS® environment, allowing users to produce high quality *rose diagrams* and *stereographic projections* and minding the gap between spatial orientation of data and its geographic location.

ArcStereoNet tools promote user accessibility thanks to their dynamic interface, providing extended support to structural geology and mineralogy studies. Typical applications include the analysis of **geo-structural field data** ([example](https://doi.org/10.1080/17445647.2022.2057876)) or the identification of **preferred mineral orientations** from rock thin section data ([example](https://doi.org/10.3390/ijgi10020051)). However, ArcStereoNet can easily process any other type of oriented datasets.

This repository includes:

* An updated version of the [original *ArcStereoNet*](https://doi.org/10.3390/ijgi10020050) (compatible with ArcGIS Desktop - ArcMap), published in the *International Journal of Geo-Information* under the CC BY license ([https://www.mdpi.com/openaccess](https://www.mdpi.com/openaccess)).
* *ArcStereoNet Pro*, a more recent and enhanced version of the toolbox, compatible with ArcGIS Pro.

# Citing This Work

If you use this software as part of your research, please cite the original paper:

Ortolano, G., D’Agostino, A., Pagano, M., Visalli, R., Zucali, M., Fazio, E., Alsop, I., & Cirrincione, R. (2021). ArcStereoNet: A New ArcGIS® Toolbox for Projection and Analysis of Meso- and Micro-Structural Data. *ISPRS International Journal of Geo-Information*, *10(2)*, *50*. [https://doi.org/10.3390/ijgi1002005](https://doi.org/10.3390/ijgi1002005)

Export citation file: [BibTeX](Cite/ijgi-v10-i02_20250225.bib) | [EndNote](Cite/ijgi-v10-i02_20250225.enw) | [RIS](Cite/ijgi-v10-i02_20250225.ris)

# User Guide

ArcStereoNet includes three tools:

* **Stereoplots**, which allows the generation of lower hemisphere stereographic projections and includes a collection of statistical algorithms for oriented data analysis.
* **Rose Diagrams**, useful to create weighted and unweighted rose diagrams and to extract mean vectors directions.
* **Graph To Hyperlink**, that allows linking the generated plots to the geographic location of plotted data.

The new *ArcStereoNet Pro* (for ArcGIS Pro) includes a full in-tool documentation, accessible from the ArcGIS application. The full user guide for the old *ArcStereoNet* (ArcMap version) is available [here](https://github.com/albdag/ArcStereoNet/wiki).
