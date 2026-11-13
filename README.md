# Carbosys – Computer Vision Experiments

This repository contains computer vision experiments developed as part of the **Carbosys project**, focusing on image processing techniques for vegetation analysis and environmental monitoring.

The experiments explore two vegetation indices, GLI and NDVI, using RGB photographs and multispectral satellite imagery. The goal is to investigate how image processing can extract useful environmental information and contribute to future features of the Carbosys platform.

## Projects

### 1. GLI – Green Leaf Index

The GLI experiment uses RGB images to distinguish vegetation from non-vegetated areas based on color information.

**Methodology:**
- Calculation of the Green Leaf Index (GLI) for each pixel.
- Comparison between Otsu automatic thresholding and a fixed threshold (GLI > 0.10).
- Morphological operations to reduce noise and fill small gaps.
- Contour extraction to identify the boundaries of classified vegetation regions.

**Formula:**

GLI = (2G - R - B) / (2G + R + B)

**Results:**

After morphological processing:
- Otsu thresholding classified approximately 35.85% of the image as vegetation.
- Fixed thresholding (GLI > 0.10) classified approximately 47.62% as vegetation.

These results demonstrate how different thresholding methods can influence vegetation segmentation.

**Technologies:** Python, OpenCV, NumPy and Matplotlib.

### 2. NDVI – Normalized Difference Vegetation Index

The NDVI experiment uses multispectral images from the Sentinel-2 satellite to analyze vegetation through red and near-infrared spectral information.

**Methodology:**
- Processing of Sentinel-2 Level-2A imagery.
- Use of band B04 (Red) and band B08 (Near Infrared).
- Calculation of the NDVI for each pixel.
- Generation of vegetation maps and classification using a threshold of 0.30.

**Formula:**

NDVI = (NIR - Red) / (NIR + Red)

**Results:**
- Minimum NDVI: approximately -0.34.
- Maximum NDVI: approximately 0.91.
- Approximately 77.82% of valid pixels had NDVI values above 0.30.

This experiment demonstrates how multispectral satellite imagery can provide information about vegetation conditions.

**Technologies:** Python, Rasterio, NumPy and Matplotlib.

## Relationship with Carbosys

The experiments explore image processing techniques that may contribute to future Carbosys features:

- **GLI:** vegetation segmentation and contour extraction may help identify vegetation regions and support agricultural plot delimitation.
- **NDVI:** vegetation indicators can support environmental monitoring and provide numerical inputs for machine learning models.

NDVI is also one of the vegetation indices used in the research by Alecsander Mergen, which combines satellite-derived information and meteorological variables to estimate Net Ecosystem Exchange (NEE) of CO₂.

The research uses Eddy Covariance tower measurements as reference data to train and evaluate a neural network. The experiments in this repository are introductory demonstrations of image processing techniques, not a reproduction of the complete predictive model.

## Future Improvements

- Explore additional vegetation indices, such as EVI, SAVI and NDWI.
- Analyze satellite images from different dates.
- Improve vegetation segmentation and plot delimitation.
- Integrate vegetation indices with meteorological variables.
- Explore machine learning models for environmental analysis.

## Repository Structure

```text
carbosys/
├── gli/
│   ├── imagens/
│   └── resultados/
├── ndvi/
│   ├── imagens/
│   └── resultados/
├── .gitignore
└── README.md
```

## Author

**Júlia Bechaire**

Federal University of Santa Maria
