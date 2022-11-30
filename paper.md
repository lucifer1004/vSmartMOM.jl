---
title: 'vSmartMOM.jl: an Open-Source Julia Package for Atmospheric Radiative Transfer and Remote Sensing Tools'

tags:
  - Julia
  - radiative transfer
  - atmospheric radiation
  - vSmartMOM
authors:
  - name: Rupesh Jeyaram
    orcid: 0000-0003-0142-7367
    affiliation: 1
  - name: Suniti Sanghavi 
    orcid: 0000-0003-0754-9154
    affiliation: 2
  - name: Christian Frankenberg
    orcid: 0000-0002-0546-5857
    affiliation: "1, 2"
affiliations:
 - name: California Institute of Technology 
   index: 1
 - name: Jet Propulsion Laboratory 
   index: 2
date: 10 January 2022
bibliography: paper.bib
---

# Summary

Remote sensing researchers use radiative transfer modeling to interpret satellite data for studying Earth's atmospheric and surface properties. The field plays a key role in how scientists understand many aspects of our rapidly changing planet – from climate change and pollution to the carbon and water cycles. Astronomers use radiative transfer to study the atmospheres of stars and substellar objects such as brown dwarfs and exoplanets. 

**vSmartMOM.jl** is a [Julia](https://julialang.org) package that enables the fast computation of atmospheric optical properties and fully polarized multiple-scattering radiance simulations [@Sanghavi:2013a], based on the Matrix Operator Method, also known as Discrete Space Theory [@grant1969adiscrete; @grant1969bdiscrete; @hunt1969discrete]. Users are free to customize simulation parameters and atmospheric properties, including trace-gas profiles, aerosol distributions, microphysical properties, surface reflectance, and quadrature schemes. Independent submodules can also be imported individually; for example, **Absorption.jl** can be used for computing gaseous absorption and **Scattering.jl** for computing scattering phase-functions. 

The Julia language provides many exciting opportunities to modernize radiative transfer software. For example, using the ForwardDiff.jl package [@Revels:2016], Jacobians can be calculated alongside computations using a technique called automatic differentiation. This technique uses the chain rule, dual numbers, and Julia's multiple dispatch paradigm to propagate not only the numerical computations at each step, but also the derivative. 

We further use Julia's multiple dispatch feature to create a software architecture that is clean, flexible, and reusable. 

Additionally, optimized techniques have been implemented to speed up the package’s performance on both CPU and GPU by orders of magnitude compared to existing radiative transfer codes. 

**vSmartMOM.jl** has already been used in research projects, ranging from methane-plume simulation to aerosol profile fitting and spectropolarimetric simulations of brown dwarfs and exoplanets. It has also been used in graduate-level remote sensing coursework. Ultimately, **vSmartMOM.jl** aims to accelerate the pace of atmospheric research through efficient software while lowering the barrier of entry for researchers and students in remote sensing. 

# Statement of need

For historical reasons, much of the scientific work in remote sensing is based on legacy code, written in Fortran or C/C++, mixed with “glue languages” such as Python. Researchers who developed these codes also placed greater emphasis on science results than software engineering best practices. As a result, many parts of key codebases are aging, convoluted, and hard to improve by both incoming graduate students and experienced researchers. 

Rather than simply *porting* these codes to a new language, **vSmartMOM.jl** designs a radiative transfer code from the ground up to include new functionalities such as GPU acceleration and automatic differentiation – features that have become computationally feasible and widespread only in the last decade. These features are highly beneficial to radiative transfer parameter fitting, which has historically been time consuming and computationally intensive. 

# Overview of functionality

The package has a modular architecture, allowing users to import just the specific module(s) that they need.

![Sample atmospheric reflectance under default atmospheric parameters, calculated using vSmartMOM.jl](joss_1.png)

**vSmartMOM.jl** is the top-level module that uses absorption and scattering submodules to carry out radiative transfer simulations. Specifically, it: 

- Enables 1D vectorized plane-parallel radiative transfer modeling based on the Matrix Operator Method [@Sanghavi:2013a]
- Incorporates fast, high-fidelity simulations of scattering atmospheres containing haze and clouds, including pressure- and temperature-resolved absorption profiles of gaseous species in the atmosphere
- Enables GPU-accelerated computations of the resulting hyperspectral multiple-scattering radiative transfer
- Enables auto-differentiation of the output spectrum with respect to various input parameters, allowing for spectral fitting routines to estimate atmospheric parameters

![Sample absorption spectrum of CO2 with 0.01 cm^-1 step size resolution, calculated using Absorption.jl](joss_2.png)

**Absorption.jl** enables absorption cross-section calculations of atmospheric gases at different pressures, temperatures, and wavelengths. It uses the HITRAN [@Gordon:2017] database for calculations. For exoplanets and brown dwarfs, developments are underway to obtain spectra from the ExoMOL [@yurchenko2012exomol:2012] and HITEMP [@rothman2010hitemp:2010] databases, as described in @Sanghavi:2019. While the module enables lineshape calculations from scratch, it also allows users to create and save an interpolator object at specified wavelength, pressure, and temperature grids. The module also supports auto-differentiation of the profile, with respect to pressure and temperature. Calculations can be computed either on CPU or GPU (CUDA).

![Sample scattering phase functions of aerosols (I $\rightarrow$ I and the I $\rightarrow$ Q transition), calculated using Scattering.jl ($r_m$ = 10.0 $\mu$m, $\sigma$ = 1.1, $n_r$ = 1.3, $n_i$ = 0.0, $\lambda$ = 0.65 $\mu$m). Positive values are <span style="color:green">*green*</span>, and negative values are <span style="color:red">*red*</span>. ](joss_3.png)

**Scattering.jl** is used for calculating Mie scattering phase-functions for aerosols with specified size distributions and refractive indices. This module enables scattering phase-function calculation of atmospheric aerosols with different size distributions, incident wavelengths, and refractive indices. It can perform the calculation using either numerical integration using quadrature points (NAI-2) [@Siewert:1982] or using precomputed tabulations of Wigner 3-j symbols (PCW) [@Domke:1975] with recent corrections [@Sanghavi:2013b]. State-of-the-art methods such as $\delta$-truncation [@Hu:2000] and $\delta$-BGE truncation [@Sanghavi:2015] are used for scalar and vector radiative transfer computations, respectively. The module also supports auto-differentiation of the phase function, with respect to the aerosol's size distribution parameters and its complex refractive index. 

# Benchmarks

Standard reference tables from the literature [@Natraj:2009] are used to validate **vSmartMOM.jl** simulation output and the key result is that simulated reflectance output from vSmartMOM.jl closely matches published standard values – always within 0.1% for I, 0.0005 for Q, and 0.0001 for U. 

Runtime duration for a given simulation is also compared between using CPU and GPU architectures. (CPU architecture is single-threaded, AMD EPYC 7H12 64-Core Processor; GPU is parallel on an NVIDIA A100 Tensor Core (40Gb))

![Runtime comparison between CPU and GPU for sample radiative transfer simulation with varying number of spectral points](agu_4.png)

With very small spectral resolution (approximately ten points), the GPU computation is slower than the CPU computation due to the overhead cost of data transfer onto the GPU. However, at higher spectral resolutions (more than one hundred spectral points), we see the computational gains of running these calculations on the GPU. A nearly 100x speedup is observed at higher resolutions beyond ten thousand points. These scenarios are more reflective of real-world scenarios with high-resolution satellite data. 

Different codes use various approximations like the correlated-k method [@goody:1989], principal component based radiative transfer modeling of hyperspectral measurements (PCRTM) [@liu:2016], and low stream approximations [@spurr:2011] in order to avoid the high computational cost of line-by-line spectral computations. GPU acceleration allows vSmartMOM to competitively employ line-by-line computations, achieving a tremendous speed up over CPU-only computations.

Hardware acceleration, in addition to algorithmic efficiencies and performance optimizations in **vSmartMOM.jl**, suggest that this package can greatly accelerate the pace of remote sensing research. 

# Acknowledgements

We thank Frankenberg lab members for their enthusiastic support and guidance throughout this project. We also acknowledge support from Caltech’s Schmidt Academy for Software Engineering.

<div style="page-break-after: always;"></div>

# References