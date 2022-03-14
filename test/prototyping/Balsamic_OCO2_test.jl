using Revise
using Plots
using Pkg
# Pkg.activate(".")
using vSmartMOM
using vSmartMOM.Architectures
using vSmartMOM.Absorption
using vSmartMOM.Scattering
using vSmartMOM.CoreRT
using vSmartMOM.SolarModel
using InstrumentOperator
using Interpolations
using Polynomials
using ForwardDiff 
using Distributions
using NCDatasets

#---------------------------------------------<Start Functions>---------------------------------------------#
function getSolar(grid, Tsolar)
    T = 5777.0
    λ_grid = reverse(1e4 ./ grid) #collect(757:0.01:777)
    black_body = planck_spectrum_wl(T, λ_grid) * 2.1629e-05 * π   
    black_body = SolarModel.watts_to_photons(λ_grid, black_body)

    # Get solar transmittance spectrum 
    #solar_transmission = solar_transmission_from_file("/home/rjeyaram/RadiativeTransfer/src/solar_merged_20160127_600_26316_100.out", parameters.spec_bands[1])
    solar_transmission = Tsolar(grid)
    # Get outgoing solar radiation
    sun_out = reverse(solar_transmission) .* black_body;
    return sun_out
end
###################################################


include("test/prototyping/runner.jl")
#---------------------------------------------<End Functions>----------------------------------------------#

## Atmospheric Radiative Transfer

# Load parameters from file
parameters = parameters_from_yaml("test/test_parameters/3BandParameters.yaml")
#parameters.architecture = CPU()
FT = Float64

# Load OCO Data: 
# File names:

L1File   = "/net/fluo/data1/group/oco3/L1bSc/oco3_L1bScSC_15935a_220226_B10311_220226124444.h5" #"/net/fluo/data1/group/oco2/L1bSc/oco2_L1bScND_26780a_190715_B10003r_200429212407.h5"
metFile  = "/net/fluo/data1/group/oco3/L2Met/oco3_L2MetSC_15935a_220226_B10311_220226124455.h5" #"/net/fluo/data1/group/oco2/L2Met/oco2_L2MetND_26780a_190715_B10003r_200429212406.h5"
dictFile = "/home/cfranken/code/gitHub/InstrumentOperator.jl/json/oco2.yaml"
# Load L1 file (could just use filenames here as well)
oco = InstrumentOperator.load_L1(dictFile,L1File, metFile);
grnd_pxl = 5
aa = Dataset(L1File)
bb = aa.group["SoundingGeometry"];
cc = bb["sounding_pcs_mode"];
pcs = [cc[grnd_pxl,i] for i=1:size(cc,2)];
iSam = findall(pcs .== "AM")

# Pick some bands as tuple (or just one)
bands = (1,2,3);
#bands = (1,3);
# Indices within that band:
indices = (92:885,114:845,50:916);
#indices = (92:885,50:916);

ii = iSam[1:36:end]
#y=[];
#oco_soundings=[];
# Get data for that sounding:
GeoInd = [grnd_pxl,ii[1]];
oco_sounding = InstrumentOperator.getMeasurement(oco, bands, indices, GeoInd);
oco_soundings = [oco_sounding];
y = [oco_soundings[1].SpectralMeasurement];

for i=2:length(ii)
    # Geo Index (footprint,sounding):
    GeoInd = [grnd_pxl,ii[i]];
    # Get data for that sounding:
    oco_sounding = InstrumentOperator.getMeasurement(oco, bands, indices, GeoInd);
    if i==1    
        oco_soundings = [oco_sounding];
        y = [oco_sounding.SpectralMeasurement];
    else
        oco_soundings = vcat(oco_soundings, oco_sounding);
        y = vcat(y, oco_sounding.SpectralMeasurement)
    end
end
###################################################
# Produce black-body in wavenumber range
Tsolar = solar_transmission_from_file("/home/rjeyaram/vSmartMOM/src/SolarModel/solar.out")
Tsolar_interp = LinearInterpolation(Tsolar[:, 1], Tsolar[:, 2])

#Initial guess state vector 
x = FT[0.2377, -3, -0.00244, 0, 0,0, 0.4, 0, 0.4, 0,1,400e-6,400e-6,400e-6, 1.0, 1.5, 0.0, -0.69, 0.3, 690., 0.3]
# State vector box-limits
limnᵣ = [1.3,1.7]
limnᵢ = [0.0,0.2]
limσᵣ = [0.05,1.6]
limlnr₀ = [-2.3,1.6] #r₀ ≈ 0.1, 5.0  
limp₀ = [100.,1100.]
limσₚ = [0.05,1.6]
# Run FW model:
# @time runner(x);
#y = oco_sounding.SpectralMeasurement;
Fx = zeros(length(y));
#ind = 92:885

for i=1:5
    K = ForwardDiff.jacobian(runner!, Fx, x);
    dx = K \ (y-Fx);
    x += dx;

    if x[16]<limnᵣ[1]
        x[16] = limnᵣ[1];
    elseif x[16]>limnᵣ[2]
        x[16] = limnᵣ[2];
    end

    if x[17]<limnᵢ[1]
        x[17] = limnᵢ[1];
    elseif x[17]>limnᵢ[2]
        x[17] = limnᵢ[2];
    end
    
    if x[18]<limlnr₀[1]
        x[18] = limlnr₀[1];
    elseif x[18]>limlnr₀[2]
        x[18] = limlnr₀[2];
    end

    if x[19]<limσᵣ[1]
        x[19] = limσᵣ[1];
    elseif x[19]>limσᵣ[2]
        x[19] = limσᵣ[2];
    end

    if x[20]<limp₀[1]
        x[20] = limp₀[1];
    elseif x[20]>limp₀[2]
        x[20] = limp₀[2];
    end

    if x[21]<limσₚ[1]
        x[21] = limσₚ[1];
    elseif x[21]>limσₚ[2]
        x[21] = limσₚ[2];
    end
    @show x;
end
runner!(Fx,x)

ν = oco_sounding.SpectralGrid*1e3
plot(y/1e20, label="Meas")
plot!(Fx/1e20, label="Mod")
plot!((y-Fx)/1e20, label="Meas-mod")
#=
a = Dataset("/net/fluo/data1/group/oco2/oco2_L2DiaND_26780a_190715_B10004r_200618191413.h5")
g = a.group["RetrievalGeometry"]
s = a.group["SpectralParameters"]
OP_results = a.group["RetrievalResults"]
i = findall(abs.(g["retrieval_latitude"][:].-41.736668).<0.0001)
fp_meas = s["measured_radiance"][:,i]
fp_mod  = s["modeled_radiance"][:,i]
=#
#=
fil = "/net/fluo/data1/group/oco3/L1bSc/oco3_L1bScSC_15935a_220226_B10311_220226124444.h5"
aa = Dataset(fil)
bb = aa.group["SoundingGeometry"];
cc = bb["sounding_pcs_mode"];
pcs = [cc[1,i] for i=1:size(cc,2)];
iSam = findall(pcs .== "AM")
#scatter(bb["sounding_longitude"][1,iSam],bb["sounding_latitude"][1,iSam])
SZA = bb["sounding_solar_zenith"][1,iSam[1:36:end]]
SAz = bb["sounding_solar_azimuth"][1,iSam[1:36:end]]
LZA = bb["sounding_zenith"][1,iSam[1:36:end]]
LAz = bb["sounding_azimuth"][1,iSam[1:36:end]]
plot(LZA)
=#