#!/usr/bin/env julia --startup-file=no

"""
Linear-Stability-Calculators: Quasi-Hydrostatic Model
=====================================================

Linear Stability Calculation of an inertially unstable jet using spectral methods

The basic state is that of a barotropic Bickley jet (to be generalized).
The perturbation is assumed to be periodic in both the zonal (x) and vertical (z) directions.

Calculations are done in serial.  See inertial_instability_mpi.py for parallel execution.
"""

include("src/quasihydrostatic_setup.jl")
include("src/parameters.jl")
include("src/Compute_Eigenvalues.jl")
include("src/Plotting_Functions.jl")

## label the output directory desired
output_dir = "./linear_instability_calculator/outputs/inertial_instability/"
mkpath(output_dir)

files   = Files(         nc = output_dir * "data_spectrum_serial.nc",
                       json = output_dir * "data_parameters_serial.json",
                 plotgrowth = output_dir * "growth_serial.png",
                 plotslicem = output_dir * "growth_slice_m_serial.png",
                plotmodes1D = output_dir * "plot_modes1D_serial",
                plotmodes2D = output_dir * "plot_modes2D_serial"
                )
grid    = Grid(Ly = 1.0, Lz = 1.0, Ny = 100, θ₀ = π/32, method=Cheb())
physics = Physics(N = sqrt(3.0), ν = 1e-6, θ₀ = grid.θ₀, NT = 1)
jet     = LinearShear(grid, physics)

Output_Parameters(grid, physics, jet, files.json)

Neigs  = 10                            
dk, Nk = 100000000*1e-6, 41
dm, Nm = 100000*1e-4, 80

ks = collect(0 :dk:(Nk-1)*dk)
ms = collect(dm:dm:Nm*dm)

ωs    = zeros(Complex, (Neigs, Nm, Nk))
modes = zeros(Complex, (3*grid.Ny+1, Neigs, Nm, Nk))

for (ik, k) in enumerate(ks)
    for (im, m) in enumerate(ms)
        A = build_matrix(k, m, grid, physics, jet)

        ωs[:, im, ik], modes[:, :, im, ik] = compute_spectrum(A, Neigs)

        print("k = ", ks[ik], " m = ", ms[im], " growth = ", imag(ωs[1, im, ik]), "\n")
    end
end

save_spectrum(ωs, modes, ks, ms, Neigs, grid.y, grid.Ny, files.nc)

plot_growth_slice(files.nc, files.json, files.plotslicem)
plot_growth(      files.nc, files.json, files.plotgrowth)

plot_modes_1D(files.nc, files.json, files.plotmodes1D, Neigs)
plot_modes_2D(files.nc, files.json, files.plotmodes2D, Neigs)
