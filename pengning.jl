using DelimitedFiles
using SparseArrays
using LinearAlgebra
using NLopt
using PyPlot
using KrylovKit
using FiniteDifferences
using Statistics
using Revise
using TopologyOptimizationHelper

Lx = Ly = 4
ω = 2π
res = 40
ρ = zeros(120, 120)
ρ = pad_matrix(ρ, 2, 2, 0; resolution=res)
offset = -round(Int, res * (0.2 + 1.5))
N, M = size(ρ)
b = zeros(N, M)
b[N÷2 + offset, M÷2] = 1

A, x, y = Maxwell2d(Lx, Ly, ones(N,M), ω; resolution=res)
v = A \ vec(b)
LDOS_norm = -imag(v' * vec(b))

LDOS_init, ε_init, _, _, _, _ = LDOS_Optimize2d(
    Lx, Ly, ρ, ω, b;
    resolution=res,
    ftol=0,
    max_eval=1000,
    design_dimensions=(3, 3),
    mat_loss=1e-4,
    max_mat=5
)
println("init_enhancement = ", LDOS_init / LDOS_norm)
ρ_init = real.(ε_init .- 1) / 5
writedlm("penging_rho_init.txt", ρ_init)


mod_LDOS_opt, mod_ε_opt, mod_LDOS_vals, mod_omegas, x, y = mod_LDOS_Optimize2d(
    Lx, Ly, ρ_init, ω, b, vec(b);
    resolution=res,
    ftol=0,
    max_eval=20_000,
    design_dimensions=(3, 3),
    mat_loss=1e-4,
    max_mat=5
)
println("mod_enhancement = ", mod_LDOS_opt / LDOS_norm)

mod_ρ_opt = real.(mod_ε_opt .- 1) / 5
mod_LDOS_vals ./= LDOS_norm
mod_LDOS_vals = vcat(fill.(mod_LDOS_vals, 2)...)
mod_Qs = -real.(mod_omegas) ./ (2 * imag.(mod_omegas))
mod_Qs = vcat(fill.(mod_Qs, 2)...)

writedlm("penging_mod_rho_opt.txt", mod_ρ_opt)
writedlm("pengning_mod_LDOS_vals.txt", mod_LDOS_vals)
writedlm("pengning_mod_Qs.txt", mod_Qs)