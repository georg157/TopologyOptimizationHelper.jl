##### long_test.jl

using DelimitedFiles
using SparseArrays
using LinearAlgebra
using NLopt
using PyPlot
using KrylovKit
using Zygote
using Statistics
using Revise
using TopologyOptimizationHelper

Lx = 5
Ly = 5
res = 20
intermediate = 6.5ones(60,60)
ε = pad_matrix(intermediate, 3, 3, 1)
ω = π
N, M = size(ε)
b = zeros(N,M)
offset = -round(Int, res * (0.1 + 1.5))
b[N÷2 + offset,M÷2] = 1

_, ε_8, _, _, _, _ = smooth_LDOS_Optimize2d(Lx, Ly, ε, ω, b; resolution=res, ftol=0, max_eval=100, design_dimensions=(3,3), fR=0.2, β=8)

_, ε_16, _, _, _, _ = smooth_LDOS_Optimize2d(Lx, Ly, ε_8, ω, b; resolution=res, ftol=0, max_eval=200, design_dimensions=(3,3), fR=0.2, β=16)

_, ε_40, _, _, _, _ = smooth_LDOS_Optimize2d(Lx, Ly, ε_16, ω, b; resolution=res, ftol=0, max_eval=300, design_dimensions=(3,3), fR=0.2, β=40)

_, ε_init, _, _, _, _ = smooth_LDOS_Optimize2d(Lx, Ly, ε_40, ω, b; resolution=res, ftol=0, max_eval=400, design_dimensions=(3,3), fR=0.2, β=Inf)


LDOS_opt, ε_opt, LDOS_vals, omegas, x, y = smooth_LDOS_Optimize2d(Lx, Ly, ε_init, ω, b; resolution=res, ftol=0, max_eval=500000, design_dimensions=(3,3), fR=0.2, β=Inf)
Qs = -real.(omegas) ./ 2imag.(omegas)
writedlm("long_test_epsilon_opt.txt", ε_opt)
writedlm("long_test_LDOS_vals.txt", LDOS_vals)
writedlm("long_test_Qs.txt", Qs)