##### test_outside.jl

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
writedlm("test_outside_epsilon_init.txt", ε_init)

A, x, y = Maxwell2d(Lx, Ly, ε_init, ω; resolution=res)
ω₀ = sqrt(Arnoldi_eig(A, vec(ε_init), ω, vec(b))[1])
E⁻¹ = spdiagm(1 ./ vec(ε_init))
vals, vecs, info = eigsolve(z -> E⁻¹ * A * z + ω^2 .* z, vec(b), 1, EigSorter(λ -> abs(λ - ω₀^2); rev = false), Arnoldi()) 
u₀ = vecs[1]
imshow(reshape(abs.(u₀).^2, N,M), cmap="coolwarm")
colorbar(label=L"init mode")
savefig("test_outside_mode_init.pdf")
close("all")

imshow(real.(ε_init), cmap="gray_r", vmin=1, vmax=12)
colorbar(label=L"ε_{init}")
savefig("test_outside_epsilon_init.pdf")
close("all")

LDOS_opt, ε_opt, LDOS_vals, omegas, _, _ = smooth_LDOS_Optimize2d(Lx, Ly, ε_init, ω, b; resolution=res, ftol=0, max_eval=100000, design_dimensions=(3,3), fR=0.2, β=Inf)
Qs = -real.(omegas) ./ 2imag.(omegas)
writedlm("test_outside_epsilon_opt.txt", ε_opt)
writedlm("test_outside_LDOS_vals.txt", LDOS_vals)
writedlm("test_outside_Qs.txt", Qs)


A, x, y = Maxwell2d(Lx, Ly, ε_opt, ω; resolution=res)
ω₀ = sqrt(Arnoldi_eig(A, vec(ε_opt), ω, vec(b))[1])
E⁻¹ = spdiagm(1 ./ vec(ε_opt))
vals, vecs, info = eigsolve(z -> E⁻¹ * A * z + ω^2 .* z, vec(b), 1, EigSorter(λ -> abs(λ - ω₀^2); rev = false), Arnoldi()) 
u₀ = vecs[1]
imshow(reshape(abs.(u₀).^2, N,M), cmap="coolwarm")
colorbar(label=L"opt mode")
savefig("test_outside_mode_opt.pdf")
close("all")

imshow(real.(ε_opt), cmap="gray_r", vmin=1, vmax=12)
colorbar(label=L"ε_{opt}")
savefig("test_outside_epsilon_opt.pdf")
close("all")

mod_LDOS_opt, mod_ε_opt, mod_LDOS_vals, mod_omegas, x, y = mod_smooth_Optimize2d(Lx, Ly, ε_init, ω, b, vec(b); resolution=res, ftol=0, max_eval=100000, design_dimensions=(3,3), fR=0.2, β=Inf, sgn=1)
mod_LDOS_vals =  vcat(fill.(mod_LDOS_vals, 2)...)
mod_Qs = -real.(mod_omegas) ./ 2imag.(mod_omegas)
mod_Qs = vcat(fill.(mod_Qs, 2)...)
writedlm("test_outside_mod_epsilon_opt.txt", mod_ε_opt)
writedlm("test_outside_mod_LDOS_vals.txt", mod_LDOS_vals)
writedlm("test_outside_mod_Qs.txt", mod_Qs)

A, x, y = Maxwell2d(Lx, Ly, mod_ε_opt, ω; resolution=res)
ω₀ = sqrt(Arnoldi_eig(A, vec(mod_ε_opt), ω, vec(b))[1])
E⁻¹ = spdiagm(1 ./ vec(mod_ε_opt))
vals, vecs, info = eigsolve(z -> E⁻¹ * A * z + ω^2 .* z, vec(b), 1, EigSorter(λ -> abs(λ - ω₀^2); rev = false), Arnoldi()) 
u₀ = vecs[1]
imshow(reshape(abs.(u₀).^2, N,M), cmap="coolwarm")
colorbar(label=L"mod opt mode")
savefig("test_outside_mod_mode.pdf")
close("all")

imshow(real.(mod_ε_opt), cmap="gray_r", vmin=1, vmax=12)
colorbar(label=L"mod ε_{opt}")
savefig("test_outside_mod_epsilon_opt.pdf")
close("all")


loglog(LDOS_vals, color="red")
loglog(mod_LDOS_vals, color="blue")
xlabel("No. System Solves")
ylabel("LDOS")
savefig("test_outside_LDOS_iterations.pdf")
close("all")

loglog(Qs, color="red")
loglog(mod_Qs, color="blue")
xlabel("No. System Solves")
ylabel("Quality Factor")
savefig("test_outside_Q_iterations.pdf")
close("all")