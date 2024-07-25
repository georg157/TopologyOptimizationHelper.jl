using SparseArrays
using LinearAlgebra
using NLopt
using PyPlot
using KrylovKit
using FiniteDifferences: extrapolate_fdm, central_fdm

# Approximate the operator A that solves Au = b
function Maxwell1d(L, ε, ω; dpml=2, resolution=20, Rpml=1e-20, ω_pml=ω)
    # PML σ = σ₀ x², with σ₀ chosen so that the round-trip reflection is Rpml
    σ₀ = -log(Rpml) / (4dpml^3/3)
    M = round(Int, (L+2dpml) * resolution)
    dx = (L+2dpml) / (M+1)
    x = (1:M) * dx # x grid
    
    # 1st derivative matrix
    o = ones(M)/dx
    D = spdiagm(M+1,M, -1 => -o, 0 => o)
    
    # Need PML scale factors 1/(1+iσ/ω) at x and x' points
    σ = [ξ < dpml ? σ₀*(dpml-ξ)^2 : ξ > L+dpml ? σ₀*(ξ-(L+dpml))^2 : 0.0 for ξ in x]
    sqrtΣ = spdiagm(@. sqrt(inv(1 + (im/ω_pml)*σ)))
    x′ = ((0:M) .+ 0.5).*dx # 1st-derivative grid points
    σ′ = [ξ < dpml ? σ₀*(dpml-ξ)^2 : ξ > L+dpml ? σ₀*(ξ-(L+dpml))^2 : 0.0 for ξ in x′]
    Σ′ = spdiagm(@. inv(1 + (im/ω_pml)*σ′))

    # 2nd derivative matrix and PML layer
    # Adjusting layer to make the matrix A symmetric
    x = x .- dpml
    A = sqrtΣ * D' * Σ′ * D * sqrtΣ - spdiagm(ω^2 .* ε)

    return A, x  
end

export Maxwell1d