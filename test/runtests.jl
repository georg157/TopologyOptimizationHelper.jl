using SparseArrays
using LinearAlgebra
using NLopt
using PyPlot
using KrylovKit
using FiniteDifferences
using TopologyOptimizationHelper
using Test

@testset "TopologyOptimizationHelper.jl" begin
    # Initialize parameters for testing
    L = 20
    ε = rand(480) .* 11 .+ 1
    δε = randn(length(ε)) * 1e-6
    ω = 2π
    δω = randn() * 1e-6
    dir = randn(length(ε))

    A, x = Maxwell1d(L, ε, ω)
    new_ε_A, _ = Maxwell1d(L, ε + δε, ω)
    new_ω_A, _ = Maxwell1d(L, ε, ω + δω; ω_pml = ω)

    M = length(x)
    b = zeros(M)
    b[M÷2] = 1


    # Gradient of LDOS w.r.t. ε
    ## Exact gradient versus numerical gradient
    LDOS, ∇LDOS = ∇_ε_LDOS(A, b, ω)
    new_ε_LDOS, _ = ∇_ε_LDOS(new_ε_A, b, ω)
    @test new_ε_LDOS - LDOS ≈ ∇LDOS' * δε atol=1e-8


    # Gradient of eigenvalue w.r.t. ε
    ## Exact gradient versus numerical gradient
    val, _ = Arnoldi_eig(A, ε, ω, b)
    new_val, _ = Arnoldi_eig(new_ε_A, ε + δε, ω, b)
    gradient = Eigengradient(A, ε, ω, b)
    @test sqrt(new_val) - sqrt(val) ≈ dot(δε, gradient) atol=1e-8

    ## Exact gradient versus Richardson extrapolation
    f = α -> Maxwell_omega(L, ε + α * dir, ω, b)
    @test extrapolate_fdm(central_fdm(2, 1), f, 0)[1] ≈ dir' * gradient atol=1e-8


    # Gradient of LDOS w.r.t. ω 
    ## Exact gradient versus numerical gradient
    ∇LDOS = ∇_ω_LDOS(A, b, ε, ω)
    new_ω_LDOS, _ = ∇_ε_LDOS(new_ω_A, b, ω)
    @test new_ω_LDOS - LDOS ≈ ∇LDOS[1] * real(δω) atol=1e-8


    # Improved gradient of LDOS w.r.t. ε
    ## Exact gradient versus numerical gradient
    ω₀ = Maxwell_omega(L, ε, ω, b)
    A₀, _ = Maxwell1d(L, ε, real(ω₀); ω_pml=ω)
    true_LDOS, true_∇LDOS = Improved_∇_ε_LDOS(A₀, ε, real(ω₀), b; ω_pml=ω)
    true_new_LDOS =  Just_Improved_LDOS(L, ε + δε, ω, b)
    @test true_new_LDOS - true_LDOS ≈ true_∇LDOS' * δε atol=1e-8

    ## Exact gradient versus Richardson extrapolation
    f = α -> Just_Improved_LDOS(L, ε + α * dir, ω, b)
    @test extrapolate_fdm(central_fdm(2, 1), f, 0)[1] ≈ dir' * true_∇LDOS atol=1e-8

end
