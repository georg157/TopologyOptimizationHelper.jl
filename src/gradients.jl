##### gradients.jl


# Compute the LDOS and gradient of a system given a matrix A, vector b, and frequency ω
function ∇_ε_LDOS(A, ω, b; α=0)
    # derivative of matrix A with respect to vector ε
    v = A \ b
    w = conj.(A) \ b

    LDOS = -imag(v' * b)
    ∇_epsilon_LDOS = ω^2 * imag.(v.^2)

    return LDOS, ∇_epsilon_LDOS
end
export ∇_ε_LDOS


# Compute LDOS given the necessary parameters to compute the operator A
function Just_LDOS(L, ε, ω, b; resolution=20, dpml=2, Rpml=1e-20, ω_pml=ω)
    A, x = Maxwell1d(L, ε, ω; resolution, dpml, Rpml, ω_pml)
    LDOS, _ = ∇_ε_LDOS(A, ω, b)
    
    return LDOS
end
export Just_LDOS


# Use Krylov subspace methods to solve for eigenvalues
function Arnoldi_eig(A, ε, ω, x₀)
    # Recall that we have Au = b
    ##### A(ω) = -(∇² + ω²E)
    ##### E = spdiagm(ε)

    # We want to solve for ωₖ² in -∇²xₖ = ωₖ²Ex
    # i.e. solve E⁻¹(-∇²)xₖ = ωₖ²xₖ
    # Given design ω, we solve for [E⁻¹(-∇²) - ω²I]⁻¹yₖ = [E⁻¹A(ω)]⁻¹yₖ = M⁻¹yₖ = λₖyₖ
    ##### M = E⁻¹A
    # After manipulating, this will give us the ωₖ closest to ω
    ##### yₖ = λₖMyₖ => Myₖ = λₖ⁻¹yₖ => -E⁻¹∇²yₖ = (ω² + λₖ⁻¹)yₖ = μₖyₖ

    # Return μₖ and yₖ
    E⁻¹ = spdiagm(1 ./ ε)
    M = E⁻¹ * A
    LU = lu(M)
    vals, vecs, info = eigsolve(z -> LU \ z, x₀, 1, :LM, Arnoldi())
    # If the optimizer ignores the constraint, use
    # EigSorter(λ -> real(sqrt(1 / λ + ω^2)) <= ω ? abs(λ) : -Inf, rev=true)
    # instead of :LM
    λₖ, yₖ = vals[1], vecs[1]
    return 1 / λₖ + ω^2, yₖ
end
export Arnoldi_eig


function Eigengradient(A, ε, ω, x₀)
    # We solve for ω₀² := μₖ and u₀ := yₖ
    ω₀², u₀ = Arnoldi_eig(A, ε, ω, x₀)
    ω₀ = sqrt(ω₀²)
    # Return gradient ∂ω₀/∂ε
    
    return ω₀, -ω₀ .* u₀.^2 ./ 2sum(u₀.^2 .* ε)
end
export Eigengradient


function ∇_ω_LDOS(A, ε, ω, b)
    E = spdiagm(ε)
    v = A \ b
    w = conj.(A) \ b

    return -imag(2ω .* v' * E * w)
end
export ∇_ω_LDOS


function Improved_∇_ε_LDOS(D², ε, ω, b, x₀)
    E = spdiagm(ε)
    A = D² - ω^2 .* E
    ω₀², u₀ = Arnoldi_eig(A, ε, ω, x₀)
    ω₀ = sqrt(ω₀²)
    A₀ = D² - real(ω₀)^2 .* E
    v = A₀ \ b
    w = conj.(A₀) \ b

    LDOS = -imag(v' * b)
    ∂LDOS_∂ε = -imag.(real(ω₀)^2 .* conj.(v) .* w)
    ∂LDOS_∂ω = -imag(2real(ω₀) .* v' * E * w)
    ∂ω_∂ε = -ω₀ .* u₀.^2 ./ 2sum(u₀.^2 .* ε)
    ∇LDOS = ∂LDOS_∂ε .+  ∂LDOS_∂ω .* real.(∂ω_∂ε)

    return LDOS, ∇LDOS, ω₀

    # E = spdiagm(ε)
    # A = D² - ω^2 .* E
    # ω₀, ∂ω_∂ε = Eigengradient(A, ε, ω, x₀)
    # @show real(ω₀) - ω
    # Libc.flush_cstdio()
    # A₀ = D² - real(ω₀)^2 .* E
    # LDOS, ∂LDOS_∂ε = ∇_ε_LDOS(A₀, b, real(ω₀))

    # if !isempty(grad) 
    #     ∂LDOS_∂ω = ∇_ω_LDOS(A₀, b, ε, real(ω₀))
    #     ∇LDOS = ∂LDOS_∂ε .+  ∂LDOS_∂ω .* real.(∂ω_∂ε)

    #     grad .= ∇LDOS
end


function Just_Improved_LDOS(L, ε, ω, b; resolution=20, dpml=2, Rpml=1e-20, ω_pml=ω)
    A, x = Maxwell1d(L, ε, ω; resolution, dpml, Rpml, ω_pml)
    ω₀ = Maxwell_omega(L, ε, ω, b)
    A₀, _ = Maxwell1d(L, ε, real(ω₀); ω_pml=ω)
    LDOS, _ = ∇_ε_LDOS(A₀, real(ω₀), b)
    
    return LDOS
end
export Just_Improved_LDOS


function Maxwell_omega(L, ε, ω, x₀; resolution=20, dpml=2, Rpml=1e-20, ω_pml=ω)
    A, _ = Maxwell1d(L, ε, ω; resolution, dpml, Rpml, ω_pml)
    eigval, _ = Arnoldi_eig(A, ε, ω, x₀)
    
    return sqrt(eigval)
end
export Maxwell_omega