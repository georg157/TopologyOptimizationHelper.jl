##### gradients.jl


# Compute the LDOS and gradient of a system given a matrix A, vector b, and frequency ω
function ∇_ε_LDOS(A, b, ω)
    # derivative of matrix A with respect to vector ε
    v = A\b
    w = A'\b

    LDOS = -imag(v' * b)
    ∇_epsilon_LDOS = -imag.(conj(ω)^2 .* conj.(v) .* w)

    return LDOS, ∇_epsilon_LDOS
end

export ∇_ε_LDOS


# Compute LDOS given the necessary parameters to compute the operator A
function Just_LDOS(L, ε, ω, b; resolution=20, dpml=2, ω_pml=ω)
    A, x = Maxwell1d(L, ε, ω; resolution, dpml, ω_pml)
    LDOS, _ = ∇_ε_LDOS(A, b, ω)
    
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

    #Return μₖ and yₖ
    E⁻¹ = spdiagm(1 ./ ε)
    M = E⁻¹ * A
    LU = lu(M)
    vals, vecs, info = eigsolve(z -> LU \ z, x₀, 1, :LM, Arnoldi())
    λₖ, yₖ = vals[1], vecs[1]
    return 1 / λₖ + ω^2, yₖ
end

export Arnoldi_eig


function Eigengradient(A, ε, ω, x₀)
    # We solve for ω₀² := μₖ and u₀ := yₖ
    ω₀², u₀ = Arnoldi_eig(A, ε, ω, x₀)
    # Return gradient ∂ω₀/∂ε
    
    return - sqrt(ω₀²) / 2 .* u₀.^2 ./ sum(u₀.^2 .* ε)
end

export Eigengradient

function ∇_ω_LDOS(A, b, ε, ω)
    E = spdiagm(ε)
    v = A \ b
    w = A' \ b

    return -2 * imag(ω .* v' * E * w)
end

export ∇_ω_LDOS


function Improved_∇_ε_LDOS(A, ε, ω, b; resolution=20, dpml=2, x₀=b, ω_pml=ω)
    # The matrix argument A := A(ε, Re(eval_ω))
    # We must compute Re[ω₀(ε)] and its associated A₀ to get the correct values
    LDOS, ∂LDOS_∂ε = ∇_ε_LDOS(A, b, ω)
    ∂LDOS_∂ω = ∇_ω_LDOS(A, b, ε, ω)
    ∂ω_∂ε = real.(Eigengradient(A, ε, ω, x₀))

    return LDOS, ∂LDOS_∂ε .+  ∂LDOS_∂ω .* ∂ω_∂ε
end

export Improved_∇_ε_LDOS


function Just_Improved_LDOS(L, ε, ω, b; resolution=20, dpml=2, ω_pml=ω)
    A, x = Maxwell1d(L, ε, ω; resolution, dpml, ω_pml)
    ω₀ = Maxwell_omega(L, ε, ω, b)
    A₀, _ = Maxwell1d(L, ε, real(ω₀); ω_pml=ω)
    LDOS, _ = ∇_ε_LDOS(A₀, b, real(ω₀))
    
    return LDOS
end

export Just_Improved_LDOS


function Maxwell_omega(L, ε, ω, b)
    A, _ = Maxwell1d(L, ε, ω)
    eigval, _ = Arnoldi_eig(A, ε, ω, b)
    
    return sqrt(eigval)
end

export Maxwell_omega