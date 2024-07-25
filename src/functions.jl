##### functions.jl 


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

function LDOS_Optimize(L, ε, ω; dpml=2, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500)
    A, x = Maxwell1d(L, ε, ω; resolution, dpml)
    M = length(x)
    b = zeros(M)
    b[M÷2] = 1
    LDOS_vals = Float64[]
    
    function LDOS_obj(ε, grad)
        A, x = Maxwell1d(L, ε, ω; resolution, dpml)
        LDOS, ∇LDOS = ∇_ε_LDOS(A, b, ω)
        grad .= ∇LDOS
        push!(LDOS_vals, LDOS)
        
        return LDOS
    end

    opt = Opt(:LD_CCSAQ, length(ε))
    opt.lower_bounds = 1
    opt.upper_bounds = 12
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = LDOS_obj

    (LDOS_opt, ε_opt, ret) = optimize(opt, ε)
    @show numevals = opt.numevals # the number of function evaluations
    
    return LDOS_opt, ε_opt, LDOS_vals
end

export LDOS_Optimize

# Our proposed theoretical, near optimal solution to the system above
function Fabry_Perot_epsilon(L, x; λ=1, n=round(2*L/λ - 2))
    if abs(x - L/2) < λ/4 
        return 1
    end

    d1 = λ/4
    d12 = d1/sqrt(12)
    period = d1 + d12
    for k in 0:n
        s_k = λ/4 + k * period
        if s_k < abs(x - L/2) < s_k + d12
            return 12
        end
    end

    return 1
end

export Fabry_Perot_epsilon

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
    _, ∂LDOS_∂ε = ∇_ε_LDOS(A, b, ω)
    ∂LDOS_∂ω = ∇_ω_LDOS(A, b, ε, ω)
    ∂ω_∂ε = real.(Eigengradient(A, ε, ω, x₀))

    return Just_LDOS(L, ε, ω, b; resolution, dpml, ω_pml), ∂LDOS_∂ε .+  ∂LDOS_∂ω .* ∂ω_∂ε
end

export Improved_∇_ε_LDOS

function Just_Improved_LDOS(L, ε, ω, b; resolution=20, dpml=2, ω_pml=ω)
    A, x = Maxwell1d(L, ε, ω; resolution, dpml, ω_pml)
    ω₀ = sqrt(Arnoldi_eig(A, ε, ω, b)[1])
    A₀, _ = Maxwell1d(L, ε, real(ω₀); ω_pml=ω)
    LDOS, _ = ∇_ε_LDOS(A₀, b, real(ω₀))
    
    return LDOS
end

export Just_Improved_LDOS