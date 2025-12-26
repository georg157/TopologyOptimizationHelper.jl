##### functions.jl 


# Approximate the operator A that solves Au = b
function Maxwell1d(L, ε, ω; dpml=0.5, resolution=20, Rpml=1e-20, ω_pml=ω)
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
    x′ = ((0:M) .+ 0.5) .* dx # 1st-derivative grid points
    σ′ = [ξ < dpml ? σ₀*(dpml-ξ)^2 : ξ > L+dpml ? σ₀*(ξ-(L+dpml))^2 : 0.0 for ξ in x′]
    Σ′ = spdiagm(@. inv(1 + (im/ω_pml)*σ′))

    # 2nd derivative matrix and PML layer
    # Adjusting layer to make the matrix A symmetric
    x = x .- dpml
    A = sqrtΣ * D' * Σ′ * D * sqrtΣ - spdiagm(ω^2 .* ε)

    return A, x  
end
export Maxwell1d


function LDOS_Optimize(L, ε, ω, b; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_width=L, fixed_width=0)
    A, x = Maxwell1d(L, ε, ω; dpml, resolution, Rpml)
    D² = A + spdiagm(ω^2 .* ε)
    M = length(x)
    LDOS_vals = Float64[]
    omegas = ComplexF64[]
    
    function LDOS_obj(ε, grad)
        A = D² - spdiagm(ω^2 .* ε)
        LDOS, ∇LDOS = ∇_ε_LDOS(A, ω, b)
        grad .= ∇LDOS
        push!(LDOS_vals, LDOS)

        A_now, _ = Maxwell1d(L, ε, ω; resolution, dpml, Rpml)
        ω₀_now = sqrt(Arnoldi_eig(A_now, ε, ω, b)[1])
        push!(omegas, ω₀_now)
        
        return LDOS
    end

    design_indices = (-design_width/2 .< x .- mean(x) .< design_width/2) .& .!(-fixed_width/2 .< x .- mean(x) .< fixed_width/2)
    ub = copy(ε)
    lb = copy(ε)
    ub[design_indices] .= 12
    lb[design_indices] .= 1
    opt = Opt(:LD_CCSAQ, M)
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = LDOS_obj

    (LDOS_opt, ε_opt, ret) = optimize(opt, ε)
    A_opt, _ = Maxwell1d(L, ε_opt, ω; resolution, dpml, Rpml)
    ω₀_opt = sqrt(Arnoldi_eig(A_opt, ε_opt, ω, b)[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt
    
    return LDOS_opt, ε_opt, LDOS_vals, omegas, x
end
export LDOS_Optimize


# Our proposed theoretical, near optimal solution to the system above
function Fabry_Perot_epsilon(L, x; λ=1, n=round(2*L/λ - 2))
    if x > L || x < 0 || abs(x - L/2) < λ/4
        return 1 
    end

    d1 = λ/4
    d12 = d1/sqrt(12)
    period = d1 + d12

    for k in 0:n
        s_k = λ/4 + k * period

        if s_k < abs(x - L/2) <= s_k + d12
            return 12
        end
    end

    return 1
end
export Fabry_Perot_epsilon


function mod_LDOS_Optimize(L, ε, ω, b, x₀; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, ω_pml=ω, design_width=L)
    A, x = Maxwell1d(L, ε, ω; dpml, resolution, Rpml)
    D² = A + ω^2 .* spdiagm(ε)
    M = length(x)
    LDOS_vals = Float64[]
    omegas = ComplexF64[]
    
    function mod_LDOS_obj(ε, grad)
        E = spdiagm(ε)
        E⁻¹ = spdiagm(1 ./ ε)
        A = D² - real(ω)^2 .* E
        C = E⁻¹ * A
        LU = lu(C)
        vals, vecs, _ = eigsolve(z -> LU \ z, x₀, 1, :LM, Arnoldi())
        vals = sqrt.(1 ./ vals .+ ω^2)
        ω₀, u₀ = vals[1], vecs[1]
        A₀ = D² - real(ω₀)^2 .* E
        v = A₀ \ b
        w = conj.(v)
        LDOS = -imag(v' * b)

        if !isempty(grad) 
            ∂LDOS_∂ε = real(ω₀)^2 * imag.(v.^2)
            ∂LDOS_∂ω = -2real(ω₀) * imag(v' * E * w)
            ∂ω_∂ε = -ω₀ .* u₀.^2 ./ 2sum(u₀.^2 .* ε)
            ∇LDOS = ∂LDOS_∂ε .+  ∂LDOS_∂ω .* real.(∂ω_∂ε)

            grad .= ∇LDOS
        end

        push!(LDOS_vals, LDOS)
        push!(omegas, ω₀)
        
        return LDOS
    end

    function freq_constraint(ε, grad)
        ω₀, ∂ω_∂ε = Eigengradient(A, ε, ω, x₀)
        if !isempty(grad) 
            grad .= -real.(∂ω_∂ε)
        end

        return ω - real(ω₀)
    end

    design_indices = -design_width / 2 .< x .- mean(x) .< design_width / 2
    ub = ones(M)
    ub[design_indices] .= 12
    
    opt = Opt(:LD_CCSAQ, M)
    # opt.params["verbosity"] = 1
    opt.lower_bounds = 1
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = mod_LDOS_obj
    # opt.initial_step = 1e-3
    inequality_constraint!(opt, freq_constraint)


    (LDOS_opt, ε_opt, ret) = optimize(opt, ε)
    ω₀ = omegas[end]
    Q = -real(ω₀) / 2imag(ω₀)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀
    @show Q
    
    return LDOS_opt, ε_opt, LDOS_vals, omegas, x
end
export mod_LDOS_Optimize