##### functions2d.jl

function Maxwell2d(Lx, Ly, ε, ω; dpml=0.5, resolution=20, Rpml=1e-20, ω_pml=ω)
    ε = vec(ε)
    # PML σ = σ₀ x²/dpml², with σ₀ chosen so that the round-trip reflection is Rpml
    σ₀ = -log(Rpml) / (4dpml/3)
    
    M = round(Int, (Lx+2dpml) * resolution)
    N = round(Int, (Ly+2dpml) * resolution)
    dx = (Lx+2dpml) / (M+1)
    dy = (Ly+2dpml) / (N+1)
    x = (1:M) * dx # x grid
    y = (1:N) * dy # y grid
    x′ = @. ((0:M) + 0.5) * dx # 1st-derivative grid points
    y′ = @. ((0:N) + 0.5) * dy
    
    # 1st-derivative matrices
    ox = ones(M) / dx
    oy = ones(N) / dy
    Dx = spdiagm(M+1,M, -1 => -ox, 0 => ox)
    Dy = spdiagm(N+1,N, -1 => -oy, 0 => oy)
    
    # PML complex "stretch" factors 1/(1+iσ/ω_pml) at both x and x' points:
    σx = [ξ < dpml ? σ₀*(dpml-ξ)^2 : ξ > Lx+dpml ? σ₀*(ξ-(Lx+dpml))^2 : 0.0 for ξ in x]
    sqrtΣx = spdiagm(@. sqrt(inv(1 + (im/ω_pml)*σx)))
    σx′ = [ξ < dpml ? σ₀*(dpml-ξ)^2 : ξ > Lx+dpml ? σ₀*(ξ-(Lx+dpml))^2 : 0.0 for ξ in x′]
    Σx′ = spdiagm(@. inv(1 + (im/ω_pml)*σx′))
    # similarly for y and y':
    σy = [ξ < dpml ? σ₀*(dpml-ξ)^2 : ξ > Ly+dpml ? σ₀*(ξ-(Ly+dpml))^2 : 0.0 for ξ in y]
    sqrtΣy = spdiagm(@. sqrt(inv(1 + (im/ω_pml)*σy)))
    σy′ = [ξ < dpml ? σ₀*(dpml-ξ)^2 : ξ > Ly+dpml ? σ₀*(ξ-(Ly+dpml))^2 : 0.0 for ξ in y′]
    Σy′ = spdiagm(@. inv(1 + (im/ω_pml)*σy′))
    
    # stretched 2nd-derivative matrices
    D2x = sqrtΣx * Dx' * Σx′ * Dx * sqrtΣx
    D2y = sqrtΣy * Dy' * Σy′ * Dy * sqrtΣy
    
    # combine x and y with Kronecker products
    Ix = spdiagm(ones(M))
    Iy = spdiagm(ones(N))
    x = x .- dpml
    y = y .- dpml
    return kron(Ix, D2y) + kron(D2x, Iy) - spdiagm(ω^2 .* ε), x, y
end
export Maxwell2d


function LDOS_Optimize2d(Lx, Ly, ε, ω, b; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx, Ly), α=0)
    A, x, y = Maxwell2d(Lx, Ly, ε, ω; dpml, resolution, Rpml)
    ε = vec(ε)
    b = vec(b)
    D² = A + spdiagm(ω^2 .* ε)
    M, N = length(x), length(y)
    LDOS_vals = Float64[]
    omegas = ComplexF64[]
    
    function LDOS_obj(ε, grad)
        A = D² - spdiagm(ω^2 .* ε .* (1 + α * im))
        LDOS, ∇LDOS = ∇_ε_LDOS(A, ω, b; α)
        grad .= ∇LDOS
        push!(LDOS_vals, LDOS)

        A_now, _, _ = Maxwell2d(Lx, Ly, reshape(ε, N,M), ω; resolution)
        ω₀_now = sqrt(Arnoldi_eig(A_now, ε, ω, vec(b))[1])
        push!(omegas, ω₀_now)
        return LDOS
    end

    design_x, design_y = design_dimensions
    x_indices = -design_x / 2 .< x .- mean(x) .< design_x / 2
    y_indices = -design_y / 2 .< y .- mean(y) .< design_y / 2
    ub = ones(N,M)
    ub[y_indices, x_indices] .= 12
    ub = vec(ub)
    
    opt = Opt(:LD_CCSAQ, M * N)
    opt.lower_bounds = 1
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = LDOS_obj

    (LDOS_opt, ε_opt, ret) = optimize(opt, ε)
    A_opt, _, _ = Maxwell2d(Lx, Ly, ε_opt, ω; resolution)
    ω₀_opt = sqrt(Arnoldi_eig(A_opt, vec(ε_opt), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt
    
    return LDOS_opt, reshape(ε_opt, N,M), LDOS_vals, omegas, x, y
end
export LDOS_Optimize2d


function mod_LDOS_Optimize2d(Lx, Ly, ε, ω, b, x₀; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx,Ly))
    A, x, y = Maxwell2d(Lx, Ly, ε, ω; dpml, resolution, Rpml)
    ε = vec(ε)
    b = vec(b)
    D² = A + ω^2 .* spdiagm(ε)
    M, N = length(x), length(y)
    mod_LDOS_vals = Float64[]
    mod_omegas = ComplexF64[]
    
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
        
        reals = real.(vals)
        imaginaries = imag.(vals)
        if !isempty(grad) 
            ∂LDOS_∂ε = -imag.(real(ω₀)^2 .* w.^2)
            ∂LDOS_∂ω = -imag(2real(ω₀) .* sum(w.^2 .* ε))
            ∂ω_∂ε = -ω₀ .* u₀.^2 ./ 2sum(u₀.^2 .* ε)
            ∇LDOS = ∂LDOS_∂ε .+  ∂LDOS_∂ω .* real.(∂ω_∂ε)

            grad .= ∇LDOS
        end

        push!(mod_LDOS_vals, LDOS)
        push!(mod_omegas, ω₀)

        return LDOS
    end

    function freq_constraint(ε, grad)
        ω₀, ∂ω_∂ε = Eigengradient(A, ε, ω, x₀)
        if !isempty(grad) 
            grad .= -real.(∂ω_∂ε)
        end

        return ω - real(ω₀)
    end

    design_x, design_y = design_dimensions
    x_indices = -design_x / 2 .< x .- mean(x) .< design_x / 2
    y_indices = -design_y / 2 .< y .- mean(y) .< design_y / 2
    ub = ones(N,M)
    ub[y_indices, x_indices] .= 12
    ub = vec(ub)
    
    opt = Opt(:LD_CCSAQ, M * N)
    # opt.params["verbosity"] = 1
    opt.lower_bounds = 1
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = mod_LDOS_obj
    opt.initial_step = 1e-3
    inequality_constraint!(opt, freq_constraint)

    (mod_LDOS_opt, mod_ε_opt, ret) = optimize(opt, ε)
    A_opt, _, _ = Maxwell2d(Lx, Ly, mod_ε_opt, ω; resolution)
    ω₀_opt = sqrt(Arnoldi_eig(A_opt, vec(mod_ε_opt), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt
    
    return mod_LDOS_opt, reshape(mod_ε_opt, N,M), mod_LDOS_vals, mod_omegas, x, y
end
export mod_LDOS_Optimize2d