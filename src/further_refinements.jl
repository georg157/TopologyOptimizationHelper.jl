##### further_refinements.jl

function log_LDOS_Optimize2d(Lx, Ly, ε, ω, b; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx, Ly), α=0)
    A, x, y = Maxwell2d(Lx, Ly, ε, ω; dpml, resolution, Rpml)
    ε = vec(ε)
    b = vec(b)
    D² = A + spdiagm(ω^2 .* ε)
    M, N = length(x), length(y)
    log_LDOS_vals = Float64[]
    log_omegas = ComplexF64[]
    
    function log_LDOS_obj(ε, grad)
        A = D² - spdiagm(ω^2 .* ε .* (1 + α * im))
        LDOS, ∇LDOS = ∇_ε_LDOS(A, ω, b; α)
        log_LDOS = log(abs(LDOS))
        grad .= ∇LDOS / LDOS
        push!(log_LDOS_vals, log_LDOS)

        A_now, _, _ = Maxwell2d(Lx, Ly, reshape(ε, N,M), ω; resolution)
        ω₀_now = sqrt(Arnoldi_eig(A_now, ε, ω, vec(b))[1])
        push!(log_omegas, ω₀_now)
        return log_LDOS
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
    opt.max_objective = log_LDOS_obj

    (log_LDOS_opt, log_ε_opt, ret) = optimize(opt, ε)
    A_opt, _, _ = Maxwell2d(Lx, Ly, log_ε_opt, ω; resolution)
    ω₀_opt = sqrt(Arnoldi_eig(A_opt, vec(log_ε_opt), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt
    
    return log_LDOS_opt, reshape(log_ε_opt, N,M), log_LDOS_vals, log_omegas, x, y
end
export log_LDOS_Optimize2d


function mod_log_LDOS_Optimize2d(Lx, Ly, ε, ω, b, x₀; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx,Ly))
    A, x, y = Maxwell2d(Lx, Ly, ε, ω; dpml, resolution, Rpml)
    ε = vec(ε)
    b = vec(b)
    D² = A + ω^2 .* spdiagm(ε)
    M, N = length(x), length(y)
    mod_log_LDOS_vals = Float64[]
    mod_log_omegas = ComplexF64[]
    
    function mod_log_LDOS_obj(ε, grad)
        E = spdiagm(ε)
        E⁻¹ = spdiagm(1 ./ ε)
        A = D² - ω^2 .* E
        C = E⁻¹ * A
        LU = lu(C)
        vals, vecs, _ = eigsolve(z -> LU \ z, x₀, 1, :LM, Arnoldi(tol=1e-4, maxiter=5000))
        vals = sqrt.(1 ./ vals .+ ω^2)
        ω₀, u₀ = vals[1], vecs[1]
        A₀ = D² - real(ω₀)^2 .* E
        v = A₀ \ b
        w = conj.(v)
        LDOS = -imag(v' * b)
        log_LDOS = log(abs(LDOS))
        
        reals = real.(vals)
        imaginaries = imag.(vals)
        if !isempty(grad) 
            ∂log_LDOS_∂ε = -imag.(real(ω₀)^2 .* w.^2) / LDOS
            ∂log_LDOS_∂ω = -imag(2real(ω₀) .* sum(w.^2 .* ε)) / LDOS
            ∂ω_∂ε = -ω₀ .* u₀.^2 ./ 2sum(u₀.^2 .* ε)
            ∇LDOS = ∂log_LDOS_∂ε .+  ∂log_LDOS_∂ω .* real.(∂ω_∂ε)

            grad .= ∇LDOS
        end

        push!(mod_log_LDOS_vals, log_LDOS)
        push!(mod_log_omegas, ω₀)

        return log_LDOS
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
    opt.max_objective = mod_log_LDOS_obj
    opt.initial_step = 1e-3
    inequality_constraint!(opt, freq_constraint)

    (mod_log_LDOS_opt, mod_log_ε_opt, ret) = optimize(opt, ε)
    A_opt, _, _ = Maxwell2d(Lx, Ly, mod_log_ε_opt, ω; resolution)
    ω₀_opt = sqrt(Arnoldi_eig(A_opt, vec(mod_log_ε_opt), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt
    
    return mod_log_LDOS_opt, reshape(mod_log_ε_opt, N,M), mod_log_LDOS_vals, mod_log_omegas, x, y
end
export mod_log_LDOS_Optimize2d


function smooth_LDOS_Optimize2d(Lx, Ly, ε, ω, b; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx, Ly), fR=0.1, β=5, η=0.5)
    A, x, y = Maxwell2d(Lx, Ly, ε, ω; dpml, resolution, Rpml)
    ε = vec(ε)
    b = vec(b)
    D² = A + spdiagm(ω^2 .* ε)
    M, N = length(x), length(y)
    smooth_LDOS_vals = Float64[]
    smooth_omegas = ComplexF64[]
    
    design_x, design_y = design_dimensions
    size_x,size_y = design_dimensions .* resolution

    x_indices = -design_x / 2 .< x .- mean(x) .< design_x / 2
    y_indices = -design_y / 2 .< y .- mean(y) .< design_y / 2
    ub = ones(N,M)
    ub[y_indices, x_indices] .= 12
    ub = vec(ub)
    design_inds = findall(ub .== 12)

    pad_x_dpml = Lx - design_x + 2dpml
    pad_y_dpml = Ly - design_y + 2dpml

    master_pad = ones(N*M)
    
    function LDOS_obj(ε, grad)
        ε = ε[design_inds]
        ζ, back = Zygote.pullback(z -> my_smoother(size_x, size_y, z; resolution, fR, β, η), ε)
        master_pad[design_inds] = ζ
        ζ_padded = master_pad
        
        A = D² - spdiagm(ω^2 .* ζ_padded)
        LDOS, ∇LDOS = ∇_ε_LDOS(A, ω, b)
        ∇LDOS = ∇LDOS[design_inds]
        grad .= 0.0
        grad[design_inds] .= back(∇LDOS)[1]
        push!(smooth_LDOS_vals, LDOS)

        A_now, _, _ = Maxwell2d(Lx, Ly, reshape(ζ_padded, N,M), ω; resolution)
        ω₀_now = sqrt(Arnoldi_eig(A_now, ζ_padded, ω, vec(b))[1])
        push!(smooth_omegas, ω₀_now)

        return LDOS
    end
    
    opt = Opt(:LD_CCSAQ, M * N)
    opt.lower_bounds = 1
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = LDOS_obj

    (smooth_LDOS_opt, ε_opt, ret) = optimize(opt, ε)
    ε_opt = ε_opt[design_inds]
    smooth_ε_opt = my_smoother(size_x, size_y, ε_opt; resolution, fR, β, η)
    # smooth_ε_opt = reshape(smooth_ε_opt, size_y,size_x)
    # smooth_ε_opt = vec(pad_matrix(smooth_ε_opt, pad_x_dpml, pad_y_dpml, 1; resolution))
    master_pad[design_inds] = smooth_ε_opt
    A_opt, _, _ = Maxwell2d(Lx, Ly, master_pad, ω; resolution)
    ω₀_opt = sqrt(Arnoldi_eig(A_opt, vec(master_pad), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt
    
    return smooth_LDOS_opt, reshape(master_pad, N,M), smooth_LDOS_vals, smooth_omegas, x, y
end
export smooth_LDOS_Optimize2d


function mod_smooth_Optimize2d(Lx, Ly, ε, ω, b, x₀; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx,Ly), fR=0.1, β=5, η=0.5, sgn=1)   
    A, x, y = Maxwell2d(Lx, Ly, ε, ω; dpml, resolution, Rpml)
    ε = vec(ε)
    b = vec(b)
    D² = A + ω^2 .* spdiagm(ε)
    M, N = length(x), length(y)
    mod_smooth_LDOS_vals = Float64[]
    mod_smooth_omegas = ComplexF64[]

    design_x, design_y = design_dimensions
    size_x,size_y = design_dimensions .* resolution   

    x_indices = -design_x / 2 .< x .- mean(x) .< design_x / 2
    y_indices = -design_y / 2 .< y .- mean(y) .< design_y / 2
    ub = ones(N,M)
    ub[y_indices, x_indices] .= 12
    ub = vec(ub)
    design_inds = findall(ub .== 12)

    pad_x_dpml = Lx - design_x + 2dpml
    pad_y_dpml = Ly - design_y + 2dpml

    master_pad = ones(N*M)

    function mod_smooth_LDOS_obj(ε, grad)
        ε = ε[design_inds]
        ζ, back = Zygote.pullback(z -> my_smoother(size_x, size_y, z; resolution, fR, β, η), ε)
        # ζ = reshape(ζ, size_y,size_x)
        # ζ_padded = pad_matrix(ζ, pad_x_dpml, pad_y_dpml, 1; resolution)
        # ζ_padded = vec(ζ_padded)
        master_pad[design_inds] = ζ
        ζ_padded = master_pad

        E = spdiagm(ζ_padded)
        E⁻¹ = spdiagm(1 ./ ζ_padded)
        A = D² - ω^2 .* E
        C = E⁻¹ * A
        LU = lu(C)
        vals, vecs, _ = eigsolve(z -> LU \ z, x₀, 1, :LM, Arnoldi(tol=1e-4, maxiter=5000))
        vals = sqrt.(1 ./ vals .+ ω^2)
        ω₀, u₀ = vals[1], vecs[1]
        A₀ = D² - real(ω₀)^2 .* E
        v = A₀ \ b
        w = conj.(v)
        LDOS = -imag(b' * w)
        
        if !isempty(grad) 
            ∂LDOS_∂ζ = -imag.(real(ω₀)^2 .* w.^2)
            ∂LDOS_∂ω = -imag(2real(ω₀) .* sum(w.^2 .* ζ_padded))
            ∂ω_∂ζ = -ω₀ .* u₀.^2 ./ 2sum(u₀.^2 .* ζ_padded)
            ∇_ζLDOS = ∂LDOS_∂ζ .+  ∂LDOS_∂ω .* real.(∂ω_∂ζ) 

            ∇_ζLDOS = ∇_ζLDOS[design_inds]
            grad .= 0.0 
            grad[design_inds] .= back(∇_ζLDOS)[1]
        end

        push!(mod_smooth_LDOS_vals, LDOS)
        push!(mod_smooth_omegas, ω₀)
        
        return LDOS
    end

    function freq_constraint(ε, grad)
        ε = ε[design_inds]
        ζ, back = Zygote.pullback(z -> my_smoother(size_x, size_y, z; resolution, fR, β, η), ε)
        # ζ = reshape(ζ, size_y,size_x)
        # ζ_padded = pad_matrix(ζ, pad_x_dpml, pad_y_dpml, 1; resolution)
        # ζ_padded = vec(ζ_padded)
        master_pad[design_inds] = ζ
        ζ_padded = master_pad

        E = spdiagm(ζ_padded)
        A = D² - ω^2 .* E
        ω₀, ∂ω_∂ε = Eigengradient(A, ζ_padded, ω, x₀)

        if !isempty(grad)
            ∂ω_∂ε = ∂ω_∂ε[design_inds]
            grad .= 0.0
            grad[design_inds] .= -sgn * back(real.(∂ω_∂ε))[1]
        end

        return sgn * (ω - real(ω₀))
    end

    opt = Opt(:LD_CCSAQ, M * N)
    #opt.params["verbosity"] = 1
    opt.lower_bounds = 1
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = mod_smooth_LDOS_obj
    # opt.initial_step = 1e-3
    inequality_constraint!(opt, freq_constraint)

    (mod_smooth_LDOS_opt, mod_ε_opt, ret) = optimize(opt, ε)
    mod_ε_opt = mod_ε_opt[design_inds]
    mod_smooth_ε_opt = my_smoother(size_x, size_y, mod_ε_opt; resolution, fR, β, η)
    mod_smooth_ε_opt = reshape(mod_smooth_ε_opt, size_y,size_x)
    mod_smooth_ε_opt = vec(pad_matrix(mod_smooth_ε_opt, pad_x_dpml, pad_y_dpml, 1; resolution))
    A_opt, _, _ = Maxwell2d(Lx, Ly, mod_smooth_ε_opt, ω; resolution)
    ω₀_opt = sqrt(Arnoldi_eig(A_opt, vec(mod_smooth_ε_opt), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt

    return mod_smooth_LDOS_opt, reshape(mod_smooth_ε_opt, N,M), mod_smooth_LDOS_vals, mod_smooth_omegas, x, y
end
export mod_smooth_Optimize2d