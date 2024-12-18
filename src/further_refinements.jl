##### further_refinements.jl

function log_LDOS_Optimize2d(Lx, Ly, ε, ω, b, x₀; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx,Ly))
    A, x, y = Maxwell2d(Lx, Ly, ε, ω; dpml, resolution, Rpml)
    ε = vec(ε)
    b = vec(b)
    D² = A + ω^2 .* spdiagm(ε)
    M, N = length(x), length(y)
    log_LDOS_vals = Float64[]
    log_omegas = ComplexF64[]
    
    function log_LDOS_obj(ε, grad)
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

        push!(log_LDOS_vals, log_LDOS)
        push!(log_omegas, ω₀)

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
    opt.max_objective = log_LDOS_obj
    opt.initial_step = 1e-3
    inequality_constraint!(opt, freq_constraint)

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