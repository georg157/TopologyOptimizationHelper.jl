##### log_tests_shifted.jl

## ----- LOAD DEPENDENCIES ----- ##

using DelimitedFiles
using SparseArrays
using LinearAlgebra
using NLopt
using KrylovKit
using Zygote
using Statistics
using FFTW

## ----- FUNCTIONS ----- ##

function my_pad(X, pad_x, pad_y, pad_val; resolution=20)
    rows, cols = size(X)
    new_rows = round(Int, rows + pad_y * resolution)
    new_cols = round(Int, cols + pad_x * resolution)

    # Compute how much padding goes to each side
    pad_top = div(pad_y .* resolution, 2)
    pad_left = div(pad_x .* resolution, 2)

    # Create padded matrix filled with pad_val
    Y = ones(new_rows, new_cols) * pad_val

    # Copy X into the centered position
    top_edge = round(Int, pad_top + 1)
    bottom_edge = round(Int, pad_top + rows)
    left_edge = round(Int, pad_left + 1)
    right_edge = round(Int, pad_left + cols)
    
    Y[top_edge:bottom_edge,left_edge:right_edge] .= X

    return Y
end

function edge_pad(arr, pad)
    # fill sides
    left = repeat(arr[1, 1:end], outer=(1, pad[1][1]))'  # left side
    right = repeat(arr[end, 1:end], outer=(1, pad[1][2]))'  # right side
    top = repeat(arr[1:end, 1], outer=(1, pad[2][1]))  # top side
    bottom = repeat(arr[1:end, end], outer=(1, pad[2][2]))  # bottom side)

    # fill corners
    top_left = repeat([arr[1, 1]], outer=(pad[2][1], pad[1][1]))'  # top left
    top_right = repeat([arr[end, 1]], outer=(pad[2][1], pad[1][2]))'  # top right
    bottom_left = repeat([arr[1, end]], outer=(pad[2][2], pad[1][1]))'  # bottom left
    bottom_right = repeat([arr[end, end]], outer=(pad[2][2], pad[1][2]))'  # bottom right

    if (pad[1][1] > 0) & (pad[1][2] > 0) & (pad[2][1] > 0) & (pad[2][2] > 0)
        return hcat(
            vcat(top_left, top, top_right),
            vcat(left, arr, right),
            vcat(bottom_left, bottom, bottom_right),
        )
    elseif (pad[1][1] == 0) & (pad[1][2] == 0) & (pad[2][1] > 0) & (pad[2][2] > 0)
        return hcat((top, arr, bottom))
    elseif (pad[1][1] > 0) & (pad[1][2] > 0) & (pad[2][1] == 0) & (pad[2][2] == 0)
        return hcat((left, arr, right))
    elseif (pad[1][1] == 0) & (pad[1][2] == 0) & (pad[2][1] == 0) & (pad[2][2] == 0)
        return arr
    else
        throw(ErrorException("Not Implemented"))
    end
end

function proper_pad(arr, pad_to)
    pad_size = pad_to .- (2 .* size(arr)) .+ 1

    top = zeros(pad_size[1], size(arr,2))
    bottom = zeros(pad_size[1], size(arr,2) - 1)
    middle = zeros(pad_to[1], pad_size[2])

    top_left = arr[1:end, 1:end]
    top_right = reverse(arr[2:end, 1:end], dims=1)
    bottom_left = reverse(arr[1:end, 2:end], dims=2)
    bottom_right = reverse(
        reverse(arr[2:end, 2:end], dims=2), dims=1
    )  

    return hcat(
        vcat(top_left, top, top_right),
        middle,
        vcat(bottom_left, bottom, bottom_right),
    )
end

function centered_fft(arr, newshape)
    currshape = size(arr)
    startind = floor.((currshape .- newshape) ./ 2)
    endind = startind .+ newshape
    myslice = [Int(st)+1:Int(en) for (st, en) ∈ zip(startind, endind)]
    return arr[myslice...]
end

function convolve_fft(x; h)
    # Reshape image and kernel for fft convolution
    sx, sy = size(x)

    (kx, ky) = size(h)

    npx = Integer(
        ceil((2 * kx - 1) / sx)
    )  # 2*kx-1 is the size of a complete kernel in the x direction
    npy = Integer(
        ceil((2 * ky - 1) / sy)
    )  # 2*ky-1 is the size of a complete kernel in the y direction
    if npx % 2 == 0
        npx += 1  # Ensure npx is an odd number
    end
    if npy % 2 == 0
        npy += 1  # Ensure npy is an odd number
    end
    x = repeat(
        x, outer=(npx, npy)
    )
    x = edge_pad(
        x, ((0, 0), (0, 0))
    )  
    h = proper_pad(
        h, (npx * sx, npy * sy),
    )

    h = h ./ sum(h)
    xout = centered_fft(
        real(ifft(fft(x) .* fft(h))), [sx, sy]
    )
end

function conicfilter(radius, Lx, Ly, resolution)
    xv = 0: 1/resolution: ceil(2 * radius / Lx) * Lx / 2
    yv = 0: 1/resolution: ceil(2 * radius / Ly) * Ly / 2

    X = repeat(xv, 1, length(yv))
    Y = repeat(yv', length(xv), 1)
    mask = X.^2 + Y.^2 .< radius.^2
    iftruemask = (1 .- sqrt.(abs.(X.^2 .+ Y.^2)) ./ radius)
    iffalsemask = zeros(size(X))
    h = ifelse.(mask,iftruemask,iffalsemask)

    # Filter the response
    return h
end

function filtergrid(x, fR, Lx, Ly, resolution)
    # Get filter

    h = conicfilter(fR, Lx, Ly, resolution)
    
    xout = convolve_fft(x; h)

    return xout
end

function thresh(x, β, η)
    """
    Application of a thresholding by means of a smoothed Heaviside-like function.
    """
    if β == Inf
        return ifelse.(x .> η, 1.0, 0.0) 
    end
    
    return ((tanh(β * η) .+ tanh.(β .* (x .- η))) ./
            (tanh(β * η) .+ tanh(β * (1.0 .- η))))
end

function computegradient(input_array)
    # Compute gradients
    gradient_x = (input_array[3:end, :] .- input_array[1:end-2, :]) ./ 2.0
    gradient_x_left = reshape(input_array[2, :] .- input_array[1, :], 1, size(input_array, 2))
    gradient_x_right = reshape(input_array[end, :] .- input_array[end-1, :], 1, size(input_array, 2))
    gradient_x = vcat(gradient_x_left, gradient_x, gradient_x_right)

    gradient_y = (input_array[:, 3:end] .- input_array[:, 1:end-2]) ./ 2.0
    gradient_y_top = reshape(input_array[:, 2] .- input_array[:, 1], size(input_array, 1), 1)
    gradient_y_bottom = reshape(input_array[:, end] .- input_array[:, end-1], size(input_array, 1), 1)
    gradient_y = hcat(gradient_y_top, gradient_y, gradient_y_bottom)
    
    # Return gradients as a tuple
    return gradient_x, gradient_y
end

function mysmoother(size_x, size_y, ε; resolution=20, fR=0.1, β=5, η=0.5)
    size_y, size_x = round(Int, size_y), round(Int, size_x) 
    ε = reshape(ε, size_y,size_x)
    F₁ = (ε .- 1) ./ 11
    F₂ = filtergrid(F₁, fR, size_x/resolution, size_y/resolution, resolution)
    x_smoothed = reshape(F₂, size_y,size_x)

    dx = dy = 1 / resolution
    pixel_radius = 0.55dx
    x_projected = thresh(x_smoothed, β, η)

    x_grad_x, x_grad_y = computegradient(x_smoothed)
    x_grad_helper = (x_grad_x / dx) .^ 2 + (x_grad_y / dy) .^ 2
    nonzero_norm = abs.(x_grad_helper) .> 0
    x_grad_norm = sqrt.(ifelse.(nonzero_norm, x_grad_helper, 1))
    x_grad_norm_eff = ifelse.(nonzero_norm, x_grad_norm, 1)
    d = (η .- x_smoothed) ./ x_grad_norm_eff
    needs_smoothing = nonzero_norm .& (abs.(d) .<= pixel_radius)

    R = d ./ pixel_radius
    fill_factor = ifelse.(
        needs_smoothing,
        0.5 .- 15/16 * R .+ 5/8 * R .^ 3 .- 3 / 16 * R .^ 5,
        1,
    )
    x_minus = x_smoothed .- x_grad_norm * pixel_radius
    x_plus = x_smoothed .+ x_grad_norm * pixel_radius
    x_minus_eff_pert = (x_smoothed .* d .+ x_minus .* (pixel_radius .- d)) / pixel_radius
    x_minus_eff = ifelse.(
        (d .> 0),
        x_minus_eff_pert,
        x_minus,
    )
    x_plus_eff_pert = (-x_smoothed .* d + x_plus .* (pixel_radius .+ d)) / pixel_radius
    x_plus_eff = ifelse.(
        (d .> 0),
        x_plus,
        x_plus_eff_pert,
    )

    x_plus_eff_projected = thresh(x_plus_eff, β, η)
    x_minus_eff_projected = thresh(x_minus_eff, β, η)

    x_projected_smoothed = (1 .- fill_factor) .* x_minus_eff_projected .+ (fill_factor) .* x_plus_eff_projected
    final = vec(ifelse.(
        needs_smoothing,
        x_projected_smoothed,
        x_projected,
    ))

    return 11final .+ 1
end

function optimizer_2d(Lx, Ly, ε, ω, b; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx, Ly), fR=0.1, β=5, η=0.5)
    A, x, y = helmholtz(Lx, Ly, ε, ω; dpml, resolution, Rpml)
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

    iter = 1
    function LDOS_obj(ε, grad)
        ε = ε[design_inds]
        ζ, back = Zygote.pullback(z -> mysmoother(size_x, size_y, z; resolution, fR, β, η), ε)
        master_pad[design_inds] = ζ
        ζ_padded = master_pad
        A = D² - spdiagm(ω^2 .* ζ_padded)
        v = A \ b
        w = conj.(v)
        LDOS = -imag(b' * w)
        ∇LDOS = -imag.(ω^2 .* w.^2)
        ∇LDOS /= LDOS
        ∇LDOS = ∇LDOS[design_inds]
        grad .= 0.0
        grad[design_inds] .= back(∇LDOS)[1]
        push!(smooth_LDOS_vals, LDOS)

        A_now = A
        ω₀_now = sqrt(get_eigenpair(A_now, ζ_padded, ω, vec(b))[1])
        push!(smooth_omegas, ω₀_now)
        @show iter, LDOS
        iter += 1

        return log(LDOS)
    end
    
    opt = Opt(:LD_CCSAQ, M * N)
    opt.lower_bounds = 1
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = LDOS_obj

    (smooth_LDOS_opt, ε_opt, ret) = optimize(opt, ε)
    ε_opt = ε_opt[design_inds]
    smooth_ε_opt = mysmoother(size_x, size_y, ε_opt; resolution, fR, β, η)
    master_pad[design_inds] = smooth_ε_opt
    A_opt = D² - spdiagm(ω^2 * master_pad)
    ω₀_opt = sqrt(get_eigenpair(A_opt, vec(master_pad), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt
    
    return smooth_LDOS_opt, reshape(master_pad, N,M), smooth_LDOS_vals, smooth_omegas, x, y
end

function mod_optimizer_2d(Lx, Ly, ε, ω, b, x₀; dpml=0.5, resolution=20, Rpml=1e-20, ftol=1e-4, max_eval=500, design_dimensions=(Lx,Ly), fR=0.1, β=5, η=0.5, sgn=1)   
    A, x, y = helmholtz(Lx, Ly, ε, ω; dpml, resolution, Rpml)
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
    
    iter = 1
    function mod_smooth_LDOS_obj(ε, grad)
        ε = ε[design_inds]
        ζ, back = Zygote.pullback(z -> mysmoother(size_x, size_y, z; resolution, fR, β, η), ε)
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
            ∇_ζLDOS /= LDOS           # comment out for max_p LDOS
            ∇_ζLDOS = ∇_ζLDOS[design_inds]
            grad .= 0.0 
            grad[design_inds] .= back(∇_ζLDOS)[1]
        end

        push!(mod_smooth_LDOS_vals, LDOS)
        push!(mod_smooth_omegas, ω₀)
        @show iter, LDOS
        iter += 1
        
        # return LDOS
        return log(LDOS)
    end

    function freq_constraint(ε, grad)
        ε = ε[design_inds]
        ζ, back = Zygote.pullback(z -> mysmoother(size_x, size_y, z; resolution, fR, β, η), ε)
        master_pad[design_inds] = ζ
        ζ_padded = master_pad

        E = spdiagm(ζ_padded)
        A = D² - ω^2 .* E
        ω₀, ∂ω_∂ε = get_eigengradient(A, ζ_padded, ω, x₀)

        if !isempty(grad)
            ∂ω_∂ε = ∂ω_∂ε[design_inds]
            grad .= 0.0
            grad[design_inds] .= -sgn * back(real.(∂ω_∂ε))[1] / ω
        end

        return sgn * (1 - real(ω₀)/ω + 1e-2)
    end

    opt = Opt(:LD_CCSAQ, M * N)
    #opt.params["verbosity"] = 1
    opt.lower_bounds = 1
    opt.upper_bounds = ub
    opt.ftol_rel = ftol
    opt.maxeval = max_eval
    opt.max_objective = mod_smooth_LDOS_obj
    # opt.initial_step = 1e-1
    inequality_constraint!(opt, freq_constraint)

    (mod_smooth_LDOS_opt, mod_ε_opt, ret) = optimize(opt, ε)
    mod_ε_opt = mod_ε_opt[design_inds]
    mod_smooth_ε_opt = mysmoother(size_x, size_y, mod_ε_opt; resolution, fR, β, η)
    mod_smooth_ε_opt = reshape(mod_smooth_ε_opt, size_y,size_x)
    mod_smooth_ε_opt = vec(my_pad(mod_smooth_ε_opt, pad_x_dpml, pad_y_dpml, 1; resolution))
    A_opt = D² - spdiagm(ω^2 * master_pad)
    ω₀_opt = sqrt(get_eigenpair(A_opt, vec(mod_smooth_ε_opt), ω, vec(b))[1])
    Q_opt = -real(ω₀_opt) / 2imag(ω₀_opt)

    @show numevals = opt.numevals # the number of function evaluations
    @show ω₀_opt
    @show Q_opt

    return mod_smooth_LDOS_opt, reshape(mod_smooth_ε_opt, N,M), mod_smooth_LDOS_vals, mod_smooth_omegas, x, y
end

function helmholtz(Lx, Ly, ε, ω; dpml=0.5, resolution=20, Rpml=1e-20, ω_pml=ω)
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

function get_eigenpair(A, ε, ω, x₀)
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
    vals, vecs, info = eigsolve(z -> LU \ z, x₀, 3, :LM, Arnoldi())
    # If the optimizer ignores the constraint, use
    # EigSorter(λ -> real(sqrt(1 / λ + ω^2)) <= ω ? abs(λ) : -Inf, rev=true)
    # instead of :LM
    λₖ, yₖ = vals[1], vecs[1]
    return 1 / λₖ + ω^2, yₖ
end

function get_eigengradient(A, ε, ω, x₀)
    # We solve for ω₀² := μₖ and u₀ := yₖ
    ω₀², u₀ = get_eigenpair(A, ε, ω, x₀)
    ω₀ = sqrt(ω₀²)
    # Return gradient ∂ω₀/∂ε
    
    return ω₀, -ω₀ .* u₀.^2 ./ 2sum(u₀.^2 .* ε)
end

## ----- OUR SETUP ----- ##

Lx = 5
Ly = 5
res = 20
intermediate = 6.5ones(60,60)
ε = my_pad(intermediate, 3, 3, 1)
ω = π
N, M = size(ε)
b = zeros(N,M)
offset = -round(Int, res * (0.1 + 1.5))
b[N÷2 + offset,M÷2] = 1

## ----- UNSHIFTED INITIALIZATION ----- ##

_, ε_8, _, _, _, _ = optimizer_2d(Lx, Ly, ε, ω, b; resolution=res, ftol=0, max_eval=100, design_dimensions=(3,3), fR=0.2, β=8)
_, ε_16, _, _, _, _ = optimizer_2d(Lx, Ly, ε_8, ω, b; resolution=res, ftol=0, max_eval=200, design_dimensions=(3,3), fR=0.2, β=16)
_, ε_40, _, _, _, _ = optimizer_2d(Lx, Ly, ε_16, ω, b; resolution=res, ftol=0, max_eval=300, design_dimensions=(3,3), fR=0.2, β=40)
_, ε_init, _, _, _, _ = optimizer_2d(Lx, Ly, ε_40, ω, b; resolution=res, ftol=0, max_eval=400, design_dimensions=(3,3), fR=0.2, β=Inf)
writedlm("log_epsilon_init.txt", ε_init)

## ----- SHIFTED OPTIMIZATION (500,000 ITERATIONS) ----- ##

mod_LDOS_opt, mod_ε_opt, mod_LDOS_vals, mod_omegas, x, y = mod_optimizer_2d(Lx, Ly, ε_init, ω, b, vec(b); resolution=res, ftol=0, max_eval=500000, design_dimensions=(3,3), fR=0.2, β=Inf, sgn=1)
writedlm("log_mod_epsilon_opt.txt", mod_ε_opt)
mod_LDOS_vals =  vcat(fill.(mod_LDOS_vals, 2)...)
writedlm("log_mod_LDOS_vals.txt", mod_LDOS_vals)
mod_Qs = -real.(mod_omegas) ./ 2imag.(mod_omegas)
mod_Qs = vcat(fill.(mod_Qs, 2)...)
writedlm("log_mod_Qs.txt", mod_Qs)
