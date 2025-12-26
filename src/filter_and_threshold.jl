##### filter_and_threshold.jl
### THANKS IAN && BENAT ~~~

deflatten(pvec; gridx, gridy) = reshape(pvec,(length(gridx), length(gridy)))

function pad_matrix(X, pad_x, pad_y, pad_val; resolution=20)
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
export pad_matrix

function edgepad(arr, pad)
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
export edgepad


function properpad(arr, pad_to)
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
export properpad


function centeredfft(arr, newshape)
    currshape = size(arr)
    startind = floor.((currshape .- newshape) ./ 2)
    endind = startind .+ newshape
    myslice = [Int(st)+1:Int(en) for (st, en) ∈ zip(startind, endind)]
    return arr[myslice...]
end
export centeredfft


function convolvefft(x; h)
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
    x = edgepad(
        x, ((0, 0), (0, 0))
    )  
    h = properpad(
        h, (npx * sx, npy * sy),
    )

    h = h ./ sum(h)
    xout = centeredfft(
        real(ifft(fft(x) .* fft(h))), [sx, sy]
    )
end
export convolvefft


function conic_filter(radius, Lx, Ly, resolution)
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
export conic_filter


function filter_grid(x, fR, Lx, Ly, resolution)
    # Get filter

    h = conic_filter(fR, Lx, Ly, resolution)
    
    xout = convolvefft(x; h)

    return xout
end
export filter_grid


function threshold(x, β, η)
    """
    Application of a thresholding by means of a smoothed Heaviside-like function.
    """
    if β == Inf
        return ifelse.(x .> η, 1.0, 0.0) 
    end
    
    return ((tanh(β * η) .+ tanh.(β .* (x .- η))) ./
            (tanh(β * η) .+ tanh(β * (1.0 .- η))))
end
export threshold


function compute_gradient(input_array)
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
export compute_gradient


function my_smoother(size_x, size_y, ε; resolution=20, fR=0.1, β=5, η=0.5)
#     size_y, size_x = round(Int, size_y), round(Int, size_x) 
#     ε = reshape(ε, size_y,size_x)
#     dx = dy = 1 / resolution
#     pixel_radius = dx / 2

#     x_grad_x, x_grad_y = compute_gradient(ε)
#     x_grad_helper = (x_grad_x / dx) .^ 2 + (x_grad_y / dy) .^ 2
#     nonzero_norm = abs.(x_grad_helper) .> 0
#     x_grad_norm = sqrt.(ifelse.(nonzero_norm, x_grad_helper, 1))
#     x_grad_norm_eff = ifelse.(nonzero_norm, x_grad_norm, 1)

#     ρ = (ε .- 1) ./ 11
#     x_smoothed = filter_grid(ρ, fR, size_x/resolution, size_y/resolution, resolution)
#     x_projected = threshold(x_smoothed, β, η)

#     d = (η .- x_smoothed) ./ x_grad_norm_eff
#     needs_smoothing = nonzero_norm .& (abs.(d) .<= pixel_radius)

#     R = d ./ pixel_radius
#     fill_factor = ifelse.(
#         needs_smoothing,
#         0.5 .- 15/16 * R .+ 5/8 * R .^ 3 .- 3 / 16 * R .^ 5,
#         1,
#     )

#     x_minus = x_smoothed .- x_grad_norm * pixel_radius
#     x_plus = x_smoothed .+ x_grad_norm * pixel_radius
#     x_minus_eff_pert = (x_smoothed .* d .+ x_minus .* (pixel_radius .- d)) / pixel_radius
#     x_minus_eff = ifelse.((d .> 0), x_minus_eff_pert, x_minus)

#     x_plus_eff_pert = (-x_smoothed .* d + x_plus .* (pixel_radius .+ d)) / pixel_radius
#     x_plus_eff = ifelse.((d .> 0), x_plus, x_plus_eff_pert)

#     x_plus_eff_projected = threshold(x_plus_eff, β, η)
#     x_minus_eff_projected = threshold(x_minus_eff, β, η)

#     x_projected_smoothed = (1 .- fill_factor) .* x_minus_eff_projected .+ (fill_factor) .* x_plus_eff_projected
#     x_projected_smoothed = 11 * x_projected_smoothed .+ 1

#     x_projected = 11 * x_projected .+ 1

#     return vec(ifelse.(
#         needs_smoothing,
#         x_projected_smoothed,
#         x_projected,
#     ))    
# end

    size_y, size_x = round(Int, size_y), round(Int, size_x) 
    ε = reshape(ε, size_y,size_x)
    F₁ = (ε .- 1) ./ 11
    F₂ = filter_grid(F₁, fR, size_x/resolution, size_y/resolution, resolution)
    x_smoothed = reshape(F₂, size_y,size_x)

    dx = dy = 1 / resolution
    pixel_radius = 0.55dx
    # pixel_radius = fR
    x_projected = threshold(x_smoothed, β, η)

    x_grad_x, x_grad_y = compute_gradient(x_smoothed)
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

    x_plus_eff_projected = threshold(x_plus_eff, β, η)
    x_minus_eff_projected = threshold(x_minus_eff, β, η)

    x_projected_smoothed = (1 .- fill_factor) .* x_minus_eff_projected .+ (fill_factor) .* x_plus_eff_projected
    final = vec(ifelse.(
        needs_smoothing,
        x_projected_smoothed,
        x_projected,
    ))

    return 11final .+ 1
end

# function my_smoother(size_x, size_y, ε; resolution=20, fR=0.5, β=5, η=0.5)
#     size_y, size_x = round(Int, size_y), round(Int, size_x) 
#     ε = reshape(ε, size_y,size_x)
#     F₁ = (ε .- 1) ./ 11
#     F₂ = filter_grid(F₁, fR, size_x/resolution, size_y/resolution, resolution)
#     F₃ = threshold(F₂, β, η)
#     ε_smooth = 11F₃ .+ 1
#     return vec(ε_smooth)
# end
export my_smoother