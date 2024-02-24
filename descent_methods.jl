using LinearAlgebra

function grad_desc(grad, α, x0, retsteps)
    x = x0
    steps = Vector{Vector{Float64}}()

    while !isapprox(norm(grad(x)), 0, atol = 1e-3)
        push!(steps, x)
        x = x - α * grad(x)
    end

    x_otim = x
    nsteps = length(steps)
    if retsteps
        return x_otim, nsteps, steps
    else
        return x_otim, nsteps
    end
end
function newthon_desc(grad, hess, α, x0, retsteps)
    x = x0
    steps = Vector{Vector{Float64}}()

    while !isapprox(norm(grad(x)), 0, atol = 1e-3)
        push!(steps, x)
        x = x - α * hess(x)/grad(x)
    end
    
    x_otim = x
    nsteps = length(steps)
    if retsteps
        return x_otim, nsteps, steps
    else
        return x_otim, nsteps
    end
end
function mod_newthon_desc(grad, hess, α, x0, retsteps)
    x = x0
    steps = Vector{Vector{Float64}}()

    while !isapprox(norm(grad(x)), 0, atol = 1e-3)
        push!(steps, x)
        x = x - α * (hess(x) + ϵ*I(length(x))) / grad(x)
    end
    
    x_otim = x
    nsteps = length(steps)
    if retsteps
        return x_otim, nsteps, steps
    else
        return x_otim, nsteps
    end
end
function conjug_grad_desc(grad, α, x0, retsteps)

    itr = 1
    x = x0

    steps = Vector{Vector{Float64}}()
    direc = Vector{Vector{Float64}}()

    push!(direc, -grad(x0))
    while !isapprox(norm(grad(x)), 0, atol = 1e-3)

        itr += 1
        push!(steps, x)

        x_km1 = grad(steps[itr - 1])
        x_k   = grad(x)

        cgrad = dot(x_k - x_km1, x_k) / dot(x_km1, x_k)
        x = x + α * (- x_k + cgrad * direc[itr - 1])
    end
    
    x_otim = x
    nsteps = length(steps)
    if retsteps
        return x_otim, nsteps, steps
    else
        return x_otim, nsteps
    end
end
