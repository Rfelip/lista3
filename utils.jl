using ForwardDiff

function get_derivatives(f)
    grad(x) = ForwardDiff.gradient(f, x) 
    hess(x) = ForwardDiff.hessian(f, x)
    return grad, hess   
end