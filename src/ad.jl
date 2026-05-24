"""
    correct(c, zhat) -> zstar

The differentiable forward map of a constraint layer: returns the corrected
output `zstar` for constraint `c` and raw input `zhat`. A `ChainRulesCore.rrule`
makes this usable inside reverse-mode AD such as Zygote, so the layer can sit
anywhere in a model and still train end to end.

The backward pass uses the adjoint VJP of Sections 4.2/4.4 of PLAN.md, which
differentiates the optimality conditions rather than the solver iterations. On a
non-success projection no gradient is claimed (I10): the pullback errors instead
of silently returning a wrong or zero gradient.
"""
correct(c, zhat) = project(c, zhat).zstar

function ChainRulesCore.rrule(::typeof(correct), c, zhat)
    res = project(c, zhat)
    function correct_pullback(zbar)
        gx = vjp(c, res, unthunk(zbar))
        return (NoTangent(), NoTangent(), gx)
    end
    return res.zstar, correct_pullback
end
