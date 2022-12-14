using Turing
using DifferentialEquations
using StatsPlots
using LinearAlgebra: I

"
Multivariate LogNormal distribution with fixed sigma
"
MvLN(x) = MvLogNormal(log.(x), 0.1 * I)

"
Lotka-Voltera Competition model for arbitrary number of species.
Function is based on the following Differential Equation system.

```
Ṅ = N ⊙ r ⊙ (1 - (A × N))
```
"
function mult_spec(dN, N, p, t)
    r, A = p
    dims = length(r)
    dN .= N .* r .* (1 .- reshape(A, (dims, dims))' * N)
    return nothing
end


"
Simulate population of arbitrary number of species
given an model and parameters. Default values are given
for competitive Lotka-Voltera model with coexistence.

"
function simulate_spec(
    prob_func;
    A = [[1 0.4]; [0.5 1]],
    u0 = [0.5, 0.3],
    r = [10, 12],
    t = LinRange(0, 1, 30),
    noise = 0.5,
    return_params = false,
)
    p = (r, A)
    tspan = (minimum(t), maximum(t))
    prob = ODEProblem(prob_func, u0, tspan, p)
    solution = Array(solve(prob, Tsit5(); saveat = t))
    data = max.(0, Array(solution) + randn(size(solution)) * (noise / 100))
    if return_params
        return data, prob, p
    else
        return data, prob
    end
end

"
Function to plot simulation data and ground truth system
"
function simulation_plotter(
    data,
    prob,
    params,
    t;
    sim_plot = missing,
    points = true,
    ln_kwargs = (),
    pt_kwargs = (),
)
    n_species = size(data)[1]
    labels = permutedims(["Species $x" for x = 1:n_species][:, :])
    colors = permutedims([1:n_species;;])

    sol = solve(prob, Tsit5(); p = params, saveat = t)
    if sim_plot === missing
        sim_plot = plot()
    end
    plot!(t, sol'; color = colors, label = labels, ln_kwargs...)
    if points
        scatter!(t, data'; color = colors, label = false, pt_kwargs...)
    end
    xlabel!("Time (AU)")
    ylabel!("Population (AU)")
    return sim_plot
end

"
Modular function to construct and return an ecological 
coexistence model with observed data and prior distributions
for parameters. If some values are previously known, a Dirac
(deterministic) distribution can be passed. 
"
function get_eco_model(
    data,
    prob,
    t,
    Aₚ,
    rₚ;
    σₚ = [2, 3],
    ADist = MvLN,
    rDist = MvLN,
    σDist = InverseGamma,
    Aₚkw = (),
    rₚkw = (),
    σₚkw = (),
    kₚkw = (),
)

    @model function eco_model(data, prob, t)
        A ~ ADist(Aₚ...; Aₚkw...)
        r ~ rDist(rₚ...; rₚkw...)
        σ ~ σDist(σₚ...; σₚkw...)
        p = (r, A)

        predicted = solve(prob, Tsit5(); p = p, saveat = t)
        data .~ Normal.(predicted, σ^2)
        return nothing
    end

    return eco_model(data, prob, t)
end


"
Simple helper function to obtain multiple sample chains for a given model.
"
function get_chain(
    model;
    progress = false,
    iterations = 1000,
    n_chains = 2,
    method = NUTS(0.65),
)
    chain = sample(model, method, MCMCThreads(), iterations, n_chains; progress = progress)
    return chain
end


"
Make predictions of model parameters given the posterior 
distribution of chain samples
"
function chain_retrodiction(chain, t; n_samples = 300, replace = false, show_plot = false)

    params = [string(x) for x in chain.name_map.parameters]
    A_p = [x for x in params if occursin("A[", x)]
    r_p = [x for x in params if occursin("r[", x)]
    posterior_samples = Array(sample(chain[vcat(A_p, r_p)], n_samples; replace = replace))
    A_ps = posterior_samples[:, 1:length(A_p)]
    r_ps = posterior_samples[:, 1+length(A_p):length(A_p)+length(r_p)]

    predicitons = Array{Float64}(undef, n_samples, length(r_p), length(t))
    for i = 1:size(A_ps)[1]
        predicitons[i, :, :] =
            Array(solve(prob, Tsit5(); p = (r_ps[i, :], A_ps[i, :]), saveat = t))
    end
    return predicitons
end


"
Plot model predictions along with data it was inferred from if specified.
"
function plot_retrodiction(predictions, t; data = missing, alpha = 0.2, colors = missing)
    n_species = size(predictions)[1]
    if colors === missing
        colors = permutedims([1:n_species;;])
    end
    predict_plot = plot()
    for i = 1:size(predictions)[1]
        plot!(t, predictions[i, :, :]'; alpha = alpha, color = colors, label = false)
    end
    if data !== missing
        labels = permutedims(["Species $x" for x = 1:n_species][:, :])
        scatter!(t, data'; color = colors, label = labels)
    end
    xlabel!("Time (AU)")
    ylabel!("Population (AU)")
    return predict_plot
end
