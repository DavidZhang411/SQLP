using SQLP, JuMP, Gurobi

using SparseArrays
function get_coefficients(ex::AffExpr, sp::SQLP.spStageProblem)
    last = spzeros(length(sp.last_stage_vars))
    current = spzeros(length(sp.current_stage_vars))

    for (i, last_var) in enumerate(sp.last_stage_vars)
        if last_var in keys(ex.terms)
            last[i] = ex.terms[last_var]
        end
    end

    for (i, current_var) in enumerate(sp.current_stage_vars)
        if current_var in keys(ex.terms)
            current[i] = ex.terms[current_var]
        end
    end

    return last, current
end

function projection!(sp_proj::SQLP.spStageProblem, x::Vector{Float64})
    xref = sp_proj.current_stage_vars
    obj = sum((xref - x).^2)
    set_objective(sp_proj.model, MIN_SENSE, obj)
    set_start_value.(xref, x)

    optimize!(sp_proj.model)

    return value.(xref)
end

function norm(x)
    return x'*x
end

# cor = SQLP.read_cor(joinpath("spInput", "lands", "lands.cor"))
# tim = SQLP.read_tim(joinpath("spInput", "lands", "lands.tim"))
# sto = SQLP.read_sto(joinpath("spInput", "lands", "lands.sto"))
prob_name = "lands3"
cor = SQLP.read_cor(joinpath("spInput", "$prob_name", "$prob_name.cor"))
tim = SQLP.read_tim(joinpath("spInput", "$prob_name", "$prob_name.tim"))
sto = SQLP.read_sto(joinpath("spInput", "$prob_name", "$prob_name.sto"))

# the stage templates
sp1 = SQLP.get_smps_stage_template(cor, tim, 1)
sp1_proj = SQLP.get_smps_stage_template(cor, tim, 1)

sp2 = SQLP.get_smps_stage_template(cor, tim, 2)

set_optimizer(sp1.model, Gurobi.Optimizer)
set_optimizer(sp2.model, Gurobi.Optimizer)
set_optimizer(sp1_proj.model, Gurobi.Optimizer)

set_silent(sp1.model)
set_silent(sp2.model)
set_silent(sp1_proj.model)


# Initialize
direction = zeros(length(sp1.current_stage_vars))
scenario_set = []

optimize!(sp1.model)
x = value.(sp1.current_stage_vars)

MAX_ITER = 300
lr = 0.1

hist_lower = []

for iter = 1:MAX_ITER
        
    scenario = SQLP.sample_scenario(sto)
    empty!(scenario_set)
    push!(scenario_set, scenario)

    subgrad = zeros(length(sp1.current_stage_vars))
    obj_lower = 0.0

    for scenario in scenario_set
        SQLP.instantiate!(sp2, scenario)
        fix.(sp2.last_stage_vars, x; force=true)
        optimize!(sp2.model)
        @assert(termination_status(sp2.model) == OPTIMAL)
        
        prob_subgrad = dual.(FixRef.(sp2.last_stage_vars))
        
        obj_lower += objective_value(sp2.model) / length(scenario_set)
        subgrad += prob_subgrad / length(scenario_set)
    end


    f = objective_function(sp1.model)

    _, c = get_coefficients(f, sp1)

    subgrad += Vector(c)
    if norm(direction) <=1e-6
        direction=subgrad
    elseif norm(subgrad-direction) <= 1e-6
        direction=subgrad
    else
        lambda=(-direction' * subgrad + norm(subgrad))/norm(-direction+subgrad)
        if lambda <=0
            direction=subgrad
        elseif lambda>=1
            direction=direction
        else
            direction=lambda*direction+(1-lambda)*subgrad
        end
    end

    obj_lower += c' * x

    global x = x - lr *direction
    global x = projection!(sp1_proj, x)
    
    push!(hist_lower, obj_lower)
end

using Plots
plot(hist_lower)

x