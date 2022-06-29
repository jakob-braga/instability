using Oceananigans
using Printf
using NCDatasets
using Statistics
using CUDA
using Adapt

CUDA.allowscalar(false)         # need to run on GPU()

grid = RectilinearGrid(
    GPU(),
    size=(64, 64, 64), halo=(3, 3, 3),
    x = (0, 2),
    y = (0, 2),
    z = (0, 2),
    topology=(Periodic, Bounded, Periodic)
)

## background consts
const f = 1.0 # coriolis
const fₕ = 1.0 # nontraditional coriolis
const M² = 0.0 # (∂u/∂z)^2

## non dimentional
const F²_nd = 0.1
const N²_nd = 0.1

const F² = F²_nd*f^2
const α  = 2 # F²_nd*f - f # ∂u/∂y
const N² = 3.0 #N²_nd*f^2
const ν  = 1e-6

# Background Fields
B(x, y, z, t) = N² .* z
U(x, y, z, t) = α .* y + (sqrt(M²)) .* z 

# Initial Fields
uᵢ(x, y, z) = 0.001*rand()

# initialize model
model = NonhydrostaticModel(; grid,
    background_fields = (u=U, b=B),
    coriolis = NonTraditionalBetaPlane(fz = f, fy = fₕ, β = 0.0, γ = 0.0),
    buoyancy = BuoyancyTracer(),
    advection = WENO5(),
    closure = ScalarDiffusivity(ν=ν),
    tracers = (:b,)
)
set!(model, u = uᵢ)

x, y, z = nodes((Center, Center, Center), grid)

function progress(sim)
    umax = maximum(abs, sim.model.velocities.u)
    @info @sprintf("Iter: %d, time: %.2e, max|u|: %.2e",
       iteration(sim), time(sim), umax)

    return nothing
end

simulation = Simulation(model; Δt=1e-3, stop_time=25.0)
simulation.callbacks[:p] = Callback(progress, IterationInterval(100))


# Store the following info
u, v, w = model.velocities
ω = ∂x(v) - ∂y(u)
#instability_cond = Field(Average(∂x(v)))
b = model.tracers.b
outputs = (; v, u, b, ω) #instability_cond)
# nc attrbiutes
global_attributes = Dict(
    "N^2" => string(N²),
    "F^2" => string(F²),
    "M^2" => string(M²),
    "f" => string(f)
)
simulation.output_writers[:fields] = NetCDFOutputWriter(
    model, outputs;
    filename = "./data/inertial_instability.nc",
    schedule = TimeInterval(0.05),
    global_attributes=global_attributes,
    overwrite_existing = true
)


run!(simulation)