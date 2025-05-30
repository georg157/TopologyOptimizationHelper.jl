##### TopologyOptimizationHelper.jl


module TopologyOptimizationHelper
using SparseArrays
using LinearAlgebra
using NLopt
using PyPlot
using KrylovKit
using FiniteDifferences
using Statistics

include("functions.jl")
include("functions2d.jl")
include("gradients.jl")
include("further_refinements.jl")

end