module BiDRA

export inferMetrics

using Turing, MCMCChains
include("functions.jl")


function inferMetrics(logConc::Array, percentResp::Array, isAscending::Int)
    local HDRmixture
    local LDRμ
  
    λ = [0.4, 0.5, 0.1]
    if isAscending == 1
        HDRmixture = MixtureModel([SkewNormal(100, 10, 1), Uniform(0, 100), SkewNormal(0, 10, -1)], λ)
        LDRμ = 0
    else
        HDRmixture= MixtureModel([SkewNormal(0, 10, 1), Uniform(0, 100), SkewNormal(100, 20, -5)], λ)
        LDRμ = 100
    end
  
    nChain = 4
    nIte = 1000
    nAdapt = 1000
    δ = 0.65
    Turing.setprogress!(false)
  
    model_toRun_bidra = model_BIDRA(expData.concentration, expData.response, HDRmixture, LDRμ, isAscending)
    sampler = NUTS(nAdapt, δ)
  
    chains_bidra = DataFrame(sample(model_toRun_bidra, sampler, MCMCThreads(), nIte, nChain))[:, [:HDR, :LDR, :ic50, :slope, :σ]]

    return chains_bidra
  end

end
