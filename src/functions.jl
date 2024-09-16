function llogistic(params::Array, isAscending::Bool) 
    HDR, LDR, ic50, slope = params 

    if isAscending
        return x -> LDR + ((HDR - LDR) / (1 + 10^(slope * (ic50 - x))))
    else
        return x -> HDR + ((LDR - HDR) / (1 + 10^(slope * (x - ic50))))
    end
end

@model function model_BIDRA(xs::Array, ys::Array, HDRmixture::MixtureModel, LDRμ::Int, isAscending::Int) 
    HDR ~ HDRmixture
    LDR ~ Normal(LDRμ, 10)
    ic50 ~ Normal(0,10)
    slope ~ LogNormal(0.5, 1)
    
    σ ~ LogNormal(1, 1)

    for i in 1:length(xs)
        f = llogistic([HDR, LDR, ic50, slope], isAscending)
        ys[i] ~ Normal(f(xs[i]), σ)
    end
end