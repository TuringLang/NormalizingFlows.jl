module NormalizingFlowsMooncakeExt

using Mooncake: Mooncake, DefaultCtx
using NormalizingFlows: NormalizingFlows

# Mooncake does not read the ChainRules rule on `_device_draw`.
Mooncake.@zero_derivative DefaultCtx Tuple{
    typeof(NormalizingFlows._device_draw),Any,Any,Any
}

end
