module NormalizingFlowsMooncakeExt

using Mooncake: Mooncake, DefaultCtx
using NormalizingFlows: NormalizingFlows

# Mooncake imports ChainRules rules one signature at a time, so it does not see the
# `@non_differentiable` on the device draw and traces into the allocation instead.
Mooncake.@zero_derivative DefaultCtx Tuple{
    typeof(NormalizingFlows._device_draw),Any,Any,Any
}

end
