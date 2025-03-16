julia --project=../../ --threads 20 train.jl
wait 
echo "training done,s tarting stability test"
julia --project=../../ --threads 20 stability.jl
wait 
echo "stability test done, starting plot"
julia --project=../../ plotting.jl
