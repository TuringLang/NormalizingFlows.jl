julia --project=../../../ --threads 20 train.jl
wait 
echo "training done,s tarting stability test"
julia --project=../../../ --threads 4 stability.jl
wait 
echo "stability test done, starting plot"
julia --project=../../../ --threads 20 plotting.jl
