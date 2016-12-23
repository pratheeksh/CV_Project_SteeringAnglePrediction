nohup qlua main.lua -model $1 -nEpochs $2 -LR 0.00001 -p -batchsize 32 -nThreads 4 >> logs/$1-regression-e$2-lr1-bs32.log &
nohup qlua main.lua -model $1 -nEpochs $2 -LR 0.0001 -p -batchsize 32 -nThreads 4 >> logs/$1-regression-e$2-lr2-bs32.log &
nohup qlua main.lua -model $1 -nEpochs $2 -LR 0.001 -p -batchsize 32 -nThreads 4 >> logs/$1-regression-e$2-lr3-bs32.log &
nohup qlua main.lua -model $1 -nEpochs $2 -LR 0.01 -p -batchsize 32 -nThreads 4 >> logs$1-regression-e$2-lr4-bs32.log &

