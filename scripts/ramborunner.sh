cd ..
for ep in 128 64 32 16
do


qlua bpmain.lua -model rambo  -nEpochs 25 -LR 0.0001 -p -batchsize $ep -nThreads 4 -output rambo-regression-e25-lre0001-bs${ep}  | tee scripts/rambo-regression-e25-lre0001-bs${ep}.log
done


