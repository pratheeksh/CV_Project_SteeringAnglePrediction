cd ..
for LR in 0.00001 0.0001 0.001 0.01 0.1
do

qlua bpmain.lua -model rambo  -nEpochs 25 -LR $LR -p -batchsize 32 -nThreads 4 -output rambo-regression-e5-lre${LR}-bs32  | tee scripts/rambo-regression-e5-lre${LR}-bs32.log
done


