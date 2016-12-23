require 'torch'
require 'cutorch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
require'lfs'
require 'utils'

names = {}
test_names =  {}

--[[ Adding data paths --]]

trainDataPath = "/home/pratheeksha/CV_Project_SteeringAnglePrediction/data_/csv/center.csv" 
testDataPath = "/home/pratheeksha/CV_Project_SteeringAnglePrediction/data_/csv/test_center.csv"
trainDir = [[/home/pratheeksha/CV_Project_SteeringAnglePrediction/data_/train_images_center/]]
testDir = [[/home/pratheeksha/CV_Project_SteeringAnglePrediction/data_/test_center/]]



trainDataPath = "data_/csv/center.csv"
testDataPath = "data_/csv/test_center.csv"
trainDir =  [[data_/train_images_center/]]
testDir = [[data_/test_center/]]


local END = 12

-- torch.save('names.t7',names)

require 'cunn'
require 'cudnn' -- faster convolutions

local csv2tensor = require 'csv2tensor'
local os = require 'os'
local math  = require 'math'
local trainData, column_names = csv2tensor.load(trainDataPath) 

local testData = csv2tensor.load(testDataPath)
local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 320, 160 -- 128,128 -- 320,140
local DATA_PATH = (opt.data ~= '' and opt.data or './data_/')


torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)

print("Parallel Iterator option ", opt.p)
if opt.p == true then 
   print("Loading parallel data set iterator")
   function getIterator(dataset) 
	local lopt = opt
	local lbias = bias
 	return tnt.ParallelDatasetIterator{
		nthread = opt.nThreads, 
		init = function()	
			opt = lopt
			bias = lbias
			require 'torchnet'
			assert(loadfile("dataload.lua"))(opt)
		end,
		closure = function()
			return  tnt.BatchDataset {
				batchsize = opt.batchsize,
				dataset = dataset
			}
		end
	}
   end 
else
   print("Loading normal Iterator") 
   assert(loadfile("dataload.lua"))(opt)  
   function getIterator(dataset)
	return  tnt.DatasetIterator{
	    	dataset =  tnt.ShuffleDataset{        
			dataset = tnt.BatchDataset{
            			batchsize = opt.batchsize,
           			dataset = dataset
        		}
    		}
	}
   end
end

--[[
function getTrainSample(dataset, idx)
    r = dataset[idx]
    file = string.format("%19d.jpg", r[1])
    name = string.sub(file,1,END)
       --print(file,names[name],name)
    return transformInput(image.load(DATA_PATH .. 'train_images_center/'..names[name]))
end

function getTrainLabel(dataset, idx)
    -- return torch.LongTensor{dataset[idx][9] + 1}
--     print(dataset[idx][2
--]]
assert(loadfile("dataload.lua"))(opt)

local listData = tnt.ListDataset{
    list = torch.range(1, trainData:size(1)):long(), 
    load = function(idx)
        return {
            input = getTrainSample(trainData, idx),
            target = getTrainLabel(trainData, idx)
        }
    end
}


local shuffledData = tnt.ShuffleDataset{
    dataset = listData
}

if opt.reweight == true then
    print('Computing class probabilities and uniform lists')
    -- class probabilities
    local classCounts = torch.zeros(40)
    
    classTables = {}
    for i = 1, 40 do
        classTables[i] = {}
    end
--    print(classTables[1])
  --  table.insert(classTables[1], 1)    
    for i = 1, trainData:size(1) do

        if i <= torch.floor(shuffledData:size() * (1.0 - opt.val)) then
            sample = shuffledData:get(i)
	    angleclass = torch.ceil((sample.target[1] + 2.4) * 100) % 40 + 1
 --	    print(sample.target[1], angleclass)
            table.insert(classTables[angleclass], i)
            classCounts[angleclass] = classCounts[angleclass] + 1
        end
    end
    classProbs = classCounts / classCounts:sum()

    -- the bias towards uniform distribution.
    -- 1: uniform distribution, 0: data distribution
    bias = 1.0
else
    bias = 0.0
end

print("Printing class labels", classTables)
if opt.reweight == true then
    shuffledData = tnt.ResampleDataset{
        dataset = shuffledData,
        sampler = function(dataset, idx)
            -- in SplitDataset, the partition names are sorted, and val comes after train.
            -- the last 0.1 of the dataset belongs to val, so we don't do resampling there.
            if torch.rand(1)[1] < 1 - bias or
                   idx > torch.floor(dataset:size() * (1.0 - opt.val)) then
                return idx
            end
        	random_num = torch.random(1,40)  
--		print("ranom num", random_num)  
 		chooser = classTables[random_num]
	  return chooser[torch.random(1, #chooser)]
        end,
        size = shuffledData:size()
    }
end

if opt.reweight == false then 
trainDataset = tnt.SplitDataset{
    partitions = {train=0.9 , val=0.1},
    initialpartition = 'train',  
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}
else 

print("Printing shuffledData", shuffledData)

trainDataset = tnt.SplitDataset{
    partitions = {train=1.0-opt.val, val=opt.val},
    initialpartition = 'train',
    dataset = shuffledData
}
end


function getSampleId(dataset, idx)
        file = string.format("%19d", dataset[idx])
        chopped =  string.sub(test_names[string.sub(file,1,12)], 1, 19)
        return chopped
        --return torch.LongTensor{tonumber(chopped)}
end
testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = getSampleId(testData, idx) --torch.LongTensor{testData[idx]}
        }
    end
}



local model = require("models/".. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.MSECriterion() -- nn.CrossEntropyCriterion() --nn.SmoothL1Criterion()-- nn.MSECriterion()--nn.CrossEntropyCriterion()
-- local criterion =nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1
model:cuda()
criterion:cuda()


local meters = {
   val = tnt.AverageValueMeter(),
   train = tnt.AverageValueMeter(),
  -- clerr = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   ap = tnt.APMeter(),
}

function meters:reset()
   self.val:reset()
   self.train:reset()
--   self.clerr:reset()
   self.ap:reset()
end
-- Support functions
local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end
-- end
 print(model)


engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
   if opt.p == true then
        dataSize = state.iterator:execSingle('size')
    else
        dataSize = state.iterator:exec('size')
    end

    if opt.reweight == true then
      --  bias = (opt.reweightEnd - epoch) / (opt.reweightEnd - opt.reweightStart)
        bias = 0.2 -- math.max(0, math.min(1, bias))
    end
end


local input  = torch.CudaTensor()
local target = torch.CudaTensor()
engine.hooks.onSample = function(state)
  input:resize( 
      state.sample.input:size()
  ):copy(state.sample.input)
  state.sample.input  = input
  if state.sample.target then
      target:resize( state.sample.target:size()):copy(state.sample.target)
      state.sample.target = target
--     print("State sample target size ", state.sample.target:size())
  end   
end


engine.hooks.onForwardCriterion = function(state)
    if state.training then
	meters.train:add(state.criterion.output)
    else
        meters.val:add(state.criterion.output)
    end

   
    meter:add(state.criterion.output)
 --[[
    print("target type", type(state.sample.target))
    print("output type", type(state.network.output))

    print("target size", state.sample.target:size())
    print("output size", state.network.output:size())

 print("target size values", state.sample.target)
    print("output size values ", state.network.output) 
   --]] 
   --clerr:add(state.network.output, state.sample.target)
    	--[[if mode == 'Val' then 
		print(state.network.output:cat(state.sample.target),1)
	end--]] 
	--[[ print(model:getParameters()) --:forward(state.sample.input))
	if mode == 'Val' then 
		-- image.display(state.sample.input)
		print(model:forward(state.sample.input))
		print(state.criterion.output)
		sleep(5)
	end--]]
-- THINGS TO CHECK 
-- END

    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, dataSize, meter:value() ,clerr:value{k = 1}))
    else
        xlua.progress(batch, dataSize)
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1
local error_out = assert(io.open("outputs/".. opt.output .. "errors.csv", "w"))
while epoch <= opt.nEpochs do
    meters:reset()
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.adam,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,

            --[[momentum = opt.momentum,
		learningRateDecay = .01,
		weightDecay = .001--]]
        }
    }
    trainloss = meters.train:value()
    logs.train_loss[#logs.train_loss + 1] = meters.train:value()
    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    logs.val_loss[#logs.val_loss+ 1] = meters.val:value()
    valloss = meters.val:value()
    error_out:write(trainloss .. ' ' .. valloss .. ' \n')
    print('Done with Epoch '..tostring(epoch))
--    print(logs.val_loss)
    epoch = epoch + 1
end

error_out:close()
local submission = assert(io.open(opt.logDir .. "/" .. opt.output ..".csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local pred = state.network.output
    for i = 1, pred:size(1) do
	submission:write(fileNames[i]..','..string.format("%f\n", pred[i][1]))
      --  submission:write(string.format("%05d,%f\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, dataSize)
    batch = batch + 1
end


engine.hooks.onEnd = function(state)
    -- Plotting stuff
    --log(logs)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

--torch.save('resnet-e50-lr001.t7',model:clearState())
print("The End!")
