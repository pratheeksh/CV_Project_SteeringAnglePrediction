require 'torch'
require 'cutorch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
require'lfs'
names = {}
local END = 12
for file in lfs.dir[[/home/pratheeksha/CV_Project_SteeringAnglePrediction/data_/train_images_center/]] do
    --if lfs.attributes(file,"mode") == "file" then 
    -- print(file,string.sub(file,0,END))
    name = string.sub(file,0,END)
    --end
    names[name] = file
end

-- torch.save('names.t7',names)

require 'cunn'
-- require 'cudnn' -- faster convolutions

--[[
--  Hint:  Plot as much as you can.
--  Look into torch wiki for packages that can help you plot.
--]]

-- local trainData = torch.load(DATA_PATH..'train.t7')
-- local testData = torch.load(DATA_PATH..'test.t7')
local csv2tensor = require 'csv2tensor'
local trainData, column_names = csv2tensor.load("/home/pratheeksha/CV_Project_SteeringAnglePrediction/data_/images_to_angles_center_ready.csv") 

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 128,128
local DATA_PATH = (opt.data ~= '' and opt.data or './data_/')

torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)

function resize(img)
    modimg = img[{{},{200,480},{}}]
    return image.scale(modimg,WIDTH,HEIGHT)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    return f(inp)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    file = string.format("%19d.jpg", r[1])
    name = string.sub(file,1,END)
    return transformInput(image.load(DATA_PATH .. 'train_images_center/'..names[name]))
    -- replaces names[name]
    --[[classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))--]]
end

function getTrainLabel(dataset, idx)
    -- return torch.LongTensor{dataset[idx][9] + 1}
	return torch.DoubleTensor{100.00*dataset[idx][2]}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end

function getIterator(dataset)
    --[[
    -- Hint:  Use ParallelIterator for using multiple CPU cores
    --]]
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end


trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    --[[
    --  Hint:  Use a resampling strategy that keeps the
    --  class distribution even during initial training epochs
    --  and then slowly converges to the actual distribution
    --  in later stages of training.
    --]]
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

--[[testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}
]]

--[[
-- Hint:  Use :cuda to convert your model to use GPUs
--]]
local model = require("models/".. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.MSECriterion()--nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1
model:cuda()
criterion:cuda()

-- print(model)

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
end

--[[
-- Hint:  Use onSample function to convert to
--        cuda tensor for using GPU
--]]
-- engine.hooks.onSample = function(state)
-- end
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
  end 
end


engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    -- clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value())) -- , clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.adam,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            -- momentum = opt.momentum
        }
    }

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

print("The End!")
