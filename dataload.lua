require 'torch'
local opt, datamean, datastd = ...
local tnt = require 'torchnet'
local image = require 'image'
local WIDTH, HEIGHT = opt.imageSize, opt.imageSize
local DATA_PATH = (opt.data ~= '' and opt.data or './data_/')

torch.setdefaulttensortype('torch.DoubleTensor')


names = {}
test_names =  {}
trainDataPath = "data_/csv/centerclasses.csv"
testDataPath = "data_/csv/test_center.csv"
trainDir =  [[data_/train_images_center/]]
testDir = [[data_/test_center/]]
local END = 12
for file in lfs.dir(trainDir) do
    name = string.sub(file,1,END)
    names[name] = file
end

for file in lfs.dir(testDir) do
--    print(file)
    name = string.sub(file,1,END)
    test_names[name] = file
end


count = 0
function  resize(img)
--   print("image size", img:size())
   -- image.save('images/original' ..count .. '.png', img)
    modimg = img[{{},{100,480},{}}]
   -- image.save('images/resized' ..count ..'.png', modimg)
--   print("image size", modimg:size())  
--image.display(modimg)
count = count + 1
  return image.scale(modimg,WIDTH,HEIGHT)
end

function yuv(img)
    return image.rgb2yuv(img)
end
function norm(img)
        --[[maxer = torch.max(img)
        miner = torch.min(img)
        new = img       
        new = new - torch.mean(new)
        new = new/math.max(maxer,-1*miner)
        -- print(torch.max(new))
        -- print(torch.min(new))--]]
        return img
end
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
     --   [2] = yuv,
       -- [3] = norm
    }
    -- image.display(f(inp))
    return f(inp)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    file = string.format("%19d.jpg", r[1])
    name = string.sub(file,1,END)
       --print(file,names[name],name)
    return transformInput(image.load(DATA_PATH .. 'train_images_center/'..names[name]))
end

function getTrainLabel(dataset, idx)
    -- return torch.LongTensor{dataset[idx][9] + 1}
--     print(dataset[idx][2])
        return torch.DoubleTensor{opt.scale*dataset[idx][2]}
end

function getTestSample(dataset, idx)
    file = string.format("%19d.jpg", dataset[idx])
    name = string.sub(file,1,END)
    file_name = DATA_PATH .. "/test_center/" .. test_names[name]
    return transformInput(image.load(file_name))
end

