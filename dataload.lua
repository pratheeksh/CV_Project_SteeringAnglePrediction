require 'torch'
local opt, datamean, datastd = ...
local tnt = require 'torchnet'
local image = require 'image'
local WIDTH, HEIGHT = 128,128 -- 320, 140 -- opt.imageSize, opt.imageSize

names = {}
test_names =  {}
trainDataPath = "data_/csv/centerclasses.csv"
testDataPath = "data_/csv/test_center.csv"
trainDir =  [[data_/train_images_center/]]
testDir = [[data_/test_center/]]


if  string.find(opt.model, 'nvidia') ~= nil  then 
	WIDTH, HEIGHT =  320, 140
end
if  string.find(opt.model, 'rambo') ~= nil  then
        WIDTH, HEIGHT =  320, 160
	trainDataPath = "data_/csv/center.csv"
end


if string.find(opt.model, 'class') ~= nil then 
-- print("Some classification stuff")
	trainDataPath = "data_/csv/centerclasses.csv"
end
local DATA_PATH = (opt.data ~= '' and opt.data or './data_/')

torch.setdefaulttensortype('torch.DoubleTensor')

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

function hsv(img) 
	return image.rgb2hsv(img)
end


function colortransform(img)
--	print(opt.t)
    if (opt.t == 'rgb') then 	
	return img
    elseif (opt.t == 'hsv') then 
	return image.rgb2hsv(img)
    elseif (opt.t == 'yuv') then 
	return image.rgb2yuv(img)
    else 
	return img
     end    
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
function addNoise(original_image)
    local rand_angle = (torch.uniform(-.2,.2))
    local rand_position_x = (torch.randn(1)*20)[1]
    local rand_position_y = (torch.randn(1)*20)[1]
    image_data = original_image:clone()
    image_data = image.rotate(image_data, rand_angle)
    image_data = image.translate(image_data, rand_position_x, rand_position_y)
    return image_data
end
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize,
       [2] = colortransform,
       -- [3] = norm
    }
    --[[if opt.noise or torch.random(0,1) < 1 then
        inp = addNoise(inp)
    end --]]   
    -- image.display(f(inp))
    return f(inp)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
--    print(r)
    file = string.format("%19d.jpg", r[1])
    name = string.sub(file,1,END)
       --print(file,names[name],name)
    return transformInput(image.load(DATA_PATH .. 'train_images_center/'..names[name]))
end

function getTrainLabel(dataset, idx)
    -- return torch.LongTensor{dataset[idx][9] + 1}
--     print(dataset[idx][2])
     return torch.DoubleTensor{opt.scale*dataset[idx][2] + 1}
end

function getTestSample(dataset, idx)
    file = string.format("%19d.jpg", dataset[idx])
    name = string.sub(file,1,END)
    file_name = DATA_PATH .. "/test_center/" .. test_names[name]
    return transformInput(image.load(file_name))
end


function getSampleId(dataset, idx)
        file = string.format("%19d", dataset[idx])
        chopped =  string.sub(test_names[string.sub(file,1,12)], 1, 19)
        return chopped
        --return torch.LongTensor{tonumber(chopped)}
end
