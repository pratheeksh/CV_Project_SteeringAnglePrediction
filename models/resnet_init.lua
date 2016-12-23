local nninit = require 'nninit' 
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local function shortcut(nInputPlane, nOutputPlane, stride) 
   -- return nn.Sequential()
   --       :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
   --       :add(SBatchNorm(nOutputPlane))
   
	if nInputPlane ~= nOutputPlane then
		return nn.Sequential()
            	:add(nn.SpatialAveragePooling(1, 1, stride, stride))
            	:add(nn.Concat(2)
               	:add(nn.Identity())
               	:add(nn.MulConstant(0)))
	else   
		return  nn.Identity()
	end
end
local iChannels
local function basicblock(n, stride)

   local nInputPlane = iChannels
   iChannels = n
   -- print (nInputPlane, iChannels, n)

   local s = nn.Sequential()
   s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
   s:add(SBatchNorm(n))
   s:add(ReLU(true))
   s:add(Convolution(n,n,3,3,1,1,1,1))
   s:add(SBatchNorm(n))

   return nn.Sequential()
      :add(nn.ConcatTable()
      :add(s)
      :add(shortcut(nInputPlane, n, stride)))
      :add(nn.CAddTable(true))
      :add(ReLU(true))
end

local function layer(features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(basicblock(features, i == 1 and stride or 1))
      end
      return s
   end
local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
end
local model = nn.Sequential()
local n = 3
iChannels = 16
model:add(Convolution(3,16,3,3,1,1,1,1):init('weight', nninit.kaiming, {
  dist = 'uniform',
  gain = {'lrelu', leakiness = 0.3}
}))
model:add(layer(16, n, 1))
model:add(layer(32, n, 2))
model:add(layer(64, n, 2))
model:add(ReLU(true))
model:add(Avg(8, 8, 1, 1))
model:add(nn.View(64*25*25):setNumInputDims(3))
model:add(nn.Linear(64*25*25, 1))
-- print(model)--]]
model:cuda()
input = torch.CudaTensor(16,3,128,128)
out = model:forward(input)                        
print(out:size())
return model

