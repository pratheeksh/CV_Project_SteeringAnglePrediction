local nn = require 'nn'
--- : 3x48x48-100C7-MP2-150C4-150MP2-250C4-250MP2-300N-43N


--  2x48x48-100C5-MP2-100C5-MP2-100C4-MP2-300N-100N-6N represents a net with 2
-- input images of size 48x48, a convolutional layer with 100 maps and 5x5 filters, a max-pooling layer over
-- non overlapping regions of size 2x2, a convolutional layer with 100 maps and 4x4 filters, a max-pooling
-- layer over non overlapping regions of size 2x2, a fully connected layer with 300 hidden units, a fully
-- connected layer with 100 hidden units and a fully connected output layer with 6 neurons (one per class).


local Convolution = nn.SpatialConvolutionMM
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local cnn = nn.Sequential()
local function addconv(nin, nout, kw, kh, sw, sh, pw, ph)
	local cnnLocal = nn.Sequential()
	cnnLocal:add(nn.SpatialConvolution(nin, nout, kw, kh, sw, sh, pw, ph))
	cnnLocal:add(nn.SpatialBatchNormalization(nout))
	cnnLocal:add(nn.ReLU())
	return cnnLocal
end
  cnn:add(addconv(3,24,5,5,2,2,0,0))
  -- cnn:add(nn.Dropout(0.25))
  -- cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  cnn:add(addconv(24,36,5,5,2,2,0,0))
  -- cnn:add(nn.Dropout(0.25))
  -- cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  cnn:add(addconv(36,48,5,5,2,2,0,0))
  -- cnn:add(nn.Dropout(0.5))
  -- cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  cnn:add(addconv(48,64,3,3,1,1,0,0))
  -- cnn:add(nn.Dropout(0.4))
  -- cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  cnn:add(addconv(64,64,3,3,1,1,0,0))
  -- cnn:add(nn.Dropout(0.4))
  -- cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))-]]
  cnn:add(nn.View(64*33*10):setNumInputDims(3))
  --cnn:add(nn.Dropout(0.5))
  cnn:add(nn.Linear(64*33*10, 1164))
  cnn:add(nn.ReLU())
  cnn:add(nn.Dropout(0.5))
  cnn:add(nn.ReLU())
  cnn:add(nn.Linear(1164, 100))
  cnn:add(nn.ReLU())
  cnn:add(nn.Linear(100, 50))
  cnn:add(nn.ReLU())
  cnn:add(nn.Linear(50,10))
  cnn:add(nn.Tanh())
  cnn:add(nn.Linear(10,1))
  --cnn:add(nn.LogSoftMax())
-- -- end

-- local conv1 = getconv(3, 100)
-- local conv2 = getconv(100, 150)
-- local model = nn.Sequential()
-- print(conv1):wq
--
-- print(conv2)
-- model:add(Max(2,2,2,2))
-- model:add(conv1)
-- -- model:add(conv2)

-- -- model:add(View(2250))
-- -- model:add(Linear(2250, 300))
-- -- model:add(ReLU())
-- -- model:add(Linear(300, 43))
-- print(cnn)--]]
input = torch.Tensor(1,3,320,140)
-- input = torch.Tensor(1,1,320,140)
out = cnn:forward(input)                        
print(out:size())
print(out)
return cnn
