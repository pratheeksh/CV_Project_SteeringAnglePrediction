
local nn = require 'nn'
local Convolution2D = nn.SpatialConvolution
local PReLU = nn.PReLU
local SubSample = nn.SpatialSubSampling
local cnn = nn.Sequential()
cnn:add(Convolution2D(3, 16, 8, 8,2,2))
cnn:add(SubSample(16,4,4))
cnn:add(PReLU())
cnn:add(Convolution2D(16,32, 5, 5,2,2))
cnn:add(SubSample(32,2,2))
cnn:add(PReLU())
cnn:add(Convolution2D(32, 64, 5, 5))
cnn:add(SubSample(64,2,2))

cnn:add(nn.View(64*29*69):setNumInputDims(3))
cnn:add(nn.Dropout(.2))
cnn:add(PReLU())
cnn:add(nn.Linear(64*29*69, 512))
cnn:add(PReLU())
cnn:add(nn.Linear(512,1))
input = torch.Tensor(1,3,320, 160)
-- input = torch.Tensor(1,1,320,140)
out = cnn:forward(input)
print(out:size())
--print(out)

return cnn
--
--model.compile(optimizer="adam", loss="mse")


