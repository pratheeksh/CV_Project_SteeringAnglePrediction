local nn = require 'nn'
local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local Relu = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout
local model  = nn.Sequential()

model:add(Convolution(3, 16, 5, 5))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(Convolution(16, 128, 5, 5))
model:add(Dropout(0.5))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(View(3200))
model:add(Dropout(0.5))
model:add(Linear(3200, 1600))
model:add(Tanh())
model:add(Dropout(0.5))
model:add(Linear(1600, 64))
model:add(Tanh())
model:add(Dropout(0.5))
model:add(Linear(64, 43))
model:cuda()
return model
