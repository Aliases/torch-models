-- http://ccvl.stat.ucla.edu/ccvl/DeepLab-MSc-COCO-LargeFOV/train.prototxt
-- Model from above link. Above has caffe model

require 'nn'
require 'nngraph'

local MaxPooling2D = nn.SpatialMaxPooling
local Convolution = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local UpConvolution = nn.SpatialFullConvolution
local Identity = nn.Identity
local ReLU = nn.ReLU
local Dropout = nn.Dropout

-- Usage (nn.JoinTable(1):forward{x, y})
-- joins 2 tables along the dimension specified
-- 1 for along increment rows, 2 for increment columns
local Join = nn.JoinTable
  -- net:add(Convolution(nIn, nOut, kernelX, kernelY, strideX, strideY, padX, padY))

-- In caffe layers, default stride = 1 , pad = 0
-- Same in torch
local function directPath0(nIn, nOut_)
  -- Input is data
  local net = nn.Sequential()
  local noChannels = 1
  local noClasses = 5
  local nInp = nIn or noChannels
  local nOut = nOut_ or 128
  net:add(Convolution(nInp, nOut, 3, 3, 8, 8, 1, 1 ))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, nOut, 1, 1, 1, 1)) -- padding not specified
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, noClasses, 1, 1)) -- padding and stride not specified --data_ms
  return net
end

local function network(nIn, nOut_)
  -- Input is data
  local net = nn.Sequential()
  local noChannels = 1
  local noClasses = 5
  local nInp = nIn or noChannels
  local nOut = nOut_ or 64
  net:add(Convolution(noChannels, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(MaxPooling2D(3, 3, 2, 2, 1, 1)) -- pool1
  return net
end

local function directPath1(nIn, nOut_) -- nIn is 64 from above network
  -- Input is pool1 i.e. output of above network
  local nInp = nIn or 64
  local nOut = nOut_ or 2*nInp
  local net = nn.Sequential()
  local noClasses = 5
  net:add(Convolution(nInp, nOut, 3, 3, 4, 4, 1, 1))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, nOut, 1, 1)) -- padding and stride not specified
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, noClasses, 1, 1)) -- padding and stride not specified --pool1_ms
  return net
end

local function intermediate1(nIn, nOut_) -- nIn is 64 from above network
  -- Input is pool1 i.e. output of  network
  local nInp = nIn or 64
  local nOut = nOut_ or 2*nInp
  local net = nn.Sequential()
  net:add(Convolution(nInp, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(MaxPooling2D(3, 3, 2, 2, 1, 1)) -- pool2
  return net
end

local function directPath2(nIn, nOut_) -- nIn is 64 from above network
  -- Input is pool2 i.e. output of intermediate1
  local nInp = nIn or 128
  local nOut = nOut_ or nInp
  local net = nn.Sequential()
  local noClasses = 5
  net:add(Convolution(nInp, nOut, 3, 3, 2, 2, 1, 1))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, nOut, 1, 1)) -- padding and stride not specified
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, noClasses, 1, 1)) -- padding and stride not specified -- pool2_ms
  return net
end

local function intermediate2(nIn, nOut_) -- nIn is 128 from above intermediate1
  -- Input is pool2 i.e. output of  network
  local nInp = nIn or 128
  local nOut = nOut_ or 2*nInp
  local net = nn.Sequential()
  net:add(Convolution(nInp, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(MaxPooling2D(3, 3, 2, 2, 1, 1)) -- pool3
  return net
end

local function directPath3(nIn, nOut_) -- nIn is 256 from above network
  -- Input is pool3 i.e. output of intermediate2
  local nInp = nIn or 256
  local nOut = nOut_ or nInp/2
  local net = nn.Sequential()
  local noClasses = 5
  net:add(Convolution(nInp, nOut, 3, 3, 1, 1, 1, 1))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, nOut, 1, 1, 1, 1)) -- padding not specified
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, noClasses, 1, 1)) -- padding and stride not specified -- pool3_ms
  return net
end

local function intermediate3(nIn, nOut_) -- nIn is 256
  -- Input is pool3 i.e. output of  network
  local nInp = nIn or 256
  local nOut = nOut_ or 2*nInp
  local net = nn.Sequential()
  net:add(Convolution(nInp, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(MaxPooling2D(3, 3, 1, 1, 1, 1)) -- pool4
  return net
end

local function directPath4(nIn, nOut_) -- nIn is 512 from above network
  -- Input is pool4 i.e. output of intermediate3
  local nInp = nIn or 512
  local nOut = nOut_ or 128
  local net = nn.Sequential()
  local noClasses = 5
  net:add(Convolution(nInp, nOut, 3, 3, 1, 1, 1, 1))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, nOut, 1, 1, 1, 1)) -- padding not specified
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, noClasses, 1, 1)) -- padding and stride not specified -- pool4_ms
  return net
end

local function lastPart(nIn, nOut_)
  local net = nn.Sequential()
  local noClasses = 5
  local nInp = nIn or 512
  -- input is pool4
  net:add(Convolution(512, 512, 5, 5, 1, 1, 2, 2)) -- stride not specified -- hole = 2, ker =3, ker_eff = ker_h + (ker_h -1 )*(hole-1),
  net:add(ReLU(true))
  net:add(Convolution(512, 512, 5, 5, 1, 1, 2, 2)) -- stride not specified -- hole = 2
  net:add(ReLU(true))
  net:add(Convolution(512, 512, 5, 5, 1, 1, 2, 2)) -- stride not specified -- hole = 2
  net:add(ReLU(true))
  net:add(MaxPooling2D(3, 3, 1, 1, 1, 1)) -- pool4
  net:add(nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
  net:add(Convolution(512, 1024, 25, 25, 1, 1, 12, 12)) -- stride not specified - hole =12
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(1024, 1024, 1, 1))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(1024, noClasses, 1, 1)) -- fc8_$EXP
  -- Fuse layers after this
  return net
end


function deeplab(options)
  options = options or {}
  local noClasses = options.noClasses or 5
  local noChannels = options.noChannels or 1

  local input = nn.Identity()()


  local data_ms = directPath0()(input)

  local pool1 = network()(input)
  local pool1_ms = directPath1()(pool1)

  local pool2 = intermediate1()(pool1)
  local pool2_ms = directPath2()(pool2)

  local pool3 = intermediate2()(pool2)
  local pool3_ms = directPath3()(pool3)

  local pool4 = intermediate3()(pool3)
  local pool4_ms = directPath4()(pool4)

  local fc8_ = lastPart()(pool4)

  local temp = nn.CAddTable()({data_ms, pool1_ms, pool2_ms, pool3_ms, pool4_ms, fc8_})

  -- temp = UpConvolution(noClasses, noClasses, 2, 2, 2, 2)(temp)
  -- temp = UpConvolution(noClasses, noClasses, 2, 2, 2, 2)(temp)
  -- temp = UpConvolution(noClasses, noClasses, 2, 2, 2, 2)(temp)

  temp = nn.SpatialUpSamplingNearest(8)(temp)

  -- fuse layers
  -- fuse data_ms , pool1_ms , pool2_ms , pool3_ms, pool4_ms, fc8_$Exp
  -- local fused = nn.Sum(data_ms, pool1_ms, pool2_ms, pool3_ms, fc8_)

  -- local output = fused
  -- label shrinking : unknown

  -- Softmax loss
  -- return nn.gModule({input}, {data_ms, pool1_ms, pool2_ms, pool3_ms, pool4_ms, fc8_, temp})
  return nn.gModule({input}, {temp})

end
