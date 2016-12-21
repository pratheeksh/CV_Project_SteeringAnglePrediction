
require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'

--[[
function meters:reset()
   self.val:reset()
   self.train:reset()
   self.clerr:reset()
   self.ap:reset()
end --]] logs = {
   train_loss_full = {},
   train_loss = {},
   val_loss = {},
   map = {},
   clerr = {},
}

function log(meters, logs)
   local gnuplot = require 'gnuplot'
   gnuplot.pngfigure(paths.concat('outputs' ,opt.output .. 'valtrain_' .. #logs.train_loss ..'.png'))
-- gnuplot.pngfigure(paths.concat('outputs','valtrain_1' .. #logs.train_loss ..'.png'))
   gnuplot.plot({'train loss',
            torch.range(1, #logs.train_loss),torch.Tensor(logs.train_loss)}, {'val loss',torch.Tensor(logs.val_loss)})
   gnuplot.title('loss per epoch' .. opt.output)

   gnuplot.plotflush()

end

