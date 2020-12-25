import os
import time
import warnings

import albumentations as albu
import math
import numpy
import torch

from FFA import FFA

# from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')
from option import model_name, opt
from data_utils import RESIDE_Dataset
from torch.utils.data import DataLoader
from metrics import ssim, psnr

print('model_name:', model_name)

models_ = {
    'ffa': FFA(gps=opt.gps, blocks=opt.blocks),
}

transform_list = [
    albu.RandomRotate90(),
    albu.Flip(0.75),
    albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    albu.OneOf([  # 畸变相关操作
        albu.OpticalDistortion(p=0.3),
        albu.GridDistortion(p=0.1),
        albu.IAAPiecewiseAffine(p=0.3),
    ], p=0.2)
]

if opt.crop:
    transform_list.append(albu.RandomCrop(opt.crop_size, opt.crop_size))

train_transform = albu.Compose(transform_list, p=1)

loaders_ = {
    'nh_train': DataLoader(dataset=RESIDE_Dataset('NH-HAZE_train', transform=train_transform), batch_size=opt.bs,
                             shuffle=True),
    'nh_test': DataLoader(dataset=RESIDE_Dataset('NH-HAZE_valid'), batch_size=1,
                            shuffle=False),
}

start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')
    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)
        out = net(x)
        loss = criterion[0](out, y)

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|'
              f'lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
              end='', flush=True)
        # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
        #	writer.add_scalar('data/loss',loss,step)
        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
            # 	writer.add_scalar('data/ssim',ssim_eval,step)
            # 	writer.add_scalar('data/psnr',psnr_eval,step)
            # 	writer.add_scalars('group',{
            # 		'ssim':ssim_eval,
            # 		'psnr':psnr_eval,
            # 		'loss':loss
            # 	},step)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict()
                }, opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')


def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    # s=True
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device);
        targets = targets.to(opt.device)
        pred = net(inputs)
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        # vutils.save_image(targets.cpu(),'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    # if (psnr1>max_psnr or ssim1 > max_ssim) and s :
    #		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
    #		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
    #		s=False
    return numpy.mean(ssims), numpy.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.benchmark = True
    criterion = []
    criterion.append(torch.nn.L1Loss().to(opt.device))
    optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion)
