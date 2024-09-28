from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from loss.label_smoothing import LabelSmoothingCrossEntropy
from models.domain_adaptation.vmamba_da import Feature, Predictor, init_weights
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from data.custom_dataset import CustomDataset
from utils.const import MEAN, STD

# Training settings
class Solver(object):
    def __init__(self, config, batch_size=32, learning_rate=1e-4, interval=2, optimizer='adamw'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.source = config.DATA.SOURCE_PATH
        self.target = config.DATA.TARGET_PATH
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.all_use = all_use
        self.learning_rate = learning_rate
        self.img_size = config.DATA.IMG_SIZE
        self.dims = config.MODEL.SLICED_WASSERSTEIN_DISTANCE
        self.EPOCHS = config.TRAIN.EPOCHS
        self.interval = interval

        print('dataset loading')
        self.load_dataset()
        print('load finished!')
        self.G = Feature(
            pretrained='pretrain/vssm1_tiny_0230s_ckpt_epoch_264.pth',
            depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
            patch_size=4, in_chans=3, num_classes=3, channel_first = True,
            ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
            ssm_init="v0", forward_type="v05_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln2d",
            downsample_version="v3", patchembed_version="v2",
            use_checkpoint=False, posembed=False, imgsize=224
        )

        self.C1 = Predictor(self.G.num_features)
        init_weights(self.C1)
        

        self.C2 = Predictor(self.G.num_features)
        init_weights(self.C2)
        
        if config.EVAL_MODE:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, config.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, config.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, config.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.set_optimizer(which_opt=optimizer)
        self.set_scheduler()

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))

    def discrepancy_slice_wasserstein(self, p1, p2, dims = 128):
        p1 = F.softmax(p1)
        p2 = F.softmax(p2)
        s = p1.shape
        if s[1] > 1:
            proj = torch.randn(s[1], dims).cuda()
            proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
            p1 = torch.matmul(p1, proj)
            p2 = torch.matmul(p2, proj)
        p1 = torch.topk(p1, s[0], dim=0)[0]
        p2 = torch.topk(p2, s[0], dim=0)[0]
        dist = p1 - p2
        wdist = torch.mean(torch.mul(dist, dist))

        return wdist

    def set_scheduler(self):
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=10, gamma=0.1)
        self.scheduler_C1 = torch.optim.lr_scheduler.StepLR(self.optimizer_C1, step_size=10, gamma=0.1)
        self.scheduler_C2 = torch.optim.lr_scheduler.StepLR(self.optimizer_C2, step_size=10, gamma=0.1)

    def set_optimizer(self, which_opt='momentum', momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=self.learning_rate, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=self.learning_rate, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=self.learning_rate, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=self.learning_rate, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=self.learning_rate, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=self.learning_rate, weight_decay=0.0005)
        
        if which_opt == 'adamw':
            self.optimizer_G = optim.AdamW(self.G.parameters(),
                                    lr=self.learning_rate, weight_decay=0.0005)

            self.optimizer_C1 = optim.AdamW(self.C1.parameters(),
                                     lr=self.learning_rate, weight_decay=0.0005)
            self.optimizer_C2 = optim.AdamW(self.C2.parameters(),
                                     lr=self.learning_rate, weight_decay=0.0005)

    def zero_grad(self):
        self.optimizer_G.zero_grad()
        self.optimizer_C1.zero_grad()
        self.optimizer_C2.zero_grad()

    def load_dataset(self):
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomResizedCrop(size=self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        train_source_dataset = CustomDataset(self.source + '/train', train_transform)
        val_source_dataset = CustomDataset(self.source + '/val', val_test_transform)
        test_source_dataset = CustomDataset(self.source + '/test', val_test_transform)

        train_target_dataset = CustomDataset(self.target + '/train', train_transform)
        val_target_dataset = CustomDataset(self.target + '/val', val_test_transform)
        test_target_dataset = CustomDataset(self.target + '/test', val_test_transform)


        # Create Source DataLoaders
        self.train_source_loader = DataLoader(train_source_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)
        self.val_source_loader = DataLoader(val_source_dataset, batch_size=self.batch_size,  drop_last=True, num_workers=4)
        self.test_source_loader = DataLoader(test_source_dataset, batch_size=self.batch_size,  drop_last=True, num_workers=4)

        # Create Source DataLoaders
        self.train_target_loader = DataLoader(train_target_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)
        self.val_target_loader = DataLoader(val_target_dataset, batch_size=self.batch_size,  drop_last=True, num_workers=4)
        self.test_target_loader = DataLoader(test_target_dataset, batch_size=self.batch_size,  drop_last=True, num_workers=4)

        # Print DataLoader lengths and class names
        print("="*60)
        print(f"Batch size : {self.batch_size}")
        print(f"Train Source DataLoader length: {len(self.train_source_loader)}")
        print(f"Validation Source  DataLoader length: {len(self.val_source_loader)}")
        print(f"Test Source DataLoader length: {len(self.test_source_loader)}")
        print("="*60)
        print(f"Train Target DataLoader length: {len(self.val_target_loader)}")
        print(f"Validation Target DataLoader length: {len(self.val_target_loader)}")
        print(f"Test Target DataLoader length: {len(self.test_target_loader)}")
        print("="*60)

    def train(self, epoch):
        print(f'Epoch {epoch} is training')
        print('='*60)
        criterion = LabelSmoothingCrossEntropy().cuda()

        self.G.train()
        self.C1.train()
        self.C2.train()

        len_dataloader = min(len(self.train_source_loader), len(self.train_target_loader))
        data_zip = enumerate(zip(self.train_source_loader, self.train_target_loader))
        for batch_idx, ((img_s, labels),(img_t,_)) in tqdm(data_zip):
            img_s = img_s.cuda()
            labels = labels.cuda()
            img_t = img_t.cuda()

            self.zero_grad()
            feature_s = self.G(img_s)
            out_s1 = self.C1(feature_s)
            out_s2 = self.C2(feature_s)

            loss_s1 = criterion(out_s1, labels)
            loss_s2 = criterion(out_s2, labels)
            loss_s = loss_s1 + loss_s2

            loss_s.backward()
            self.optimizer_G.step()
            self.optimizer_C1.step()
            self.optimizer_C2.step()

            self.zero_grad()

            feature_s = self.G(img_s)
            out_s1 = self.C1(feature_s)
            out_s2 = self.C2(feature_s)
            feature_t = self.G(img_t)
            out_t1 = self.C1(feature_t)
            out_t2 = self.C2(feature_t)

            loss_s1 = criterion(out_s1, labels)
            loss_s2 = criterion(out_s2, labels)
            loss_s = loss_s1 + loss_s2
            loss_discrepancy = self.discrepancy(out_t1, out_t2)
            # loss_discrepancy = self.discrepancy_slice_wasserstein(out_t1, out_t2, self.dims)
            loss = loss_s - loss_discrepancy


            loss.backward()
            self.optimizer_C1.step()
            self.optimizer_C2.step()

            self.zero_grad()
            ###
            for _ in range(self.num_k):
                feature_t = self.G(img_t)
                out_t1 = self.C1(feature_t)
                out_t2 = self.C2(feature_t)
                loss_discrepancy = self.discrepancy(out_t1, out_t2)
                # loss_discrepancy = discrepancy_slice_wasserstein(out_t1, out_t2, dims)
                loss_discrepancy.backward()
                self.optimizer_G.step()
                self.zero_grad()

            if (batch_idx+1) % self.interval == 0:
                print("Epoch: {}/{} [{}/{}]: C1 Source Loss ={:.5f}, C2 Source Loss={:.5f}, Discrepancy Loss={:.5f}"
                    .format(epoch+1, self.EPOCHS, batch_idx + 1, len_dataloader, loss_s1.item(), loss_s2.item(),loss_discrepancy.item()))

        self.scheduler_G.step()
        self.scheduler_C1.step()
        self.scheduler_C2.step()

    def test(self, max = 0):
        self.max = max
        os.makedirs('Weights/MCD', exist_ok=True)
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for (imgs, labels) in tqdm(self.val_target_loader):
                imgs = imgs.cuda()
                labels = labels.cuda().long()
                features = self.G(imgs)
                output1 = self.C1(features)
                output2 = self.C2(features)
                output_of_two = output1 + output2
                pred_of_two = output_of_two.data.max(1)[1]
                correct += pred_of_two.eq(labels.data).cpu().sum()

            size = len(self.val_target_loader.dataset)
            acc_private = 100. * correct / size
            if acc_private > self.max:
                print(f'Maxium acc on target: {acc_private:2f}')
                self.max = acc_private
                torch.save(self.G.state_dict(),'Weights/MCD/best_G.pth')
                torch.save(self.C1.state_dict(),'Weights/MCD/best_C1.pth')
                torch.save(self.C2.state_dict(),'Weights/MCD/best_C2.pth')
            print('Test:  Accuracy on target val: {:.0f}% '.format(acc_private))

            correct = 0
            for (imgs, labels) in tqdm(self.train_source_loader):
                imgs = imgs.cuda()
                labels = labels.cuda().long()
                features = self.G(imgs)
                output1 = self.C1(features)
                output2 = self.C2(features)
                output_of_two = output1 + output2
                pred_of_two = output_of_two.data.max(1)[1]
                correct += pred_of_two.eq(labels.data).cpu().sum()

            size = len(self.train_source_loader.dataset)
            acc_source_train = 100. * correct / size
            print('Test:  Accuracy on Source train: {:.0f}% '.format(acc_source_train))

            correct = 0
            for (imgs, labels) in tqdm(self.val_source_loader):
                imgs = imgs.cuda()
                labels = labels.cuda().long()
                features = self.G(imgs)
                output1 = self.C1(features).float()
                output2 = self.C2(features).float()
                output_of_two = output1 + output2
                pred_of_two = output_of_two.data.max(1)[1]
                correct += pred_of_two.eq(labels.data).cpu().sum()

            size = len(self.val_source_loader.dataset)
            acc_source_test = 100. * correct / size
            print('Accuracy on source val: {:.0f}% '.format(acc_source_test))

        return self.max