import torch
import tqdm
from utils import get_optimizer
import cxr_dataset as CXR
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from readData import ChestXrayDataSet
import torch.backends.cudnn as cudnn
from sklearn import metrics
import AmoebaNetAll as amoeba

import random

class Trainer:

    def __init__(self, model, normal_ops, reduction_ops, optimizer, loss_fn=None, device=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.normal_ops = normal_ops
        self.reduction_ops = reduction_ops
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.task_id = None
        self.device = device

    def set_id(self, num):
        self.task_id = num

    def save_checkpoint(self, checkpoint_path):
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          optim_state_dict=self.optimizer.state_dict(),
                          normal_ops=self.normal_ops,
                          reduction_ops=self.reduction_ops)
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.optimizer = get_optimizer(self.model ,0.01)
        self.normal_ops = checkpoint['normal_ops']
        self.reduction_ops = checkpoint['reduction_ops']
        self.model = amoeba.amoebanet(14, 3, 100, self.normal_ops, self.reduction_ops)
        self.model.load_state_dict(checkpoint['model_state_dict'])


    def train(self):
        self.model.train()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transformed_dataset = CXR.CXRDataset(
            path_to_images="/mnt/scratch2/users/40175159/chest-data/chest-images",
            fold='train',
            transform=data_transforms)

        dataloader = torch.utils.data.DataLoader(
            transformed_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=8)

        lr = 0.01
        for epoch in range(3):
            for x, y, _ in dataloader:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device)).float()
                output = self.model(x)
                
                loss = self.loss_fn(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            lr *= 0.95
            self.optimizer = get_optimizer(self.model, lr)


    def compute_AUCs(self, gt, pred):
        AUROCs = []
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        for i in range(14):
            AUROCs.append(metrics.roc_auc_score(gt_np[i], pred_np[i]))
        return AUROCs


    def eval(self):

        self.model.eval()

        cudnn.benchmark = True
        """Evaluate model on the provided validation or test set."""

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transformed_dataset = CXR.CXRDataset(
            path_to_images="/mnt/scratch2/users/40175159/chest-data/chest-images",
            fold='test',
            transform=data_transforms)

        dataloader = torch.utils.data.DataLoader(
            transformed_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=8)

        gt = torch.FloatTensor()
        gt = gt.to(self.device)
        pred = torch.FloatTensor()
        pred = pred.to(self.device)

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        test_dataset = ChestXrayDataSet(data_dir="/mnt/scratch2/users/40175159/chest-data/chest-images", image_list_file="./labels/test_list.txt",
                                    transform=transforms.Compose([
                                        transforms.Resize(340),
                                        transforms.TenCrop(299),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
        test_loader = DataLoader(dataset=test_dataset, batch_size=16,
                             shuffle=False, num_workers=0, pin_memory=True)

        for i, (inp, target) in enumerate(test_loader):
            with torch.no_grad():
                target = target.to(self.device)
                gt = torch.cat((gt, target), 0)
                bs, n_crops, c, h, w = inp.size()
                input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(self.device))
                output = self.model(input_var)
                output_mean = output.view(bs, n_crops, -1).mean(1)
                pred = torch.cat((pred, output_mean.data), 0)

        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()

        return AUROC_avg
                                        