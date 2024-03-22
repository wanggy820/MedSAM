import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import network
from ISBIDataset import ISBIDataset
import os
from tqdm import tqdm
from torch.nn import functional as F

def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # torch.mean(score) 就是我们的dice系数
    dice_loss = 1 - torch.mean(score)

    return dice_loss

#batch_size > 1
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    # intersection = (output_ & target_).sum()
    # union = (output_ | target_).sum()
    iou = 0.
    if len(output)>1:
        for i in range(len(output)):
            union = (output_[i] | target_[i]).sum()
            intersection = (output_[i] & target_[i]).sum()
            iou += (intersection + smooth) / (union + smooth)
    else:
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
    return iou


epoch = 100
crop_size = 513
lr = 0.01
train_transform = transforms.Compose([
    transforms.Resize([crop_size, crop_size]),
    transforms.ToTensor(),
])


test_transform = transforms.Compose([
    transforms.Resize([crop_size, crop_size]),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 5
model_path = "result/model.pt"
train_dataset = ISBIDataset("./datasets/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv", train_transform)
test_dataset = ISBIDataset("./datasets/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv", test_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=2, output_stride=16)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
criterion = nn.CrossEntropyLoss(weight=None,
                                ignore_index=-100,
                                reduction='mean')

writer = SummaryWriter("result/train_logs")
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model.to(device)
model.train()

for i in range(epoch):
    total_loss = 0
    cur_num = 0
    pbar = tqdm(total=len(train_dataloader))
    for (images, labels) in train_dataloader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        np_loss = loss.detach().cpu().numpy()
        total_loss += np_loss
        cur_num += 1
        pbar.set_description("train Epoch %d, Loss=%f" %
                             (i + 1, total_loss/cur_num))
        pbar.update(1)

    writer.add_scalar("train loss:", total_loss/len(train_dataloader), i+1)
    print("\ntrain Epoch %d,Loss=%f" % (i+1, total_loss/len(train_dataloader)))
    scheduler.step()
    torch.save(model.state_dict(), model_path)
    pbar.close()

    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_dataloader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)

            loss = criterion(outputs, labels)
            np_loss = loss.detach().cpu().numpy()
            total_loss += np_loss

        print("test Epoch %d,Loss=%f\n" % (i + 1, total_loss / len(test_dataloader)))


writer.close()
