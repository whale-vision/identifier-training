from dataloader import WhaleDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torchrefs.engine import train_one_epoch, evaluate
from torchrefs import utils

NUM_CLASSES = 3
# EPOCHS = 10
SAVE_FREQ = 10

if __name__ == '__main__':
    # load dataset
    dataset = WhaleDataset()

    # create model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES) 

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    tsize = int(len(dataset)*test_split)
    dataset_train = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset, indices[-tsize:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=10, shuffle=True, num_workers=4,
        collate_fn = utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=10, shuffle=False, num_workers=4,
        collate_fn = utils.collate_fn)

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f"training on {device}")
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # for epoch in range(EPOCHS):
    epoch = 0
    while epoch < 1000:
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=SAVE_FREQ)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        stats = evaluate(model, data_loader_test, device=device)

        print(stats)

        if (epoch % SAVE_FREQ == 0):
            torch.save(model.state_dict(), f'./models/model_{epoch}.pth')

        epoch += 1

    # save model
    torch.save(model.state_dict(), './models/final_model.pth')