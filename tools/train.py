import torch
import argparse
import os
import sys
import numpy as np
import yaml
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import FasterRCNN
from tqdm import tqdm
from dataset import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    #Read the config file
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Define initial seed
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        
    voc = VOCDataset(
            "train",
            im_dir=dataset_config["im_train_path"],
            ann_dir=dataset_config['ann_train_path']
        )
    
    train_dataset = DataLoader(voc, batch_size=1, shuffle=True, num_workers=4)
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config["num_classes"])
    
    faster_rcnn_model.train() # set faster in train mode
    faster_rcnn_model.to(device)
    
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    
    optimizer = torch.optim.SGD(
            lr=train_config["lr"], # learning rate
            params=filter(lambda p: p.requires_grad,
                        faster_rcnn_model.parameters()),
            weight_decay=5E-4, # L2 weight decay
            momentum=0.9 # momentum coeficient
        )
    #Decays the learning rate of optmizer by gamma once the number of epoch reaches one of the milestones.
    #Eg.: The initial learning rate will be used for epochs 0 through 11.
    # 1)At the beginning of epoch 12, the learning rate will be multiplied by gamma.
    # 2)The new, lower learning rate will be used for epochs 12 through 15.
    # 3)At the beginning of epoch 16, the learning rate will be multiplied by gamma again.
    # 4)This final learning rate will be used for all subsequent epochs.
    scheduler = MultiStepLR(
            optimizer, 
            milestones=train_config['lr_steps'],  
            gamma=0.1
        )
    #accumulation steps = specifies how many mini-batches of gradients to accumulate before performing a weight update with the optimizer
    acc_steps = train_config['acc_steps'] 
    num_epochs = train_config['num_epochs']
    step_count = 1
    
    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        optimizer.zero_grad()
        
        for im, target, fname in tqdm(train_dataset):
            im = im.float().to(device)
            target["bboxes"] = target["bboxes"].float().to(device)
            target["labels"] = target["labels"].long().to(device)
            
            # Calculate Faster R-CNN loss
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss'] # RPN loss
            frcnn_loss = frcnn_output["frcnn_classification_loss"] + frcnn_output['frcnn_localization_loss'] # Faster R-CNN loss
            loss = rpn_loss + frcnn_loss # Final loss
            
            # Save individual losses for each batch
            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())

            loss = loss / acc_steps # average the loss over the accumulated batches
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1
         
        print('Finished epoch {}'.format(i))
        optimizer.step()
        optimizer.zero_grad()
        torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['ckpt_name']))
        
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
        scheduler.step()
    print('Done Training...')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument("--config", dest="config_path", default="config/voc.yaml", type=str)
    
    args = parser.parse_args()
    train(args)