import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out,
                              kernel_size=(3, 3), stride=stride)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layer_config = ((64, 2), (64, 1), (128, 2), (128, 1))

        ch_in = 1
        block_list = []
        for ch_out, stride in layer_config:
            block = ConvBlock(ch_in, ch_out, stride)
            block_list.append(block)
            ch_in = ch_out

        self.backbone = nn.Sequential(*block_list)

        # Insert information bottleneck layer
        bottleneck_num_ch = 2
        self.info_bottleneck = nn.Linear(layer_config[-1][0], bottleneck_num_ch)

        self.head = nn.Linear(bottleneck_num_ch, num_classes)

        # Add softmax function as class member
        self.softmax = nn.Softmax()


    def forward(self, input):
        featuremap = self.backbone(input)
        squashed = F.adaptive_avg_pool2d(featuremap, output_size=(1, 1))
        squeezed = squashed.view(squashed.shape[0], -1)

        # Insert information bottleneck layer
        self.bottleneck_output = self.info_bottleneck(squeezed)

        pred = self.head(self.bottleneck_output)

        # Apply softmax function and store the result
        self.softmax_pred = self.softmax(pred)
        return pred

    @classmethod
    def loss(cls, pred, gt):
        return F.cross_entropy(pred, gt)


class Trainer:
    def __init__(self):

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.7, 1.1)),
            transforms.ToTensor(),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = FashionMNIST("./data", train=True,
                                     transform=self.train_transform,
                                     download=True)

        # Make validation dataset a class member to access it later
        self.val_dataset = FashionMNIST("./data", train=False,
                                   transform=self.val_transform,
                                   download=True)


        # Define class names and samples holders
        self.class_labels = ["T-shirt or top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        self.class_samples = defaultdict(list)

        # Choose 10 instances of every class
        for sample_idx, sample in enumerate(self.val_dataset):
            class_label = sample[-1]
            if len(self.class_samples[class_label]) < 10:
                self.class_samples[class_label].append(self.val_dataset[sample_idx])


        batch_size = 1024
        self.train_loader = data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        self.val_loader = data.DataLoader(self.val_dataset, 
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=4)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = NeuralNet()
        self.net.to(self.device)

        self.logger = SummaryWriter()
        self.i_batch = 0

    def train(self):

        num_epochs = 100

        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        for i_epoch in range(num_epochs):
            # Store current epoch
            self.i_epoch = i_epoch
            self.net.train()

            for feature_batch, gt_batch in self.train_loader:
                feature_batch = feature_batch.to(self.device)
                gt_batch = gt_batch.to(self.device)

                pred_batch = self.net(feature_batch)

                loss = NeuralNet.loss(pred_batch, gt_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.add_scalar("train/loss", loss.item(), self.i_batch)

                if self.i_batch % 100 == 0:
                    print(f"batch={self.i_batch} loss={loss.item():.6f}")

                self.i_batch += 1

            self.validate()

            torch.save(self.net, "model.pth")

    def validate(self, show_matrix=False, show_hard_samples=False):
        self.net.eval()

        loss_all = []
        pred_all = []
        gt_all = []

        # Add list for softmax results
        softmax_all = []


        # Proceed information bottleneck output and serialize it to file
        # Serializing is made to replot data easily, without waiting for CNN to learn
        file_out = open("./data_to_plot.dat", "a+")
        # Concatenating all samples into a batch
        for cloth_label, cloth_type_list in self.class_samples.items():
            all_samples = [cloth_type_list[i][0][None, :, :, :] for i in range(len(cloth_type_list))]
            feature_batch = torch.cat(all_samples, 0).to(self.device)

            with torch.no_grad():
                pred = self.net(feature_batch).to(self.device)

            x = self.net.bottleneck_output[:, 0].cpu().detach().numpy()
            y = self.net.bottleneck_output[:, 1].cpu().detach().numpy()

            if not show_matrix or not show_hard_samples:
                for instance_idx in range(0, len(x)):
                    file_out.write(f"{self.i_epoch} {cloth_label} {instance_idx} {x[instance_idx]} {y[instance_idx]}\n")

        file_out.close()

        for feature_batch, gt_batch in self.val_loader:
            feature_batch = feature_batch.to(self.device)
            gt_batch = gt_batch.to(self.device)

            with torch.no_grad():
                pred_batch = self.net(feature_batch)
                loss = NeuralNet.loss(pred_batch, gt_batch)
                softmax_batch = self.net.softmax_pred

            loss_all.append(loss.item())
            pred_all.append(pred_batch.cpu().numpy())
            gt_all.append(gt_batch.cpu().numpy())
            softmax_all.append(softmax_batch.cpu().numpy())

        loss_mean = np.mean(np.array(loss_all))
        pred_all = np.argmax(np.concatenate(pred_all, axis=0), axis=1)
        gt_all = np.concatenate(np.array(gt_all))
        softmax_all = np.concatenate(np.array(softmax_all))

        # Save confusion matrix
        if show_matrix:
            conf_mat = confusion_matrix(gt_all, pred_all)
            disp = ConfusionMatrixDisplay(conf_mat, display_labels=self.class_labels)
            fig, ax = plt.subplots(figsize=(12, 12))
            fig.suptitle('Confusion matrix', fontsize=16)
            disp.plot(ax=ax)
            plt.savefig("./pictures/confusion_matrix.png")


        # Save 5 hardest samples of every class

        # For a particular class we find a sample with mis-classification and
        # maximum deviation and find what the most rated prediction is for that sample
        # Then we save this sample as an image and write out softmax value to
        # the file associated with this particular sample
        if show_hard_samples:
            mis_classifications_indices = [idx for idx in range(0, len(gt_all)) if pred_all[idx] != gt_all[idx]]
            class_misclassifications = defaultdict(list)
            for idx in mis_classifications_indices:
                true_label = self.val_dataset[idx][-1]
                class_misclassifications[true_label].append([idx, softmax_all[idx]])

            for true_label in range(0, len(self.class_labels)):
                current_class_misses = sorted(class_misclassifications[true_label], key=lambda x: max(x[1]), reverse=True)
                for i in range(0, min(5, len(current_class_misses))):
                    current_miss = current_class_misses[i]
                    current_idx = current_miss[0]
                    current_softmax = current_miss[1]
                    miss_label = np.argmax(current_softmax)

                    current_sample = self.val_dataset[current_idx]
                    if not os.path.exists(f"./pictures/hard_samples/true_label({self.class_labels[true_label]})"):
                        os.mkdir(f"./pictures/hard_samples/true_label({self.class_labels[true_label]})")

                    plt.imsave(f"./pictures/hard_samples/true_label({self.class_labels[true_label]})/miss_label({self.class_labels[miss_label]})_{i}.png", current_sample[0][0])
                    file_softmax = open(f"./pictures/hard_samples/true_label({self.class_labels[true_label]})/miss_label({self.class_labels[miss_label]})_{i}.txt", "w+")
                    file_softmax.write(f"softmax = {current_softmax[miss_label]}")
                    file_softmax.close()

        accuracy = np.sum(np.equal(pred_all, gt_all)) / len(pred_all)

        self.logger.add_scalar("val/loss", loss_mean, self.i_batch)
        self.logger.add_scalar("val/accuracy", accuracy, self.i_batch)

        print(f"Val_loss={loss_mean} val_accu={accuracy:.6f}")


def main():
    trainer = Trainer()
    #trainer.train()

    trainer.net = torch.load("model.pth")

    # And finally show confusion matrix and 5 hardest samples
    trainer.validate(show_matrix=True, show_hard_samples=True)

    print("Done!")


if __name__ == "__main__":
    main()
