# Internal state of neural net

### Subtask 1: Information Bottleneck
#### We insert Information Bottleneck layer at the very end of layer chain. Then we take 10 samples of every class, 100 samples in total, and plot corresponding points in the bottleneck layer

<p align="center">
    <img src="./pictures/epochs.gif" width="750" height="500" />
</p>

#### Each color represents individual class. Legend of the colors is shown on the plot


### Subtask 2: Hard Samples & Confusion Matrix
#### As far as the net is trained, we validate it. We find mis-classified samples for every class. After that we find 5 most missed samples per class (i. e. those samples of current class label which are most rated for some different class). Comparing this data we write out true label of this class and most rated label (note that it can't be true class label by definition). Complete list of true labels and corresponding hard mis-classified labels one can find in `pictures/hard_samples/`


#### Also we draw a confusion matrix for trained net
![](./pictures/confusion_matrix.png)


#### You can see results on [drive](https://disk.yandex.com/d/bnnbcz6Uf_q_3g)