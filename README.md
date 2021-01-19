# BCINet: An Optimized Convolutional Neural Network for EEG-Based Brain-Computer Interface Applications

[A. K. Singh and X. Tao, "BCINet: An Optimized Convolutional Neural Network for EEG-Based Brain-Computer Interface Applications," 2020 IEEE Symposium Series on Computational Intelligence (SSCI), Canberra, Australia, 2020, pp. 582-587, doi: 10.1109/SSCI47803.2020.9308292.](https://ieeexplore.ieee.org/abstract/document/9308292)


# Requirements

- Python == 3.7 or 3.8
- tensorflow == 2.X (both for CPU and GPU)
- PyRiemann >= 0.2.5
- scikit-learn >= 0.20.1
- matplotlib >= 2.2.3

# How to run

- Input Data Format: Number of EEG Channels x Number of Samples X Number of Trials for EEG data and Labels as vector. See testData.mat for references.
- Provide Data related information in op.py such as data path, sampling rate, number of classes, etc.
- Execute the following line of code

```
python main.py
```

# Models implemented/used
- EEGNet [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013) 
- DeepConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [[3]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)

EEGNet, DeepConvNet, and ShallowConvNet implementation are based on EEGNet repo [[4]](https://github.com/vlawhern/arl-eegmodels)
