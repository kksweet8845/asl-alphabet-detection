# asl-alphabet project


## Setup

```
$ pip install -r requirement.txt
```


## Instruction

1. Downloads the dataset from [asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet?fbclid=IwAR1qJn-0tivnRrB6WmZpbq_PuWqtMCVJzMF8CK3RchnHGz466v5kDfJcnBY) and [asl-alphabet-test](https://www.kaggle.com/danrasband/asl-alphabet-test?fbclid=IwAR0adhGL68wEA9JQocOB9q1RFTKutnq1otecvStcLSmDiU4Rwk30R1CJLgs)


2. Create a directory named `data`, then extract the dataset and put them into `data`. The structure of `data` directory is shown as following. You should rearrange the tree structure of directory like this. And each sub directory contains the images.
```
./data
├── test
│   ├── A
│   ├── B
│   ├── C
│   ├── X
│   ├── Y
│   └── Z...
└── train
    ├── A
    ├── B
    ├── C
    ├── D
    ├── W
    ├── X
    ├── Y
    └── Z...

60 directories
```

3. If you finish the step 2, you can verify by running `read-img.py`, it will output the meta info of each image named `preview.log` and a tfrecord file transformed from dataset.

```
$ python read-img.py -i [Path of the directory] -o [The path of output file]
```

4. If you suceed in step 3, you can forward to `training.py` to change the train and test value to train your model. 

