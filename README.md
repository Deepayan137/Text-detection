# Text-detection
A faster-RCNN approach towards text detction and it subsequent recognition

## Datasets

data present in the given [link](https://drive.google.com/file/d/1ezofR3RsWuUQ4vFUd7fWoC-64VlnOXNU/view?usp=sharing)

The fRCNN code only runs on VOC format inorder to convert.
Inorder to convert you data into the Pascal VOC format run the following codes.

`python -m loader`

change the book path in the code to desired location

then run

```
python3.6 vod_converter/main.py --from udacity-crowdai --from-path give/Image/path/name --to voc --to-path target/path/name

```
## To train your f-RCNN code run the follwing command

```
python -m train
```

## Note

The above code is heavily borrowed from [simple-faster-RCNN](https://github.com/chenyuntc/simple-faster-rcnn-pytorch). Install all the dependencies as mentioned in the repo. 

Also refer this [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8270290) 

The above paper proposes multiple region proposals, which I haven' implemented yet.