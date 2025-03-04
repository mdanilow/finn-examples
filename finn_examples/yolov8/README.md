YOLOv8 demo

```shell
python batch_driver.py --bitfile bitfile/finn-accel.xclbin --batchsize 90 --output_dir output --sequence_dir img1
```

```shell
python simple_driver.py --input cocoimgs/000000007281.jpg --bitfile bitfile/finn-accel.xclbin --conf_thres 0.3
```