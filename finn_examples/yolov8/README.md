YOLOv8 demo

```shell
python batch_driver.py --bitfile bitfile/finn-accel.xclbin --batchsize 90 --output_dir output
```

```shell
python simple_driver.py --input cocoimgs/resized.jpg --bitfile bitfile/finn-accel.xclbin --conf_thres 0.3
```