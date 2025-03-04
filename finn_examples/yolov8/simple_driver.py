
# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device

from os.path import join
from time import time
import cv2

from yolov8 import plot_one_box, DetectorDriver, scale_coords

# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['INT21'], DataType['INT21'], DataType['INT21']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 192, 320, 3)],
    "oshape_normal" : [(1, 24, 40, 144), (1, 12, 20, 144), (1, 6, 10, 144)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 192, 320, 3, 1)],
    "oshape_folded" : [(1, 24, 40, 144, 1), (1, 12, 20, 144, 1), (1, 6, 10, 144, 1)],
    "ishape_packed" : [(1, 192, 320, 3, 1)],
    "oshape_packed" : [(1, 24, 40, 144, 3), (1, 12, 20, 144, 3), (1, 6, 10, 144, 3)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0', 'odma1', 'odma2'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 3,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="alveo")
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights', default="runtime_weights/")
    parser.add_argument('--sequence_dir', help='path to the folder with input images', type=str, default='img1')
    parser.add_argument('--input', help='path to input image', type=str)
    parser.add_argument('--conf_thres', help='confidence threshold', type=float, default=0.3)
    # parse arguments
    args = parser.parse_args()
    platform = args.platform
    bitfile = args.bitfile
    input_img = args.input
    runtime_weight_dir = args.runtime_weight_dir
    devID = args.device
    device = Device.devices[devID]
    output_file = 'demo_result.jpg'

    detector_driver = DetectorDriver(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = 1,
        runtime_weight_dir = runtime_weight_dir, device=device
    )

    img = cv2.imread(input_img)
    detections = detector_driver.single_inference(img, conf_thres=args.conf_thres)       
    visualized_img = detector_driver.visualize(img, detections)
    cv2.imwrite(output_file, visualized_img)
    print('Result saved as', output_file)
