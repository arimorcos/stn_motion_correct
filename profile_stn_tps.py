import lasagne
import networks
import numpy as np
import shutil
import os

if __name__ == "__main__":

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    batch_size = 16

    # Create network
    network = networks.stn_tps(batch_size=batch_size, save_every=250,
                               initialization='he_normal', max_norm=5,
                               alpha=0.0005)

    # set temp directory
    temp_dir = "/temp"

    # generate fake input
    net_input = np.random.normal(0, 1, size=(batch_size, 2, 512, 512))
    net_input = lasagne.utils.floatX(net_input)

    # process
    network.process(net_input)

    # delete directory
    shutil.rmtree(temp_dir)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
