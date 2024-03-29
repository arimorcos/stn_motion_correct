import networks
import os
import image_manager
import datetime
import numpy as np
import lasagne


def do_train(batch_size=32, cost_gap=500, anneal_thresh=0.0005,
             anneal_val=2, anneal_delay=1000):
    ################ NETWORK ###############
    # Create network
    network = networks.stn_tps(batch_size=batch_size, save_every=100,
                               initialization='he_normal', max_norm=5,
                               alpha=0.001)
    # network = networks.stn_affine(batch_size=batch_size, save_every=100,
    #                               initialization='glorot_normal', max_norm=15,
    #                               alpha=0.001)

    # force init
    # network.set_parameters('/media/arimorcos/4TB
    # External/stn_conv_net/160114_125908/Epoch_0000_weights.pkl')

    ############### RUN #################
    # Train network
    num_batches = 1000 * 1000

    # Get directory based on date
    root_dir = '/media/arimorcos/4TB External/stn_conv_net'
    new_log_dir = os.path.join(root_dir,
                               datetime.datetime.now().
                               strftime('%y%m%d_%H%M%S'))
    network.set_log_dir(new_log_dir)
    print "log directory: {}".format(network.get_log_dir())

    print_every = 5
    cost = []
    last_anneal = 0
    # anneal_thresh = 20
    # cost_gap = 1

    for batch in range(num_batches):

        # Get data
        curr_batch_input, curr_batch_ref = im_handler.offer_data()

        temp_cost = network.train_adam(curr_batch_input, curr_batch_ref)
        cost.append(temp_cost)

        if batch % print_every == 0:
            print("Batch: {} | Cost: {:.9f}".format(batch, temp_cost.tolist()))

        # Check if the
        if len(cost) >= cost_gap and \
                        np.abs(cost[-1] - cost[-cost_gap]) < anneal_thresh:
            if last_anneal > anneal_delay:
                new_lr = network.shared_lr.get_value() / anneal_val
                network.shared_lr.set_value(lasagne.utils.floatX(new_lr))
                last_anneal = 0
                anneal_thresh /= anneal_val
                msg = "Annealing learning rate. New rate: {:.9f}" \
                    .format(float(network.shared_lr.get_value()))
                print(msg)
                network.logger.info(msg)
            else:
                last_anneal += 1

        # Advance the schedule
        im_handler.advance_schedule()


# Set batch size
batch_size = 16

# Create image handler
im_handler = image_manager.image_manager(batch_size=batch_size)

# Advance schedule for the first time
im_handler.advance_schedule()

do_train(batch_size=batch_size)
