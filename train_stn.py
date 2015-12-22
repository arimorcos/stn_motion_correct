import networks
import os
import image_manager
import datetime


def do_train(batch_size=16):

    ################ NETWORK ###############
    # Create network
    network = networks.stn(batch_size=batch_size, save_every=100)

    ############### RUN #################
    # Train network

    num_batches = 1000*1000
    last_cost = 100000000

    # Get directory based on date
    root_dir = '/media/arimorcos/4TB External/stn_conv_net'
    new_log_dir = os.path.join(root_dir, datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
    network.set_log_dir(new_log_dir)

    print_every = 5

    for batch in range(num_batches):

        # Get data
        curr_batch_input, curr_batch_ref = im_handler.offer_data()

        temp_cost = network.train_adam(curr_batch_input, curr_batch_ref)

        # if temp_cost > 1.5*last_cost:
        #     del network
        #     print("Bad initialization...restarting...")
        #     do_train()
        #     return

        if batch % print_every == 0:
            print("Batch: {} | Cost: {:.5f}".format(batch, temp_cost.tolist()))

        last_cost = temp_cost

# Set batch size
batch_size = 16

# Create image handler
im_handler = image_manager.image_manager(batch_size=batch_size)

# Advance schedule for the first time
im_handler.advance_schedule()

do_train(batch_size=batch_size)
