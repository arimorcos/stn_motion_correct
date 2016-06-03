import image_manager

if __name__ == "__main__":

    # Create image handler
    im_handler = image_manager.\
        image_manager(batch_size=16, verbose=True, num_files_per_batch=2)
    im_handler.advance_schedule()

    x = 5

    # # Advance schedule for the first time
    # im_handler.advance_schedule()
    #
    # for batch in xrange(int(1e6)):
    #     # Advance the schedule
    #     im_handler.advance_schedule()
    #     print "batch: {:06d}".format(batch)
