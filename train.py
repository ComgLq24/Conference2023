import time
import os
import datetime
import copy
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    # Environment and parser settings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    opt = TrainOptions().parse()   # get training options

    # Dataset creation
    training_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    training_dataset_size = len(training_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % training_dataset_size)

    if opt.validate:
        opt_copy = copy.deepcopy(opt)
        opt_copy.phase = 'validation'
        opt_copy.dataroot = opt.validation_dataset_root
        opt_copy.batch_size = opt.validation_batch_size
        opt_copy.load_size = opt.validation_load_size
        opt_copy.preprocess = opt.validation_preprocess
        validation_dataset = create_dataset(opt_copy)
        validation_dataset.dataset.infos *= 1 + len(training_dataset) // len(validation_dataset)
        validation_dataset_size = len(validation_dataset)
        print('The number of validation images = %d' % validation_dataset_size)


    # Training model creation
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Visualizer setup
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    # training settings
    total_iters = 0                # the total number of training iterations
    training_start_time = time.time()

    # training procedure
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        # training settings for each epoch
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.curr_epoch += 1    
        if opt.validate:       # update epoch and iteration number
            validation_loader = enumerate(validation_dataset)

        # update learning rate
        model.update_learning_rate()   

        # enumerate dataset
        for i, training_data in enumerate(training_dataset): 
            
            # training settings for each iteration
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # train model
            model.set_input(training_data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # validate model
            if opt.validate:
                if opt.eval:
                    model.eval()
                validation_index, validation_data = validation_loader.__next__()
                with torch.no_grad():
                    model.validate(validation_data)    
                if opt.eval:
                    model.train()
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / training_dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()


            model.curr_iter += 1

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    print("Total training time: %d mins" % ((time.time() - training_start_time) / 60))
