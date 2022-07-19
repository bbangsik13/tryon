"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
if "STY" not in os.environ.keys():
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import wandb
from tqdm import tqdm

def main():
    # parse options
    opt = TrainOptions().parse()
    if opt.use_wandb:
        wandb.init(project=opt.name,reinit=True)
        wandb.config.update(opt)
    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader(opt)

    # create trainer for our model
    trainer = Pix2PixTrainer(opt)
    if opt.use_wandb:
        wandb.watch(trainer.pix2pix_model)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)

    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            trainer.run_discriminator_one_step(data_i)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                if opt.use_wandb:
                    errors ={}
                    for tag, value in losses.items():
                        value = value.mean().float()
                        errors[tag]=value
                    wandb.log(errors)
                #visualizer.print_current_errors(epoch, iter_counter.epoch_iter,losses, iter_counter.time_per_iter)
                #visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                #print(trainer.get_semantics().shape)
                visuals = OrderedDict([('input_label', trainer.get_semantics().cpu()[0:1,:-1,:,:]),
                                       ('mask', trainer.get_semantics().cpu()[0:1, -1, :, :]),
                                    ('synthesized_image', trainer.get_latest_generated()),
                                    ('real_image', data_i['ground truth img']),
                                    ('masked', trainer.get_mask())])



                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
        epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    print('Training was successfully finished.')

if __name__ == '__main__':
    main()