from argparse import ArgumentParser
import tensorflow as tf
import os
from train import RealTimeSrganTrainer
from data import DIV2K
import traceback

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
parser.add_argument('--hr_size', default=256, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                            help='The number of iterations to save the tensorboard summaries and models.')

def main():
    args = parser.parse_args()

    if not os.path.exists('model-weights'):
        os.makedirs('model-weights')


    train_loader = DIV2K(scale=4,
                         downgrade='bicubic',
                         subset='train')

    ds = train_loader.dataset(batch_size=args.batch_size,
                            random_transform=True,
                            repeat_count=1)

    # Define the directory for saving pretrainig loss tensorboard summary.
    pretrain_summary_writer = tf.summary.create_file_writer('logs/pretrain')
    train_summary_writer = tf.summary.create_file_writer('logs/train')
    
    RealTime_srgan_trainer = RealTimeSrganTrainer(args, train_summary_writer, pretrain_summary_writer) 

    #RealTime_srgan_trainer.pretrain_generator(ds)

    # Run training.
    RealTime_srgan_trainer.train(ds, args.save_iter, args.epochs)


if __name__=='__main__':
    # dynamically allocate GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_visible_devices(gpus[1],'GPU')
        #for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpus[1], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs', '\nUsing GPU : ', gpus[1])
      except Exception:
        traceback.print_exc()

    main()
