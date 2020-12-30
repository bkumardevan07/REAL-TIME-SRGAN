from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganGeneratorTrainer, SrganTrainer
import tensorflow as tf
import traceback
from UTILS.logging import SummaryManager

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


train_loader = DIV2K(scale=4,             
                     downgrade='bicubic', 
                     subset='train')      

train_ds = train_loader.dataset(batch_size=16,         
                                random_transform=True, 
                                repeat_count=None)     

valid_loader = DIV2K(scale=4,             
                     downgrade='bicubic', 
                     subset='valid')      

valid_ds = valid_loader.dataset(batch_size=1,           
                        random_transform=False, 
                        repeat_count=None)      

# Create a training context for the generator (SRResNet) alone.
pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pretrain-mse_3e-6_con_from_scratch')

# Pre-train the generator with 100,000 steps
pre_trainer.train(train_ds, valid_ds.take(10), steps=100000, evaluate_every=1000) 

# Save weights of pre-trained generator (needed for fine-tuning with GAN).
pre_trainer.model.save_weights('weights/srgan/pretrainer.h5')

#import sys
print('Pre generator done.')
#sys.exit(1)

prev_generator = generator()
prev_generator.load_weights('weights/srgan/pre_generator.h5')
gan_generator = generator()
gan_generator.load_weights('weights/srgan/pre_generator.h5')

# Create a training context for the GAN (generator + discriminator).
gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())

# Train the GAN with 200,000 steps.
gan_trainer.train(train_ds, steps=200000)

# Save weights of generator and discriminator.
gan_trainer.generator.save_weights('weights/srgan/generator.h5')
gan_trainer.discriminator.save_weights('weights/srgan/discriminator.h5')
