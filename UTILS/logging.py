from pathlib import Path

import tensorflow as tf

from UTILS.display import tight_grid, buffer_image
from UTILS.vec_ops import norm_tensor
from UTILS.decorators import ignore_exception

def control_frequency(f):
    def apply_func(*args, **kwargs):
        # args[0] is self
        plot_all = ('plot_all' in kwargs) and kwargs['plot_all']
        if (args[0].global_step % args[0].plot_frequency == 0) or plot_all:
            result = f(*args, **kwargs)
            return result
        else:
            return None
    
    return apply_func

class SummaryManager:
  """
  Writes tensorboard logs during training.

  :arg model: model object that is trained
  :arg log_dir: base directory where logs of a config are created
  :arg max_plot_frequency: every how many steps to plot
  """

  def __init__(self, 
               model: tf.keras.models.Model,
               log_dir: str,
               max_plot_frequency= 10,
               default_writer = 'log_dir'):
    self.model = model
    self.log_dir = Path(log_dir)
    self.plot_frequency = max_plot_frequency
    self.default_writer = default_writer
    self.writers = {}
    self.add_writer(tag= default_writer, path= self.log_dir, default= True)

  def add_writer(self, path, tag= None, default= False):
    """
     Adds a writer to self.writers if the writer does not exist already.
    To avoid spamming on disk.

      : return the writer on path with tag or path
      
    """

    if not tag:
      tag = path
    if tag not in self.writers.keys():
      self.writers[tag] = tf.summary.create_file_writer(str(path))
    if default:
      self.default_writer = tag
    return self.writers[tag]

  @property
  def global_step(self):
    return self.model.generator_optimizer.iterations

  #def add_scalars(self, tag, dictionary):
  #  for k in dictionary.keys():
  #    with self.add_writer(str(self.log_dir / k)).as_default():
  #      tf.summary.scalar(name=tag, data=dictionary[k], step=self.global_step)
 
  
  def add_image(self, tag, img, iterations):
    with self.writers[self.default_writer].as_default():
      tf.summary.image(name=tag, data=img, step=iterations, max_outputs=4)

  def add_scalar(self, tag, scalar_value, iterations):
    with self.writers[self.default_writer].as_default():
      tf.summary.scalar(name=tag, data=scalar_value, step=iterations)

  @ignore_exception
  def display_loss(self, loss_val, iterations, tag= '', plot_all= False):
    #self.add_scalars(tag=f'{tag}/losses', dictionary=output['losses'])
    self.add_scalar(tag=f'{tag}/loss', scalar_value=loss_val, iterations=iterations)

  @control_frequency
  @ignore_exception
  def display_scalar(self, tag, scalar_value, plot_all= False):
    self.add_scalar(tag=tag, scalar_value=scalar_value)

  def display_image(self, images, tag, iterations):
      titles = ['LR', 'SR (PRE)', 'SR (GAN)']
      for i, (img, title) in enumerate(zip(images, titles)):
          tag = tag+'/'+title
          self.add_image(tag, tf.expand_dims(img,0), iterations)





