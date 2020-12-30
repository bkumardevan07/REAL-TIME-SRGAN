import time
import tensorflow as tf

from model import evaluate
from model import srgan
from model.real_time_srgan import RealTimeSRGAN

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/srgan'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

class SrganGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganTrainer:
    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        if content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be 'VGG54'")

        self.content_loss = content_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_dataset, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        for lr, hr in train_dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                pls_metric.reset_states()
                dls_metric.reset_states()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            mse_loss = self._mse_loss(hr, sr)
            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss + mse_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss
	
    @tf.function
    def _mse_loss(self, hr, sr):
        return self.mean_squared_error(hr, sr)

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss

class RealTimeSrganTrainer:
    def __init__(self, args, train_writer, pretrain_writer):
        self.model = RealTimeSRGAN(args)
        self.writer = train_writer
        self.pretrain_writer = pretrain_writer

    @tf.function
    def pretrain_step(self, x, y):
        with tf.GradientTape() as tape:
            fake_hr = self.model.generator(x)
            loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)
    
        grads = tape.gradient(loss_mse, self.model.generator.trainable_variables)
        self.model.gen_optimizer.apply_gradients(zip(grads, self.model.generator.trainable_variables))
    
        return loss_mse

    def pretrain_generator(self, dataset):
        with self.pretrain_writer.as_default():
            iteration = 0
            for _ in range(5):
                for x, y in dataset:
                    loss = self.pretrain_step(x, y)
                    if iteration % 20 == 0:
                        tf.summary.scalar('MSE Loss', loss, step=tf.cast(iteration, tf.int64))
                        self.pretrain_writer.flush()
                    if iteration % 50 == 0 :
                        print(f'MSE loss: {loss}')
                    iteration += 1

    @tf.function
    def train_step(self, x, y):
        # Label smoothing for better gradient flow
        valid = tf.ones((x.shape[0],) + self.model.disc_patch)
        fake = tf.zeros((x.shape[0],) + self.model.disc_patch)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_hr = self.model.generator(x)
    
            valid_prediction = self.model.discriminator(y)
            fake_prediction = self.model.discriminator(fake_hr)
    
            # Generator loss
            content_loss = self.model.content_loss(y, fake_hr)
            adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
            mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
            perceptual_loss = content_loss + adv_loss + mse_loss
    
            # Discriminator loss
            valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
            fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
            d_loss = tf.add(valid_loss, fake_loss)
    
        # Backprop on Generator
        gen_grads = gen_tape.gradient(perceptual_loss, self.model.generator.trainable_variables)
        self.model.gen_optimizer.apply_gradients(zip(gen_grads, self.model.generator.trainable_variables))
    
        # Backprop on Discriminator
        disc_grads = disc_tape.gradient(d_loss, self.model.discriminator.trainable_variables)
        self.model.disc_optimizer.apply_gradients(zip(disc_grads, self.model.discriminator.trainable_variables))
        return d_loss, adv_loss, content_loss, mse_loss


    def train(self, dataset, log_iter, epochs):

        with self.writer.as_default():
            for _ in range(epochs):
                for x, y in dataset:
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    disc_loss, adv_loss, content_loss, mse_loss = self.train_step(x, y)
                    if self.model.iterations % log_iter == 0:
                        tf.summary.scalar('Adversarial Loss', adv_loss, step=self.model.iterations)
                        tf.summary.scalar('Content Loss', content_loss, step=self.model.iterations)
                        tf.summary.scalar('MSE Loss', mse_loss, step=self.model.iterations)
                        tf.summary.scalar('Discriminator Loss', disc_loss, step=self.model.iterations)
                        tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=self.model.iterations)
                        tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=self.model.iterations)
                        tf.summary.image('Generated', tf.cast(255 * (self.model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                         step=self.model.iterations)
                        self.model.generator.save('model-weights/generator.h5')
                        self.model.discriminator.save('model-weights/discriminator.h5')
                        self.writer.flush()
                    self.model.iterations += 1
                    if self.model.iterations % 50==1:
                        print(f'Iteration {self.model.iterations}, disc loss: {disc_loss}, adv loss: {adv_loss}, content loss: {content_loss}, mse loss: {mse_loss}')

