import tensorflow as tf
from .sgdr_decay import sgdr_decay
import math
import time
import os
from src.utils import count_params
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits as ce_loss
from tensorflow.contrib.tensorboard.plugins import projector


class Estimator:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        data.create_embedding_metadata()

        # Create placeholders for loss, we will run summary op only once in a while but we
        # still would like to log loss over the whole dataset
        self.train_rec_loss_p = tf.placeholder(tf.float32, name='train_reconstruction_loss')
        self.train_disc_loss_p = tf.placeholder(tf.float32, name='train_discriminator_loss')
        self.train_enc_loss_p = tf.placeholder(tf.float32, name='train_encoder_loss')

        self.val_rec_loss_p = tf.placeholder(tf.float32, name='val_reconstruction_loss')

        self.global_step = tf.Variable(0, trainable=False)

        # One learning rate for all submodules
        self.learning_rate = sgdr_decay(0.05, self.global_step, t_0=1000, mult_factor=1, name='lr')

        # Encoder part
        z_train = self.model.encoder(queue=data.train_batch)
        z_val = self.model.encoder(queue=data.val_batch, reuse=True)

        # Decoder part
        x_train_rec = self.model.decoder(z_train)
        x_val_rec = self.model.decoder(z_val, reuse=True)

        # Prior on z
        z_sam = self.model.sampler()

        # Discriminator part (on latent representation z)
        t_fake = self.model.discriminator(z_train)
        t_real = self.model.discriminator(z_sam, reuse=True)

        # Gan part

        x_train_fake = self.model.gan(x_train_rec)
        x_val_fake = self.model.gan(x_val_rec, reuse=True)

        t_gan_fake = self.model.critic(x_train_fake)
        t_gan_real = self.model.critic(data.train_batch[0], reuse=True)

        # Losses
        # Reconstruction loss
        self.train_rec_loss = tf.reduce_mean(tf.square(x_train_rec - data.train_batch[0]))
        self.val_rec_loss = tf.reduce_mean(tf.square(x_val_rec - data.val_batch[0]))

        # Discriminator loss
        self.train_disc_loss = (tf.reduce_mean(ce_loss(logits=t_real, labels=tf.ones_like(t_real))) +
                                tf.reduce_mean(ce_loss(logits=t_fake, labels=tf.zeros_like(t_fake))))/2.0

        # Encoder loss
        self.train_enc_loss = tf.reduce_mean(ce_loss(logits=t_fake, labels=tf.ones_like(t_real)))

        # Critic loss
        self.train_critic_loss = (tf.reduce_mean(ce_loss(logits=t_gan_real, labels=tf.ones_like(t_gan_real))) +
                                  tf.reduce_mean(ce_loss(logits=t_gan_fake, labels=tf.zeros_like(t_gan_fake))))/2.0

        # Gan loss
        self.train_gan_loss = tf.reduce_mean(ce_loss(logits=t_gan_fake, labels=tf.ones_like(t_gan_real)))

        # Summaries
        train_summaries = []
        val_summaries = []

        train_summaries.append(tf.summary.scalar('learning_rate', self.learning_rate))

        train_summaries.append(tf.summary.scalar("train_rec_loss", self.train_rec_loss))
        val_summaries.append(tf.summary.scalar("val_rec_loss", self.val_rec_loss))

        train_summaries.append(tf.summary.scalar("train_disc_loss", self.train_disc_loss))
        train_summaries.append(tf.summary.scalar("train_enc_loss", self.train_enc_loss))

        train_summaries.append(tf.summary.histogram("z_train", z_train))
        val_summaries.append(tf.summary.histogram("z_val", z_val))

        val_summaries.append(tf.summary.tensor_summary("z_val", z_val))

        val_summary_size = 1

        val_summaries.append(tf.summary.image("Val_Img", tf.transpose(self.unroll_images(data.val_batch[0]), [0, 2, 3, 1]),
                                              max_outputs=val_summary_size))
        val_summaries.append(tf.summary.image("Val_Rec", tf.transpose(self.unroll_images(x_val_rec), [0, 2, 3, 1]),
                                              max_outputs=val_summary_size))
        val_summaries.append(tf.summary.image("Val_Gan", tf.transpose(self.unroll_images(x_val_fake), [0, 2, 3, 1]),
                                              max_outputs=val_summary_size))

        self.train_summaries = tf.summary.merge(train_summaries)
        self.val_summaries = tf.summary.merge(val_summaries)

        # Embeddings
        embedding_var = tf.get_variable(name='z_embedding',
                                        shape=[self.model.batch_size, self.model.z_dim+self.model.y_dim])
        self.embedding_op = tf.assign(embedding_var, z_val)
        self.embedding_config = projector.ProjectorConfig()
        self.embedding = self.embedding_config.embeddings.add()
        self.embedding.tensor_name = 'z_embedding'
        self.embedding.metadata_path = 'metadata/metadata.tsv'
        self.embedding.sprite.image_path = 'metadata/sprite.png'
        self.embedding.sprite.single_image_dim.extend([model.image_size, model.image_size])

        # Optimizers
        self.rec_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)\
            .minimize(self.train_rec_loss, global_step=self.global_step,
                      var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='(encoder|decoder)'))
        self.disc_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0)\
            .minimize(self.train_disc_loss,
                      var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
        self.enc_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0) \
            .minimize(self.train_enc_loss,
                      var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder'))
        self.gan_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0) \
            .minimize(self.train_gan_loss,
                      var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gan'))
        self.critic_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0) \
            .minimize(self.train_critic_loss,
                      var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))

    def run_training(self, iterations):
        sv = tf.train.Supervisor(logdir=os.path.join('logs', time.strftime("%Y%m%d-%H%M%S")), summary_op=None,
                                 global_step=self.global_step, save_model_secs=3600)

        with sv.managed_session() as sess:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            tf.logging.log(tf.logging.INFO, "Number of parameters %d" % count_params())

            train_rec_loss = 0
            train_disc_loss = 0
            train_enc_loss = 0
            log_time = 100
            for i in range(1, iterations+1):
                r = sess.run([self.train_rec_loss, self.rec_optimizer,
                              self.train_disc_loss, self.disc_optimizer,
                              self.train_enc_loss, self.enc_optimizer,
                              self.gan_optimizer, self.critic_optimizer],
                             feed_dict={self.model.is_training: True})
                loss_r, _, loss_d, _, loss_e, _, _, _ = r
                train_rec_loss += loss_r
                train_disc_loss += loss_d
                train_enc_loss += loss_e
                # Compute summary on training on every 100th iteration
                if i % log_time is 0:
                    t_s = sess.run(self.train_summaries, feed_dict={self.model.is_training: False,
                                                                    self.train_rec_loss_p: train_rec_loss/log_time,
                                                                    self.train_disc_loss_p: train_disc_loss/log_time,
                                                                    self.train_enc_loss_p: train_enc_loss/log_time})
                    tf.logging.log(tf.logging.INFO, "\nIteration %d" % i)
                    tf.logging.log(tf.logging.INFO, "Reconstruction Loss %g" % (train_rec_loss/log_time))
                    tf.logging.log(tf.logging.INFO, "Discriminator Loss %g" % (train_disc_loss/log_time))
                    tf.logging.log(tf.logging.INFO, "Encoder Loss %g" % (train_enc_loss/log_time))
                    #tf.logging.log(tf.logging.INFO, "Gan Loss %g" % (train_enc_loss/log_time))
                    #tf.logging.log(tf.logging.INFO, "Critic Loss %g" % (train_enc_loss/log_time))
                    sv.summary_computed(sess, t_s)
                    train_rec_loss = 0
                    train_disc_loss = 0
                    train_enc_loss = 0

                    self.run_validation(sv, sess)

    # Validation pass is cheap, just 5 batches
    def run_validation(self, sv, sess):
        val_rec_loss = 0
        val_iterations = self.data.val_size // self.model.batch_size
        for _ in range(val_iterations-1):
            loss = sess.run(self.val_rec_loss, feed_dict={self.model.is_training: False})
            val_rec_loss += loss

        _, v_s = sess.run([self.embedding_op, self.val_summaries],
                          feed_dict={self.model.is_training: False,
                                     self.val_rec_loss_p: val_rec_loss/(val_iterations-1)})

        tf.logging.log(tf.logging.INFO, "Reconstruction Loss (Validation) %g\n" % (val_rec_loss / (val_iterations-1)))
        sv.summary_computed(sess, v_s)
        projector.visualize_embeddings(sv.summary_writer, self.embedding_config)

    # Takes all Images and creates one big image
    def unroll_images(self, c_i):
        batch_size = self.model.batch_size
        img_size = self.model.image_size

        x_len = int(math.sqrt(batch_size))
        y_len = int(batch_size // x_len)
        x_len = x_len + 1 if (x_len * y_len < batch_size) else x_len

        blank_images_n = x_len * y_len - batch_size
        c_i = tf.concat([c_i, tf.zeros([blank_images_n, 3, img_size, img_size])], axis=0)

        c_i = tf.split(c_i, x_len, axis=0)
        c_i = tf.concat(c_i, axis=2)
        c_i = tf.split(c_i, y_len, axis=0)
        c_i = tf.concat(c_i, axis=3)
        return c_i









