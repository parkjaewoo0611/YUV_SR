from ops import *
import os


class CbCrSRNN(object):
    # 5x5 64개, 3x3 32개, 3x3 (2^2)x2개 - 2
    def __init__(self, learning_rate=0.00001):
        self.learning_rate = learning_rate

    def build(self):
        self.x = tf.placeholder(tf.float32, name='x')
        self.y = tf.placeholder(tf.float32, name='y')

        with tf.variable_scope('conv1') as scope:
            self.w_conv1, self.b_conv1 = conv_variable([5, 5, 2, 64], 'conv1')

        with tf.variable_scope('conv2') as scope:
            self.w_conv2, self.b_conv2 = conv_variable([3, 3, 64, 32], 'conv2')

        with tf.variable_scope('conv3') as scope:
            self.w_conv3, self.b_conv3 = conv_variable([3, 3, 32, 8], 'conv3')

        h1 = conv2d(self.x, self.w_conv1, self.b_conv1)

        h1_relu = relu(h1)

        h2 = conv2d(h1_relu, self.w_conv2, self.b_conv2)

        h2_relu = relu(h2)

        h3 = conv2d(h2_relu, self.w_conv3, self.b_conv3)

        self.test_result = h3

        self.im_result = PS(h3, 2)

        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.im_result - self.y, name='loss'))
        tf.summary.scalar("loss", self.loss)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # merge all summaries into a single "operation" which we can execute in a session
        self.summary_op = tf.summary.merge_all()

        log_path = "./tensorboard"
        self.writer = tf.summary.FileWriter(log_path)
        self.saver = tf.train.Saver()

    def save(self, session, save_path):
        self.saver.save(session, os.path.join(save_path, 'model.ckpt'))

    def load(self, session, load_path):
        self.saver.restore(session, os.path.join(load_path, 'model.ckpt'))

    def training(self, session, X, Y, X_val=None, Y_val=None, epochs=10, batch_size=500):
        # For training the model
        data_num = Y.shape[0]
        num_steps = int(data_num / batch_size)
        loss = 0
        least_loss = math.inf
        for ep in range(epochs):
            loss = 0
            for step in range(num_steps):
                offset = (step * batch_size)
                # Generate a minibatch.
                mini_batch_data = X[step * (batch_size): (step + 1) * batch_size]
                mini_batch_labels = Y[step * (batch_size): (step + 1) * batch_size]

                feed_dict = {self.x: mini_batch_data, self.y: mini_batch_labels}

                [_, mini_batch_loss] = session.run([self.optimizer, self.loss],
                                                   feed_dict=feed_dict)

                summary = session.run(self.summary_op, feed_dict=feed_dict)
                self.writer.add_summary(summary, ep * num_steps + step)
                loss = loss + mini_batch_loss
            print("loss at epoch %d: %f" % (ep, loss / num_steps))

            if ep % 100 == 0:
                val_batch_size = 8
                val_data_num = Y_val.shape[0]
                val_num_steps = int(val_data_num / val_batch_size)
                val_loss = 0
                for step in range(val_num_steps):
                    offset = (step * val_batch_size)
                    # Generate a minibatch.
                    val_mini_batch_data = X_val[step * (val_batch_size): (step + 1) * val_batch_size]
                    val_mini_batch_labels = Y_val[step * (val_batch_size): (step + 1) * val_batch_size]

                    feed_dict = {self.x: val_mini_batch_data, self.y: val_mini_batch_labels}

                    val_mini_batch_loss = session.run(self.loss, feed_dict=feed_dict)
                    val_loss = val_loss + val_mini_batch_loss

                    # summary = session.run(self.summary_op, feed_dict=feed_dict)
                    # self.writer.add_summary(summary, ep * num_steps + step)

                print("Loss on Validataion Dataset %.4f" % (val_loss / val_num_steps))
                if loss < least_loss:
                    least_loss = loss
                    self.save(session, 'model_checkpoints')

        self.writer.close()
        print("Training done!")


    def test(self, session, X):

        feed_dict = {self.x: X}

        output = session.run(self.test_result, feed_dict=feed_dict)

        return output



