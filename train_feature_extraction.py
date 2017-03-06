import time
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

nb_classes = 43
# Hyper params
rate = 1e-4
EPOCHS = 40
BATCH_SIZE = 128

# TODO: Load traffic signs data.
training_file = 'train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    train['features'], train['labels'], test_size=0.33, random_state=42)
# TODO: Define placeholders and resize operation.
#(?, 32, 32, 3)
img_shape = X_train[0].shape
CHANNEL = img_shape[2]
x = tf.placeholder(tf.float32, (None, img_shape[0], img_shape[1], CHANNEL))
labels = tf.placeholder(tf.int64, None)
# Required by AlexNet
resized = tf.image.resize_images(x, (227, 227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8_w = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=1e-2))
fc8_b = tf.Variable(tf.zeros(nb_classes))
#logits = tf.matmul(fc7, fc8_w) + fc8_b
logits = tf.nn.xw_plus_b(fc7, fc8_w, fc8_b)
print(logits.get_shape())
probs = tf.nn.softmax(logits)
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
# WHEN TO USE sparse?
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss_operation = tf.reduce_mean(cross_entropy)
# Should we use AdamOptimizer with certain learning rate?
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8_w, fc8_b])

# TODO: Train and evaluate the feature extraction model.
correct_prediction = tf.equal(tf.argmax(logits, 1), 1)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data, sess):
#  num_examples = len(X_data)
  num_examples = X_data.shape[0]
  total_accuracy = 0
  sess = tf.get_default_sesssion()
  for offset in range(0, num_examples, BATCH_SIZE):
    end = offset + BATCH_SIZE
    batch_x, batch_y = X_data[offset: end], y_data[offset:end]
    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, labels: batch_y})
    total_accuracy += (accuracy * len(batch_x))
  return total_accuracy/ num_examples

# Run
t = time.time()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  nums_examples = len(X_train)
  print("Training...")
  print()
  for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, nums_examples, BATCH_SIZE):
      end = offset + BATCH_SIZE
      batch_x, batch_y = X_train[offset:end], y_train[offset:end]
      sess.run(training_operation, feed_dict={x:batch_x, labels:batch_y})
     # batch_accuracy = sess.run(training_operation, feed_dict={x:batch_x, labels:batch_y})
    validation_accuracy = evaluate(X_valid, y_valid, sess)
    print("EPOCH {}...".format(i + 1))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))

print("Time: %.3f seconds"%(time.time() - t))
