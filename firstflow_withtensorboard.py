import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime


def log_scalar(writer, tag, value, step):
    """Log a scalar variable.
    Parameter
    ----------
    tag : basestring
        Name of the scalar
    value
    step : int
        training iteration
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                 simple_value=value)])
    writer.add_summary(summary, step)

print ("Tensorflow version "+str(tf.__version__))
x = tf.Variable(3,name="x")
y = tf.Variable(4,name="y")
f = x*x*y+y+2

init = tf.global_variables_initializer()
session = tf.InteractiveSession()
init.run()
result = f.eval()



print("result: "+str(result))  # should give 42 - the answer to everything


# Model parameters - initialize to something
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b   # the model we are trying to find the best value of W and b for our data
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares of the difference
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data - - in this example they fit perfectly to the model : y = -1 * x + 1,
# so we should in an optimal training situation be able to achieve an error of 0 in theory.
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training init
init = tf.global_variables_initializer()
sess = tf.Session()


# setup a logdir for tensorflow to use
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir,datetime.utcnow().strftime("%Y%m%d%H%M%S"))
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
sess.run(init) # set values to our initialials
error = []
slope = []
bvalue = []
# training
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train}) # train using our data - and minimize the loss function
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    error.append(curr_loss)
    log_scalar(file_writer,"error",curr_loss,i)
    slope.append(curr_W)
    bvalue.append(curr_b)

file_writer.close()
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
session.close()
plt.plot(error,label='Error')
plt.plot(slope,label='Slope')
plt.plot(bvalue,label='constant,b')
plt.legend()
plt.title("Tensorflow training to match data to the linear model of y = -1*x + 1")
plt.xlabel("Training iteration")
plt.show()