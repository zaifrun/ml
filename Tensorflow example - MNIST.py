
# coding: utf-8

# # This notebook shows how to use Tensorflow to train on the MNIST dataset using a neural network.

# In this example we will load the MNIST dataset (this is a labelled dataset), then setup a neural network with 3 hidden layers to train on this data

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt


# After importing Tensorflow we define some helper methods we can then reuse in other projects maybe - after checking tensorflow is working

# In[2]:


print ("Tensorflow version "+str(tf.__version__))


# In[3]:


def neural_network_model(inputdata):
    # we use an input layer of 784 because the MNIST dataset are images of
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    # connecting the input data to the first hidden layer
    l1 = tf.add(tf.matmul(inputdata, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #using the RELU activation function
    l1 = tf.nn.relu(l1)

    # connecting the first hidden layer to the second layer 
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    #connecting the second hidden layer to the third hidden layer
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    # connecting the third hidden layer to the output layer
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


# We also need another helper function for doing the actual training of the network. 

# In[4]:


# x represents the input - in this case it is just a placeholder for the data, which will be loaded later inside this method
def train_neural_network(x):
    #use the input data to initialize the model to make our predictor
    model = neural_network_model(x)
    #This is the error function that we try to optimize
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    #This is the optimizing method we are using - you could user others such as gradient descent.
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
       # initialize the tensorflow network.
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # how many iterations should we do in this iteration
            for iter in range(int(mnist.train.num_examples / batch_size)):
                # get the X data and Y laebls from the MNIST dataset
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                #use this data to run the network through our model using our optimizeer
                # and our cost function with our x and y data from MNIST
                iter, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                #add the error to the total error for this epoch
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            error.append(epoch_loss)


        #training done - evalute on test test - how many did we correctly identify
        correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
       #evaluate the accurary on the MNIST TEST dataset - so this is a different dataset than
       # the training dataset
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


# Now after defining the helper methods we can define the main program

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data ## load input data
#loading the mnist data, that we then use inside the helper methods
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print(mnist.train)

#for saving the errors for each epoch - for printing.
error = []

#the following constants are used inside the helper methods, but are defined here.
#Number of nodes in hidden layer 1
n_nodes_hl1 = 500
#Number of nodes in hidden layer 2
n_nodes_hl2 = 500
#Number of nodes in hidden layer 3
n_nodes_hl3 = 500
#how many output neurons should we have (10 because there are 10 digits : 0...9)
n_classes = 10
#batch size
batch_size = 100
#nr of epocs
hm_epochs = 100 # number of training iterations

Now that the initial setup is over, we can call our helper methods
# In[ ]:



# the placeholder for the input - we have a single float array of 784 = 28x28 which is the resolution of our images
x = tf.placeholder('float', [None, 784])
# The output should just be a single value - i.e. the predicted number
y = tf.placeholder('float')

#calling our helper training function
train_neural_network(x)

#plotting our error
plt.plot(error,label='Error')
plt.legend()
plt.title("Tensorflow training on MNIST")
plt.xlabel("Training iteration")
plt.show()

