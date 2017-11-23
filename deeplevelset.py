'''
by timemao
2017/11/17-2017/11/17
refer to
1.dilated arch.
https://github.com/ndrplz/dilation-tensorflow.git
https://github.com/fyu/dilation
'''
# module 1: import modules
import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import os
import time

# module 2: parameters
batch_size,init_learning_rate,iteratives,eval_every= 32,1e-4,20000,1 # lr=1e-3 nan
step_rate,decay=1000,0.95
# pretrained
pretrained_file='vgg16_weights.npz'
pretrained_model = np.load(pretrained_file)
# checkpoint
checkpoint_dir='./checkpoint'
save_path='checkpoint/%d.ckpt'%iteratives
# data dir
data_path=os.path.abspath('/home/timemao/Documents/dataset/MSRA10K_Imgs_GT/Imgs/')
image_names = os.listdir(data_path)
image_train_names = []
for x in image_names:
    if x[-3:-1] == 'jp':image_train_names.append(x)
image_mean = np.divide([123.68,116.779,103.939],255.0)
# test
#visualize_path=os.path.join(data_path,image_train_names[np.random.choice(len(image_train_names),1)])
visualize_path=os.path.join(data_path,'75.jpg')#,os.path.join(data_path,'3048.jpg')] #3048
# module 3: configuration C net aritechture create
def createnetwork(input):
    # part 1
    w_conv1_1,b_conv1_1=tf.Variable(pretrained_model['conv1_1_W']),tf.Variable(pretrained_model['conv1_1_b'])
    w_conv1_2, b_conv1_2 = tf.Variable(pretrained_model['conv1_2_W']),tf.Variable(pretrained_model['conv1_2_b'])
    w_conv2_1,b_conv2_1=tf.Variable(pretrained_model['conv2_1_W']),tf.Variable(pretrained_model['conv2_1_b'])
    w_conv2_2, b_conv2_2 = tf.Variable(pretrained_model['conv2_2_W']),tf.Variable(pretrained_model['conv2_2_b'])

    w_conv3_1,b_conv3_1=tf.Variable(pretrained_model['conv3_1_W']),tf.Variable(pretrained_model['conv3_1_b'])
    w_conv3_2, b_conv3_2 = tf.Variable(pretrained_model['conv3_2_W']),tf.Variable(pretrained_model['conv3_2_b'])
    w_conv3_3, b_conv3_3 = tf.Variable(pretrained_model['conv3_3_W']),tf.Variable(pretrained_model['conv3_3_b'])

    w_conv_dil4_1,b_conv_dil4_1=weight_dilated_variable([3,3,256,512]),bias_variable([512])
    w_conv4_2, b_conv4_2 = tf.Variable(pretrained_model['conv4_2_W']),tf.Variable(pretrained_model['conv4_2_b'])
    w_conv4_3, b_conv4_3 = tf.Variable(pretrained_model['conv4_3_W']),tf.Variable(pretrained_model['conv4_3_b'])

    w_conv_dil5_1,b_conv_dil5_1=weight_dilated_variable([3,3,512,512]),bias_variable([512])
    w_conv_dil5_2, b_conv_dil5_2= weight_dilated_variable([3, 3, 512, 512]), bias_variable([512])
    w_conv_dil5_3, b_conv_dil5_3 = weight_dilated_variable([3, 3, 512, 512]), bias_variable([512])

    w_conv_dil6_1, b_conv_dil6_1 = weight_dilated_variable([3, 3, 512, 512]), bias_variable([512])
    w_conv6_2, b_conv6_2 = weight_varibale([3, 3, 512, 512]), bias_variable([512])

    w_conv7,b_conv7=weight_varibale([3,3,512,1]),bias_variable([1])
    # part 2
    h_conv1_1=tf.nn.relu(simpleconv2d(input,w_conv1_1,1)+b_conv1_1)
    h_conv1_2 = tf.nn.relu(simpleconv2d(h_conv1_1, w_conv1_2, 1) + b_conv1_2)
    max_pool1 = tf.nn.max_pool(h_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
    h_conv2_1=tf.nn.relu(simpleconv2d(max_pool1,w_conv2_1,1)+b_conv2_1)
    h_conv2_2 = tf.nn.relu(simpleconv2d(h_conv2_1, w_conv2_2, 1) + b_conv2_2)
    max_pool2 = tf.nn.max_pool(h_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
    h_conv3_1=tf.nn.relu(simpleconv2d(max_pool2,w_conv3_1,1)+b_conv3_1)
    h_conv3_2 = tf.nn.relu(simpleconv2d(h_conv3_1, w_conv3_2, 1) + b_conv3_2)
    h_conv3_3 = tf.nn.relu(simpleconv2d(h_conv3_2, w_conv3_3, 1) + b_conv3_3)

    h_conv_dil4_1 = tf.nn.relu(tf.nn.atrous_conv2d(h_conv3_3, w_conv_dil4_1, rate=2, padding='SAME') + b_conv_dil4_1)
    h_conv4_2 = tf.nn.relu(simpleconv2d(h_conv_dil4_1, w_conv4_2, 1) + b_conv4_2)
    h_conv4_3 = tf.nn.relu(simpleconv2d(h_conv4_2, w_conv4_3, 1) + b_conv4_3)

    h_conv_dil5_1 = tf.nn.relu(tf.nn.atrous_conv2d(h_conv4_3, w_conv_dil5_1, rate=2, padding='SAME') + b_conv_dil5_1)
    h_conv_dil5_2=tf.nn.relu(tf.nn.atrous_conv2d(h_conv_dil5_1,w_conv_dil5_2,rate=2,padding='SAME')+b_conv_dil5_2)
    h_conv_dil5_3 = tf.nn.relu(tf.nn.atrous_conv2d(h_conv_dil5_2, w_conv_dil5_3, rate=2, padding='SAME') + b_conv_dil5_3)

    h_conv_dil6_1 = tf.nn.relu(tf.nn.atrous_conv2d(h_conv_dil5_3, w_conv_dil6_1, rate=2, padding='SAME') + b_conv_dil6_1)
    h_conv6_2 = tf.nn.relu(simpleconv2d(h_conv_dil6_1, w_conv6_2, 1) + b_conv6_2)

    #with tf.device('/cpu:0'):
    #    h_conv7=tf.nn.max_pool(h_conv6_2,ksize=[1,1,1,512],strides=[1,1,1,512],padding='SAME',name='max_pooling')
    #size=input.get_shape()[0]
    #size=lambda train: 1 if train=='no' else batch_size
    h_conv7=simpleconv2d(h_conv6_2,w_conv7,1)+b_conv7

    #h_conv7=tf.reshape(tf.reduce_max(h_conv6_2,axis=3),[batch_size,56,56,1])
    h_conv_sig=tf.nn.sigmoid(h_conv7)

    h_final=h_conv_sig
    # h_final=tf.image.resize_bilinear(h_conv_sig,[224,224])
    return h_final
# module 4: train, plot
def train_network(input,h_final,sess,first_train):
    # part 1: loss
    groudtruth = tf.placeholder("float32", shape=[None, 56, 56, 1])
    loss = tf.reduce_sum(
        -tf.multiply(groudtruth, tf.log(h_final)) - tf.multiply((1 - groudtruth), tf.log(1 - h_final)))
    diff_binary_pixels_loss = tf.div(tf.reduce_sum(tf.square(groudtruth - h_final)), batch_size)
    # part 2: optimizer decay
    global_step = tf.placeholder(np.float32, shape=[])
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, step_rate, decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    if first_train == 'no':
        previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables('./checkpoint')]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables}
        tf.contrib.framework.init_from_checkpoint('./checkpoint', restore_map)
        print('not first train, use previous')
    sess.run(tf.global_variables_initializer())
    train_loss,binary_loss,iter = [],[],[]
    for k in range(iteratives):
        train_batch, label_batch = get_train_label_batch()
        train_dict = {global_step: k, input: train_batch, groudtruth: label_batch}
        before = time.clock()
        train_step.run(feed_dict=train_dict)
        temp_train_loss = sess.run(loss, feed_dict=train_dict)
        temp_binary_loss = sess.run(diff_binary_pixels_loss, feed_dict=train_dict)
        drtime = time.clock() - before
        if (k) % eval_every == 0:
            iter.append(k)
            train_loss.append(temp_train_loss)
            binary_loss.append(temp_binary_loss)
            print(' Iterative ' + str(k) + ' Temp_train_loss ' + str(temp_train_loss),
                  ' Temp_binary_loss ' + str(temp_binary_loss),
                  'Learning rate: %f' % (sess.run(optimizer._lr, feed_dict=train_dict)),
                  ' Used time ', drtime)
    return sess, iter, train_loss, binary_loss
# module 5: basic functions
def weight_varibale(shape):
    initial=tf.truncated_normal(shape,stddev=0.001)
    return tf.Variable(initial)
def weight_dilated_variable(shape):
    initial=np.array(np.zeros(shape),np.float32)
    initial[1,1,np.random.choice(shape[2],shape[2]//2),np.random.choice(shape[3],shape[2]//2)]=1
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.01,shape=shape)
    return tf.Variable(initial)
def simpleconv2d(x,w,stride):
    return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding="SAME")
def get_train_label_batch():
    train_data_index = np.random.choice(len(image_train_names), batch_size)
    train_names = [os.path.join(data_path,image_train_names[x]) for x in train_data_index]

    train_batch= dls_imread(train_names[0],3)
    label_name = train_names[0][0:-3] + 'png'
    label_batch= dls_imread(label_name, 1)
    for k in range(1,len(train_names)):
        batch_one=dls_imread(train_names[k],3)
        train_batch=np.append(train_batch,batch_one,axis=0)

        label_name = train_names[k][0:-3] + 'png'
        label_one=dls_imread(label_name,1)
        label_batch=np.append(label_batch,label_one,axis=0)
    return train_batch,label_batch
def dls_imread(image_path,color_gray):
    img = skimage.io.imread(image_path)
    if color_gray == 3:
        img_resize = np.array(skimage.transform.resize(img, (224, 224)), np.float32)
        img_resize[:, :, 0] -= image_mean[0]
        img_resize[:, :, 1] -= image_mean[1]
        img_resize[:, :, 2] -= image_mean[2]
        batch_one = img_resize.reshape((1, 224, 224, 3))
    else:
        img_resize = np.array(skimage.transform.resize(img, (56,56)), np.float32)
        batch_one = img_resize.reshape((1, 56, 56, 1))
    return batch_one
def visualize_groudtruth(visualize_path):
    original = skimage.io.imread(visualize_path)
    img_resize = np.array(skimage.transform.resize(original, (224, 224)), np.float32)
    groudtruth = np.squeeze(dls_imread(visualize_path[0:-3] + 'png', 1))
    plt.figure(1)
    plt.imshow(img_resize)
    plt.figure(2)
    plt.imshow(groudtruth, cmap='gray')
def train_loss_plot(iter,train_loss,binary_loss):
    plt.subplot(1,2,1)
    plt.plot(iter,train_loss,'r-')
    plt.title('train loss vs. iterative')
    plt.xlabel('iterative')
    plt.ylabel('train loss')
    plt.subplot(1,2,2)
    plt.plot(iter, binary_loss, 'r-')
    plt.title('sum binary loss vs. iterative')
    plt.xlabel('iterative')
    plt.ylabel('sum binary loss')
    plt.show()
# module 6 : test_programming
def main():
    sess=tf.InteractiveSession()
    input = tf.placeholder("float32", [None, 224, 224, 3])
    h_final=createnetwork(input)
    my_save = tf.train.Saver(max_to_keep=iteratives // eval_every)
    train,first_train='yes','no'
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if  checkpoint and checkpoint.model_checkpoint_path and first_train!='yes':
        my_save.restore(sess,checkpoint.model_checkpoint_path)
        print('loaded checkpoint file')
        if train=='no':
            visualize_groudtruth(visualize_path)
            img = dls_imread(visualize_path, 3)
            seg_out = tf.squeeze(h_final)
            seg_out = sess.run(seg_out, feed_dict={input: img})
            plt.figure(3)
            plt.title('predicted segment')
            plt.imshow(np.asarray(seg_out, np.float32), cmap='gray')
            plt.show()
        else:
            print('now continue to train...')
            sess, iter, train_loss, binary_loss = train_network(input, h_final, sess,first_train)
            train_loss_plot(iter, train_loss, binary_loss)
            my_save.save(sess, save_path)
    else:
        print('no checkpoint file, now to train...')
        sess,iter,train_loss,binary_loss=train_network(input,h_final,sess,'yes')
        train_loss_plot(iter,train_loss,binary_loss)
        my_save.save(sess,save_path)
def test1():
    x=weight_dilated_variable([3,3,256,512])
    return x

if __name__=="__main__":
    main()