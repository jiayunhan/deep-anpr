import cv2
import numpy
import tensorflow as tf
import sys
import keras
import model
import matplotlib.pyplot as plt
import math
import common

def valid_imshow_data(data):
    data = numpy.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False

def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))

def attack(im, param_vals):
	scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))
	img = scaled_ims[2]
	input = numpy.stack([img])

	x_hat = tf.Variable(tf.zeros(input.shape), dtype = tf.float32)

	assign_op = tf.assign(x_hat,input)

	_, y, params = model.get_detect_model(x_hat)

	y_val = tf.reduce_mean(y)

	optim_step = tf.train.GradientDescentOptimizer(
		1e-1).minimize(y_val,var_list= [x_hat])

	with tf.Session(config = tf.ConfigProto()) as sess:
		sess.run(assign_op)
		feed_dict = {}
		feed_dict.update(dict(zip(params, param_vals)))
		for i in range(1):
			sess.run(optim_step, feed_dict = feed_dict)
			print(sess.run(y_val, feed_dict = feed_dict))
		adv = x_hat.eval()
	adv = adv[0,:,:]
	print(adv)
	print(adv.shape)
	print(img)
	print(img.shape)
	#valid_imshow_data(adv.shape)
	plt.imshow(adv)
	plt.show()

def detect_presense(im, param_vals):
	scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))
	img = scaled_ims[2]
	x, y, params = model.get_detect_model()

	with tf.Session(config = tf.ConfigProto()) as sess:
		feed_dict = {x: numpy.stack([img])}
		feed_dict.update(dict(zip(params, param_vals)))
		y_val = sess.run(y, feed_dict = feed_dict)
		y_val[0, :, :, 0]
	if(len(numpy.argwhere(y_val[0, :, :, 0] > -math.log(1./0.99 - 1)))):
		print('[FOUND PLATE]')

def detect(im, param_vals):

    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))
    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
    	#print(i)
    	#print(numpy.argwhere(y_val[0, :, :, 0] > -math.log(1./0.99 - 1)))
        #print(-math.log(1./0.99 - 1))
        #print(numpy.argwhere(y_val[0, :, :, 0] >-math.log(1./0.99 - 1)))
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])
            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs

if __name__ == "__main__":
	print(sys.argv[1])
	img = cv2.imread(sys.argv[1])
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.

	#cv2.imwrite(sys.argv[3], img_gray)
	f = numpy.load(sys.argv[2])
	param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

	attack(img_gray, param_vals)
	#attack(img_gray, param_vals)
	#for _, _, present_prob, letter_probs in detect(img_gray,param_vals):
	#	pass

	#sess = tf.Session()
	#keras.backend.set_session(sess)
