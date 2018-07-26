#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines to detect number plates.

Use `detect` to detect all bounding boxes, and use `post_process` on the output
of `detect` to filter using non-maximum suppression.

"""


__all__ = (
	'detect',
	'post_process',
)


import collections
import itertools
import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model

from detect_get_logits import get_attack_logits 

def make_scaled_ims(im, min_shape):
	ratio = 1. / 2 ** 0.5
	shape = (im.shape[0] / ratio, im.shape[1] / ratio)

	while True:
		shape = (int(shape[0] * ratio), int(shape[1] * ratio))
		if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
			break
		yield cv2.resize(im, (shape[1], shape[0]))


def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    # Convert the image to various scales.
    #scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))
    scaled_ims = hide_attack(im, param_vals)
    #print(scaled_ims[0].shape)

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    print(y_vals[0].shape)
    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
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

def hide_attack_color(im, param_vals):
	scaled_ims_perturb = hide_attack(im, param_vals)

	scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))

	def get_avg_loss(scaled_ims_perturb, scaled_ims):
		loss = 0
		length = len(scaled_ims)

		return loss/length 
	
	# optim_step = tf.train.GradientDescentOptimizer(1e-1).minimize



def hide_attack(im, param_vals):
	scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))
	print('LENGTH', len(scaled_ims))

	def perturb(img):
		input = numpy.stack([img])
		x_hat = tf.Variable(tf.zeros(input.shape))
		assign_op = tf.assign(x_hat, input)
		_, y, params = model.get_detect_model(x_hat)

		y_mean = tf.reduce_mean(y)

		optim_step = tf.train.GradientDescentOptimizer(1e-1).minimize(y_mean, var_list = [x_hat])
		adv = []
		init = tf.global_variables_initializer()

		with tf.Session(config=tf.ConfigProto()) as sess:
			sess.run(init)
			sess.run(assign_op)
			feed_dict = {}
			feed_dict.update(dict(zip(params, param_vals)))
			for i in range(10):
				sess.run(optim_step,feed_dict = feed_dict)
				print(sess.run(y_mean, feed_dict = feed_dict))
			adv = (x_hat.eval())

		list0 = []
		for i in adv[0]:
			list0.append(i)
			arr = numpy.array(list0)
		arr = arr*255 #scales the values so cv2 can write image to file properly 
		for i in range(arr.shape[0]): 
			for j in range(arr.shape[1]):
				if arr[i][j] <0:
					arr[i][j] = 0
		cv2.imwrite('NewScaledImage', arr)
		return arr


	for i in range(len(scaled_ims)):
		scaled_ims[i] = perturb(scaled_ims[i])
		print(scaled_ims[i])

	return scaled_ims

def attack(im, param_vals, attack_targets):

	x, y, params = model.get_detect_model()
	#letter_prob_arr = []
	# Execute the model at each scale.
	print(zip(params, param_vals))
	y = sess.run(y, feed_dict=dict(zip(params, param_vals)))

	logits_arr = get_attack_logits(im, param_vals, attack_targets, sess)
   
	print("Y: ", y)

	# #print(tf.reshape(y[:, 1:],[-1, len(common.CHARS)]))
	# for tl, br, present_prob, letter_probs in post_process(detect(im, param_vals)):
	#     letter_prob_arr.append(letter_probs)
	  

	#Modified PGD code from www.anishathalye.com/2017/07/25/synthesizing-adversarial-examples/
	def pgd_attack(im, target_class, y, class_num):
		#need to change image dimensions
		image = tf.Variable(tf.zeros((im.shape[0], im.shape[1], 3))) 
		x_hat = image 
		x = tf.placeholder(tf.float32, (im.shape[0], im.shape[1], 3))

		assign_op = tf.assign(x_hat, x)
		learning_rate = tf.placeholder(tf.float32, ())
		y_hat = tf.placeholder(tf.int32, ())

		#26 letters + 10 numbers = 36 total classes 
		labels = tf.one_hot(y_hat, class_num)

		print("labels: ", labels)
		#want to minimize y and y_hat
		loss = tf.nn.softmax_cross_entropy_with_logits(logits = y, labels=[labels])
		optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list = [x_hat])
		
		epsilon = tf.placeholder(tf.float32, ())
		below = x - epsilon
		above = x + epsilon 
		projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)

		#projected gradient descent 
		with tf.control_dependencies([projected]):
			project_step = tf.assign(x_hat, projected)
			train_epsilon = 2.0/255.0
			train_lr = 1e-1
			train_steps = 100
			train_target = target_class
			sess.run(assign_op, feed_dict={x: im})

			for i in range(train_steps):
				_, loss_value = sess.run([optim_step, loss], feed_dict = {learning_rate: train_lr, y_hat:train_target})
				sess.run(project_step, feed_dict={x: im, epsilon: train_epsilon})
				print('step %d, loss = %g' % (i+1, loss_value))

		adv = x_hat.eval()
		return adv
	
	#iterate PGD for each letter we want to skew the license plate to 
	for i in range(len(logits_arr)):
		print('Letter Index: ', i+1)
		#36 class numbers, 10nums + 26letters 
		im = pgd_attack(im, attack_targets[i], y, 36)

	return im 

def _overlaps(match1, match2):
	bbox_tl1, bbox_br1, _, _ = match1
	bbox_tl2, bbox_br2, _, _ = match2
	return (bbox_br1[0] > bbox_tl2[0] and
			bbox_br2[0] > bbox_tl1[0] and
			bbox_br1[1] > bbox_tl2[1] and
			bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
	matches = list(matches)
	num_groups = 0
	match_to_group = {}
	for idx1 in range(len(matches)):
		for idx2 in range(idx1):
			if _overlaps(matches[idx1], matches[idx2]):
				match_to_group[idx1] = match_to_group[idx2]
				break
		else:
			match_to_group[idx1] = num_groups 
			num_groups += 1

	groups = collections.defaultdict(list)
	for idx, group in match_to_group.items():
		groups[group].append(matches[idx])

	return groups


def post_process(matches):
	"""
	Take an iterable of matches as returned by `detect` and merge duplicates.

	Merging consists of two steps:
	  - Finding sets of overlapping rectangles.
	  - Finding the intersection of those sets, along with the code
		corresponding with the rectangle with the highest presence parameter.

	"""
	groups = _group_overlapping_rectangles(matches)

	for group_matches in groups.values():
		mins = numpy.stack(numpy.array(m[0]) for m in group_matches)
		maxs = numpy.stack(numpy.array(m[1]) for m in group_matches)
		present_probs = numpy.array([m[2] for m in group_matches])
		letter_probs = numpy.stack(m[3] for m in group_matches)

		yield (numpy.max(mins, axis=0).flatten(),
			   numpy.min(maxs, axis=0).flatten(),
			   numpy.max(present_probs),
			   letter_probs[numpy.argmax(present_probs)])
def print_help():
	print("\nToo few arguments. Expected: 3")
	print("Usage: python detect.py [input_image] [weights] [output_image]\n")

def letter_probs_to_code(letter_probs):
	return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


if __name__ == "__main__":
    if(len(sys.argv)<4):
        print_help()
        exit()
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    f = numpy.load(sys.argv[2])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    #hide_attack(im_gray, param_vals)

    # list0 = []
    # for i in image[0]:
    # 	list0.append(i)
    # arr = numpy.array(list0)
    # print(arr)


    for pt1, pt2, present_prob, letter_probs in post_process(
                                                  detect(im_gray, param_vals)):
        pt1 = tuple(reversed(map(int, pt1)))
        pt2 = tuple(reversed(map(int, pt2)))

        code = letter_probs_to_code(letter_probs)

        color = (0.0, 255.0, 0.0)
        cv2.rectangle(im, pt1, pt2, color)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (0, 0, 0),
                    thickness=5)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (255, 255, 255),
                    thickness=2)

    cv2.imwrite(sys.argv[3], im)
	



