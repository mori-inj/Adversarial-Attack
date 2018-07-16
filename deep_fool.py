from keras.models import Model
from keras.losses import categorical_crossentropy
from .base_generator import BaseGenerator

import tensorflow as tf
import numpy as np
import os
import math
import copy

class DeepFool(BaseGenerator):
    def __init__(self, network, max_iter=100):
        super(DeepFool, self).__init__(network)

        self.max_iter = max_iter

        # LOG: the initialized algorithm and the network
        self.image_placeholder = tf.placeholder(shape=self.network.get_input_shape(), name='input', dtype=tf.float32)

        base_model = self.network.model

        original_preprocessed = tf.reshape(self.network.preprocess_op(self.image_placeholder), (1,) + self.network.get_input_shape())
        self.original_prediction = base_model(original_preprocessed)
        self.class_placeholder = tf.placeholder(dtype=tf.int32)
        self.gradient = tf.gradients(self.original_prediction[0][self.class_placeholder], self.image_placeholder)

        self.network.sess.graph.finalize()
        # LOG: the finished state

    def generate(self, dataset, path):
        if not os.path.exists(path):
            os.mkdir(path)
        trial = 0
        success = 0
        
                
        class_num = int(max(dataset.get_test_data(), key=lambda item: item[1])[1]) + 1
                
        for i, (x, y) in enumerate(dataset.get_test_data()):
            subdir = os.path.join(path, self.format_filename(str(y)))
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            result = self._generate(x, y, class_num, os.path.join(
                subdir, '{}.png'.format(str(i))))
            if result:
                success += 1
            trial += 1
            if i % 20 == 0:
                print('[+] Generated {} adv : fooling rate = {}'.format(i, success / trial))

    def _generate(self, source_image, source_label, class_num, output_path):
        perturbation = self.deepfool(source_image, int(str(source_label)), class_num)
        adversarial = source_image + perturbation
        adv_prediction = self.network.sess.run(self.original_prediction, feed_dict={
            self.image_placeholder: adversarial
        })
        adv_prediction = np.argmax(adv_prediction)

        if str(source_label) == str(adv_prediction):
            success =  False
        else:
            success = True


        if os.path.splitext(output_path)[1] == '.png':
            self.network.save_image(adversarial, output_path)
        elif os.path.splitext(output_path)[1] == '.npy':
            self.network.save_numpy(adversarial, output_path)
        else:
            raise Exception('Unknown save format')
        return success

    def deepfool(self, source_image, source_label, class_num):
        x = copy.deepcopy(source_image)
        current_label = source_label
        itr = 0
        w = np.zeros(x.shape)
        r = np.zeros(x.shape)

        while str(current_label) == str(source_label) and itr < self.max_iter:
            l = np.inf
            grad_s = self.network.sess.run(self.gradient, feed_dict={
                self.image_placeholder: x,
                self.class_placeholder: source_label
            })
            grad_s = np.asarray(grad_s)[0]

            for k in range(class_num):
                grad_k = self.network.sess.run(self.gradient, feed_dict={
                    self.image_placeholder: x,
                    self.class_placeholder: k
                })
                grad_k = np.asarray(grad_k)[0]
                w_k = grad_k - grad_s
                
                f_ = self.network.sess.run(self.original_prediction, feed_dict={
                    self.image_placeholder: x
                })[0]
                f_k = f_[k] - f_[source_label]

                l_k = abs(f_k) / (np.linalg.norm(w_k.flatten(), ord=2) + 1e-14)
                if l_k < l and str(k) != str(source_label):
                    l = l_k
                    w = w_k
            
            r += l * w / (np.linalg.norm(w.flatten(), ord=2) + 1e-14)
            x = source_image + r
            itr += 1

            current_label = self.network.sess.run(self.original_prediction, feed_dict={
                self.image_placeholder: x
            })
            current_label = np.argmax(current_label)
        
        if np.linalg.norm(r) != np.linalg.norm(r):
            print("!")
        return r

