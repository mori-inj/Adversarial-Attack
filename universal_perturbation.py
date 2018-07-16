from keras.models import Model
from keras.losses import categorical_crossentropy
from .base_generator import BaseGenerator

import tensorflow as tf
import numpy as np
import os
import math
import copy

class UniversalPerturbation(BaseGenerator):
    def __init__(self, network, max_iter=20, df_max_iter=10, c=0.2, p=np.inf, delta=0.2, xi=10, m=200):
        super(UniversalPerturbation, self).__init__(network)
        #Universal Perturbation
        self.max_iter = max_iter
        self.df_max_iter = df_max_iter
        self.p = p # L2 or L-inf
        self.delta = delta # 1 - delta is target fooling rate on X
        self.xi = xi # perturbation size (used in projection)
        self.m = m # number of data in X
        
        #Target Optimization
        self.init_c = c
        # LOG: the initialized algorithm and the network
        self.image_placeholder = tf.placeholder(shape=self.network.get_input_shape(), name='input', dtype=tf.float32)
        self.target_label = tf.placeholder(shape=self.network.get_output_shape(), name='output', dtype=tf.float32)

        # self.r = tf.Variable(tf.zeros(self.network.get_input_shape()), name='perturbation', dtype=tf.float32)
        self.r = tf.get_variable('r', shape=self.network.get_input_shape(), dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.c = tf.get_variable('c', dtype=tf.float32, initializer=tf.constant(0.))
        self.c_placeholder = tf.placeholder(shape=(), name='c_ph', dtype=tf.float32)

        base_model = self.network.model

        self.adversarial = self.image_placeholder + self.r
        preprocessed = tf.reshape(self.network.preprocess_op(self.adversarial), (1,) + self.network.get_input_shape())
        self.prediction = base_model(preprocessed)

        original_preprocessed = tf.reshape(self.network.preprocess_op(self.image_placeholder), (1,) + self.network.get_input_shape())
        self.original_prediction = base_model(original_preprocessed)
        self.class_placeholder = tf.placeholder(dtype=tf.int32)
        self.gradient = tf.gradients(self.original_prediction[0][self.class_placeholder], self.image_placeholder)
        loss_f = -tf.reduce_sum(self.target_label * tf.log(self.prediction))
        loss_d = tf.scalar_mul(self.c, tf.norm(self.r))
        penalty = loss_d + loss_f

        lower_bound = np.vectorize(lambda x: 0. - x)
        upper_bound = np.vectorize(lambda x: 255. - x)
        self.bound = lambda x: (lower_bound(x), upper_bound(x))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                penalty, var_list=[self.r], method='L-BFGS-B', options={'maxiter': 100, 'ftol': 0., 'gtol': 1e-18}
        )
        self.assign = tf.assign(self.c, self.c_placeholder)

        self.network.sess.graph.finalize()
        # LOG: the finished state

    def apply_bound(self, optimizer, var_to_bounds):
        left_packed_bounds = []
        right_packed_bounds = []
        for var in optimizer._vars:
            shape = var.get_shape().as_list()
            bounds = (-np.infty, np.infty)
            if var in var_to_bounds:
                bounds = var_to_bounds[var]
            left_packed_bounds.extend(list(np.broadcast_to(bounds[0], shape).flat))
            right_packed_bounds.extend(list(np.broadcast_to(bounds[1], shape).flat))
        packed_bounds = list(zip(left_packed_bounds, right_packed_bounds))
        return packed_bounds

    def proj_lp(self, v, xi, p):
        #Copied from https://github.com/LTS4/universal/blob/matster/python/universal_pert.py
        #Project on the lp ball centered at 0 and of radius xi
        if p == 2:
            v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')
        return v
    
    
    def generate(self, dataset, path):
        if not os.path.exists(path):
            os.mkdir(path)
        trial = 0
        success = 0
        
                
        class_num = int(max(dataset.get_test_data(), key=lambda item: item[1])[1]) + 1
        universal_perturbation = self.calc_universal_perturbation(dataset, class_num)
                
        for i, (x, y) in enumerate(dataset.get_test_data()):
            subdir = os.path.join(path, self.format_filename(str(y)))
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            result = self._generate(x, y, class_num, universal_perturbation, os.path.join(
                subdir, '{}.png'.format(str(i))))
            if result:
                success += 1
            trial += 1
            if i % 20 == 0:
                print('[+] Generated {} adv : fooling rate = {}'.format(i, success / trial))


    def _generate(self, source_image, source_label, class_num, universal_perturbation, output_path):
        adversarial = source_image + universal_perturbation
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

    def calc_universal_perturbation(self, dataset, class_num):
        fooling_rate = 0.0
        itr = 0
        v = 0

        X = []
        for k, (x,y) in enumerate(dataset.get_training_data(), itr*self.m):
            X.append((x,y))
            if k % self.m == self.m - 1:
                break
        X = np.asarray(X)

        while fooling_rate < 1 - self.delta and itr < self.max_iter:
            np.random.shuffle(X)

            for k, (x,y) in enumerate(X):
                adv_prediction = self.network.sess.run(self.original_prediction, feed_dict={
                    self.image_placeholder: x+v
                })
                adv_prediction = np.argmax(adv_prediction) 
                if str(y) == str(adv_prediction):
                    #perturbation = self.min_target_opt(x+v, y, class_num)[1]
                    perturbation = self.deepfool(x+v, int(str(y)), class_num)

                    if itr < self.max_iter-1:
                        v = v + perturbation
                        v = self.proj_lp(v, self.xi, self.p)

                if k % (self.m/10) == 0:
                    print(str(k) + "/" + str(self.m) +" in iter " + str(itr))
                
            
            print(" iter:"+str(itr))
            fooling_occasion = 0.0
            for i, (x,y) in enumerate(X):
                adv_prediction = self.network.sess.run(self.original_prediction, feed_dict={
                    self.image_placeholder: x+v
                })
                adv_prediction = np.argmax(adv_prediction) 
                if str(y) != str(adv_prediction):
                    fooling_occasion += 1.0
            fooling_rate = fooling_occasion / self.m
            print("fooling rate of X: "+str(fooling_occasion)+"/"+str(self.m)+"="+str(fooling_rate)+", "+str(len(X)))
            itr += 1
        return v

    def deepfool(self, source_image, source_label, class_num):
        x = copy.deepcopy(source_image)
        current_label = source_label
        itr = 0
        w = np.zeros(x.shape)
        r = np.zeros(x.shape)

        while str(current_label) == str(source_label) and itr < self.df_max_iter:
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

        
    
    def min_target_opt(self, source_image, source_label, class_num):
        target_opt_result = {}
        for i in range(class_num):
            if str(i) != str(source_label):
                target_opt_result[i] = self.target_opt(source_image, str(i))
        
        target_opt_result = {k:v for k,v in target_opt_result.items() if v[0]}
        min_perturbation_class = min(target_opt_result, key=lambda item: np.linalg.norm(target_opt_result[item][1].flatten(), ord=2))
        min_perturbation = target_opt_result[min_perturbation_class][1]
        return min_perturbation_class, min_perturbation 

    def target_opt(self, source_image, label):
        # Set bounds for optimizer
        self.network.sess.run(self.c.initializer)
        self.network.sess.run(self.r.initializer)

        target_label = self.network.convert_label(label)

        self.optimizer._packed_bounds = self.apply_bound(self.optimizer, {self.c: (0., np.infty), self.r: self.bound(source_image)})

        c = self.init_c
        # TODO: set better line search scheme
        upper_bound = 10.0
        lower_bound = 1e-6

        last_success = None

        for i in range(10):
            # Assign c
            self.network.sess.run(self.assign, feed_dict={
                self.c_placeholder: c
            })
            # Try with c
            self.optimizer.minimize(session=self.network.sess, feed_dict={
                self.image_placeholder: source_image,
                self.target_label: target_label,
            })

            adversarial = self.network.sess.run(self.adversarial, feed_dict={
                self.image_placeholder: source_image,
            })
            adv_prediction = self.network.sess.run(self.prediction, feed_dict={
                self.image_placeholder: source_image,
                self.target_label: target_label,
            })
            success = self.network.compare(adv_prediction, label)
            if success:
                # Fooling has succeeded, so increase the distance factor c
                last_success = adversarial, adv_prediction, c
                lower_bound = c
                c = math.sqrt(c * upper_bound)
            else:
                # Fooling has failed, so decrease the distance factor c
                upper_bound = c
                c = math.sqrt(c * lower_bound)

        if last_success is None:
            success = False
            print(c)
        else:
            success = True
            adversarial, adv_prediction, c = last_success

        perturbation = adversarial - source_image
           
        return success, perturbation
