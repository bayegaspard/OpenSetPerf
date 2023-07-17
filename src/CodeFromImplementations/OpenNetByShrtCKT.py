#Note: I have NO IDEA how to get this to work with our pytorch model nicely 
#       because it is modifying how the model is trained.
#NOTE: DECIDED TO WRITE MY OWN VERSION BASED ON THE PAPER
import torch
#https://arxiv.org/pdf/1802.04365.pdf
#https://github.com/shrtCKT/opennet/blob/master/opennet/opennet.py
import tensorflow as tf
#Tensorflow and pytorch conversion
#https://stackoverflow.com/a/60724300

class OpenNetBase(object):
    #This is not the original initializer
    def __init__(self):
        #These are all being set because of the comments that say "ii-loss"
        self.dist = "mean_separation_spread"
        self.enable_recon_loss = False
        self.enable_intra_loss = True
        self.enable_inter_loss = True
        self.div_loss = False

        #I am not sure if crossentropy loss should be on?
        self.enable_ce_loss = True


    def loss_fn_training_op(self, x, y, z, logits, x_recon, class_means):
        """ Computes the loss functions and creates the update ops.
        :param x - input X
        :param y - labels y
        :param z - z layer transform of X.
        :param logits - softmax logits if ce loss is used. Can be None if only ii-loss.
        :param recon - reconstructed X. Experimental! Can be None.
        :class_means - the class means.
        """
        # Calculate intra class and inter class distance
        if self.dist == 'class_mean':   # For experimental pupose only
            self.intra_c_loss, self.inter_c_loss = self.inter_intra_diff(
                z, tf.cast(y, tf.int32), class_means)
        elif self.dist == 'all_pair':   # For experimental pupose only
            self.intra_c_loss, self.inter_c_loss = self.all_pair_inter_intra_diff(
                z, tf.cast(y, tf.int32))
        elif self.dist == 'mean_separation_spread':  # ii-loss
            self.intra_c_loss, self.inter_c_loss = self.inter_separation_intra_spred(
                z, tf.cast(y, tf.int32), class_means)
        elif self.dist == 'min_max':   # For experimental pupose only
            self.intra_c_loss, self.inter_c_loss = self.inter_min_intra_max(
                z, tf.cast(y, tf.int32), class_means)

        # Calculate reconstruction loss
        if self.enable_recon_loss:    # For experimental pupose only
            self.recon_loss = tf.reduce_mean(tf.squared_difference(x, x_recon))

        if self.enable_intra_loss and self.enable_inter_loss:        # The correct ii-loss
            self.loss = tf.reduce_mean(self.intra_c_loss - self.inter_c_loss)
        elif self.enable_intra_loss and not self.enable_inter_loss:  # For experimental pupose only
            self.loss = tf.reduce_mean(self.intra_c_loss)
        elif not self.enable_intra_loss and self.enable_inter_loss:  # For experimental pupose only
            self.loss = tf.reduce_mean(-self.inter_c_loss)
        elif self.div_loss:                                          # For experimental pupose only
            self.loss = tf.reduce_mean(self.intra_c_loss / self.inter_c_loss)
        else:                                                        # For experimental pupose only
            self.loss = tf.reduce_mean((self.recon_loss * 1. if self.enable_recon_loss else 0.)
                                       + (self.intra_c_loss * 1. if self.enable_intra_loss else 0.)
                                       - (self.inter_c_loss * 1. if self.enable_inter_loss else 0.)
                                      )

        # Classifier loss
        if self.enable_ce_loss:
            self.ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

        tvars = tf.trainable_variables()
        e_vars = [var for var in tvars if 'enc_' in var.name ]
        classifier_vars = [var for var in tvars if 'enc_' in var.name or 'classifier_' in var.name]
        recon_vars = [var for var in tvars if 'enc_' in var.name or 'dec_' in var.name]

        # Training Ops
        if self.enable_recon_loss:
            self.recon_train_op = self.recon_opt.minimize(self.recon_loss, var_list=recon_vars)

        if self.enable_inter_loss or self.enable_intra_loss or self.div_loss:
            #COMPLETELY CHANGED LINE
            #self.train_op = self.opt.minimize(self.loss, var_list=e_vars)
            self.loss.backward()

        if self.enable_ce_loss:
            self.ce_train_op = self.c_opt.minimize(self.ce_loss, var_list=classifier_vars)

    def bucket_mean(self, data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    def sq_difference_from_mean(self, data, class_mean):
        """ Calculates the squared difference from clas mean.
        """
        sq_diff_list = []
        for i in range(self.y_dim):
            sq_diff_list.append(tf.reduce_mean(
                tf.squared_difference(data, class_mean[i]), axis=1))

        return tf.stack(sq_diff_list, axis=1)

    def inter_intra_diff(self, data, labels, class_mean):
        """ Calculates the intra-class and inter-class distance
        as the average distance from the class means.
        """
        sq_diff = self.sq_difference_from_mean(data, class_mean)

        inter_intra_sq_diff = self.bucket_mean(sq_diff, labels, 2)
        inter_class_sq_diff = inter_intra_sq_diff[0]
        intra_class_sq_diff = inter_intra_sq_diff[1]
        return intra_class_sq_diff, inter_class_sq_diff

    def inter_separation_intra_spred(self, data, labels, class_mean):
        """ Calculates intra-class spread as average distance from class means.
        Calculates inter-class separation as the distance between the two closest class means.
        Returns:
        intra-class spread and inter-class separation.
        """
        intra_class_sq_diff, _ = self.inter_intra_diff(data, labels, class_mean)

        ap_dist = self.all_pair_distance(class_mean)
        dim = tf.shape(class_mean)[0]
        not_diag_mask = tf.logical_not(tf.cast(tf.eye(dim), dtype=tf.bool))
        inter_separation = tf.reduce_min(tf.boolean_mask(tensor=ap_dist, mask=not_diag_mask))
        return intra_class_sq_diff, inter_separation

    def all_pair_distance(self, A):
        r = tf.reduce_sum(A*A, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(A, A, transpose_b=True) + tf.transpose(r)
        return D


