import tensorflow as tf
import os
 
 
DATA_PATH = "/home/fabsta/projects/deeplearning/vae_playground/CDN_Molecule/"
 
# for your Tensorflow program.
# It is then accessible under tf.app.flags.FLAGS, i.e. tf.app.flags.FLAGS.dev_sample_percentage
# Data loading params
tf.app.flags.DEFINE_float("dev_sample_percentage", .03, "Percentage of the training data to use for validation")
 
tf.app.flags.DEFINE_string("data_file", os.path.join(DATA_PATH,"data/TrainVectors.pickle"), "Data source.")
 
#tf.app.flags.DEFINE_string("data_file", os.path.join(DATA_PATH,"data/TrainVectors_python2.pickle"), "Data source.")
 
tf.app.flags.DEFINE_string("parameters_file", None, "Checkpoint directory for training restart")
tf.app.flags.DEFINE_boolean("load_param", False, "Load trained models if True")
 
# Model Hyperparameters
tf.app.flags.DEFINE_integer("vocab_size", 37, "number of chars in SMILES vocab)")
tf.app.flags.DEFINE_integer("max_molecule_length", 50, "number of chars in SMILES vocab)")
tf.app.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5,6", "Comma-separated filter sizes")
tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda ")
tf.app.flags.DEFINE_integer("unit_gaussian_dim", 300, "number of gaussians")
tf.app.flags.DEFINE_float("initial_learning_rate", 1e-3, "initial learning rate")
 
# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size")  # originally: 64
tf.app.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")  # was 100
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.app.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps")  # originally: 5000
tf.app.flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store")
tf.app.flags.DEFINE_string('f', '', 'kernel')