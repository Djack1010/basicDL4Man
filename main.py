import argparse
import datetime
import re
import numpy as np
import pathlib
import tensorflow as tf
from models.basic_cnn import BasicCNN
from models.basic_mlp import BasicMLP
from models.VGG16 import VGG16_19
from utils.config import timeExec, MAIN_PATH, time, os


def main(arguments):
    print("LOADING AND PRE-PROCESSING DATA")

    dataset_base = MAIN_PATH + arguments.dataset

    # Create dataset of filepaths
    train_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/train/*/*")
    val_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/val/*/*")
    final_training_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/*/*/*")
    test_paths_ds = tf.data.Dataset.list_files(dataset_base + "/test/*/*")

    # STATS
    size_train = sum([len(files) for r, d, files in os.walk(dataset_base + "/training/train")])
    size_val = sum([len(files) for r, d, files in os.walk(dataset_base + "/training/val")])
    size_test = sum([len(files) for r, d, files in os.walk(dataset_base + "/test")])
    nclasses = len(CLASS_NAMES)

    # Print information on log
    # EXECUTION Info
    print_log(f"INFO EXECUTION:"
              f"\nmode = {arguments.mode}\nmodel = {arguments.model}\ndataset = {arguments.dataset}"
              f"\noutput_model = {arguments.output_model}\nepochs = {arguments.epochs}\nbatch_size = {arguments.batch_size}"
              f"\n----------------")

    # DATA Info
    print_log(f"INFO DATA:"
              f"\nnum_classes = {nclasses}\nclass_names= {CLASS_NAMES}"
              f"\nSize train-val-test= {size_train}-{size_val}-{size_test}"
              f"\nsize_img = {arguments.image_size}x{arguments.channels}")

    print_log("STARTING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    # SELECTING MODELS
    model = _model_selection(arguments, nclasses)

    # --------------  TRAINING and VALIDATION part  --------------------

    if arguments.mode == "train-val":
        #  Use Dataset.map to create a dataset of image, label pairs
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        lab_train_ds = train_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        lab_val_ds = val_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)

        train_ds = prepare_for_training(lab_train_ds)
        val_ds = prepare_for_training(lab_val_ds)

        print_log('Start Training for {} epochs  '.format(arguments.epochs), print_on_screen=True)

        # Initialize callbacks for Tensorboard
        log_fit = "tensorboard/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback_fit = tf.keras.callbacks.TensorBoard(log_dir=log_fit, histogram_freq=1)

        train_results = model.fit(x=train_ds, batch_size=arguments.batch_size, epochs=arguments.epochs,
                                  validation_data=val_ds, callbacks=[tensorboard_callback_fit])

        print_log("\ttrain_loss:{} \n\ttrain_acc:{} \n\ttrain_prec:{} \n\ttrain_rec:{} \n"
                  "\tval_loss:{} \n\tval_acc:{} \n\tval_prec:{} \n\tval_rec:{}"
                  .format(train_results.history['loss'], train_results.history['prec'], train_results.history['rec'],
                          train_results.history['acc'],
                          train_results.history['val_loss'], train_results.history['val_acc'],
                          train_results.history['val_prec'], train_results.history['val_rec']))

        del train_ds, val_ds

    # --------------  FINAL TRAINING and TEST part  --------------------

    if arguments.mode == "train-test":
        #  Use Dataset.map to create a dataset of image, label pairs
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        lab_final_train_ds = final_training_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        lab_test_ds = test_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)

        fin_train_ds = prepare_for_training(lab_final_train_ds)
        test_ds = prepare_for_training(lab_test_ds)

        # Train the model over the entire total_training set and then test
        print_log('Start Final Training for {} epochs  '.format(arguments.epochs), print_on_screen=True)
        final_train_results = model.fit(x=fin_train_ds, batch_size=arguments.batch_size, epochs=arguments.epochs)
        print_log("\ttrain_loss:{} \n\ttrain_acc:{} \n\ttrain_prec:{} \n\ttrain_rec:{} \n"
                  .format(final_train_results.history['loss'], final_train_results.history['prec'],
                          final_train_results.history['rec'], final_train_results.history['acc']))

        # Test the trained model over the test set
        print_log('Start Test', print_on_screen=True)
        results = model.evaluate(test_ds)
        print_log("\ttest loss:{} \n\ttest accuracy:{}".format(results[0], results[1]), print_on_screen=True)
        print_log("\tPrec:{} \n\tRecall:{}".format(results[2], results[3]), print_on_screen=True)
        try:
            # F-measure calculated as (2 * Prec * Recall)/(Prec + Recall)
            print_log("\tF-Measure:{} \n\tAUC:{}"
                      .format((2 * results[2] * results[3]) / (results[2] + results[3]), results[4]), print_on_screen=True)
        except ZeroDivisionError:
            print_log("\tF-Measure:{} \n\tAUC:{}"
                      .format("Error", results[4]), print_on_screen=True)

        del fin_train_ds, test_ds

        # save model and architecture to single file
        if arguments.output_model is not None:
            model.save(MAIN_PATH + 'model_saved/{}_m{}'.format(arguments.output_model, arguments.model))
            model.save_weights(MAIN_PATH + 'model_saved/{}_m{}_weights'.format(arguments.output_model, arguments.model))

    print_log("ENDING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep Learning Image Malware Classification')
    group = parser.add_argument_group('Arguments')
    # REQUIRED Arguments
    group.add_argument('-m', '--model', required=True, type=str, choices=['BASIC_CNN', 'BASIC_MLP', 'VGG16'],
                       help='DATA, BASIC_[CNN|MLP], or VGG16')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='the dataset path, must have the folder structure: training/train, training/val and test,'
                            'in each of this folders, one folder per class (see dataset_test)')
    # OPTIONAL Arguments
    group.add_argument('-o', '--output_model', required=False, type=str, default=None,
                       help='Name of model to store')
    group.add_argument('-e', '--epochs', required=False, type=int, default=10,
                       help='number of epochs')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=32)
    group.add_argument('-i', '--image_size', required=False, type=str, default="250x1",
                       help='FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input '
                            '(reshape will be applied)')
    group.add_argument('-w', '--weights', required=False, type=str, default=None,
                       help="If you do not want random initialization of the model weights "
                            "(ex. 'imagenet' or path to weights to be loaded)")
    group.add_argument('--mode', required=False, type=str, default='train-val', choices=['train-val', 'train-test',
                                                                                         'test'],
                       help="Choose which mode run between 'train-val' (default) or 'train-test'."
                            "The 'train-val' mode will run a phase of training and validation on the training and "
                            "validation set, while the 'train-test' mode will run a phase of training on the "
                            "training+validation sets and then test on the test set.")
    # FLAGS
    group.add_argument('--exclude_top', dest='include_top', action='store_false',
                       help='Exclute the fully-connected layer at the top pf the network (default INCLUDE)')
    group.set_defaults(include_top=True)
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if re.match(r"^\d{2,4}x([13])$", arguments.image_size):
        img_size = arguments.image_size.split("x")[0]
        channels = arguments.image_size.split("x")[1]
        setattr(arguments, "image_size", int(img_size))
        setattr(arguments, "channels", int(channels))
    else:
        print('Invalid image_size, exiting...')
        exit()
    if not os.path.isdir(MAIN_PATH + arguments.dataset):
        print('Cannot find dataset in {}, exiting...'.format(arguments.dataset))
        exit()
    # Check Dataset struct: should be in folder tree training/[train|val] e test
    if not os.path.isdir(MAIN_PATH + arguments.dataset + "/test") or \
            not os.path.isdir(MAIN_PATH + arguments.dataset + "/training/val") or \
            not os.path.isdir(MAIN_PATH + arguments.dataset + "/training/train"):
        print(f"Dataset '{arguments.dataset}' should contain folders 'test, training/train and training/val'...")
        exit()


def _model_selection(arguments, nclasses):
    model = None
    print("INITIALIZING MODEL")

    if arguments.model == "BASIC_CNN":
        model_class = BasicCNN(nclasses, arguments.image_size, arguments.channels)
    elif arguments.model == "BASIC_MLP":
        print("TODO: integrate MLP model! Code available for educational purpose only, exiting...")
        exit()
        # model_class = BasicMLP(nclasses, <VECTOR_SIZE>)
    elif arguments.model == "VGG16":
        # NB. Setting include_top=True and thus accepting the entire struct, the input Shape MUST be 224x224x3
        # and in any case, channels has to be 3
        if arguments.channels != 3:
            print("VGG requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        model_class = VGG16_19(nclasses, arguments.image_size, arguments.channels,
                      weights=arguments.weights, include_top=arguments.include_top)
    else:
        print("model {} not implemented yet...".format(arguments.model))
        exit()

    model = model_class.build()
    return model


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # cast to float32 for one_hot encode (otherwise TRUE/FALSE tensor)
    return tf.cast(parts[-2] == CLASS_NAMES, tf.float32)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=CHANNELS)  # tf.image.decode_jpeg(img, channels=CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_DIM, IMG_DIM])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, loop=False):
    # IF it is a small dataset, only load it once and keep it in memory.
    # OTHERWISE use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    if loop:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def print_log(string, print_on_screen=False, print_on_file=True):
    if print_on_screen:
        print(string)
    if print_on_file:
        with open(MAIN_PATH + 'logs/' + timeExec + ".results", 'a') as logfile:
            logfile.write(string + "\n")


if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()
    _check_args(args)
    # GLOBAL SETTINGS FOR THE EXECUTIONS
    # Reduce verbosity for Tensorflow Warnings and set dtype for layers
    # tf.keras.backend.set_floatx('float64')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CHANNELS = args.channels
    IMG_DIM = args.image_size
    CLASS_NAMES = np.array([item.name for item in pathlib.Path(MAIN_PATH + args.dataset + "/training/train").glob('*')])
    BATCH_SIZE = args.batch_size

    # Check if tensorflow can access the GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print('GPU device not found...')
    else:
        print('Found GPU at: {}'.format(device_name))

    main(args)
    end = time.perf_counter()
    print()
    print_log("EX. TIME: {} ".format(str(datetime.timedelta(seconds=end-start))), print_on_screen=True)
