# Typical setup to include TensorFlow.
import tensorflow as tf
import time
import os

# images dir
# path = "../../data/fish/train-fix/"
# files = os.listdir(path)
# # print files
# list_files = []
# for file in files:
#     list_files.append(path + file)

#labels dir
# path = "../../data/fish/label-train-fix/"

# # Make a queue of file names including all the JPEG images files in the relative
# # image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("../../data/fish/train-fix/*.jpg"))

filename_queue_label = tf.train.string_input_producer(
    tf.train.match_filenames_once("../../data/fish/label-train-fix/*.txt"))

# filename_queue = tf.train.string_input_producer(list_files)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()
label_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_ , image_file = image_reader.read(filename_queue)
_ , label_file = label_reader.read(filename_queue_label)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)
image.set_shape([976, 1732, 3])
label = tf.string_to_number(label_file, out_type=tf.int32)
# image = tf.bitcast(image,tf.float32)
# label.set_shape([0,0])
print "image", image
print "label", label
# queue = tf.RandomShuffleQueue(
#     capacity=50,
#     min_after_dequeue=30,
#     # dtypes=[tf.float32, tf.string],
#     # shapes=[[1000, 1540, 3], [1]],
#     dtypes=[tf.uint8],
#     # shapes=[1540, 3],
#     name="random_shuffle_queue"
    # )

# Enqueue and dequeue operations
# enqueue_op = queue.enqueue_many([image, label])
# enqueue_op = queue.enqueue_many(image)
# queue_image, queue_label = queue.dequeue_many(15)

# queue_size = queue.size()
# square_op = tf.square(queue_image)
# print image
# decoded.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
# image.set_shape([720, 1280, 3])
# print image
# print label
# image_raw = tf.image.decode_jpeg(image_file)
# print image_raw
# image = tf.image.resize_images(image_raw, [299, 299])
# print image

images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=5, 
    capacity=500,
    min_after_dequeue=100)

square_op = tf.square(tf.cast(images_batch, tf.float16))

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    # tf.initialize_all_variables().run()
    init = tf.global_variables_initializer()
    print "init"
    sess.run(init)
    
    print "coordinator"
    # # Coordinate the loading of image files.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    # tf.train.start_queue_runners(sess=sess)

    sleep = 30
    print "queue sleep in seg:", sleep
    time.sleep(sleep)
    print "end queue sleep"
    # print "enqueue_op"
    # sess.run(enqueue_op)

    # num_examples = 0
    # try:
    #     step = 0
    #     while not coord.should_stop():
    #         # start_time = time.time()
    #         e, l = sess.run([images_batch, labels_batch])
    #         print "grabbing"
    #         e, l = sess.run([images_batch, labels_batch])
    #         num_examples = num_examples + e.shape[0]
    #         print "num_examples = " + str(num_examples)
    #         # duration = time.time() - start_time

    # except tf.errors.OutOfRangeError:
    #     print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    # finally:
    #     # When done, ask the threads to stop.
    #     coord.request_stop()

    #     # Wait for threads to finish.
    #     coord.join(threads)
    #     sess.close()

    # Make sure the queueu is filled with some examples (n = 500)
    # num_samples_in_queue = 0
    # while num_samples_in_queue < 500:
    #     num_samples_in_queue = sess.run(queue_size)
    #     print("Initializing queue, current size = %i" % num_samples_in_queue)
    #     time.sleep(1)

    # Get an image tensor and print its value.
    for x in xrange(0,5):
        # image_tensor = sess.run([image])
        # image_tensor, labels_tensor = sess.run([images_batch, labels_batch])
        # print(image_tensor)
        # print len(labels_tensor)
        # print str(labels_tensor[0])
        # print len(image_tensor)
        # print image.get_shape()
        sess.run(square_op)
        print x


    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    sess.close()