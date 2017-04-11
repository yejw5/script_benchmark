#!/usr/bin/env python2.7

from __future__ import print_function

import multiprocessing
import sys
import time
import threading

from grpc.beta import implementations
import numpy
import tensorflow as tf

import predict_pb2
import prediction_service_pb2
import mnist_input_data


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_integer('request_delay', 1000, 'Delay of requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.app.flags.DEFINE_string('model_name', '', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

lock = multiprocessing.Lock()
counter = multiprocessing.Value("i", 0)
real_test_num = multiprocessing.Value("i", 0)
finish_time = multiprocessing.Value("d", 0.0)


def _create_rpc_callback(label, event):
  def _callback(result_future):
    global lock, counter
    with lock:
      counter.value += 1
    event.set()
    exception = result_future.exception()
    if exception:
      print(exception)
    else:
      pass
      #sys.stdout.write('.')
  return _callback


def do_inference(process_num, hostport, num_tests, image, label):
  #print("Begin process: {}".format(process_num))
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  events = []
  for _ in range(num_tests):
    event = threading.Event()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model_name
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
    result_future = stub.Predict.future(request, FLAGS.request_delay)
    result_future.add_done_callback(
        _create_rpc_callback(label[0], event))
    events.append(event)

  for event in events:
    event.wait()
  global lock, counter, real_test_num, start_time, finish_time
  with lock:
    if real_test_num.value == 0:
      real_test_num.value = counter.value
      finish_time.value = time.time()
      #print("{} {}".format(finish_time.value, time.time()))
  #print("Finish process: {}".format(process_num))


def main(_):
  if not FLAGS.server:
    print('please specify server host:port')
    return
  if not FLAGS.model_name:
    print('please specify model_name')
    return

  test_data_set = mnist_input_data.read_data_sets(FLAGS.work_dir).test
  image, label = test_data_set.next_batch(1)
  #import ipdb; ipdb.set_trace()
  #do_inference(0, FLAGS.server, FLAGS.num_tests, image, label)

  pool = multiprocessing.Pool(processes=FLAGS.concurrency)
  results = []
  start_time = time.time()
  print("The time start predicting: {}".format(start_time))
  for process_num in range(FLAGS.concurrency):
    pool.apply_async(do_inference, (process_num, FLAGS.server, FLAGS.num_tests, image, label))
  pool.close()
  pool.join()


  global lock, finish_time, real_test_num
  with lock:
    duration = finish_time.value - start_time
    #total_num_tests = FLAGS.num_tests * FLAGS.concurrency
    total_num_tests = real_test_num.value
    qps = total_num_tests / duration
    print("The time end predicting: {}".format(finish_time.value))
    print("Total predict time: {}(s)".format(duration))
    print("Total predict num: {}".format(total_num_tests))
    print("qps: {}".format(qps))

if __name__ == '__main__':
  tf.app.run()
