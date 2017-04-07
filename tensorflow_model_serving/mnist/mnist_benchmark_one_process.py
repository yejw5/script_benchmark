#!/usr/bin/env python2.7

from __future__ import print_function

import sys
import multiprocessing
import time

from grpc.beta import implementations
import numpy
import tensorflow as tf

import predict_pb2
import prediction_service_pb2
import mnist_input_data

tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_integer('request_delay', 1000, 'Delay of requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.app.flags.DEFINE_string('model_name', '', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

def _create_rpc_callback(label):
  def _callback(result_future):
    exception = result_future.exception()
    if exception:
      print(exception)
    else:
      pass
      sys.stdout.write('.')
  return _callback


def do_inference(hostport, num_tests, image, label):
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  for _ in range(num_tests):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model_name
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
    result_future = stub.Predict.future(request, FLAGS.request_delay)
    result_future.add_done_callback(
        _create_rpc_callback(label[0]))


def main(_):
  if not FLAGS.server:
    print('please specify server host:port')
    return
  if not FLAGS.model_name:
    print('please specify model_name')
    return

  test_data_set = mnist_input_data.read_data_sets(FLAGS.work_dir).test
  image, label = test_data_set.next_batch(1)

  start = time.time()
  print("The time start predicting: {}".format(start))

  do_inference(FLAGS.server, FLAGS.num_tests, image, label)

  end = time.time()
  duration = end - start
  qps = FLAGS.num_tests / duration
  print("The time end predicting: {}".format(end))
  print("Total predict time: {}(s)".format(duration))
  print("Total predict num: {}".format(FLAGS.num_tests))
  print("qps: {}".format(qps))

if __name__ == '__main__':
  tf.app.run()
