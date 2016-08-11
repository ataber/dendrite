import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from time import strftime

class Graph:
  def __init__(self, geometry, debug=True):
    self.geometry = geometry
    self.debug = debug
    self.summary_writer = None
    self.initialize_session()

  @contextmanager
  def tensorboard_logging(self, tag_prefix):
    try:
      if self.debug:
        self.summary_writer = tf.train.SummaryWriter("log/Run-" + strftime("%d-%m_%H-%M-%S"), self.session.graph)
        self.tag_prefix = tag_prefix
        self.run_step = 0
      yield
    finally:
      self.summary_writer = None
      self.tag_prefix = None
      self.run_step = 0

  def run(self, ops, feed={}):
    feed_dict = self.merge_feeds(feed)

    if self.summary_writer is not None:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      result = self.session.run(
        ops,
        feed_dict=feed_dict,
        options=run_options,
        run_metadata=run_metadata
      )
      self.summary_writer.add_run_metadata(run_metadata, self.tag_prefix + str(self.run_step))
      self.run_step += 1
    else:
      result = self.session.run(ops, feed_dict=feed_dict)

    return result

  def run_summary(self, summary_op, feed):
    if self.summary_writer is not None:
      feed_dict = self.merge_feeds(feed)
      self.summary_writer.add_summary(self.session.run(summary_op, feed_dict=feed_dict))

  def merge_feeds(self, feed):
    feed_dict = feed.copy()

    for name, placeholder in self.geometry.placeholders.items():
      feed_dict[placeholder] = self.geometry.parameters[name]

    return feed_dict

  def initialize_session(self):
    # http://stackoverflow.com/questions/35905830/can-tensorflow-cache-sub-graph-computations
    optimizer_opts = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1)
    graph_opts = tf.GraphOptions(optimizer_options=optimizer_opts)
    config = tf.ConfigProto(graph_options=graph_opts, log_device_placement=self.debug)
    self.session = tf.Session(config=config)
    return self.session
