from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf

import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import googletest


def sgdr_decay(learning_rate, global_step, t_0, mult_factor=2, name=None):
  
  if global_step is None:
    raise ValueError("global_step is required for sgdr_decay.")
  with ops.name_scope(name, "SGDRDecay",
                      [learning_rate, global_step,
                       t_0, mult_factor]) as name:
    global_step = global_step

    learning_rate = ops.convert_to_tensor(learning_rate, tf.float32, name="lr")
    global_step = math_ops.cast(global_step, tf.float32)
    mult_factor = ops.convert_to_tensor(mult_factor, tf.float32)
    t_0 = ops.convert_to_tensor(t_0, tf.float32)
    pi = ops.convert_to_tensor(math.pi, tf.float32)

    x = tf.cond(tf.equal(mult_factor, 1.),
                lambda: tf.mod(global_step, t_0)/t_0,
                lambda: tf.mod(tf.log(1. - global_step*(1-mult_factor)/t_0)/tf.log(mult_factor), 1))

    fac = (math_ops.cos(x * pi)+1)*0.5

    return fac * learning_rate


class SGDRDecayTest(test_util.TensorFlowTestCase):

  def testSGDR(self):
    k = 100
    initial_lr = 0.5
    t_0 = 2
    mult_factor = 12
    step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
                                   name="step", container="", shared_name="")
    assign_step = state_ops.assign(step, 0)
    increment_step = state_ops.assign_add(step, 1)
    sgdr_lr = sgdr_decay(initial_lr, step, t_0, mult_factor)

    with self.test_session():
      assign_step.op.run()
      for i in range(k+1):
        lr = sgdr_lr.eval()
        print(lr)
        increment_step.op.run()


if __name__ == "__main__":
  googletest.main()
