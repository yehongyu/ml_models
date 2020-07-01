from absl import app
from absl import flags
 
FLAGS = flags.FLAGS # 用法和TensorFlow的FLAGS类似，具有谷歌独特的风格。
flags.DEFINE_string("name", None, "Your name.")
flags.DEFINE_integer("num_times", 1,
                     "Number of times to print greeting.")
 
# 指定必须输入的参数
flags.mark_flag_as_required("name")
 
def main(argv):
  del argv  # 无用
  for i in range(0, FLAGS.num_times):
    print('Hello, %s!' % FLAGS.name)
 
 
if __name__ == '__main__':
  app.run(main)  # 和tf.app.run()类似
