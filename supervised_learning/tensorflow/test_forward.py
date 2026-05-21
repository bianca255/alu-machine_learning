import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop

tf.reset_default_graph()
x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.relu, tf.nn.sigmoid])

ops = [op.name for op in tf.get_default_graph().get_operations() if 'layer' in op.name]
for o in ops:
    if any(act in o for act in ['Tanh', 'Relu', 'Sigmoid', 'BiasAdd']):
        print(o)
