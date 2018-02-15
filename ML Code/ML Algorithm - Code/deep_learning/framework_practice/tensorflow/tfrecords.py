
# TFRecords Demystified !

# I - Writing Example
# II - Writing SequenceExample
# III - Reading Example

# I - Example
# tf.Example equivalent with Python dict
my_dict = {
    'features' : {
        'my_ints': [5,6],
        'my_float': [2.7],
        'my_bytes': ['data']
}}

# tf.Example with Tensorflow
import tensorflow as tf

# Init of Example
my_example = tf.train.Example(features=tf.train.Features(feature={
                'my_ints': tf.train.Feature(int64_list=tf.train.Int64List(value=[5,6])),
                'my_float': tf.train.Feature(float_list=tf.train.FloatList(value=[2.7])),
                'my_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=['data']))
            }
    ))

# Get features of Example
print "Content of 'my_ints' : ", my_example.features.feature['my_ints'].int64_list.value
print "Content of 'my_float' : ", my_example.features.feature['my_float'].float_list.value
print "Content of 'my_bytes' : ", my_example.features.feature['my_bytes'].bytes_list.value

# Serializing the Example
my_example_str = my_example.SerializeToString()

writer = tf.python_io.TFRecordWriter('my_example.tfrecords')
writer.write(my_example_str)
writer.close()
# II - SequenceExample
# Use for a static 'context' : "Name of Movie"
# with a dynamic sequence of data 'features list' : "Reviews during time"

# In python dict
my_seq_dict = {
    'context' : {
        'my_bytes':
            ['data']},
    'feature_lists' : {
        'my_ints': [
            [5, 6],
            [7, 8, 9]
        ]},
    }

#In Tensorflow
my_seq_example = tf.train.SequenceExample(
    context=tf.train.Features(feature={
        'my_bytes':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=['data']))
    }),
    feature_lists=tf.train.FeatureLists(feature_list={
        'my_ints':
            tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 6])),
                tf.train.Feature(int64_list=tf.train.Int64List(value=[7, 8, 9])),                
            ])
    })
)

# Get Element in Sequence
print "Content of S 'my_bytes' : ", my_seq_example.context.feature['my_bytes'].bytes_list.value
print "Content of S 'my_ints' : ", my_seq_example.feature_lists.feature_list['my_ints']

# Writing Sequence Example
my_seq_example_str = my_seq_example.SerializeToString()

writer = tf.python_io.TFRecordWriter('my_seq_example.tfrecords')
writer.write(my_seq_example_str)
writer.close()

# III - Reading Example from File and parsing inside TensorFlow Graph

# we get an iterator for this file
reader = tf.python_io.tf_record_iterator('my_example.tfrecords')
my_readed_example = [tf.train.Example().FromString(example_str)
        for example_str in reader]

print "EXAMPLE : ",my_readed_example

# we get an iterator for this file
reader = tf.python_io.tf_record_iterator('my_seq_example.tfrecords')
my_readed_example = [tf.train.SequenceExample().FromString(seq_example_str)
        for seq_example_str in reader]

print "SEQ_EXAMPLE : ",my_readed_example


