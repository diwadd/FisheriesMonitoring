import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def cnn_12_model(x_image, verbose=False):

    # Dropout probabilities
    dropout_one= 0.1
    dropout_two= 0.1
    dropout_three= 0.1
    dropout_four= 0.1
    dropout_five= 0.1
    dropout_six= 0.1
    dropout_seven= 0.1
    dropout_eight= 0.1
    dropout_nine= 0.1
    dropout_ten= 0.1
    dropout_eleven= 0.1
    dropout_twelve= 0.1

    # 1st conv layer
    kernel_one = tf.Variable(tf.random_normal([32, 32, 3, 64]))
    bias_one = tf.Variable(tf.random_normal([64]))
    conv_layer_one = tf.nn.relu(tf.nn.conv2d(x_image, kernel_one, strides=[1, 1, 1, 1], padding='SAME') + bias_one)
    conv_layer_one_drop = tf.nn.dropout(conv_layer_one, dropout_one)

    # 2nd conv layer
    kernel_two = tf.Variable(tf.random_normal([32, 32, 64, 64]))
    bias_two = tf.Variable(tf.random_normal([64]))
    conv_layer_two = tf.nn.relu(
        tf.nn.conv2d(conv_layer_one_drop, kernel_two, strides=[1, 1, 1, 1], padding='SAME') + bias_two)
    conv_layer_two_drop = tf.nn.dropout(conv_layer_two, dropout_two)

    # 3rd conv layer
    kernel_three = tf.Variable(tf.random_normal([32, 32, 64, 64]))
    bias_three = tf.Variable(tf.random_normal([64]))
    conv_layer_three = tf.nn.relu(
        tf.nn.conv2d(conv_layer_two_drop, kernel_three, strides=[1, 2, 2, 1], padding='SAME') + bias_three)
    conv_layer_three_drop = tf.nn.dropout(conv_layer_three, dropout_three)

    # 4th conv layer
    kernel_four = tf.Variable(tf.random_normal([4, 4, 64, 192]))
    bias_four = tf.Variable(tf.random_normal([192]))
    conv_layer_four = tf.nn.relu(
        tf.nn.conv2d(conv_layer_three_drop, kernel_four, strides=[1, 1, 1, 1], padding='SAME') + bias_four)
    conv_layer_four_drop = tf.nn.dropout(conv_layer_four, dropout_four)

    # 5th conv layer
    kernel_five = tf.Variable(tf.random_normal([4, 4, 192, 192]))
    bias_five = tf.Variable(tf.random_normal([192]))
    conv_layer_five = tf.nn.relu(
        tf.nn.conv2d(conv_layer_four_drop, kernel_five, strides=[1, 1, 1, 1], padding='SAME') + bias_five)
    conv_layer_five_drop = tf.nn.dropout(conv_layer_five, dropout_five)

    # 6th conv layer
    kernel_six = tf.Variable(tf.random_normal([4, 4, 192, 192]))
    bias_six = tf.Variable(tf.random_normal([192]))
    conv_layer_six = tf.nn.relu(
        tf.nn.conv2d(conv_layer_five_drop, kernel_six, strides=[1, 2, 2, 1], padding='SAME') + bias_six)
    conv_layer_six_drop = tf.nn.dropout(conv_layer_six, dropout_six)

    # 7th conv layer
    kernel_seven = tf.Variable(tf.random_normal([4, 4, 192, 256]))
    bias_seven = tf.Variable(tf.random_normal([256]))
    conv_layer_seven = tf.nn.relu(
        tf.nn.conv2d(conv_layer_six_drop, kernel_seven, strides=[1, 1, 1, 1], padding='SAME') + bias_seven)
    conv_layer_seven_drop = tf.nn.dropout(conv_layer_seven, dropout_seven)

    # 8th conv layer
    kernel_eight = tf.Variable(tf.random_normal([4, 4, 256, 512]))
    bias_eight = tf.Variable(tf.random_normal([512]))
    conv_layer_eight = tf.nn.relu(
        tf.nn.conv2d(conv_layer_seven_drop, kernel_eight, strides=[1, 2, 2, 1], padding='SAME') + bias_eight)
    conv_layer_eight_drop = tf.nn.dropout(conv_layer_eight, dropout_eight)

    # 9th conv layer
    kernel_nine = tf.Variable(tf.random_normal([4, 4, 512, 512]))
    bias_nine = tf.Variable(tf.random_normal([512]))
    conv_layer_nine = tf.nn.relu(
        tf.nn.conv2d(conv_layer_eight_drop, kernel_nine, strides=[1, 1, 1, 1], padding='SAME') + bias_nine)
    conv_layer_nine_drop = tf.nn.dropout(conv_layer_nine, dropout_nine)

    # 10th conv layer
    kernel_ten = tf.Variable(tf.random_normal([4, 4, 512, 512]))
    bias_ten = tf.Variable(tf.random_normal([512]))
    conv_layer_ten = tf.nn.relu(
        tf.nn.conv2d(conv_layer_nine_drop, kernel_ten, strides=[1, 2, 2, 1], padding='SAME') + bias_ten)
    conv_layer_ten_drop = tf.nn.dropout(conv_layer_ten, dropout_ten)

    # 11th conv layer
    kernel_eleven = tf.Variable(tf.random_normal([4, 4, 512, 512]))
    bias_eleven = tf.Variable(tf.random_normal([512]))
    conv_layer_eleven = tf.nn.relu(
        tf.nn.conv2d(conv_layer_ten_drop, kernel_eleven, strides=[1, 1, 1, 1], padding='SAME') + bias_eleven)
    conv_layer_eleven_drop = tf.nn.dropout(conv_layer_eleven, dropout_eleven)

    # 12th conv layer
    kernel_twelve = tf.Variable(tf.random_normal([4, 4, 512, 1024]))
    bias_twelve = tf.Variable(tf.random_normal([1024]))
    conv_layer_twelve = tf.nn.relu(
        tf.nn.conv2d(conv_layer_eleven_drop, kernel_twelve, strides=[1, 2, 2, 1], padding='SAME') + bias_twelve)
    conv_layer_twelve_drop = tf.nn.dropout(conv_layer_twelve, dropout_twelve)

    # Fully connected layer
    conv_layer_twelve_drop_array = tf.reshape(conv_layer_twelve_drop, [-1, 16 * 16 * 1024])
    matrix_final = weight_variable([16 * 16 * 1024, 8])
    bias_final = bias_variable([8])

    # Network output
    y_conv = tf.matmul(conv_layer_twelve_drop_array, matrix_final) + bias_final

    if verbose == True:
        print("conv_layer_one size: " + str(conv_layer_one.get_shape()))
        print("conv_layer_two size: " + str(conv_layer_two.get_shape()))
        print("conv_layer_three size: " + str(conv_layer_three.get_shape()))
        print("conv_layer_four size: " + str(conv_layer_four.get_shape()))
        print("conv_layer_five size: " + str(conv_layer_five.get_shape()))
        print("conv_layer_six size: " + str(conv_layer_six.get_shape()))
        print("conv_layer_seven size: " + str(conv_layer_seven.get_shape()))
        print("conv_layer_eight size: " + str(conv_layer_eight.get_shape()))
        print("conv_layer_nine size: " + str(conv_layer_nine.get_shape()))
        print("conv_layer_ten size: " + str(conv_layer_ten.get_shape()))
        print("conv_layer_eleven size: " + str(conv_layer_eleven.get_shape()))
        print("conv_layer_twelve size: " + str(conv_layer_twelve.get_shape()))

    return y_conv


def cnn_7_model(x_image, verbose=False):

    # Dropout probabilities
    dropout_one= 0.1
    dropout_two= 0.1
    dropout_three= 0.1
    dropout_four= 0.1
    dropout_five= 0.1
    dropout_six= 0.1
    dropout_seven= 0.1

    # 1st conv layer
    kernel_one = tf.Variable(tf.random_normal([5, 5, 3, 64]))
    bias_one = tf.Variable(tf.random_normal([64]))
    conv_layer_one = tf.nn.relu(
        tf.nn.conv2d(x_image, kernel_one, strides=[1, 1, 1, 1], padding='SAME') + bias_one)
    conv_layer_one_drop = tf.nn.dropout(conv_layer_one, dropout_one)

    # 2nd conv layer
    kernel_two = tf.Variable(tf.random_normal([9, 9, 64, 128]))
    bias_two = tf.Variable(tf.random_normal([128]))
    conv_layer_two = tf.nn.relu(
        tf.nn.conv2d(conv_layer_one_drop, kernel_two, strides=[1, 4, 4, 1], padding='SAME') + bias_two)
    conv_layer_two_drop = tf.nn.dropout(conv_layer_two, dropout_two)

    # 3rd conv layer
    kernel_three = tf.Variable(tf.random_normal([3, 3, 128, 128]))
    bias_three = tf.Variable(tf.random_normal([128]))
    conv_layer_three = tf.nn.relu(
        tf.nn.conv2d(conv_layer_two_drop, kernel_three, strides=[1, 1, 1, 1], padding='SAME') + bias_three)
    conv_layer_three_drop = tf.nn.dropout(conv_layer_three, dropout_three)

    # 4th conv layer
    kernel_four = tf.Variable(tf.random_normal([3, 3, 128, 256]))
    bias_four = tf.Variable(tf.random_normal([256]))
    conv_layer_four = tf.nn.relu(
        tf.nn.conv2d(conv_layer_three_drop, kernel_four, strides=[1, 2, 2, 1], padding='SAME') + bias_four)
    conv_layer_four_drop = tf.nn.dropout(conv_layer_four, dropout_four)

    # 5th conv layer
    kernel_five = tf.Variable(tf.random_normal([3, 3, 256, 512]))
    bias_five = tf.Variable(tf.random_normal([512]))
    conv_layer_five = tf.nn.relu(
        tf.nn.conv2d(conv_layer_four_drop, kernel_five, strides=[1, 1, 1, 1], padding='SAME') + bias_five)
    #dropout_five = tf.placeholder(tf.float32)
    conv_layer_five_drop = tf.nn.dropout(conv_layer_five, dropout_five)

    # 6th conv layer
    kernel_six = tf.Variable(tf.random_normal([3, 3, 512, 1024]))
    bias_six = tf.Variable(tf.random_normal([1024]))
    conv_layer_six = tf.nn.relu(
        tf.nn.conv2d(conv_layer_five_drop, kernel_six, strides=[1, 2, 2, 1], padding='SAME') + bias_six)
    #dropout_six = tf.placeholder(tf.float32)
    conv_layer_six_drop = tf.nn.dropout(conv_layer_six, dropout_six)

    # 7th conv layer
    kernel_seven = tf.Variable(tf.random_normal([3, 3, 1024, 2048]))
    bias_seven = tf.Variable(tf.random_normal([2048]))
    conv_layer_seven = tf.nn.relu(
        tf.nn.conv2d(conv_layer_six_drop, kernel_seven, strides=[1, 2, 2, 1], padding='SAME') + bias_seven)
    #dropout_seven = tf.placeholder(tf.float32)
    conv_layer_seven_drop = tf.nn.dropout(conv_layer_seven, dropout_seven)

    # Fully connected layer
    conv_layer_twelve_drop_array = tf.reshape(conv_layer_seven_drop, [-1, 16 * 16 * 2048])
    matrix_final = weight_variable([16 * 16 * 2048, 8])
    bias_final = bias_variable([8])

    # Network output
    y_conv = tf.matmul(conv_layer_twelve_drop_array, matrix_final) + bias_final

    if verbose == True:
        print("conv_layer_one size: " + str(conv_layer_one.get_shape()))
        print("conv_layer_two size: " + str(conv_layer_two.get_shape()))
        print("conv_layer_three size: " + str(conv_layer_three.get_shape()))
        print("conv_layer_four size: " + str(conv_layer_four.get_shape()))
        print("conv_layer_five size: " + str(conv_layer_five.get_shape()))
        print("conv_layer_six size: " + str(conv_layer_six.get_shape()))
        print("conv_layer_seven size: " + str(conv_layer_seven.get_shape()))

    return y_conv


def cnn_5_model(x_image, verbose=False):

    # Dropout probabilities
    dropout_one= 0.1
    dropout_two= 0.1
    dropout_three= 0.1
    dropout_four= 0.1
    dropout_five= 0.1

    # 1st conv layer
    kernel_one = tf.Variable(tf.random_normal([5, 5, 3, 128]))
    bias_one = tf.Variable(tf.random_normal([128]))
    conv_layer_one = tf.nn.relu(
        tf.nn.conv2d(x_image, kernel_one, strides=[1, 2, 2, 1], padding='SAME') + bias_one)
    conv_layer_one_drop = tf.nn.dropout(conv_layer_one, dropout_one)

    # 2nd conv layer
    kernel_two = tf.Variable(tf.random_normal([5, 5, 128, 256]))
    bias_two = tf.Variable(tf.random_normal([256]))
    conv_layer_two = tf.nn.relu(
        tf.nn.conv2d(conv_layer_one_drop, kernel_two, strides=[1, 2, 2, 1], padding='SAME') + bias_two)
    conv_layer_two_drop = tf.nn.dropout(conv_layer_two, dropout_two)

    # 3rd conv layer
    kernel_three = tf.Variable(tf.random_normal([5, 5, 256, 512]))
    bias_three = tf.Variable(tf.random_normal([512]))
    conv_layer_three = tf.nn.relu(
        tf.nn.conv2d(conv_layer_two_drop, kernel_three, strides=[1, 2, 2, 1], padding='SAME') + bias_three)
    conv_layer_three_drop = tf.nn.dropout(conv_layer_three, dropout_three)

    # 4th conv layer
    kernel_four = tf.Variable(tf.random_normal([5, 5, 512, 1024]))
    bias_four = tf.Variable(tf.random_normal([1024]))
    conv_layer_four = tf.nn.relu(
        tf.nn.conv2d(conv_layer_three_drop, kernel_four, strides=[1, 2, 2, 1], padding='SAME') + bias_four)
    conv_layer_four_drop = tf.nn.dropout(conv_layer_four, dropout_four)

    # 5th conv layer
    kernel_five = tf.Variable(tf.random_normal([5, 5, 1024, 2048]))
    bias_five = tf.Variable(tf.random_normal([2048]))
    conv_layer_five = tf.nn.relu(
        tf.nn.conv2d(conv_layer_four_drop, kernel_five, strides=[1, 2, 2, 1], padding='SAME') + bias_five)
    #dropout_five = tf.placeholder(tf.float32)
    conv_layer_five_drop = tf.nn.dropout(conv_layer_five, dropout_five)

    # Fully connected layer
    conv_layer_twelve_drop_array = tf.reshape(conv_layer_five_drop, [-1, 16 * 16 * 2048])
    matrix_final = weight_variable([16 * 16 * 2048, 8])
    bias_final = bias_variable([8])

    # Network output
    y_conv = tf.matmul(conv_layer_twelve_drop_array, matrix_final) + bias_final

    if verbose == True:
        print("conv_layer_one size: " + str(conv_layer_one.get_shape()))
        print("conv_layer_two size: " + str(conv_layer_two.get_shape()))
        print("conv_layer_three size: " + str(conv_layer_three.get_shape()))
        print("conv_layer_four size: " + str(conv_layer_four.get_shape()))
        print("conv_layer_five size: " + str(conv_layer_five.get_shape()))

    return y_conv

def cnn_3_model(x_image, verbose=False):

    # Dropout probabilities
    dropout_one= 0.1
    dropout_two= 0.1
    dropout_three= 0.1

    # 1st conv layer
    kernel_one = tf.Variable(tf.random_normal([5, 5, 3, 128]))
    bias_one = tf.Variable(tf.random_normal([128]))
    conv_layer_one = tf.nn.relu(
        tf.nn.conv2d(x_image, kernel_one, strides=[1, 4, 4, 1], padding='SAME') + bias_one)
    conv_layer_one_drop = tf.nn.dropout(conv_layer_one, dropout_one)

    # 2nd conv layer
    kernel_two = tf.Variable(tf.random_normal([3, 3, 128, 256]))
    bias_two = tf.Variable(tf.random_normal([256]))
    conv_layer_two = tf.nn.relu(
        tf.nn.conv2d(conv_layer_one_drop, kernel_two, strides=[1, 2, 2, 1], padding='SAME') + bias_two)
    conv_layer_two_drop = tf.nn.dropout(conv_layer_two, dropout_two)

    # 3rd conv layer
    kernel_three = tf.Variable(tf.random_normal([3, 3, 256, 512]))
    bias_three = tf.Variable(tf.random_normal([512]))
    conv_layer_three = tf.nn.relu(
        tf.nn.conv2d(conv_layer_two_drop, kernel_three, strides=[1, 2, 2, 1], padding='SAME') + bias_three)
    conv_layer_three_drop = tf.nn.dropout(conv_layer_three, dropout_three)

    # Fully connected layer
    conv_layer_twelve_drop_array = tf.reshape(conv_layer_three_drop, [-1, 32 * 32 * 512])
    matrix_final = weight_variable([32 * 32 * 512, 8])
    bias_final = bias_variable([8])

    # Network output
    y_conv = tf.matmul(conv_layer_twelve_drop_array, matrix_final) + bias_final

    if verbose == True:
        print("conv_layer_one size: " + str(conv_layer_one.get_shape()))
        print("conv_layer_two size: " + str(conv_layer_two.get_shape()))
        print("conv_layer_three size: " + str(conv_layer_three.get_shape()))

    return y_conv