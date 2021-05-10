from tensorflow.keras import layers, Model, Sequential
# This section has built the Resnet 50 model as a function

class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        # First layer：Using 1X1 convolution kernel to reduce the dimensionality
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # Second layer：Using 3X3 convolutional layer
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # Third layer：Using 1X1 convolution kernel to increase the dimensionality
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

#Building the basic convolutional network structure
    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
          identity = self.downsample(inputs)
        # First layer
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        # Second layer
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        # Third layer
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        # Addition of channels
        x = self.add([x, identity])
        x = self.relu(x)
        return x

# When stride not equal to 1 or output dimensions not equal to input dimensions, a 1X1 convolutional layer need to be added
def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")
    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))
    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))
    return Sequential(layers_list, name=name)

#Buliding the basic resnet structure
def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # Define the input (batch, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # First layer: conv1
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)
    # conv2_x
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    # conv3_x
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    # conv4_x
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    # conv5_x
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)
    # Global average pooling
    if include_top:
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x
    model = Model(inputs=input_image, outputs=predict)
    return model

#Assemble the resnet50 network structure
def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


