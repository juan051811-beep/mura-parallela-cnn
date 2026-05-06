import tensorflow as tf

def cargar_cnn():
    modelo = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    modelo.trainable = False
    return modelo