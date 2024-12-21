# custom_losses.py
import tensorflow as tf
from tensorflow.keras import backend as K

def categorical_focal_crossentropy(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class classification
    Formula:
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    Args:
        y_true: true labels, one-hot encoded.
        y_pred: predicted labels, probabilities.
        gamma: focusing parameter.
        alpha: balancing parameter.
    """
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    loss = K.sum(loss, axis=1)
    return loss
