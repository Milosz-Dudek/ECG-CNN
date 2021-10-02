import numpy as np
import pandas as pd
import tensorflow as tf 
import os 
from sklearn.model_selection import train_test_split


class CNN(object):
    def __init__(self, batch_size=64,
                 epochs=20, learning_rate=1e-4, 
                 dropout=0.5,
                 shuffle=True, random_seed=123):
        self.random_seed = np.random.seed(random_seed)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.shuffle = shuffle

        ## tworzy graf
        g = tf.Graph()
        with g.as_default():
            tf.random.set_seed(self.random_seed)
            ## buduje graf
            self.build_cnn()

            ## inicjator
            self.init_op = \
                tf.compat.v1.global_variables_initializer()

            ## obiekt zapisujący
            self.saver = tf.compat.v1.train.Saver()
            
        ## tworzy sesję
        self.sess = tf.compat.v1.Session(graph=g)

    def batch_generator(self, X, y):
        
        idx = np.arange(y.shape[0])
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_seed)
            rng.shuffle(idx)
            X = X[idx]
            y = y[idx]
        
        for i in range(0, X.shape[0], self.batch_size):
            yield (X[i:i+self.batch_size, :], y[i:i+self.batch_size])

    ## funkcje opakowujące

    def conv_layer(self, input_tensor, name,
                kernel_size, n_output_channels, 
                padding_mode='SAME', strides=2):
        with tf.compat.v1.variable_scope(name):
            ## pobiera n_input_channels::
            ##   wymiary tensora wejściowego: 
            ##   [grupa x szerokość x wysokość x kanały]
            input_shape = input_tensor.get_shape().as_list()
            n_input_channels = input_shape[-1] 

            weights_shape = (list(kernel_size) + 
                            [n_input_channels, n_output_channels])

            weights = tf.compat.v1.get_variable(name='_wagi',
                                    shape=weights_shape)
            print(weights)
            biases = tf.compat.v1.get_variable(name='_obciazenia',
                                    initializer=tf.zeros(
                                        shape=[n_output_channels]))
            print(biases)
            conv = tf.nn.conv1d(input=input_tensor, 
                                filters=weights,
                                stride=strides, 
                                padding=padding_mode)
            print(conv)
            conv = tf.nn.bias_add(conv, biases, 
                                name='przed-aktywacja_calkowita')
            print(conv)
            conv = tf.nn.relu(conv, name='aktywacja')
            print(conv)
            
            return conv
        
    def fc_layer(self, input_tensor, name, 
                n_output_units, activation_fn=None):
        with tf.compat.v1.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()[1:]
            n_input_units = np.prod(input_shape)
            if len(input_shape) > 1:
                input_tensor = tf.reshape(input_tensor, 
                                        shape=(-1, n_input_units))

            weights_shape = [n_input_units, n_output_units]

            weights = tf.compat.v1.get_variable(name='_wagi',
                                    shape=weights_shape)
            print(weights)
            biases = tf.compat.v1.get_variable(name='_obciazenia',
                                    initializer=tf.zeros(
                                        shape=[n_output_units]))
            print(biases)
            layer = tf.matmul(input_tensor, weights)
            print(layer)
            layer = tf.nn.bias_add(layer, biases,
                                name='przed-aktywacja_calkowita')
            print(layer)
            if activation_fn is None:
                return layer
            
            layer = activation_fn(layer, name='aktywacja')
            print(layer)
            return layer

    def build_cnn(self):
        ## Węzły zastępcze dla zmiennych X i y:
        tf_x = tf.compat.v1.placeholder(tf.float32, shape=[None, 187],
                            name='tf_x')
        tf_y = tf.compat.v1.placeholder(tf.int32, shape=[None],
                            name='tf_y')

        tf_x_reshaped = tf.reshape(tf_x, shape=[-1, 187, 1],
                                name='tf_x_przeksztalcony')
        ## Kodowanie „gorącojedynkowe”:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=5,
                                dtype=tf.float32,
                                name='tf_y_goracojedynkowe')

        ## Pierwsza warstwa: splot_1
        print('\nBudowanie pierwszej warstwy: ')
        h1 = self.conv_layer(tf_x_reshaped, name='splot_1',
                        kernel_size=[5], 
                        padding_mode='SAME',
                        n_output_channels=64)
        ## Łączenie maksymalizujące
        h1_pool = tf.nn.max_pool1d(h1, 
                                ksize=5,
                                strides=2,
                                padding='SAME')
        ## Druga warstwa: splot_2
        print('\nBudowanie drugiej warstwy: ')
        h2 = self.conv_layer(h1_pool, name='splot_2', 
                        kernel_size=[5], 
                        padding_mode='SAME',
                        n_output_channels=64)
        ## Łączenie maksymalizujące 
        h2_pool = tf.nn.max_pool1d(h2, 
                                ksize=5,
                                strides=2, 
                                padding='SAME')
        ## Trzecia warstwa: splot_3
        print('\nBudowanie trzeciej warstwy: ')
        h3 = self.conv_layer(h2_pool, name='splot_3', 
                        kernel_size=[5], 
                        padding_mode='SAME',
                        n_output_channels=64)
        ## Łączenie maksymalizujące 
        h3_pool = tf.nn.max_pool1d(h3, 
                                ksize=5,
                                strides=2, 
                                padding='SAME')
        ## Czwarta warstwa: w pełni połączona
        print('\nBudowanie czwartej warstwy:')
        h4 = self.fc_layer(h3_pool, name='pp_4',
                    n_output_units=256, 
                    activation_fn=tf.nn.relu)
        ## Piąta warstwa
        print('\nBudowanie piątej warstwy:')
        h5 = self.fc_layer(h4, name='pp_5',
                    n_output_units=128, 
                    activation_fn=tf.nn.relu)

        ## Porzucanie
        keep_prob = tf.compat.v1.placeholder(tf.float32, name='pp_prawd_pozost')
        h5_drop = tf.nn.dropout(h5, rate=1-keep_prob, 
                                name='warstwa_porzucania')

        ## Szósta warstwa: w pełni połączona (aktywacja liniowa)
        print('\nBudowanie szóstej warstwy:')
        h6 = self.fc_layer(h5_drop, name='pp_6',
                    n_output_units=5, 
                    activation_fn=None)

        ## Prognozowanie
        predictions = {
            'probabilities' : tf.nn.softmax(h6, name='prawdopodobienstwa'),
            'labels' : tf.cast(tf.argmax(h6, axis=1), tf.int32,
                            name='etykiety')
        }    

        ## Funkcja straty i optymalizacja
        cross_entropy_loss = tf.reduce_mean(
            tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                logits=h6, labels=tf_y_onehot),
            name='strata_entropia_krzyzowa')

        ## Optymalizator:
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss,
                                    name='op_uczenia')

        ## Obliczanie dokładności przewidywań
        correct_predictions = tf.equal(
            predictions['labels'], 
            tf_y, name='prawidlowe_predy')

        self.accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='dokladnosc')

    def train(self, training_set, validation_set=None, initialize=True):

        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])
        training_loss = []

        ## inicjowanie zmiennych
        if initialize:
            self.sess.run(self.init_op)

        np.random.seed(self.random_seed) # do tasowania w batch_generator
        for epoch in range(1, self.epochs+1):
            batch_gen = self.batch_generator(X_data, y_data)
            avg_loss = 0.0
            for i,(batch_x,batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0': batch_x, 
                        'tf_y:0': batch_y, 
                        'pp_prawd_pozost:0': self.dropout}
                loss, _= self.sess.run(
                        ['strata_entropia_krzyzowa:0', 'op_uczenia'],
                        feed_dict=feed)
                avg_loss += loss

            feed = {'tf_x:0': X_data, 
                        'tf_y:0': y_data, 
                        'pp_prawd_pozost:0': 1.0}

            print('Epoka %02d. Uśredniona strata w czasie uczenia: %7.3f '% \
            (epoch, avg_loss), end=' ')
            if validation_set is not None:
                feed = {'tf_x:0': validation_set[0],
                        'tf_y:0': validation_set[1],
                        'pp_prawd_pozost:0':1.0}
                valid_acc = self.sess.run('dokladnosc:0', feed_dict=feed)
                print('Dokładność walidacji: %7.3f' % valid_acc)
            else:
                print()

    def predict(self, X_test, return_proba=True):
        feed = {'tf_x:0': X_test, 
                'pp_prawd_pozost:0': 1.0}
        if return_proba:
            return self.sess.run('prawdopodobienstwa:0', feed_dict=feed)
        else:
            return self.sess.run('etykiety:0', feed_dict=feed)

    def save(self, epoch, path='C:\Projekty\ECG-CNN\model_saver'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Zapisywanie modelu w %s' % path)
        self.saver.save(self.sess, os.path.join(path,'cnn-model.ckpt'),
                global_step=epoch)

    def load(self, epoch, path='C:\Projekty\ECG-CNN\model_saver'):
        print('Wczytywanie modelu z %s' % path)
        self.saver.restore(self.sess, os.path.join(
                path, 'cnn-model.ckpt-%d' % epoch))


test_data = pd.read_csv("mitbih_test.csv", header=None)
train_data = pd.read_csv("mitbih_train.csv", header=None)

train_data = train_data.rename(columns={187: 'classes'})
test_data = test_data.rename(columns={187: 'classes'})

y_train = train_data['classes'].astype(int)
X_train = train_data.drop(columns=['classes'])

y_test = test_data['classes'].astype(int)
X_test = test_data.drop(columns=['classes'])

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)

cnn = CNN(epochs=20, learning_rate=1e-3, random_seed=123)

cnn.train(training_set=(X_train, y_train), 
          validation_set=(X_valid, y_valid))

# cnn.save(epoch=15) - napraw

preds = cnn.predict(X_test)
print('Dokładność dla danych testowych: %.2f%%' % (100*
      np.sum(y_test == preds)/len(y_test)))

y_pred = np.asarray(preds)

from sklearn.metrics import classification_report
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8, 8))
plot_confusion_matrix(cnf_matrix, normalize = True, classes=['N', 'S', 'V', 'F', 'Q'],
                      title='Confusion matrix, with normalization')
plt.show()