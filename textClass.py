import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

tf.compat.v1.disable_eager_execution()

categories = ['sci.med', 'talk.politics.guns' , 'comp.windows.x']

newsgroups_train = fetch_20newsgroups(subset = 'train', categories = categories)
newsgroups_test = fetch_20newsgroups(subset = 'test', categories = categories)

vocabulary = Counter()
for text in newsgroups_train.data:
     for word in text.split(' '):
         vocabulary[word.lower()] += 1

for text in newsgroups_test.data:
     for word in text.split(' '):
         vocabulary[word.lower()] += 1

total_words = len(vocabulary)

def get_word_2_index(vocabulary):
    word2index = {}
    for i,word in enumerate(vocabulary):
        word2index[word.lower()] = i

    return word2index

word2index = get_word_2_index(vocabulary)

# print('Індекс в словах "The": ', word2index['the'])
# print('Всього слів в базі: ', len(vocabulary))

def text_to_vector(text):
    layer = np.zeros(total_words,dtype=float)
    for word in text.split(' '):
        layer[word2index[word.lower()]] += 1
        
    return layer

def category_to_vector(category):
    y = np.zeros((3),dtype=float)
    if category == 0:
        y[0] = 1.
    elif category == 1:
        y[1] = 1.
    else:
        y[2] = 1.
        
    return y

def get_batch(df, i, batch_size):
     batches = []
     results = []

     texts = df.data[i*batch_size : i*batch_size+batch_size]
     categories = df.target[i*batch_size : i*batch_size+batch_size]

     for text in texts:
         layer = text_to_vector(text)
         batches.append(layer)

     for category in categories:
         y = category_to_vector(category)
         results.append(y)

     return np.array(batches), np.array(results)

# print('Кількість текстів та слів у 1-100 текстах - ', get_batch(newsgroups_train, 1, 100)[0].shape)
# print('Кількість текстів та категорій у 1-100 текстах - ', get_batch(newsgroups_train, 1, 100)[1].shape)
# print('Векторизовані дані - ', get_batch(newsgroups_train, 1, 100))


# Змінні-параметри
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Мережеві параметри
n_hidden_1 = 100   # 1 шар для обробки
n_hidden_2 = 100   # 2 шар для обробки
n_input = total_words  #
n_classes = 3   #       

# Локальні параметри
input_tensor = tf.compat.v1.placeholder(tf.float32,[None, n_input],name="input")
output_tensor =  tf.compat.v1.placeholder(tf.float32,[None, n_classes],name="output") 

# Функція обчислання кінцевих даних
def multiplayer_perception(input_tensor, weights, biases):
     layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
     layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
     layer_1 = tf.nn.relu(layer_1_addition)

     layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
     layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
     layer_2 = tf.nn.relu(layer_2_addition)


     out_layer_multiplication = tf.matmul(layer_2, weights['out'])
     out_layer_addition = out_layer_multiplication + biases['out']

     return out_layer_addition

weights = {
     'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
     'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
     'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes]))
}

biases = {
     'b1': tf.Variable(tf.random.normal([n_hidden_1])),
     'b2': tf.Variable(tf.random.normal([n_hidden_2])),
     'out': tf.Variable(tf.random.normal([n_classes]))
}

prediction = multiplayer_perception(input_tensor, weights, biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.compat.v1.global_variables_initializer()

saver = tf.compat.v1.train.Saver()

print('Помилок немає')

with tf.compat.v1.Session() as sess:
     sess.run(init)
     # Тренувальний цикл
     for epoch in range(training_epochs):
          avg_cost = 0. # Тип float
          total_batch = int(len(newsgroups_train.data)/batch_size)

          for i in range(total_batch):
               #х - текст, у - категорії(Достаємо їхні партії)
               batch_x, batch_y = get_batch(newsgroups_train,i,batch_size)

               c,_ = sess.run([loss,optimizer], feed_dict = {input_tensor: batch_x,output_tensor:batch_y})

               avg_cost += c / total_batch

          if epoch % display_step == 0:
               print('Epoch:', '%04d' % (epoch+1), 'loss = '< \
                     '{:,9f}', format(avg_cost))   

     print('Optimizer finished!')
     
     # Тестинг

     # Повернення вузлу, два вузли (Коди для першого тестингу)
     correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


     total_test_data = len(newsgroups_test.target)
     batch_x_test , batch_y_test = get_batch(newsgroups_test, 0, total_test_data)

     # Результати тестування
     print('Accuracy', accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test})) 

     # Зберігання сесії
     save_path = saver.save(sess, 'tmp/model.ckpt')
     print("Model saved in: %s" % save_path)
