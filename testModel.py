# -*- coding: utf-8 -*-
import data_helpers
import keras
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

x = data_helpers.my_get_input_sentence()
model = keras.models.load_model('./simple_net.h5')
y = model.predict(x)
dict = ['business', 'service', 'others', 'product',
        'platform']  # (I choose the wrong csv as the second 'service' should be 'product')
predict_label = []
# for index in range(len(x)):
#     predict_label[index] = dict[y[index]]

result = model.predict_proba(x)


print predict_label;