import numpy as np
from tensorflow.keras.models import load_model

text = open('C:/Users/meruy/Desktop/sonnets.txt', encoding='utf8').read().lower()

chars = set(text)
sorted_chars = sorted(chars)
char_index=dict()
for index, char in enumerate(sorted_chars):
    char_index.update({char:index})
reverse_char_index = dict([(value, key) for (key, value) in char_index.items()])

model = load_model('C:/Users/meruy/Desktop/poem_g/weights.h5')

def generate(length_given, diversity_given):
    user_input = []
    user_input = input('Enter first line of poem (min 40 chars):  ')
    while(len(user_input) < 50):
        user_input = []
        user_input = input('..too short, please retype\n')
    user_input = user_input[0:50]
    poem = ''
    poem += user_input
    for i in range(length_given):
        x = np.zeros((1,50, len(sorted_chars)))
        for j, ch in enumerate(user_input):
            x[0, j, char_index[ch]] = 1
        predictions = model.predict(x, verbose = 0)[0]
        next_ind = sample(predictions, diversity_given)
        next_ch = reverse_char_index[next_ind]
        poem += next_ch
        user_input = user_input[1:] + next_ch
    return poem

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

poem_length = input('Enter entire length of the poem you want:  ')

#temp = [1.5, 0.1, 0.5, 1, 2.0] #HOW TO EVALUATE THIS

#for t in temp:
#print('Temperature: ' + str(t))
print(generate(int(poem_length), 1.0))
