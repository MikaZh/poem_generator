import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('C:/Users/meruy/Desktop/poem_g/weights.h5')

#HOW TO EXTRACT MODEL FIT FROM SAVED MODEL

history_dict=model.history
loss_values=history_dict['loss']
epochs=range(1, len(history_dict['loss'])+1)
plt.plot(epochs, loss_values, 'bo', label="Training loss")
plt.title("Traning loss ")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.plot()
