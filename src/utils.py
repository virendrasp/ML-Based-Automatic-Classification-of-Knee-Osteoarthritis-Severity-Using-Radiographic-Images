import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_plot_loss_acc(model, model_name):
    plt.figure(figsize=(8, 4))
    plt.plot(model.history.history['loss'], label='Loss')
    plt.plot(model.history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(model.history.history['accuracy'], label='Accuracy')
    plt.plot(model.history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def get_evaluate(generator, dataset_name, model):
    loss, accuracy = model.evaluate(generator, verbose=0)
    print(f"{dataset_name} Loss: {loss:.4f}")
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")

def get_predict(generator, model):
    return model.predict(generator, verbose=0)