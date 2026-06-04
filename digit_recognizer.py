import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DigitRecognizer:
    def __init__(self, k=7):
        self.k = k
        self.model = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.digits = None
    
    def load_data(self):
        """Load the digits dataset"""
        self.digits = load_digits()
        print(f"Loaded {self.digits.data.shape[0]} samples")
        return self.digits
    
    def preprocess_and_split(self, test_size=0.2, random_state=42):
        """Split data into train and test"""
        X = self.digits.data
        y = self.digits.target
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def train(self):
        """Train KNN model"""
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X_train, self.Y_train)
        print(f"Model trained with K={self.k}")
    
    def predict(self, samples=None):
        """Predict on test set or provided samples"""
        if samples is None:
            samples = self.X_test
        return self.model.predict(samples)
    
    def evaluate(self):
        """Evaluate model"""
        y_pred = self.predict()
        accuracy = accuracy_score(self.Y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy, y_pred
    
    def visualize_samples(self, n=10):
        """Visualize sample digits"""
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.digits.images[i], cmap='gray')
            ax.set_title(f'Digit: {self.digits.target[i]}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('sample_digits.png')
        print('Sample images saved')
    
    def plot_accuracy_vs_k(self, ks=[1,3,5,7,9]):
        """Plot accuracy for different K"""
        accuracies = []
        for k in ks:
            temp_knn = KNeighborsClassifier(n_neighbors=k)
            temp_knn.fit(self.X_train, self.Y_train)
            y_pred = temp_knn.predict(self.X_test)
            acc = accuracy_score(self.Y_test, y_pred)
            accuracies.append(acc)
        plt.figure(figsize=(8,5))
        plt.plot(ks, accuracies, marker='o')
        plt.title('Accuracy vs K')
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig('accuracy_vs_k.png')
        print('Accuracy plot saved')
        return ks[np.argmax(accuracies)]
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        y_pred = self.predict()
        cm = confusion_matrix(self.Y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (K={self.k})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        print('Confusion matrix saved')

# Usage
if __name__ == "__main__":
    recognizer = DigitRecognizer(k=7)
    recognizer.load_data()
    recognizer.preprocess_and_split()
    recognizer.train()
    acc, _ = recognizer.evaluate()
    recognizer.visualize_samples()
    optimal_k = recognizer.plot_accuracy_vs_k()
    print(f"Optimal K: {optimal_k}")
    recognizer.plot_confusion_matrix()
