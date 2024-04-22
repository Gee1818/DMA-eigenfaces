import numpy as np

# disable scientific notation
np.set_printoptions(suppress=True, precision=2)

class n_network:
    def __init__(self, params, layers, activations, loss_function):
        self.layers = layers #list of layer clasess
        self.learning_rate = params['learning_rate']
        self.max_epochs = params['epochs']
        self.target_error = params['target_error'] 
        self.weights = []
        self.biases = []
        self.activations = activations #list of activations per layer
        self.loss = loss_function #CrossEntropyLoss class in functions_nn.py
        self.accuracy = 0
        
        
    def train(self, X, Y):
        epoch = 0
        error = 10
        
        while error > self.target_error and epoch < self.max_epochs:
            error = 0
            for x, y in zip(X, Y):
                
                # Forward
                activation = x # input
                
                # hidden layers
                for layer , activf in zip(self.layers[:-1] , self.activations[:-1]):
                    output = layer.forward(activation) # z_j
                    activation = activf.forward(output) #a_j
                    
                # output layer
                output = self.layers[-1].forward(activation) #z_n
                activation_output = self.activations[-1].forward(output) #a_n
                
                
                # Loss function
                gradient = self.loss.prime(y, activation_output)
                
                # Backward        
                layers_and_activs = [val for pair in zip(self.layers, self.activations) for val in pair] # https://stackoverflow.com/a/7946825
                for layer in reversed(layers_and_activs):
                    gradient = layer.backward(gradient, self.learning_rate)
                
                # Error
                error += self.loss.calc(y, activation_output)
        
            # Update loop params
            epoch += 1
            error /= len(X)
            
            # Save parameters
            self.save_params()
            
            # print epoch error
            if epoch % 20 == 0:
                print(f"Epoch {epoch} Error {error}")
            if epoch % 4000 == 0:
                learning_rate = learning_rate / 10

        
       
    def predict(self, X):
        
        if not self.weights: 
            print("The model is not trained")
            return
        
        predictions = [] 
        
        for x in X:
            activation = x
            
            # hidden layers
            for layer, activf in zip(self.layers[:-1] , self.activations[:-1]):
                output = layer.forward(activation)
                activation = activf.forward(output)
    
        
            # output layer
            output = self.layers[-1].forward(activation)
            activation_output = self.activations[-1].forward(output)
            
            predictions.append(np.argmax(activation_output, axis=0)[0])      
        
        return predictions
    
    def evaluate(self, X, Y):
        predictions = self.predict(X)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == np.argmax(Y[i]):
                correct += 1
        self.accuracy = correct / len(predictions)
        print(f"Accuracy: {self.accuracy}")
        
        return self.accuracy
        
    def save_params(self):
        for layer in self.layers:
            self.weights.append(layer.weights)
            self.biases.append(layer.biases)
        
    
    def __str__(self) -> str:
        result = f"Neural Network with {len(self.layers)} layers:\n"
        for i in range(len(self.layers)):
            result += f"Layer {i+1}:\n"
            result += f"Weights: {self.layers[i].weights}\n"
            result += f"Biases: {self.layers[i].biases}\n"
        return result
    
    def export_params(self):
        for i, layer in enumerate(self.layers):
            np.savetxt(f"weights_{i}.csv", layer.weights, delimiter=",")
            np.savetxt(f"biases_{i}.csv", layer.biases, delimiter=",")
        print("Parameters exported")       
        
        