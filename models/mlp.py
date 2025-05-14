import numpy as np


class MultilayerPerceptron:
    def __init__(self, p, q, m, learning_rate=0.01):
        self.p = p # Número de variáveis de entrada
        self.q = q # Lista de número de neurônios em cada camada oculta
        self.m = m # Número de neurônios na camada de saída
        self.layers = []
        self.learning_rate = learning_rate

        self.layers.append(Layer(q[0], self.p))
        for i in range(len(q)):
            self.layers.append(Layer(q[i], q[i-1] if i > 0 else q[0]))
        self.layers.append(Layer(m, q[-1]))

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, x, d, update=True):
        """
        Perform the backward pass (backpropagation) to compute gradients.

        Parameters:
        x -- Input data (xamostra)
        d -- Target output (rótulo)
        update -- If True, update the weights; if False, only compute gradients.
        """
        j = len(self.layers) - 1  # Start from the last layer

        while j >= 0:
            if j + 1 == len(self.layers):  # Output layer
                self.layers[j].delta = self.layers[j].activation_derivative() * (d - self.layers[j].output)
                y_bias = np.vstack((-np.ones((1, 1)), self.layers[j - 1].output))
                grad = self.layers[j].delta @ y_bias.T
                if update:
                    self.layers[j].w += self.learning_rate * grad
                # Optionally, store the gradient for checking:
                self.layers[j].grad = grad

            elif j == 0:  # First hidden layer
                Wb = self.layers[j + 1].w[:, 1:].T
                self.layers[j].delta = self.layers[j].activation_derivative() * (Wb @ self.layers[j + 1].delta)
                x_bias = np.vstack((-np.ones((1, 1)), x))
                grad = self.layers[j].delta @ x_bias.T
                if update:
                    self.layers[j].w += self.learning_rate * grad
                self.layers[j].grad = grad

            else:  # Hidden layers
                Wb = self.layers[j + 1].w[:, 1:].T
                self.layers[j].delta = self.layers[j].activation_derivative() * (Wb @ self.layers[j + 1].delta)
                y_bias = np.vstack((-np.ones((1, 1)), self.layers[j - 1].output))
                grad = self.layers[j].delta @ y_bias.T
                if update:
                    self.layers[j].w += self.learning_rate * grad
                self.layers[j].grad = grad

            j -= 1  # Move to the previous layer

    def train(self, x, y, epochs, tol, patience=5):
        """
        Train the MLP using backpropagation.

        Parameters:
        x -- Input data
        y -- Target output
        epochs -- Number of training epochs
        tol -- Tolerance for convergence
        """
        count = 0
        errors = []
        mse = 1
        for epoch in range(epochs):
            ise = 0
            for i in range(x.shape[1]):
                # print("Forward", x[:, i].reshape((self.p, 1)).shape)
                # print("Backward", y[i, :].reshape((self.m, 1)).shape)
                x_k = x[:, i].reshape((self.p, 1))
                u_k = self.forward(x_k)
                d_k = y[:, i].reshape((self.m, 1))
                self.backward(x_k, d_k)
                ise += np.sum((d_k - u_k) ** 2) / 2
            mse = ise / (2 * x.shape[1])
            if epoch >= 1000:
                if (mse > errors[-1] if errors else 0):
                    count += 1
                    print("Error got worse. Count:", count)
                    if count > patience:
                        print("Early stopping")
                        break
                elif count != 0:
                    print("Error improved. Count:", count)
                    count = 0
            errors.append(mse)
            print(f"Epoch: {epoch}, MSE: {mse}")
            if abs(mse) < tol:
                print(f"Epoch: {epoch}, MSE: {mse}")
                print("Training completed")
                break
        else:
            print(f"Epoch: {epochs}")
            print("Training completed")

    def mse(self, x, y):
        """
        Compute the Mean Squared Error (MSE) between predicted and target outputs.

        Parameters:
        x -- Input data
        y -- Target output

        Returns:
        Mean squared error over all samples.
        """
        total_error = 0
        num_samples = x.shape[1]
        for i in range(num_samples):
            x_k = x[:, i].reshape((self.p, 1))
            u_k = self.forward(x_k)
            d_k = y[:, i].reshape((self.m, 1))
            # Instant squared error with a 1/2 factor
            error = np.sum((d_k - u_k) ** 2) / 2
            total_error += error
        mse_value = total_error / (2 * num_samples)
        return mse_value

    def predict(self, x):
        """
        Make predictions using the trained MLP.

        Parameters:
        x -- Input data

        Returns:
        Predicted output
        """
        preds = []
        for i in range(x.shape[1]):
            y_k = self.forward(x[:, i].reshape((self.p, 1)))
            y_k = y_k.reshape((self.m, 1))
            pred = np.argmax(y_k, axis=0)
            # print("Pred", pred.shape, pred)
            preds.append(pred)
        preds = np.array(preds)
        preds = preds.reshape((x.shape[1], 1))
        preds = preds.T
        return preds

    def gradient_checking(self, x, d, epsilon=1e-7):
        """
        Perform gradient checking to verify the correctness of backpropagation.

        Parameters:
        x -- Input data (single sample)
        d -- Target output (single sample)
        epsilon -- Small value for numerical gradient approximation
        """
        # Step 1: Perform a forward pass
        self.forward(x)

        # Step 2: Perform a backward pass to compute analytical gradients
        self.backward(x, d, update=False)

        # Step 3: Compute numerical gradients
        for l, layer in enumerate(self.layers):
            numerical_gradient = np.zeros_like(layer.w)
            for i in range(layer.w.shape[0]):
                for j in range(layer.w.shape[1]):
                    # Save the original weight
                    original_weight = layer.w[i, j]

                    # Compute J(W + epsilon)
                    layer.w[i, j] = original_weight + epsilon
                    output_plus = self.forward(x)
                    loss_plus = np.sum((output_plus - d) ** 2) / 2  # MSE loss

                    # Compute J(W - epsilon)
                    layer.w[i, j] = original_weight - epsilon
                    output_minus = self.forward(x)
                    loss_minus = np.sum((output_minus - d) ** 2) / 2  # MSE loss

                    # Compute numerical gradient
                    numerical_gradient[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                    # Restore the original weight
                    layer.w[i, j] = original_weight

            # Step 4: Compare analytical and numerical gradients
            analytical_gradient = layer.delta @ layer.input.T
            difference = np.linalg.norm(analytical_gradient - numerical_gradient) / (
                np.linalg.norm(analytical_gradient) + np.linalg.norm(numerical_gradient)
            )

            print(f"Layer {l}: Gradient difference = {difference}")
            if difference > 1e-5:
                print("Warning: Gradients might not be correct!")
            # print(f"Layer {l} delta: {layer.delta}")
            # print(f"Layer {l} input: {layer.input}")


class Layer:
    def __init__(self, in_features, out_features):
        self.w = np.random.random_sample((in_features, out_features + 1))
        self.input = None
        self.delta = None
        self.output = None

    def forward(self, u):
        # Include bias
        u = np.vstack(
            (-np.ones((1, 1)), u)
        )
        self.input = u
        # print("w", self.w.shape)
        # print("u", u.shape)
        assert self.w.shape[1] == u.shape[0], f"w: {self.w.shape}, u: {u.shape}"
        self.output = self.activation(self.w @ u)
        return self.output

    def activation(self, u):
        return 1 / (1 + np.exp(-u))

    def activation_derivative(self):
        return self.output * (1 - self.output)

    def __repr__(self):
        return str(self.w)
