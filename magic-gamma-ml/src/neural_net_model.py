import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

def plot_history(history):
    """
    Generates diagnostic plots to monitor training progress and detect overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Binary Crossentropy measures the 'distance' between predicted and actual probabilities
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Learning Curve: Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy shows the percentage of correct classifications
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Learning Curve: Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/plots/neural_net_history.png")
    plt.close()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    """
    Defines and compiles a Multi-Layer Perceptron (MLP) architecture.
    """
    nn_model = tf.keras.Sequential([
        # Input layer: matched to the 10 geometric features of the MAGIC dataset
        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        # Hidden Layer 1: ReLU activation introduces non-linearity
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        # Dropout: Regularization technique to prevent overfitting by deactivating neurons
        tf.keras.layers.Dropout(dropout_prob),

        # Hidden Layer 2
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),

        # Output layer: Sigmoid squashes output to [0, 1] for binary probability
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Adam optimizer: An adaptive learning rate optimization algorithm
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # EarlyStopping: Halts training if validation loss plateaus to save compute and avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = nn_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2, # Uses 20% of training data for internal validation
        verbose=0,
        callbacks=[early_stop]
    )

    return nn_model, history

def run_neural_net(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Executes a Grid Search to find the best combination of NN hyperparameters.
    """
    print("====================================")
    print("Neural Network (Grid Search)")
    print("====================================")

    least_val_loss = float('inf')
    least_loss_model = None

    # Nested loops iterate through the hyperparameter space (Grid Search)
    for num_nodes in [16, 32]:
        for dropout_prob in [0, 0.2]:
            for lr in [0.01, 0.005]:
                for batch_size in [32, 64]:
                    print(f"Training: nodes={num_nodes}, dropout={dropout_prob}, lr={lr}, batch={batch_size}")

                    model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs=100)

                    # Monitor performance on the validation set to pick the 'best' model
                    val_loss = model.evaluate(X_valid, y_valid, verbose=0)[0]

                    if val_loss < least_val_loss:
                        least_val_loss = val_loss
                        least_loss_model = model
                        plot_history(history)

    print(f"\nBest Validation Loss: {least_val_loss:.4f}")

    # Evaluate the champion model on the final held-out test set
    y_pred = least_loss_model.predict(X_test, verbose=0)
    y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

    print(classification_report(y_test, y_pred))
