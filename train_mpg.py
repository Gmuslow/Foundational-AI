from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlp

if __name__ == "__main__":
    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    # Combine features and target into one DataFrame for easy filtering
    data = pd.concat([X, y], axis=1)

    # Drop rows where the target variable is NaN
    cleaned_data = data.dropna()

    # Split the data back into features (X) and target (y)
    X = cleaned_data.iloc[:, :-1]
    y = cleaned_data.iloc[:, -1]
    print(y)
    # Display the number of rows removed
    rows_removed = len(data) - len(cleaned_data)
    print(f"Rows removed: {rows_removed}")

    # Do a 70/30 split (e.g., 70% train, 30% other)
    X_train, X_leftover, y_train, y_leftover = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,    # for reproducibility
        shuffle=True,       # whether to shuffle the data before splitting
    )

    # Split the remaining 30% into validation/testing (15%/15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_leftover, y_leftover,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )

    # Compute statistics for X (features)
    X_mean = X_train.mean(axis=0)  # Mean of each feature
    X_std = X_train.std(axis=0)    # Standard deviation of each feature

    # Standardize X
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Compute statistics for y (targets)
    y_mean = y_train.mean()  # Mean of target
    y_std = y_train.std()    # Standard deviation of target

    # Standardize y
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    y_train = np.array([[y] for y in y_train])
    y_val = np.array([[y] for y in y_val])
    y_test = np.array([[y] for y in y_test])
    print(X_train.shape)
    print(y_val.shape)
    layers = (
        mlp.Layer(fan_in=7, fan_out=7, activation_function=mlp.Linear()),
        mlp.Layer(fan_in=7, fan_out=100, activation_function=mlp.Relu()),
        mlp.Layer(fan_in=100, fan_out=30, activation_function=mlp.Relu(), training_dropout=0.1),
        mlp.Layer(fan_in=30, fan_out=20, activation_function=mlp.Relu()),
        mlp.Layer(fan_in=20, fan_out=1, activation_function=mlp.Linear()),
    )

    multi_layer_perceptron = mlp.MultilayerPerceptron(layers)
    loss_function = mlp.SquaredError()
    train_losses, val_losses = multi_layer_perceptron.train(
        X_train.to_numpy(), 
        y_train, 
        X_val.to_numpy(), 
        y_val, 
        loss_function, 
        learning_rate=0.001,
        batch_size=32,
        epochs=3000)
    
    

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    plt.savefig('training_validation_losses_mpg.png')

    # Calculate total loss and average loss on the testing set
    y_test_pred = multi_layer_perceptron.forward(X_test.to_numpy())
    test_loss = loss_function.loss(y_test_pred, y_test)
    total_test_loss = np.sum(test_loss)
    average_test_loss = np.mean(test_loss)

    print(f"Total test loss: {total_test_loss}")
    print(f"Average test loss: {average_test_loss}")
    y_test_series = pd.Series(y_test.flatten(), index=X_test.index)

    # Select 10 random samples from the test set
    random_indices = np.random.choice(X_test.index, size=10, replace=False)
    X_test_samples = X_test.loc[random_indices]
    y_test_samples = y_test_series[random_indices]

    # Predict the MPG for the selected samples
    y_pred_samples = multi_layer_perceptron.forward(X_test_samples.to_numpy())

    # Create a DataFrame to display the results
    results = pd.DataFrame({
        'True MPG': y_test_samples.values.flatten(),
        'Predicted MPG': y_pred_samples.flatten()
    })

    print(results)