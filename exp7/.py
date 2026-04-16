
# ===============================

# 1. Import Libraries

# ===============================

import numpy as np import pandas as pd import tensorflow as tf


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler



# ===============================
 
# 2. Load Dataset

# ===============================

df = pd.read_csv('/content/Churn_Modelling.csv')



# Display dataset print(df.head())


# ===============================

# 3. Data Preprocessing

# ===============================



# Drop unnecessary columns

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)



# Encode categorical variables label_encoder = LabelEncoder()


df['Gender'] = label_encoder.fit_transform(df['Gender'])



# One-hot encoding for Geography

df = pd.get_dummies(df, columns=['Geography'], drop_first=True)



# Split features and target
 
X = df.drop('Exited', axis=1) y = df['Exited']


# ===============================

# 4. Train-Test Split

# ===============================

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42
)



# ===============================

# 5. Feature Scaling

# ===============================

sc = StandardScaler()



X_train = sc.fit_transform(X_train) X_test = sc.transform(X_test)
# ===============================

# 6. Build ANN Model

# ===============================

model = tf.keras.models.Sequential()



# Input + Hidden Layers
 
model.add(tf.keras.layers.Dense(units=16, activation='relu', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dense(units=8, activation='relu'))



# Output Layer

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # ===============================
# 7. Compile Model

# ===============================

model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
)

# ===============================

# 8. Train Model

# ===============================

history = model.fit( X_train, y_train, batch_size=32, epochs=50, validation_split=0.2
)

# ===============================
 
# 9. Evaluate Model

# ===============================

loss, accuracy = model.evaluate(X_test, y_test) print("Test Accuracy:", accuracy)


# ===============================

# 10. Predictions

# ===============================

y_pred = model.predict(X_test) y_pred = (y_pred > 0.5)


# ===============================

# 11. Confusion Matrix

# ===============================

from sklearn.metrics import confusion_matrix cm = confusion_matrix(y_test, y_pred) print("Confusion Matrix:\n", cm)
plt.plot(history.history['accuracy'], label='Training Accuracy') plt.plot(history.history['val_accuracy'], label='Validation Accuracy') plt.legend()
plt.title("Model Accuracy") plt.show()
