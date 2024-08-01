import timeit
import tensorflow as tf

# Callbacks
early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=8,
    restore_best_weights=True
)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, min_delta=0.01,
    min_lr=1e-10, patience=4, mode='auto'
)

# Model Definition
model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    input_shape=(img_shape),
    include_top=False,
    weights="imagenet",
)

for layer in model.layers:
    layer.trainable = True

model_ft = tf.keras.models.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation="softmax"),
])

model_ft.summary()

model_ft.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# Training
start_ft = timeit.default_timer()

history = model_ft.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early, plateau],
    validation_data=valid_generator,
    class_weight=train_class_weights,
    verbose=1,
)

stop_ft = timeit.default_timer()

execution_time_ft = (stop_ft - start_ft) / 60
print(f"Model fine tuning executed in {execution_time_ft:.2f} minutes")

model_ft.save(save_model_ft)
