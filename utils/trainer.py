import tensorflow as tf
from .preprocessing import parse_tfRecords
from tqdm import tqdm



def Trainer(train, test, model, epoch=5, ts=0, tts=0):
    """
        The Trainer to train the @model
        Inside this fn, you can modify the `loss object`, `optimizer` to different one.
    """
    l = ts + tts
    l = 10000 if l == 0 else l
    # Train dataset
    train_ds = parse_tfRecords(train).filter(lambda x: x['height'] == x['width'])
    train_ds = train_ds.shuffle(l).batch(32)

    # Test dataset
    test_ds = parse_tfRecords(test).filter(lambda x: x['height'] == x['width'])

    # Loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Optimizer
    optimizer = tf.keras.optimizers.Adam()



    train_loss = tf.keras.metrics.Mean(name='Train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='Train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='Test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='Test_accuracy')


    # The Train function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)


    # The Test function
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        test_loss(loss)
        test_accuracy(labels, predictions)

    pbar = tqdm(range(epoch))
    for epo in pbar:

        train_loss.reset_states()
        train_accuracy.reset_states()

        # progress = tqdm(total=l/32)
        for record in train_ds:
            if record['img_raw'].shape[0] == 1:
                imgs = tf.expand_dims(tf.io.decode_image(record['img_raw'].numpy(), channels=1), axis=0)
                imgs = tf.cast(imgs, tf.float16)
                labels = tf.expand_dims(record['labels'], axis=0)
                print(imgs.shape)
            else:
                imgs = tf.map_fn(fn=lambda t: tf.io.decode_image(t.numpy(), channels=1), elems=record['img_raw'], fn_output_signature=tf.uint8)
                imgs = tf.map_fn(fn=lambda t: tf.cast(t, tf.float16), elems=imgs, fn_output_signature=tf.float16)
                labels = record['label']

            train_step(imgs, labels)
            # progress.update(1)

        for record in test_ds:
            imgs = tf.io.decode_image(record['img_raw'].numpy(), channels=1)
            labels = record['label']
            if len(imgs.shape) <= 3:
                imgs = tf.expand_dims(imgs, axis=0)
                labels = tf.expand_dims(labels, axis=0)
            imgs = tf.cast(imgs, dtype=tf.float16)
            test_step(imgs, labels)

        pbar.set_postfix({
            "train_loss" : f"{train_loss.result():3f}",
            "train_acc" : f"{train_accuracy.result():3f}",
            "test_loss" : f"{test_loss.result():3f}",
            "test_acc" : f"{test_accuracy.result():3f}",
        })
        # print("="*60)
        # print(f'Epoch : {epo}\n train_loss: {train_loss.result()},\n train_accuracy : {train_accuracy.result()}')
        # print(f'test_loss : {test_loss.result()}, \n test_accuracy : {test_accuracy.result()}')



























