from autokeras.image.image_supervised import ImageClassifier, load_image_dataset

# we created ./all path where we copied all the images
train_path = '../data/all'
train_labels = '../data/labels_train.csv'

x_train, y_train = load_image_dataset(csv_file_path=train_labels, images_path=train_path)
#x_val, y_val = load_image_dataset(csv_file_path=validation_labels,images_path=validation_path)


clf = ImageClassifier(verbose=True)
# 4 hours search
clf.fit(x_train,y_train, time_limit = 4 * 60* 60)
best_model = clf.export_keras_model()
keras_model = best_model.produce_keras_model('asdf')
keras_model.summary()
# save it
keras_model.save('best.hdf5')

#clf.final_fit(x_train,y_train,x_val,y_val,retrain = True, trainer_args={'max_iter_num':10})
#print(clf.evaluate(x_val,y_val))
