import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import json

class DataGenerator(Sequence):
    def __init__(self, file_path, batch_size, label_encoder, shuffle=True):
        self.file_path = file_path
        self.batch_size = batch_size
        self.label_encoder = label_encoder
        self.shuffle = shuffle
        self.df_chunks = pd.read_csv(self.file_path, chunksize=batch_size)
        self.num_chunks = self.get_num_chunks()
        self.chunk_index = 0
        self.df_iterator = iter(self.df_chunks)  # Initialize DataFrame iterator
        self.reset_df_iterator()

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index):
        while True:
            try:
                batch = next(self.df_iterator)
                X, y = self.__data_generation(batch)
                return X, y
            except StopIteration:
                self.reset_df_iterator()

    def reset_df_iterator(self):
        self.df_chunks = pd.read_csv(self.file_path, chunksize=self.batch_size)
        if self.shuffle:
            self.df_chunks = [chunk.sample(frac=1).reset_index(drop=True) for chunk in self.df_chunks]
        self.df_iterator = iter(self.df_chunks)

    def get_num_chunks(self):
        num_rows = sum(1 for _ in open(self.file_path)) - 1  # Excluding header row
        return int(np.ceil(num_rows / self.batch_size))

    def on_epoch_end(self):
        self.chunk_index = 0  # Reset chunk index at the end of each epoch

    def __data_generation(self, batch):
        X = np.array([json.loads(x) for x in batch['Features']])
        X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape to add channels dimension
        y = self.label_encoder.transform(batch['Label'].values)
        # Remove one-hot encoding and convert to class labels
        y = np.argmax(to_categorical(y, num_classes=len(self.label_encoder.classes_)), axis=1)
        return X, y


    def get_labels(self):
        labels = []
        for chunk in self.df_chunks:
            labels.extend(chunk['Label'].values)
        return labels
