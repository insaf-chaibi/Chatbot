import pickle
import json
import nltk
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.optimizers.legacy import SGD
#from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential

wnl = WordNetLemmatizer()

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

words = []
classes = []
documents = []
ignore_words = ["?", "!", ",","."]
intents = json.loads(open("intents2.json").read())

for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # take each word and tokenize it
        world_list = nltk.word_tokenize(pattern)
        words.extend(world_list)

        # adding documents
        documents.append((world_list, intent["tag"]))

        # adding classes to our class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [wnl.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

#print(len(documents), "documents",documents)

#print(len(classes), "classes", classes)

#print(len(words), "unique lemmatized words", words)

pickle.dump(words, open("model/words.pkl", "wb"))
pickle.dump(classes, open("model/classes.pkl", "wb"))

# initializing training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # initializing bag of words
    bag = []

    # list of tokenized words for the pattern
    pattern_words = doc[0]

    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [wnl.lemmatize(word.lower()) for word in pattern_words]

    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print(train_x)
print(train_y)

#print("Training data created")

# actual training
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#Optional snippet - I have commented it as I did not use it.

# from keras import callbacks 
# earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
# callbacks =[earlystopping]

# fitting and saving the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("model/chatbot_model.h5", history)

print("Model created")