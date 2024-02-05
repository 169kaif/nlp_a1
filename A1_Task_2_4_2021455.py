emotions_list = ['sadness', 'joy', 'love', 'angry', 'fear', 'surprise']
os.makedirs('Sentence_samples', exist_ok=True)

for emotion in emotions_list:
    with open(os.path.join('Sentence_samples', f'gen_{emotion}.txt'), 'w') as sample_file:
        for i in range(50):
            sent, label, score = bgram.get_emotion_score(emotion=emotion)
            sample_file.write(f"{label} - {score}\n{sent}\n\n")
print("Samples Generated.")

#Load Original Corpus
with open('corpus.txt', 'r') as corpus_file:
    corpus = corpus_file.readlines()

#Generated Samples
generated_samples = []
labels = []
for emotion in emotions_list:
    with open(os.path.join('Sentence_samples', f'gen_{emotion}.txt'), 'r') as sample_file:
        samples = sample_file.readlines()
        generated_samples.extend(samples)
        labels.extend([emotion] * len(samples))

#Vectorizing the Text Data
vectorized = TfidfVectorizer()
X_original = vectorized.fit_transform(corpus)
X_generated = vectorized.transform(generated_samples)
# print(X_generated)

# Data splitting and SVC training
X_train, X_val, y_train, y_val = train_test_split(X_generated, labels,test_size=0.2)
svc = SVC()

# Grid-CV
parameters = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'],'degree':[1,2,3,4]}
grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Model Evaluation
best_svc = grid_search.best_estimator_
y_pred = best_svc.predict(X_generated)
print(classification_report(labels, y_pred))