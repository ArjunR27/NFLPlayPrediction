import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

games1 = pd.read_csv('../data/raw/games2024.csv')
games2 = pd.read_csv('../data/raw/games2025.csv')
games = pd.concat([games1, games2], ignore_index=True)

df1 = pd.read_csv('../data/raw/plays2024.csv')
data1 = df1[['gameId', 'quarter', 'down', 'yardsToGo', 'gameClock', 'absoluteYardlineNumber', 'possessionTeam', 'defensiveTeam', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability']]#, 'receiverAlignment'
result1 = df1['passResult']

df2 = pd.read_csv('../data/raw/plays2025.csv')
data2 = df2[['gameId', 'quarter', 'down', 'yardsToGo', 'gameClock', 'absoluteYardlineNumber', 'possessionTeam', 'defensiveTeam', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability']]#, 'receiverAlignment'
result2 = df2['passResult']

data = pd.concat([data1, data2], ignore_index=True)
result = pd.concat([result1, result2], ignore_index=True).tolist()

times = []
#yards = []
possessionScore = []
defenceScore = []
pointDiff = []
for i in range(len(data)):
    if data['quarter'][i] == 1 or data['quarter'][i] == 3:
        additional = 15 * 60
    else:
        additional = 0
    #print(df['gameClock'][i])
    time = data['gameClock'][i].split(':')
    times.append(additional + int(time[0]) * 60 + int(time[1]))

    #yard = df['yardlineNumber'][i]
    #if df['yardlineSide'][i] == df['possessionTeam'][i]:
    #    yard = 100 - int(yard)
    #yards.append(yard)

    gameid = data['gameId'][i]
    pos = data['possessionTeam'][i]
    defence = data['defensiveTeam'][i]
    row = games[games['gameId'] == gameid]
    if pos == row.iloc[0]['homeTeamAbbr']:
        possessionScore.append(data['preSnapHomeScore'][i])
        defenceScore.append(data['preSnapVisitorScore'][i])
        pointDiff.append(data['preSnapHomeScore'][i] - data['preSnapVisitorScore'][i])
    else:
        possessionScore.append(data['preSnapVisitorScore'][i])
        defenceScore.append(data['preSnapHomeScore'][i])
        pointDiff.append(data['preSnapVisitorScore'][i] - data['preSnapHomeScore'][i])

data['timeLeftHalf'] = times
#data['yardsToGoal'] = yards
#data['possessionScore'] = possessionScore
#data['defenceScore'] = defenceScore
data['pointDifferential'] = pointDiff

data = data.drop(['gameClock', 'gameId', 'preSnapHomeScore', 'preSnapVisitorScore'], axis=1)
label_encoder = LabelEncoder()
data['possessionTeam'] = label_encoder.fit_transform(data['possessionTeam'])
data['defensiveTeam'] = label_encoder.fit_transform(data['defensiveTeam'])
data['offenseFormation'] = label_encoder.fit_transform(data['offenseFormation'])
#data['receiverAlignment'] = label_encoder.fit_transform(data['receiverAlignment'])

# 0 for pass 1 for run
target = [0 if type(i) == type("") else 1 for i in result]

print(data[:10])
print(result[:10])
print(target[:10])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Logistic Regression (Baseline)
# Best parameters found:  {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}

logr = LogisticRegression(max_iter=1000000, C=0.1, penalty='l1', solver='liblinear')
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
cm = confusion_matrix(y_pred, y_test)
print(f"Logistic Regression Accuracy: {accuracy}")
print(cm)

plt.figure(figsize=(10, 6))
plt.scatter(X_test['offenseFormation'], X_test['timeLeftHalf'], c=y_pred, cmap=plt.cm.Paired, edgecolor='k', s=50)
plt.title('Logistic Regression Predictions based on offenseFormation and timeLeftHalf')
plt.xlabel('offenseFormation')
plt.ylabel('timeLeftHalf')
plt.show()
"""param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers for optimization
    'penalty': ['l1', 'l2']  # Type of regularization
}

grid_search = GridSearchCV(estimator=logr, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=1, n_jobs=-1)


grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)

best_logr = grid_search.best_estimator_
y_pred = best_logr.predict(X_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))"""



# Random Forest
rf = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42) # Best model: 89.0% n_estimators=150, max_depth=25, random_state=42
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred))]

print("\nTotal distribution of downs:")
downs = {1: 0, 2: 0, 3: 0, 4: 0}
for idx in indices:
    #print(f"Index: {idx}, True label: {y_test[idx]}, Predicted label: {y_pred[idx]}")
    #print(f"Features: {X_test.iloc[idx].to_dict()}")
    downs[X_test.iloc[idx].to_dict()['down']] += 1

print(downs)
print(f"1st: {downs[1]/len(indices)}, 2nd: {downs[2]/len(indices)}, 3rd: {downs[3]/len(indices)}, 4th: {downs[4]/len(indices)}")

misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]

print("\nIncorrectly classified downs:")
mis_downs = {1: 0, 2: 0, 3: 0, 4: 0}
for idx in misclassified_indices:
    #print(f"Index: {idx}, True label: {y_test[idx]}, Predicted label: {y_pred[idx]}")
    #print(f"Features: {X_test.iloc[idx].to_dict()}")
    mis_downs[X_test.iloc[idx].to_dict()['down']] += 1

print(mis_downs)
print(f"1st: {mis_downs[1]/len(misclassified_indices)}, 2nd: {mis_downs[2]/len(misclassified_indices)}, 3rd: {mis_downs[3]/len(misclassified_indices)}, 4th: {mis_downs[4]/len(misclassified_indices)}")
print(f"Accuracy by down:\n1st: {(downs[1] - mis_downs[1])/downs[1]}, 2nd: {(downs[2] - mis_downs[2])/downs[2]}, 3rd: {(downs[3] - mis_downs[3])/downs[3]}, 4th: {(downs[4] - mis_downs[4])/downs[4]}")

print(f"Features: {X_train.columns}")
print(f"Importances: {rf.feature_importances_}")


"""print(X_test[10:20])
print(y_test[10:20])
print(y_pred[10:20])"""

# K-Fold Cross Validation

k_folds = KFold(n_splits=10)

scores = cross_val_score(rf, X_train, y_train, cv=k_folds)

print(f"CV Scores: {scores}")