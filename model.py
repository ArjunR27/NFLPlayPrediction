import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)

games1 = pd.read_csv('games.csv')
games2 = pd.read_csv('games25.csv')
games = pd.concat([games1, games2], ignore_index=True)

df1 = pd.read_csv('plays.csv')
data1 = df1[['gameId', 'quarter', 'down', 'yardsToGo', 'gameClock', 'absoluteYardlineNumber', 'possessionTeam', 'defensiveTeam', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation']]#, 'receiverAlignment'
result1 = df1['passResult']

df2 = pd.read_csv('plays25.csv')
data2 = df2[['gameId', 'quarter', 'down', 'yardsToGo', 'gameClock', 'absoluteYardlineNumber', 'possessionTeam', 'defensiveTeam', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation']]#, 'receiverAlignment'
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

"""print(X_test[10:20])
print(y_test[10:20])
print(y_pred[10:20])"""
