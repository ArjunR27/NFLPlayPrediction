import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)

games = pd.read_csv('games.csv')

df = pd.read_csv('plays.csv')
data = df[['quarter', 'down', 'yardsToGo', 'possessionTeam', 'defensiveTeam', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation']]
result = df['passResult'].tolist()

times = []
yards = []
possessionScore = []
defenceScore = []
for i in range(len(df)):
    if df['quarter'][i] == 1 or df['quarter'][i] == 3:
        additional = 15 * 60
    else:
        additional = 0
    time = df['gameClock'][i].split(':')
    times.append(additional + int(time[0]) * 60 + int(time[1]))

    yard = df['yardlineNumber'][i]
    if df['yardlineSide'][i] == df['possessionTeam'][i]:
        yard = 100 - yard
    yards.append(yard)

    """gameid = df['gameId'][i]
    pos = df['possessionTeam'][i]
    defence = df['defensiveTeam'][i]
    for index, row in games.iterrows():
        if row['gameId'] == gameid:
            if pos == row['homeTeamAbbr']:
                possessionScore.append(df['preSnapHomeScore'][i])
                defenceScore.append(df['preSnapVisitorScore'][i])
            else:
                possessionScore.append(df['preSnapVisitorScore'][i])
                defenceScore.append(df['preSnapHomeScore'][i])
        break"""
data['timeLeftHalf'] = times
data['yardsToGoal'] = yards
#data['possessionScore'] = possessionScore
#data['defenceScore'] = defenceScore

label_encoder = LabelEncoder()
data['possessionTeam'] = label_encoder.fit_transform(data['possessionTeam'])
data['defensiveTeam'] = label_encoder.fit_transform(data['defensiveTeam'])
data['offenseFormation'] = label_encoder.fit_transform(data['offenseFormation'])

target = [0 if type(i) == type("") else 1 for i in result]

print(data[:10])
print(result[:10])
print(target[:10])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
