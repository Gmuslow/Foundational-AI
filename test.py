import requests
# Send results
predictions = [100]
data = {
    "name": "Dr. McSwagger",
    "predictions": predictions
}
response = requests.post("https://csc7700leaderboard-d4fce9d9h2b5h8ab.centralus-01.azurewebsites.net/submit", json=data)
print(response.json())