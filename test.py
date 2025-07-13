import requests

resp = requests.get("http://localhost:8080/joke", params={"tokens": 200})
print(resp.json().get("joke", "No joke generated."))  # Adjusted to match the expected response structure
