from github import Github

# Replace 'your_token' with your personal access token
g = Github("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")

# Get the authenticated user
user = g.get_user()

# Create a new repository
repo = user.create_repo("FraudDetection")

print(f"Repository created at {repo.clone_url}")