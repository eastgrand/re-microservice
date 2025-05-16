#!/bin/bash
# Deploy the worker fix to Render.com

echo "Deploying worker fix to Render.com..."

# Ensure the simple_worker.py script is executable
chmod +x ./simple_worker.py

# Create a commit with the changes
git add render.yaml simple_worker.py
git commit -m "Fix: Switch worker to simple_worker.py to avoid Connection import error"

# Push to the repository, assuming the branch is main or master
# Change the branch name if yours is different
git push origin main

echo "Changes pushed to GitHub. Render will automatically deploy the changes."
echo "Wait for a few minutes and then check the worker logs on Render dashboard."
echo "Done!"
