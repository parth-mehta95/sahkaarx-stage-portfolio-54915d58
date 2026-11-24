# Git Branching & Pull Request Automation Workflow
# Step 1: Create feature branch
git checkout -b feature/eda-feature-engineering

# Step 2: Add all week 02 artifacts
git add practice/week-02-practice-branching-and-pull-requests/

# Step 3: Commit to feature branch
git commit -m "feat(pipeline): implement customer churn EDA and advanced feature engineering pipeline"

# Step 4: Switch back to main
git checkout main

# Step 5: Merge feature branch into main with non-fast-forward PR merge commit
git merge --no-ff feature/eda-feature-engineering -m "Merge pull request #2 from feature/eda-feature-engineering - Customer Churn EDA and Feature Engineering Pipeline"
