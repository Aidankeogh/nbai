
set -e
python -m src.h5_db
git add .

curr_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ curr_branch == "master" ]]
then 
    echo "$curr_branch"
    git checkout -b $1
    git commit -m "$1"
    curr_branch=$1
    git push -u origin HEAD
else
    git commit -m "$curr_branch"
    git push -u origin HEAD
fi

echo "https://github.com/Aidankeogh/nbai/compare/$curr_branch...master"