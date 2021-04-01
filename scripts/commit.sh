
python -m src.h5_db

git add .

curr_branch=$(git rev-parse --abbrev-ref HEAD)
if [ curr_branch = "master" ]
then 
    git checkout -b $1
    git commit -m "$1"
    git push origin HEAD
else
    git commit -m "$curr_branch"
    git push origin HEAD
fi