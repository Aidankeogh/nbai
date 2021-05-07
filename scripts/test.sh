
set -e

start=`date +%s`

python src/data/game.py
python src/data/play.py
python src/data/possession.py
python src/data/batch_loader.py
python src/thought_path.py
python src/h5_db.py

end=`date +%s`
echo $((end-start))

echo "☆彡(ノ^ ^)ノ CONGRATULATIONS ALL TESTS PASS ヘ(^ ^ヘ)☆彡"