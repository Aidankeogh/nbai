
set -e

start=`date +%s`

python src/data/game.py
python src/data/play.py
python src/data/possession.py
python src/data/batch_loader.py
python src/thought_path.py

end=`date +%s`

echo "time elapsed: $((end-start))s"
echo "☆彡(ノ^ ^)ノ CONGRATULATIONS ALL TESTS PASSED ヘ(^ ^ヘ)☆彡"
echo "ALL TESTS PASSED ON $(date) IN $((end-start)) SECONDS" > test_recipt.txt