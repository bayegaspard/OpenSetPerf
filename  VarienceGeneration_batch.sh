#Runs the model
python src/main/main.py --LOOP 0 --MaxPerClass 200
python src/main/main.py --LOOP 0 --attemptLoadModel 1 --testlength 1 --Mix\ unknowns\ and\ validation 0 --num_epochs 0 --ItemLogitData 1 --SaveBatchData 1 --MaxPerClass 200 --batch_size 5
python src/main/main.py --LOOP 0 --attemptLoadModel 1 --testlength 1 --Mix\ unknowns\ and\ validation 0 --num_epochs 0 --ItemLogitData 1 --SaveBatchData 1 --MaxPerClass 200 --batch_size 10
python src/main/main.py --LOOP 0 --attemptLoadModel 1 --testlength 1 --Mix\ unknowns\ and\ validation 0 --num_epochs 0 --ItemLogitData 1 --SaveBatchData 1 --MaxPerClass 200 --batch_size 50
python src/main/main.py --LOOP 0 --attemptLoadModel 1 --testlength 1 --Mix\ unknowns\ and\ validation 0 --num_epochs 0 --ItemLogitData 1 --SaveBatchData 1 --MaxPerClass 200 --batch_size 100
python src/main/main.py --LOOP 0 --attemptLoadModel 1 --testlength 1 --Mix\ unknowns\ and\ validation 0 --num_epochs 0 --ItemLogitData 1 --SaveBatchData 1 --MaxPerClass 200 --batch_size 1000