
rm -r result/*
rm -r 结果/*

python main.py   #

rm -r result_final/bagging/*

python get_final_result.py

cp -r result_final/bagging result

python 2_statistics.py

python 3_getFinalReport.py

python 4_result.py


#python statistic_analysis.py
#python t-SNE.py
