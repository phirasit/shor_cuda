#!/bin/bash

CMD="time ./shor gpu"

OUTPUT_FILE="benchmark.txt"

echo "" > $OUTPUT_FILE

for N in 15 77 129 413 1207; do
	echo "============================" >> $OUTPUT_FILE
	echo "The number to factorize $N" 	>> $OUTPUT_FILE
	echo "============================" >> $OUTPUT_FILE
	echo "The number to factorize $N"
	for SM in {1..15}; do
		CORES=$(( SM * 128 ))
		echo "====================================" >> $OUTPUT_FILE
		echo "The program run on $CORES cuda cores" >> $OUTPUT_FILE
		echo "====================================" >> $OUTPUT_FILE
		echo "The program run on $CORES cuda cores"
		$CMD $N $SM |& tee -a $OUTPUT_FILE
	done
done

#           || visible in terminal ||   visible in file   || existing
#   Syntax  ||  StdOut  |  StdErr  ||  StdOut  |  StdErr  ||   file   
# ==========++==========+==========++==========+==========++===========
#     >     ||    no    |   yes    ||   yes    |    no    || overwrite
#     >>    ||    no    |   yes    ||   yes    |    no    ||  append
#           ||          |          ||          |          ||
#    2>     ||   yes    |    no    ||    no    |   yes    || overwrite
#    2>>    ||   yes    |    no    ||    no    |   yes    ||  append
#           ||          |          ||          |          ||
#    &>     ||    no    |    no    ||   yes    |   yes    || overwrite
#    &>>    ||    no    |    no    ||   yes    |   yes    ||  append
#           ||          |          ||          |          ||
#  | tee    ||   yes    |   yes    ||   yes    |    no    || overwrite
#  | tee -a ||   yes    |   yes    ||   yes    |    no    ||  append
#           ||          |          ||          |          ||
#  n.e. (*) ||   yes    |   yes    ||    no    |   yes    || overwrite
#  n.e. (*) ||   yes    |   yes    ||    no    |   yes    ||  append
#           ||          |          ||          |          ||
# |& tee    ||   yes    |   yes    ||   yes    |   yes    || overwrite
# |& tee -a ||   yes    |   yes    ||   yes    |   yes    ||  append