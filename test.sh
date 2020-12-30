#/bin/bash
countBlankLines=$(grep '^$' ../data/abs.txt|wc -l)
if [ $countBlankLines -eq 0 ]
    then
	echo 'No empty lines'
    else
	echo $countBlankLines
fi
