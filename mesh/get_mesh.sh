#!/bin/bash

FILE="5ggs_relaxed_wH_2a_surface"
echo ${FILE}
cat ${FILE}.wrl | sed -n -e '/Coordinate/,$p' | sed '/colorPer/q' > ${FILE}_test.csv
cat ${FILE}_test.csv | sed 's/\-1,//' | sed 's/,//' > ${FILE}_mesh.csv
#rm ${FILE}_test.csv

