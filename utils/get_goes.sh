for year in `seq 2014 2015`; do
    for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
        for day in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
            # goes_file="https://satdat.ngdc.noaa.gov/sem/goes/data/full/${year}/${month}/goes14/csv/g14_xrs_2s_${year}${month}${day}_${year}${month}${day}.csv"
            # # save the file to ../data/noaa/
            # wget $goes_file  -P data/noaa/
            # # if 404 error, then download the gr15

            # if [ $? -eq 4 ]; then
            goes_file="https://satdat.ngdc.noaa.gov/sem/goes/data/full/${year}/${month}/goes15/csv/gr15_xrs_2s_${year}${month}${day}_${year}${month}${day}.csv"
            wget $goes_file  -P data/noaa/
            # fi
        done
    done
done
# https://satdat.ngdc.noaa.gov/sem/goes/data/full/2014/01/goes15/csv/