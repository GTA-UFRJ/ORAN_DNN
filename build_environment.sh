#!/bin/bash

mkdir oran_dataset
mkdir models
mkdir results

wget -bqc https://repository.library.northeastern.edu/downloads/neu:bz61jz54w?datastream_id=content
wget -bqc https://repository.library.northeastern.edu/downloads/neu:bz61jz84m?datastream_id=content
wget -bqc https://repository.library.northeastern.edu/downloads/neu:bz61k158k?datastream_id=content

unzip 'neu:bz61jz54w?datastream_id=content'
unzip 'neu:bz61jz84m?datastream_id=content'
unzip 'neu:bz61k158k?datastream_id=content'

mv ./CLEAR.bin ./oran_dataset/
mv ./LTE_1M.bin ./oran_dataset/
mv ./WIFI_1M.bin ./oran_dataset/

rm 'neu:bz61jz54w?datastream_id=content'
rm 'neu:bz61jz84m?datastream_id=content'
rm 'neu:bz61k158k?datastream_id=content'
