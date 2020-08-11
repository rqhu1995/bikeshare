#!/bin/bash
i=0
times=20
while [ $i -le $times ]
do
let 'i++'
/Users/hurunqiu/anaconda3/envs/bikeshare/bin/python /Users/hurunqiu/project/bikeshare/genetic_algorithm/schedule.py
done
