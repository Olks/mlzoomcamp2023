# mlzoomcamp2023 Homework 5
## Question 5

$docker pull svizor/zoomcamp-model:3.10.12-slim
$docker images


## Question 6

$docker build -t hm5 .
$docker run -it --rm -p 9696:9696 hm5

#### In another terminal
$python3 predict-test-docker.py

