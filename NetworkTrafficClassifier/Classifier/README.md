To retrain the classifier model, it is necessary to follow the guidence of the original authors in [repository](https://github.com/RadionBik/ML-based-network-traffic-classifier/tree/master/gpt_model) to download the pretrained models and datasets to the folder *traffic-classifier*. Follow the instructions below.

### Pre-trained models and datasets: 

It is necessary to download a MinIO client to your computer as per: https://docs.min.io/docs/minio-client-quickstart-guide.html

To get the data, execute the following commands:

```
./mc alias set ext-anon http://195.201.38.68:9000
./mc ls ext-anon/traffic-classifier
./mc cp ext-anon/traffic-classifier .
```

where the first command will prompt you for user credentials:

```
Access Key: gpt_research
Secret Key: mbmug8VDbRu5hqJ
```


*Note: opening the URL in a browser leads to the administrator console. To access the datasets and models you have to install MinIO client as mentioned above.*
