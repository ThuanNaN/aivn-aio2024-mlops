awslocal s3api create-bucket --bucket aivn-mlops
awslocal s3api list-buckets
awslocal s3 sync DATA/ s3://aivn-mlops