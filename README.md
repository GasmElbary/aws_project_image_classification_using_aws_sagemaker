# Image Classification using AWS SageMaker

In this notebook, I fine-tuned the pre-trained model EfficientNet-B7 to use on the Intel Image Classification dataset. First, the tunning is run on the SageMaker training jobs to utilize the ml.g4dn.xlarge instance type. Meanwhile, I used the model profiler and debugger to analyze the generated model. Then I deploy the model and inference random image sets.

## Project Set Up and Installation
AWS Account


## Dataset
The data set I'm using is the Intel Image Classification dataset available on (https://www.kaggle.com/datasets/puneet6060/intel-image-classification). This data set consists of images of natural scenes around the world under 6 categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

There are a total of 25k images of size 150x150, distributed between Train, Test, and Prediction files. where, 14k images are in Train, 3k in Test, and 7k in Prediction.

The dataset file structure is like this

```
.   
└───data
    └───seg_train
    │   │   
    │   └───buildings
    │   └───forest
    │   └───glacier
    │   └───mountain
    │   └───sea
    │   └───street
    │
    └───seg_test
    │   │   
    │   └───buildings
    │   └───forest
    │   └───glacier
    │   └───mountain
    │   └───sea
    │   └───street
    │
    └───seg_pred

```
Each category is in a separate file.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.



https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-debugger/pytorch_model_debugging/pytorch_script_change_smdebug.ipynb
https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#load-a-model
https://aws.amazon.com/blogs/machine-learning/create-amazon-sagemaker-models-using-the-pytorch-model-zoo/#:~:text=The%20entry_point%20parameter%20points%20to,SageMaker%20Deep%20Learning%20Container%20image
