## Installing requirements.txt
```pip install -r requirements.txt```

## Evaluation:

```python main.py -t```

Note: You need to put the test.txt file in the folder Polynomial/

Note2: Since the predict() function (from the given starter code) accepts one expression at a time I havent been able to do batch processing during prediction, making it slower.

## Training the NN
```python train.py```

Note: This will train the model on data from train.txt

## Approach discussion
Open ```Approach_explanation.pdf``` to view a brief doc on my work for this assignment.

## Other details
1. Model weights and language details are stored in ```polynomial/weights```

2. ```network.txt``` contains the printed network architecture for the best performing model.

3. Extra model weights (for model 1, 2, 3 (described in Approach_explanation.pdf)) uploaded here just in case. Although a few other changes would need to be made to the code to run using these weights.