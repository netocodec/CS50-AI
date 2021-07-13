# Traffic Program Report

## This is my report about this project.

### Python version: 3.8.5
### Report Written by: Netocodec


First thing i did it was installing all the required dependencies. Checking if all dependencies are compatible with my python version.


On Tensorflow it gave me an error "core dump", so i started search on the internet about this error and i found the Tensorflow is not compatible with old CPU but it is compatible with my raspberry pi 2 with python 3.7 which is for me perfect.

So i work with raspberries pi's a lot and i know i can work with that using my raspberry pi, i knew the dependency of the Tensorflow is working on ARM architecture. I installed all the dependecies using the "requeriments.txt" file and the result was SUCCESS, so i started to code your exercise with all the structed rules.

Now i uploaded my python code into the raspberry and i try to run it so test if it runs. I get an error about numpy version is too old so i resolved this by upgrading everything in the raspberry and then run the command "pip3 install --upgrade numpy", after all the changes i try to run again and it runs perfect on my raspberry. So let's do some code!

My first objective it was put my python program running with the first code, without any otimizations.

After searching on Tensorflow documentation i started with some layers and try to run it.

First run with the following layers:

#### Model: "sequential"

|Layer (type) |Output Shape |Param #|
|--- | --- | --- |
conv2d (Conv2D)|(None, 28, 28, 16)|448|
max_pooling2d (MaxPooling2D)|(None, 14, 14, 16)|0|
flatten (Flatten)|(None, 3136)|0|
dense (Dense)|(None, 64)|200768|

---
#### Total params: 201,216
#### Trainable params: 201,216
#### Non-trainable params: 0

---

What went wrong on the first run of my neural network? It gave me an error while learning on the first Epoch.

Error message: tensorflow.python.framework.errors_impl.InvalidArgumentError: Can not squeeze dim[1], expected a dimension of 1, got 3

So i got to resolve this error.
After a search on the internet i found that i need to change the "loss" parameter on function "compile", so i change to CategorialCrossentropy.


After a few changes i run the program again.
#### Model: "sequential"

|Layer (type) |Output Shape |Param #|
|--- | --- | --- |
conv2d (Conv2D)|(None, 28, 28, 16)|448|
max_pooling2d (MaxPooling2D)|(None, 14, 14, 16)|0|
flatten (Flatten)|(None, 3136)|0|
dense (Dense)|(None, 64)|200768|
dense_1 (Dense)|(None, 43)|2795|

---
#### Total params: 204,011
#### Trainable params: 204,011
#### Non-trainable params: 0

---

```
Train on 15984 samples

Epoch 1/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 10.0033 - acc: 0.5830
Epoch 2/10
15984/15984 [==============================] - 88s 6ms/sample - loss: 0.8285 - acc: 0.8305
Epoch 3/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.5261 - acc: 0.8900
Epoch 4/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.4247 - acc: 0.9105
Epoch 5/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.3666 - acc: 0.9282
Epoch 6/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.3121 - acc: 0.9383
Epoch 7/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.3142 - acc: 0.9394
Epoch 8/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.2873 - acc: 0.9457
Epoch 9/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.2439 - acc: 0.9539
Epoch 10/10
15984/15984 [==============================] - 89s 6ms/sample - loss: 0.3248 - acc: 0.9419
```

### Result
10656/10656 - 24s - loss: 0.7517 - acc: 0.9201

So it work well but i was not very happy with the result.
Even the program run with sucess i wanted more than 92% of accuracy.
So i went to the tensorflow documentation and i start to change some parameters on my program. 

My goal now is to optimize my AI.

Testing with the same layers but with different parameters.

So i got 88% percent with this layers and parameters.

#### Model: "sequential"

|Layer (type) |Output Shape |Param #|
|--- | --- | --- |
conv2d (Conv2D)|(None, 28, 28, 32)|896|
max_pooling2d (MaxPooling2D)|(None, 14, 14, 32)|0|
flatten (Flatten)|(None, 6272)|0|
dense (Dense)|(None, 64)|200736|
dropout (Dropout)|(None, 32)|0|
dense_1 (Dense)|(None, 43)|1419|

---


So my solution it was go back it the last parameters and try the Conv2D and MaxPooling2D layers with interactions to see if i get better results.
I added on this time the "Dropout" layer to check if the results can be better or not.

### Result of the next test

```
10656/10656 - 31s - loss: 0.5253 - acc: 0.8838
```

---

### Results after doing some changes and testings

After some tries with different layers and values i found the optimized solution for my program.
What went wrong with the other tries i made? I was put to many layers to see the result and it got worst,
so i return to the first configuration i made on the past tests and i try to adjust the layers with the weights and the dropout rate too.

I've notice that if you put more layers and nodes into you program the more and heavier it becomes to you to learn.
So for me the optmized solution it was to keep it simple and stupid (KISS concept).

There it is my otmizied solution data:

```
Train on 15984 samples
Epoch 1/10
15984/15984 [==============================] - 115s 7ms/sample - loss: 0.0674 - acc: 0.9822
Epoch 2/10
15984/15984 [==============================] - 115s 7ms/sample - loss: 0.0291 - acc: 0.9902
Epoch 3/10
15984/15984 [==============================] - 115s 7ms/sample - loss: 0.0210 - acc: 0.9930
Epoch 4/10
15984/15984 [==============================] - 116s 7ms/sample - loss: 0.0163 - acc: 0.9945
Epoch 5/10
15984/15984 [==============================] - 114s 7ms/sample - loss: 0.0133 - acc: 0.9956
Epoch 6/10
15984/15984 [==============================] - 114s 7ms/sample - loss: 0.0125 - acc: 0.9960
Epoch 7/10
15984/15984 [==============================] - 114s 7ms/sample - loss: 0.0104 - acc: 0.9967
Epoch 8/10
15984/15984 [==============================] - 114s 7ms/sample - loss: 0.0092 - acc: 0.9971
Epoch 9/10
15984/15984 [==============================] - 114s 7ms/sample - loss: 0.0091 - acc: 0.9972
Epoch 10/10
15984/15984 [==============================] - 115s 7ms/sample - loss: 0.0084 - acc: 0.9974
10656/10656 - 28s - loss: 0.0128 - acc: 0.9967
```


## Conclusion

Try to invest some time on this project, it is very good for you to understand the neural networks. 
I will leave here the links of the documentation that i search and it helped me a lot.
And i recomend a book that helps me to understand better the AI and neural networks very much.

The name of the book is "Machine Learning with Python Cookbook Practical Solutions from Preprocessing to Deep Learning" by Chris Albon, it is a very nice book that explains with code the algoritms, explains to you every step of their examples and this book helped me a lot at this course.

### Youtube Video

You can access to the video bellow.

[Click Here to access to the video](https://youtu.be/rr-OEg0blBA)

| Documentation Links |
| --- |
| [CNN Model](https://www.tensorflow.org/tutorials/images/cnn#add_dense_layers_on_top)|
| [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)|
| [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)|
| [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten)|
| [Dense Layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)|
| [Models](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile)|
| [Loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss)|
| [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)|
| Recomended Book: Machine Learning with Python Cookbook Practical Solutions from Preprocessing to Deep Learning by Chris Albon |

---
Created by: Netocodec
Edited on date: 21/04/2021


