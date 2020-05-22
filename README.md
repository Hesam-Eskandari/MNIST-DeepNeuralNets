# MNIST-DeepNeuralNets
Handwritten Digit Recognition With MNIST Database - Deep Neural Network With Python 3.8

<h2> MNIST Database</h2>
<p>This database includes 60,000 training examples and 10,000 test samples with labels. Each instance is a 28 by 28 pixels grayscale image. Download the ".gz" files from the <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">MNIST</a> directory. For ".gz" files you need to download are:</p>
<ul>
  <li>Training Examples: "train-images-idx3-ubyte.gz"</li>
  <li>Training Labels: "train-labels-idx1-ubyte.gz"</li>
  <li>Testing Samples: "t10k-images-idx3-ubyte.gz"</li>
  <li>Testing Labels: "t10k-labels-idx1-ubyte.gz"</li>
</ul>

<h2> Python Libraries </h2>
<p>We tried to implemet Neural Network functions and not to use libraries functions such as TensorFlow in this project. Because TensorFlow would make it simle without fully understading the mathematical level. We used hte following libraries:</p>
<ul>
  <li>numpy as np: for matrix computations</li>
  <li>gzip: to open database files</li>
  <li>cv2: I used OpenCV to show pictures and labels and verify that right lable is used for each sample</li>
  <li>time: to measure processing times</li>
</ul>
<p>
To install a module you can execute the following code line in Spyder or Jupyter Notebook: 
</P>

```
!pip install opencv-python
!pip install gzip
```
<p>Or install it via anaconda command prompt:</p>

```
conda install opencv-python
```
<h2>Reading ".gz" Database Files</h2>
<p>I wrote these lines of code to read training examples and labels database files:</p>

```
import gzip
trainingData = 'train-images-idx3-ubyte.gz'
trainingLabels = 'train-labels-idx1-ubyte.gz'
trD = gzip.open(trainingData,'r')
trL = gzip.open(trainingLabels,'r')
trD.read(16)
trL.read(8)
# the code to process data here
trD.close()
trL.close()
```

 
