#Written by Hesam Eskandari
import gzip
import numpy as np
import cv2
from time import time

class NeuralNets:
    def __init__(self,hLayers,trainData,trainLabels,testData,testLabels,funcType):
        self.hiddenLayers = hLayers #define hidden layers with their sizes
        self.trainData = trainData #string file name
        self.trainLabels = trainLabels #string file name
        self.testData = testData #string file name
        self.testLabels = testLabels #string file name
        self.inputSize = 28 #length of each dimensions of image examples
        self.outputSize = 10 #number of output perceptrons
        self.w = [] #weights in layers
        self.a = [] #neurons in layers
        self.layers = [] #size of layers
        self.function = ['sigmoid','RelU','LeakyReLU'] #functions to be written later
        self.funcType = funcType # it's only "sigmoid" for now
        self.alpha = 0.01 #learning rate
    
    def show(self,window,instance,iterate,out):
        #self,input instance, iterare i, desired output
        image =instance.reshape(self.inputSize,self.inputSize,1)
        image2 = cv2.resize(image,(280,280)) #just for convenient visualization
        #cv2.putText(image2,str(text[0]),(2,270),cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.putText(image2,str(out[0]),(2,40),cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255)
        cv2.putText(image2,str(iterate),(2,270),cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow(window,image2)
        key = cv2.waitKey(1) & 0xff
        return key

    def initNet(self):
        self.layers = [self.inputSize*self.inputSize]
        self.layers.extend(self.hiddenLayers)
        self.layers.append(self.outputSize)
        self.bias = []
        for i in range(len(self.layers)):
            if i != len(self.layers)-1:
                self.bias.append([0.1])
                self.w.append(np.random.random((self.layers[i]+1)*self.layers[i+1]).reshape(self.layers[i]+1,self.layers[i+1]))
            self.a.append(np.zeros((self.layers[i],1)))
        #for wLayer in self.w:
        #    print(np.shape(wLayer))
        #for aLayer in self.a:
        #    print(np.shape(aLayer))
        
    def neuronFunc(self,x,w,b):
        #vector x and matrix w
        #print(np.shape(x))
        #print(np.shape(w))
        x = np.append([b],x,0)
        if self.funcType == self.function[0]:
            return 1.0/(1+np.exp(-np.matmul(np.transpose(w),x)))
    
    def aPartialDevZ(self,a):
        if self.funcType == self.function[0]:
            return a*(1-a)
        
    
    def forwardPropagate(self,aInput):
        self.a[0] = aInput
        for i in range(1,len(self.a)):
            self.a[i] = self.neuronFunc(self.a[i-1],self.w[i-1],self.bias[i-1])
            #print(self.a[i])
            #print("++++++++++++++++++++++++++++")
    def backPropagate(self,y):
        devJ2a = 2*(self.a[-1]-y)
        for i in range(len(self.a)-1,0,-1):
            x = np.append([self.bias[i-1]],self.a[i-1],0)
            #print("a[i-1]",np.shape(self.a[i-1]))
            #print("x",np.shape(x))
            #print(np.shape(devJ2a))
            devJ2wMeshG = np.meshgrid(devJ2a * self.aPartialDevZ(self.a[i]), x)
            devJ2w = devJ2wMeshG[0] * devJ2wMeshG[1]
            #print("dwvJ2w",np.shape(devJ2w))
            devJ2a = np.matmul(self.w[i-1],devJ2a * self.aPartialDevZ(self.a[i]))[1:]
            self.w[i-1] -= self.alpha * devJ2w
            #print("here",len(self.a),len(self.bias),len(self.w),i)
        
    def trainRecursion(self):
        #gradient descent method
        #self.initNet()
        window = 'Training'
        trD = gzip.open(self.trainData,'r')
        trL = gzip.open(self.trainLabels,'r')
        trD.read(16)
        trL.read(8)
        length = self.inputSize**2
        i, error, key = 0,0,255
        #print("here",len(self.a),len(self.bias),len(self.w))
        while i<60000:
            instance = np.frombuffer(trD.read(length),'uint8').reshape(length,1)
            self.forwardPropagate(instance)
            y10 = np.frombuffer(trL.read(1),'uint8')
            y = np.zeros((self.outputSize,1))
            y[y10] = 1
            #print(y)
            error += sum((self.a[-1]-y)**2)
            self.backPropagate(y)
            #uncomment to see training samples with opencv (performance warning)
            #key = self.show(window,instance,i,y10) 
            if key == 27:
                break
            elif key != 255:
                key = cv2.waitKey(0) & 0xff
            if key == 27:
                break
            i += 1
        cv2.destroyWindow(window)
        error /= i
        #print('error',error)
        #print("here",len(self.a),len(self.bias),len(self.w))

        trD.close()
        trL.close()
        return error
        
    def train(self):
        self.initNet()
        error = [100]
        t1 = time()
        for i in range(1,1000):      
            err=self.trainRecursion()[0]
            if i%1 == 0:
                print("iteration={}, error={}%, processing time={}sec".format(i,int(err*100+0.5)/10.0,int(1000*(time()-t1)+0.5)/1000.0))
                t1 = time()
            if error[-1]-err<0.0000000000001 or err<0.01:
            #if err<0.5 or error[-1]==err:
                error.append(err)
                error.pop(0)
                break
            else:
                error.append(err)
        print(i,error)
        
    def test(self):
        window = 'Testing'
        tsD = gzip.open(self.testData,'r')
        tsL = gzip.open(self.testLabels,'r')
        tsD.read(16)
        tsL.read(8)
        length = self.inputSize**2
        i, error, key = 0,0,255
        while i<10000:
            instance = np.frombuffer(tsD.read(length),'uint8').reshape(length,1)
            self.forwardPropagate(instance)
            y10 = np.frombuffer(tsL.read(1),'uint8')
            y = np.zeros((self.outputSize,1))
            y[y10] = 1
            #print(y)
            error += sum((self.a[-1]-y)**2)
            key = self.show(window,instance,i,y10)
            if key == 27:
                break
            elif key != 255:
                key = cv2.waitKey(0) & 0xff
            if key == 27:
                break
            i += 1
        cv2.destroyWindow(window)
        error /= i
        #print('error',error)
        #print("here",len(self.a),len(self.bias),len(self.w))

        tsD.close()
        tsL.close()
        return error

def main():
    trainingData = 'train-images-idx3-ubyte.gz'
    trainingLabels = 'train-labels-idx1-ubyte.gz'
    testData = 't10k-images-idx3-ubyte.gz'
    testLabels = 't10k-labels-idx1-ubyte.gz'
    funcType = 'sigmoid'
    hiddenLayers = [30,15] #two hidden layers are used
    #hiddenLayers = [100,30,15] #three hidden layers are used
    mnist = NeuralNets(hiddenLayers,trainingData,trainingLabels,testData,testLabels,funcType) 
    mnist.train()
    errorTest = mnist.test()
    print("Test Error = {}%".format(int(100*errorTest+0.5)/10.0))
    print("Accuracy = {}%".format(int(100*(10-errorTest)+0.5)/10.0))
    return mnist

if __name__ == "__main__":
    t0 = time()
    mnistObj = main()
    print("Total Processing Time = {}sec".format(int(1000*(time()-t0)+0.5)/1000.0))
