
import numpy as np
import pandas as pd




def load_data(path, header):
    df = pd.read_csv(path, header=header, delimiter="\t")
    return df


def sigmoid(x):
    return (1/(1+np.exp(-x)))

def dsigmoid(x):
    return x*(1-x)




xtrain=load_data('train_dataset.tsv',None)





ytrain=np.array(xtrain.pop(1000))







class LogisticRegression:
    
    def fit(self,x,y,lr=0.01,max_iters=5000,tol=0.0005):
    #随机生成w矩阵 bias
        self.y=y
        w=np.random.randn(x.shape[-1]+1,1)
        bias = np.ones(x.shape[0]).T
        bias = bias.reshape(len(bias),1)   
        x = np.hstack((bias, x))
        for i in range(max_iters):
            pred = sigmoid(w.T.dot(x.T))
            delta=(y-pred)*dsigmoid(pred)
            w+=lr*x.T.dot(delta.T)/x.T.shape[0]
            loss=abs(np.mean(y-pred))
            if i%50==0:
                print('iters:%d,loss:%f'%(i,loss))
            if loss<tol:
                break
        self.w=w        
        print('训练完毕')
        self.predict(x)
    
    def predict(self,x):
        pred=sigmoid(self.w.T.dot(x.T))
        pred=np.where(pred<0.5,0,1)
        acc=np.mean(pred==self.y)
        print('acc:',acc)
        return pred
    def save_parameters(self,path):
        weights=pd.DataFrame(self.w)
        weights.to_csv(path,header=None,sep='\t')
        print('保存完毕')


if __name__ == '__main__':



	model=LogisticRegression()





	model.fit(xtrain,ytrain)




	model.save_parameters('d:/weights.tsv')







